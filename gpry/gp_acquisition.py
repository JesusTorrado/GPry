import tempfile
import warnings
from copy import deepcopy
from functools import partial
from time import time
import numpy as np
import scipy.optimize
from sklearn.base import is_regressor

from gpry.acquisition_functions import LogExp, NonlinearLogExp
from gpry.acquisition_functions import is_acquisition_function
from gpry.proposal import PartialProposer, CentroidsProposer, UniformProposer
from gpry.mpi import mpi_comm, mpi_rank, is_main_process, \
    split_number_for_parallel_processes, multi_gather_array, sync_processes
from gpry.tools import check_random_state, NumpyErrorHandling


class GPAcquisition(object):
    """Run Gaussian Process acquisition.

    Works similarly to a GPRegressor but instead of optimizing the kernel's
    hyperparameters it optimizes the Acquisition function in order to find one
    or multiple points at which the likelihood/posterior should be evaluated
    next.

    Furthermore contains a framework for different lying strategies in order to
    improve the performance if multiple processors are available

    Use this class directly if you want to control the iterations of your
    bayesian quadrature loop.

    Parameters
    ----------
    bounds : array
        Bounds in which to optimize the acquisition function,
        assumed to be of shape (d,2) for d dimensional prior

    proposer : Proposer object, optional (default: "ParialProposer", producing a mixture
        of points drawn from an "UniformProposer" and from a "CentroidsProposer")
        Proposes points from which the acquisition function should be optimized.

    acq_func : GPry Acquisition Function, optional (default: "LogExp")
        Acquisition function to maximize/minimize. If none is given the
        `LogExp` acquisition function will be used

    acq_optimizer : string or callable, optional (default: "auto")
        Can either be one of the internally supported optimizers for optimizing
        the acquisition function, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_guess, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_guess': the initial value for X, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of X
                ....
                # Returned are the best found X and
                # the corresponding value of the target function.
                return X_opt, func_min

        if set to 'auto' either the 'fmin_l_bfgs_b' or 'sampling' algorithm
        from scipy.optimize is used depending on whether gradient information
        is available or not.

        .. note::
            The default optimizers are designed to **maximize** the acquisition
            function.

    preprocessing_X : X-preprocessor, Pipeline_X, optional (default: None)
        Single preprocessor or pipeline of preprocessors for X. Preprocessing
        makes sense if the scales along the different dimensions are vastly
        different which means that the optimizer struggles to find the maximum
        of the acquisition function. If None is passed the data is not
        preprocessed.

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the maximum of the
        acquisition function. The first run of the optimizer is performed from
        the last X fit to the model if available, otherwise it is drawn at
        random.

        The remaining ones (if any) from X's sampled uniform randomly
        from the space of allowed X-values. Note that n_restarts_optimizer == 0
        implies that one run is performed.

    random_state : int or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    zeta_scaling : float, optional (default: 1.1)
        The scaling of the acquisition function's zeta parameter with dimensionality
        (Only if "LogExp" is passed as acquisition_function)

    zeta: float, optional (default: None, uses zeta_scaling)
        Specifies the value of the zeta parameter directly.

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.

    Attributes
    ----------
    gpr_ : GaussianProcessRegressor
            The GP Regressor which is currently used for optimization.

    """

    def __init__(self, bounds,
                 proposer=None,
                 acq_func="LogExp",
                 acq_optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0,
                 n_repeats_propose=0,
                 preprocessing_X=None,
                 random_state=None,
                 zeta_scaling=0.85,
                 zeta=None,
                 verbose=1):

        self.bounds = bounds
        self.n_d = np.shape(bounds)[0]
        self.proposer = proposer
        self.obj_func = None

        # If nothing is provided for the proposal, we use a centroids proposer with
        # a fraction of uniform samples.
        if self.proposer is None:
            self.proposer = PartialProposer(self.bounds, CentroidsProposer(self.bounds))
        else:
            # TODO: Catch error if it's not the right instance
            self.proposer = proposer

        if is_acquisition_function(acq_func):
            self.acq_func = acq_func
        elif acq_func == "LogExp":
            # If the LogExp acquisition function is chosen it's zeta is set
            # automatically using the dimensionality of the prior.
            self.acq_func = LogExp(
                dimension=self.n_d, zeta=zeta, zeta_scaling=zeta_scaling)
        elif acq_func == "NonlinearLogExp":
            # If the LogExp acquisition function is chosen it's zeta is set
            # automatically using the dimensionality of the prior.
            self.acq_func = NonlinearLogExp(
                dimension=self.n_d, zeta=zeta, zeta_scaling=zeta_scaling)
        else:
            raise TypeError("acq_func needs to be an Acquisition_Function or "
                            f"'LogExp' or 'NonlinearLogExp', instead got {acq_func}")

        # Configure optimizer
        # decide optimizer based on gradient information
        if acq_optimizer == "auto":
            if self.acq_func.hasgradient:
                self.acq_optimizer = "fmin_l_bfgs_b"
            else:
                self.acq_optimizer = "sampling"

        elif isinstance(acq_optimizer, str):
            if acq_optimizer == "fmin_l_bfgs_b":
                if not self.acq_func.hasgradient:
                    raise ValueError(
                        "In order to use the 'fmin_l_bfgs_b' "
                        "optimizer the acquisition function needs to be able "
                        "to return gradients. Got %s" % self.acq_func)
                self.acq_optimizer = "fmin_l_bfgs_b"
            elif acq_optimizer == "sampling":
                self.acq_optimizer = "sampling"
            else:
                raise ValueError("Supported internal optimizers are 'auto', "
                                 "'lbfgs' or 'sampling', "
                                 "got {0}".format(acq_optimizer))
        else:
            self.acq_optimizer = acq_optimizer

        self.n_restarts_optimizer = n_restarts_optimizer
        self.n_repeats_propose = n_repeats_propose

        self.preprocessing_X = preprocessing_X

        self.verbose = verbose

        self.mean_ = None
        self.cov = None

    def __call__(self, X, gpr, eval_gradient=False):
        """Returns the value of the acquision function at ``X`` given a ``gpr``."""
        return self.acq_func(X, gpr, eval_gradient=eval_gradient)

    def optimize_acquisition_function(self, gpr, i, random_state=None):
        """Exposes the optimization method for the acquisition function. When
        called it proposes a single point where for where to evaluate the true
        model next. It is internally called in the :meth:`multi_add` method.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            The GP Regressor which is used as surrogate model.

        i : int
            Internal counter which is used to enable MPI support. If you want
            to optimize from a single location and rerun the optimizer from
            multiple starting locations loop over this parameter.

        random_state : int or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.

        Returns
        -------
        X_opt : numpy.ndarray, shape = (X_dim,)
            The X value of the found optimum
        func : float
            The value of the acquisition function at X_opt
        """
        # Update proposer with new gpr
        self.proposer.update(gpr)

        # If we do a first-time run, use this
        if not self.obj_func:

            # Check whether gpr is a GP regressor
            if not is_regressor(gpr):
                raise ValueError("surrogate model has to be a GP Regressor. "
                                 "Got %s instead." % gpr)

            # Check whether the GP has been fit to data before
            if not hasattr(gpr, "X_train_"):
                raise AttributeError(
                    "The model which is given has not been fed "
                    "any points. Please make sure, that the model already "
                    "contains data when trying to optimize an acquisition "
                    "function on it as optimizing priors is not supported yet.")

            def obj_func(X, eval_gradient=False):

                # TODO: optionally suppress this checks if called by optimiser
                # Check inputs
                X = np.asarray(X)
                X = np.expand_dims(X, axis=0)
                if X.ndim != 2:
                    raise ValueError("X is {}-dimensional, however, "
                                     "it must be 2-dimensional.".format(X.ndim))
                if self.preprocessing_X is not None:
                    X = self.preprocessing_X.inverse_transform(X)

                if eval_gradient:
                    acq, grad = self.acq_func(X, gpr, eval_gradient=True)
                    return -1 * acq, -1 * grad
                else:
                    return -1 * self.acq_func(X, gpr, eval_gradient=False)

            self.obj_func = obj_func

        # Preprocessing
        if self.preprocessing_X is not None:
            transformed_bounds = self.preprocessing_X.transform_bounds(
                self.bounds)
        else:
            transformed_bounds = self.bounds

        if i == 0:
            # Perform first run from last training point
            x0 = gpr.X_train[-1]
            if self.preprocessing_X is not None:
                x0 = self.preprocessing_X.transform(x0)
            return self._constrained_optimization(self.obj_func, x0,
                                                  transformed_bounds)
        else:
            n_tries = 10 * self.bounds.shape[0] * self.n_restarts_optimizer
            x0s = np.empty((self.n_repeats_propose + 1, self.bounds.shape[0]))
            values = np.empty(self.n_repeats_propose + 1)
            ifull = 0
            for n_try in range(n_tries):
                x0 = self.proposer.get(random_state=random_state)
                value = self.acq_func(x0, gpr)
                if not np.isfinite(value):
                    continue
                x0s[ifull] = x0
                values[ifull] = value
                ifull += 1
                if ifull > self.n_repeats_propose:
                    x0 = x0s[np.argmax(values)]
                    if self.preprocessing_X is not None:
                        x0 = self.preprocessing_X.transform(x0)
                    return self._constrained_optimization(self.obj_func, x0,
                                                          transformed_bounds)
            # if there's at least one finite value try optimizing from
            # there, otherwise take the last x0 and add that to the GP
            if ifull > 0:
                x0 = x0s[np.argmax(values[:ifull])]
                if self.preprocessing_X is not None:
                    x0 = self.preprocessing_X.transform(x0)
                return self._constrained_optimization(self.obj_func, x0,
                                                      transformed_bounds)
            else:
                if self.verbose > 1:
                    print(f"of {n_tries} initial samples for the "
                          "acquisition optimizer none returned a "
                          "finite value")
                if self.preprocessing_X is not None:
                    x0 = self.preprocessing_X.transform(x0)
                return x0, -1 * value

    def multi_add(self, gpr, n_points=1, random_state=None):
        r"""Method to query multiple points where the objective function
        shall be evaluated. The strategy which is used to query multiple
        points is by using the :math:`f(x)\sim \mu(x)` strategy and and not
        changing the hyperparameters of the model.

        This is done to increase speed since then the blockwise matrix
        inversion lemma can be used to invert the K matrix. The optimization
        for a single point is done using the :meth:`optimize_acq_func` method.

        When run in parallel (MPI), returns the same values for all processes.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            The GP Regressor which is used as surrogate model.

        n_points : int, optional (default=1)
            Number of points returned by the optimize method
            If the value is 1, a single point to evaluate is returned.

            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective
            in parallel, and thus obtain more objective function evaluations
            per unit of time.

        random_state : int or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.

        Returns
        -------
        X : numpy.ndarray, shape = (X_dim, n_points)
            The X values of the found optima
        y_lies : numpy.ndarray, shape = (n_points,)
            The predicted values of the GP at the proposed sampling locations
        fval : numpy.ndarray, shape = (n_points,)
            The values of the acquisition function at X_opt
        """
        # Check if n_points is positive and an integer
        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )
        if is_main_process:
            # Initialize arrays for storing the optimized points
            X_opts = np.empty((n_points,
                               gpr.d))
            y_lies = np.empty(n_points)
            acq_vals = np.empty(n_points)
            # Copy the GP instance as it is modified during
            # the optimization. The GP will be reset after the
            # Acquisition is done.
            gpr_ = deepcopy(gpr)
        gpr_ = mpi_comm.bcast(gpr_ if is_main_process else None)
        n_acq_per_process = \
            split_number_for_parallel_processes(self.n_restarts_optimizer)
        n_acq_this_process = n_acq_per_process[mpi_rank]
        i_acq_this_process = sum(n_acq_per_process[:mpi_rank])
        proposal_X = np.empty((n_acq_this_process, gpr_.d))
        acq_X = np.empty((n_acq_this_process,))
        for ipoint in range(n_points):
            # Optimize the acquisition function to get a few possible next proposal points
            # (done in parallel)
            for i in range(n_acq_this_process):
                proposal_X[i], acq_X[i] = self.optimize_acquisition_function(
                    gpr_, i + i_acq_this_process, random_state=random_state)
            proposal_X_main, acq_X_main = multi_gather_array(
                [proposal_X, acq_X])
            # Reset the objective function, such that afterwards the correct one is used
            self.obj_func = None
            # Now take the best and add it to the gpr (done in sequence)
            if is_main_process:
                # Find out which one of these is the best
                max_pos = np.argmin(acq_X_main) if np.any(
                    np.isfinite(acq_X_main)) else len(acq_X_main) - 1
                X_opt = proposal_X_main[max_pos]
                # Transform X and clip to bounds
                if self.preprocessing_X is not None:
                    X_opt = self.preprocessing_X.inverse_transform(X_opt,
                                                                   copy=True)

                # Get the value of the acquisition function at the optimum value
                acq_val = -1 * acq_X_main[max_pos]
                X_opt = np.array([X_opt])
                # Get the "lie" (prediction of the GP at X)
                y_lie = gpr_.predict(X_opt)
                # Try to append the lie to change uncertainties (and thus acq func)
                # (no need to append if it's the last iteration)
                if ipoint < n_points - 1:
                    # Take the mean of errors as supposed measurement error
                    if np.iterable(gpr_.noise_level):
                        lie_noise_level = np.array(
                            [np.mean(gpr_.noise_level)])
                        # Add lie to GP
                        gpr_.append_to_data(
                            X_opt, y_lie, noise_level=lie_noise_level,
                            fit=False)
                    else:
                        # Add lie to GP
                        gpr_.append_to_data(X_opt, y_lie, fit=False)
                # Append the points found to the array
                X_opts[ipoint] = X_opt[0]
                y_lies[ipoint] = y_lie[0]
                acq_vals[ipoint] = acq_val
            # Send this new gpr_ instance to all mpi
            gpr_ = mpi_comm.bcast(gpr_ if is_main_process else None)
        gpr.n_eval = gpr_.n_eval  # gather #evals of the GP, for cost monitoring
        return mpi_comm.bcast(
            (X_opts, y_lies, acq_vals) if is_main_process else (None, None, None))

    def _constrained_optimization(self, obj_func, initial_X, bounds):

        if self.acq_optimizer == "fmin_l_bfgs_b":
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            opt_res = scipy.optimize.fmin_l_bfgs_b(
                obj_func, initial_X, args={"eval_gradient": True},
                bounds=bounds, approx_grad=False)
            theta_opt, func_min = opt_res[0], opt_res[1]
        elif self.acq_optimizer == "sampling":
            opt_res = scipy.optimize.minimize(
                obj_func, initial_X, args=(False), method="Powell",
                bounds=bounds)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.acq_optimizer):
            theta_opt, func_min = \
                self.acq_optimizer(obj_func, initial_X, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.acq_optimizer)

        return theta_opt, func_min

    def _has_already_been_sampled(self, gpr, new_X):
        """
        Method for determining whether points which have been found by the
        acquisition algorithm are already in the GP. This is called from the
        optimize_acq_func method to determine whether points may have been
        sampled multiple times which could break the GP.
        This is meant to be used for a **single** point new_X.
        """
        X_train = np.copy(gpr.X_train)

        for i, xi in enumerate(X_train):
            if np.allclose(new_X, xi):
                if self.verbose > 1:
                    warnings.warn("A point has been sampled multiple times. "
                                  "Excluding this.")
                return True
        return False


class Griffins(GPAcquisition):
    """
    Run Gaussian Process acquisition with Griffins.

    Uses krigging believer while it samples the acquisition function using nested
    sampling (with PolyChord).

    Parameters
    ----------
    bounds : array
        Bounds in which to optimize the acquisition function,
        assumed to be of shape (d,2) for d dimensional prior

    acq_func : GPry Acquisition Function, optional (default: "LogExp")
        Acquisition function to maximize/minimize. If none is given the
        `LogExp` acquisition function will be used

    mc_every : int
        If >1, only calls the MC sampler every `mc_steps`, and reuses previous X
        otherwise, recomputing y and sigma with the new GPR.

    random_state : int or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    zeta_scaling : float, optional (default: 1.1)
        The scaling of the acquisition function's zeta parameter with dimensionality
        (Only if "LogExp" is passed as acquisition_function)

    zeta: float, optional (default: None, uses zeta_scaling)
        Specifies the value of the zeta parameter directly.

    nlive_per_training: int
        live points per sample in the current training set.
        Not recommended to decrease it.

    nlive_per_dim_min: int
        live points min cap (times dimension).

    num_repeats_per_dim: int
        length of slice-chains times dimension.

    precision_criterion_target: float
        Cap on precision criterion of PolyChord.

    nprior_per_nlive: int
        Number of initial samples times dimension.

    preprocessing_X : X-preprocessor, Pipeline_X, optional (default: None)
        Single preprocessor or pipeline of preprocessors for X. Preprocessing
        makes sense if the scales along the different dimensions are vastly
        different which means that the optimizer struggles to find the maximum
        of the acquisition function. If None is passed the data is not
        preprocessed.

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.

    Attributes
    ----------
    gpr_ : GaussianProcessRegressor
            The GP Regressor which is currently used for optimization.

    """

    def __init__(self, bounds,
                 acq_func="LogExp",
                 mc_every=1,
                 preprocessing_X=None,
                 random_state=None,
                 zeta_scaling=0.85,
                 zeta=None,
                 nlive_per_training=3,
                 nlive_per_dim_min=40,
                 num_repeats_per_dim=8,
                 precision_criterion_target=0.005,
                 nprior_per_nlive=25,
                 verbose=1):
        try:
            # pylint: disable=import-outside-toplevel
            from pypolychord.settings import PolyChordSettings
            from pypolychord.priors import UniformPrior
        except ImportError as excpt:
            raise ImportError(
                "PolyChord needs to be installed to use this acquirer.") from excpt
        self.bounds = np.array(bounds)
        self.n_d = np.shape(bounds)[0]
        self.preprocessing_X = preprocessing_X
        self.verbose = verbose
        self.mc_every = int(mc_every)
        self.mc_every_i = 0
        if is_acquisition_function(acq_func):
            self.acq_func = acq_func
        elif acq_func == "LogExp":
            # If the LogExp acquisition function is chosen it's zeta is set
            # automatically using the dimensionality of the prior.
            self.acq_func = LogExp(
                dimension=self.n_d, zeta=zeta, zeta_scaling=zeta_scaling)
        elif acq_func == "NonlinearLogExp":
            # If the LogExp acquisition function is chosen it's zeta is set
            # automatically using the dimensionality of the prior.
            self.acq_func = NonlinearLogExp(
                dimension=self.n_d, zeta=zeta, zeta_scaling=zeta_scaling)
        else:
            raise TypeError("acq_func needs to be an Acquisition_Function or "
                            f"'LogExp' or 'NonlinearLogExp', instead got {acq_func}")
        self.acq_func_y_sigma = partial(self.acq_func.f, zeta=self.acq_func.zeta)
        # Configure nested sampler
        self.polychord_settings = PolyChordSettings(nDims=self.n_d, nDerived=0)
        # Don't write unnecessary files: take lots of space and waste time
        self.polychord_settings.read_resume = False
        self.polychord_settings.write_resume = False
        self.polychord_settings.write_live = False
        self.polychord_settings.write_dead = True
        self.polychord_settings.write_prior = True
        self.polychord_settings.feedback = verbose
        # Using rng state as seed for PolyChord
        if random_state is not None:
            self.polychord_settings.seed = \
                random_state.bit_generator.state["state"]["state"] + mpi_rank
        # Prepare precision parameters
        self.nlive_per_training = nlive_per_training
        self.nlive_per_dim_min = nlive_per_dim_min
        self.num_repeats_per_dim = num_repeats_per_dim
        self.precision_criterion_target = precision_criterion_target
        self.nprior_per_nlive = nprior_per_nlive
        self.prior = UniformPrior(*self.bounds.T)
        # Pool for storing intermediate results during parallelised acquisition
        self.pool = None
        self.last_polychord_output = None

    @property
    def pool_size(self):
        """Size of the pool of points."""
        if self.pool is None:
            return None
        return len(self.pool)

    def update_NS_precision(self, gpr):
        """
        Updates NS (PolyChord) precision parameters:
        - num_repeats: constant for now
        - nlive: `nlive_per_training` times the size of the training set, capped at
            `nlive_per_dim_cap` (typically 25) times the dimension.
        - precision_criterion: takes a line that passes through some
            (log_max_preccrit, max_logKL) and some (log_min_preccrit, min_logKL)
            and interpolates for the exponential running mean of the logKL's
        """
        self.polychord_settings.nlive = min(
            self.nlive_per_training * gpr.n,
            self.nlive_per_dim_min * self.n_d)
        self.polychord_settings.num_repeats = self.num_repeats_per_dim * self.n_d
        self.polychord_settings.precision_criterion = self.precision_criterion_target
        self.polychord_settings.nprior = \
            int(self.nprior_per_nlive * self.polychord_settings.nlive)

    def log(self, msg, level=None):
        """
        Print a message if its verbosity level is equal or lower than the given one (or
        always if ``level=None``.
        """
        if level is None or level <= self.verbose:
            print(msg)

    def get_MC_sample(self, gpr, random_state=None, sampler="polychord"):
        """

        Returns
        -------
        X, y, sigma_y
            May return None for any of y, sigma_y
        """
        if sampler.lower() == "uniform":
            return self._get_MC_sample_uniform(gpr, random_state)
        if sampler.lower() == "polychord":
            return self._get_MC_sample_polychord(gpr, random_state)
        raise ValueError(f"Sampler '{sampler}' not known.")

    def _get_MC_sample_uniform(self, gpr, random_state):
        if not is_main_process:
            return None, None, None
        proposer = UniformProposer(self.bounds)
        n_total = 8 * gpr.d
        X = np.empty(shape=(n_total, gpr.d))
        for i in range(n_total):
            X[i] = proposer.get(random_state=random_state)
        return X, None, None

    def _get_MC_sample_polychord(self, gpr, random_state):

        # Initialise "likelihood" -- returns GPR value and deals with pooling/ranking
        def logp(X):
            """
            Returns the predicted value at a given point (-inf if prior=0).
            """
            return gpr.predict(np.atleast_2d(X), return_std=False, validate=False)[0], []

        # Update PolyChord precision settings
        self.update_NS_precision(gpr)
        from pypolychord import run_polychord  # pylint: disable=import-outside-toplevel
        if is_main_process:
            # TODO: add to checkpoint folder?
            tmpdir = tempfile.TemporaryDirectory().name
            # ALT: persistent folder:
            # tmpdir = tempfile.mkdtemp()
            self.polychord_settings.base_dir = tmpdir
            self.polychord_settings.file_root = "test"
        with NumpyErrorHandling(all="ignore") as _:
            self.last_polychord_output = run_polychord(
                logp,
                nDims=self.n_d, nDerived=0,
                settings=self.polychord_settings,
                prior=self.prior)
            dummy_paramnames = [tuple(2 * [f"x_{i + 1}"]) for i in range(gpr.d)]
            self.last_polychord_output.make_paramnames_files(dummy_paramnames)
        if is_main_process:
            prior_T = np.loadtxt(self.last_polychord_output.root + "_prior.txt").T
            X_prior = prior_T[2:].T
            y_prior = - prior_T[1] / 2  # this one is stored as chi2
            dead_T = np.loadtxt(self.last_polychord_output.root + "_dead.txt").T
            X_dead = dead_T[1:].T
            y_dead = dead_T[0]  # this one stores logp
            X = np.concatenate([X_prior, X_dead])
            y = np.concatenate([y_prior, y_dead])
            return X, y, None
        return None, None, None

    def multi_add(self, gpr, n_points=1, random_state=None):
        r"""Method to query multiple points where the objective function
        shall be evaluated.

        The strategy which is used to query multiple points is by using
        the :math:`f(x)\sim \mu(x)` strategy and and not changing the
        hyperparameters of the model.

        It runs PolyChord on the mean of the GP model, tracking the value
        of the acquisition function at every evaluation, and keeping a
        pool of candidates which is re-sorted whenever a new good candidate
        is found.

        When run in parallel (MPI), returns the same values for all processes.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            The GP Regressor which is used as surrogate model.

        n_points : int, optional (default=1)
            Number of points returned by the optimize method
            If the value is 1, a single point to evaluate is returned.

            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective
            in parallel, and thus obtain more objective function evaluations
            per unit of time.

        random_state : int or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.

        Returns
        -------
        X : numpy.ndarray, shape = (X_dim, n_points)
            The X values of the found optima
        y_lies : numpy.ndarray, shape = (n_points,)
            The predicted values of the GP at the proposed sampling locations
        fval : numpy.ndarray, shape = (n_points,)
            The values of the acquisition function at X_opt
        """
        # Check if n_points is positive and an integer
        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(f"n_points should be int > 0, got {n_points}")
        # Gather an MC sample
        if is_main_process:
            start_sample = time()
        if self.mc_every_i % self.mc_every:  # reuse
            self.X, self.y, self.sigma_y = self.X, None, None
            # TODO: implement reweighting!
        else:
            self.X, self.y, self.sigma_y = self.get_MC_sample(
                gpr, random_state, sampler="polychord")
        self.mc_every_i += 1
        # Split for parallel processes (full arrays passed: a bit of overhead)
        X = mpi_comm.bcast(self.X)
        n_per_process = split_number_for_parallel_processes(len(X))
        n_this_process = n_per_process[mpi_rank]
        i_this_process = sum(n_per_process[:mpi_rank])
        this_X = X[i_this_process: i_this_process + n_this_process]
        y = mpi_comm.bcast(self.y)
        sigma_y = mpi_comm.bcast(self.sigma_y)
        if y is None:  # assume sigma_y is also None
            if len(this_X) > 0:
                this_y, this_sigma_y = gpr.predict(this_X, return_std=True, validate=False)
            else:
                this_y, this_sigma_y = np.array([], dtype=float), np.array([], dtype=float)
            self.y, self.sigma_y = multi_gather_array([this_y, this_sigma_y])
        elif sigma_y is None:
            this_y = y[i_this_process: i_this_process + n_this_process]
            if len(this_y) > 0:
                _, this_sigma_y = gpr.predict(this_X, return_std=True, validate=False)
                # assert np.allclose(this_y, _)
            else:
                this_sigma_y = np.array([], dtype=float)
            self.sigma_y = multi_gather_array(this_sigma_y)[0]
        else:
            this_y = y[i_this_process: i_this_process + n_this_process]
            this_sigma_y = sigma_y[i_this_process: i_this_process + n_this_process]
        sync_processes()
        if is_main_process:
            self.log(f"[griffins] MC sampling took {time()-start_sample} seconds")
        # Rank to get best points
        # The size of the pool should be at least the amount of points to be acquired.
        # If running several processes in parallel, it can be reduced down to the number
        #   of points to be evaluated per process, but with less guarantee to find an
        #   optimal set.
        if is_main_process:
            start_rank = time()
        self.pool = RankedPool(n_points, gpr=gpr, acq_func=self.acq_func_y_sigma)
        this_acq = self.acq_func_y_sigma(this_y, this_sigma_y)
        for X_i, y_i, sigma_y_i, acq_i in zip(this_X, this_y, this_sigma_y, this_acq):
            self.pool.add_one(X_i, y_i, sigma_y_i, acq_i)
        sync_processes()
        if is_main_process:
            self.log(f"[griffins] Ranking took {time()-start_rank} seconds")
        # MPI-merge and return
        return self.get_optimal_locations(n_points)

    def multi_add_OLD_WORKING(self, gpr, n_points=1, random_state=None):
        r"""Method to query multiple points where the objective function
        shall be evaluated.

        The strategy which is used to query multiple points is by using
        the :math:`f(x)\sim \mu(x)` strategy and and not changing the
        hyperparameters of the model.

        It runs PolyChord on the mean of the GP model, tracking the value
        of the acquisition function at every evaluation, and keeping a
        pool of candidates which is re-sorted whenever a new good candidate
        is found.

        When run in parallel (MPI), returns the same values for all processes.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            The GP Regressor which is used as surrogate model.

        n_points : int, optional (default=1)
            Number of points returned by the optimize method
            If the value is 1, a single point to evaluate is returned.

            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective
            in parallel, and thus obtain more objective function evaluations
            per unit of time.

        random_state : int or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.

        Returns
        -------
        X : numpy.ndarray, shape = (X_dim, n_points)
            The X values of the found optima
        y_lies : numpy.ndarray, shape = (n_points,)
            The predicted values of the GP at the proposed sampling locations
        fval : numpy.ndarray, shape = (n_points,)
            The values of the acquisition function at X_opt
        """
        # Check if n_points is positive and an integer
        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(f"n_points should be int > 0, got {n_points}")
        # Reset pool of points to be computed
        # The size of the pool should be at least the amount of points to be acquired.
        # If running several processes in parallel, it can be reduced down to the number
        #   of points to be evaluated per process, but with less guarantee to find an
        #   optimal set.
        self.pool = RankedPool(n_points, gpr=gpr, acq_func=self.acq_func_y_sigma)

        # Initialise "likelihood" -- returns GPR value and deals with pooling/ranking
        def logp(X):
            """
            Returns the predicted value at a given point (-inf if prior=0).
            """
            X = np.atleast_2d(X)
            logp_pred, sigma_pred = gpr.predict(X, return_std=True, validate=False)
            if sigma_pred[0] > 0:
                self.pool.add_one(X, logp_pred[0], sigma_pred[0])
            return logp_pred[0], []

        # Update PolyChord precision settings
        self.update_NS_precision(gpr)
        from pypolychord import run_polychord  # pylint: disable=import-outside-toplevel
        with NumpyErrorHandling(all="ignore") as _:
            self.last_polychord_output = run_polychord(
                logp,
                nDims=self.n_d, nDerived=0,
                settings=self.polychord_settings,
                prior=self.prior)
        dummy_paramnames = [tuple(2 * [f"x_{i + 1}"]) for i in range(gpr.d)]
        self.last_polychord_output.make_paramnames_files(dummy_paramnames)
        try:
            self.mean = self.last_polychord_output.posterior.means
            self.cov = self.last_polychord_output.posterior.cov()
        except ValueError:
            self.mean = None
            self.cov = None
        return self.get_optimal_locations(n_points)

    def mpi_gather_pools(self):
        """
        Merges the points in all pools, discarding the empty ones.

        rank-0 process returns [X, y, sigma]
        """
        # Acq value is conditional per se, but can be used as an indicator of whether a
        # position in the pool is filled, if != nan
        pool_X = mpi_comm.gather(self.pool.X[:len(self.pool)])
        pool_y = mpi_comm.gather(self.pool.y[:len(self.pool)])
        pool_sigma = mpi_comm.gather(self.pool.sigma[:len(self.pool)])
        pool_acq = mpi_comm.gather(self.pool.acq[:len(self.pool)])
        if is_main_process:
            # Discard unfilled positions
            # (NB: with PolyChord and #mpi-proceses > 1, the whole rank-0 pool is empty)
            pool_acq = np.concatenate(pool_acq)
            i_notnan = np.invert(np.isnan(pool_acq))
            pool_acq = pool_acq[i_notnan]
            pool_X = np.concatenate(pool_X)[i_notnan]
            pool_y = np.concatenate(pool_y)[i_notnan]
            pool_sigma = np.concatenate(pool_sigma)[i_notnan]
        return pool_X, pool_y, pool_sigma

    def get_optimal_locations(self, n_points):
        """
        Merges the pools of parallel processes to find ``n_points`` optimal locations.

        Returns all these locations for all processes.
        """
        pool_X, pool_y, pool_sigma = self.mpi_gather_pools()
        if is_main_process:
            self.pool.add(pool_X, pool_y, pool_sigma)
        pool = mpi_comm.bcast(self.pool)
        pool_acq = self.acq_func_y_sigma(pool.y[:n_points], pool.sigma[:n_points])
        return pool.X[:n_points], pool.y[:n_points], pool_acq


class RankedPool():
    """
    Keeps a ranked pool of sample proposal for Krigging-believer, given a GP regressor
    and an acquisition function.

    Parameters
    ----------
    size : int
        Number of points sampled proposals targeted.

    gpr : GaussianProcessRegressor
        The GP Regressor which is used as surrogate model.

    acq_func : callable
        Acquisition function used to rank the pool. Must be a function of ``(y, sigma)``
        only, partially evaluated its hyperparameters, if necessary.

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.
    """

    def __init__(self, size, gpr, acq_func, verbose=1):
        # The pool should have one more element than the number of desired points.
        self.X = np.zeros((size + 1, gpr.d))
        self.y = np.zeros((size + 1))
        self.sigma = np.zeros((size + 1))  # only used for logging purposes
        self.gpr = (size - 1) * [None]
        self.acq = np.full((size + 1), -np.inf)  # -np.inf represents an empty slot
        self._gpr = gpr
        self._acq_func = acq_func
        self.verbose = False

    def __len__(self):
        return len(self.y) - 1

    @property
    def min_acq(self):
        """
        Minimum acquisition function value in order for a point to be considered, i.e.
        conditioned acquisition function value of the last element in the pool.

        NB: while the pool is not yet fool, empty points are assigned minus infinity as
        acquisition function value, so one can still use the condition
        ``acq_value > RankedPool.min_acq`` in order to decide whether to add a point.
        """
        return self.acq[len(self) - 1]

    # TODO: abstract these to a class
    def log(self, msg, level=None):
        """
        Print a message if its verbosity level is equal or lower than the given one (or
        always if ``level=None``.
        """
        if level is None or level <= self.verbose:
            print(msg)

    def str_point(self, X, y, sigma, acq):
        """Retuns a standardised string to log a point."""
        return f"{X}, y = {y} +/- {sigma}; acq = {acq}"

    def log_pool(self, level=4, include_last=False):
        """Prints the current pools."""
        if self.verbose < level:
            return
        for i in range(len(self.X) + (-1 if not include_last else 0)):
            self.log(level=level,
                     msg=(f"{i + 1} : " + self.str_point(
                         self.X[i], self.y[i], self.sigma[i], self.acq[i])))

    def add(self, X, y=None, sigma=None):
        """
        Adds points to the pool. For stability, it sorts them first in order of growing
        acquisition value function.

        Parameters
        ----------
        X: np.ndarray (1- or 2-dimensional)
            Position of the proposed sample.

        y: np.ndarray (1 dimension fewer than X) or float, optional
            Predicted value under the GPR.

        sigma: np.ndarray (1 dimension fewer than X) or float, optional
            Predicted standard deviation under the GPR.
        """
        if len(X.shape) < 2:
            self.add(X, y, sigma)
            return
        if X.shape[1] == 1:
            self.add(X[0], y[0] if y else None, sigma[0] if sigma else None)
            return
        # Multiple points: sort in descending order of acquisition function values before
        # adding them one-by-one, for stability, since the addition order has some
        # influence on the final order.
        if y is None:
            y, sigma = self._gpr.predict(X, return_std=True, validate=False)
        acq = self._acq_func(y, sigma)
        i_sort = np.argsort(acq)[::-1]  # descending order
        X, y, sigma, acq = X[i_sort], y[i_sort], sigma[i_sort], acq[i_sort]
        for X_i, y_i, sigma_i, acq_i in zip(X, y, sigma, acq):
            self.add_one(X_i, y_i, sigma_i, acq_i)

    def add_one(self, X, y=None, sigma=None, acq=None):
        """
        Tries to add one point to the pool:

        1. Computes its acquisition function value.
           (If the pools is not full, just adds it and re-sorts.)

        2. Finds the provisional position of the point in the list, without KB.
           Notice that once KB is taken into account, the KB-ranked position can only be
           lower. (If the acq. fun. value is lower than the last point, it is discarded.)

        3. Updates its aquisition function value KB-informed by the points above, finds
           the new provisional position, and repeats until the position stabilises.

        4. Sorts the list of points below the new one, recursively applying KB.

        Parameters
        ----------
        X: np.ndarray with 1 dimension
            Position of the proposed sample.

        y: float, optional
            Predicted value under the GPR.

        sigma: float, optional
            Predicted standard deviation under the GPR.

        acq: float, optional
            Value of the acquisition function.

        Raises
        ------
        ValueError: if invalid acquisition function value.
        """
        if y is None:
            y, sigma = self._gpr.predict(
                np.atleast_2d(X), return_std=True, validate=False)
            y, sigma = y[0], sigma[0]
        if acq is None:
            acq = self._acq_func(y, sigma)
        if self.verbose >= 4:
            self.log(
                level=4, msg=("[pool.add] Checking point " +
                              self.str_point(X, y, sigma, acq)))
        # Discard the point as early as possible!
        # NB: The equals sign below takes care of the case in which we are trying to add a
        # point with -inf acq. func. to a pool which is not full (min acq. func. = -inf)
        if acq <= self.min_acq:
            self.log(level=4, msg="[pool.add] Acq. func. value too small. Ignoring.")
            return
        if np.isnan(acq):
            raise ValueError(f"Acquisition function value not a number: {acq}")
        self.log(level=4, msg="[pool.add] Initial pool:")
        self.log_pool(level=4)
        # Find its position in the list, conditioned to those on top
        # Shortcut: start from the bottom, ignore last element (just a placeholder)
        i_new_last = len(self)
        acq_cond = acq
        sigma_cond = sigma
        while True:
            try:
                # Notice that we start always from the bottom of the list, in case a point
                # with high acq. func. value is close to one of the top points, and its
                # conditioned acq. func. value drops to a small value after conditioning.
                # The equals sign below prevents -inf's from climbing up the list.
                i_new = (len(self) -
                         next(i for i in range(len(self))
                              if self.acq[-(i + 2)] >= acq_cond))
            except StopIteration:  # top of the list reached
                i_new = 0
            self.log(level=4, msg=f"[pool.add] Provisional position: [{i_new + 1}]")
            # top, same as last or last: final ranking reached
            if i_new in [0, i_new_last, len(self)]:
                break
            # Otherwise, compute conditioned acquisition value, using point above,
            # and continue to the next iteration to re-rank
            sigma_cond = self.gpr[i_new - 1].predict(
                np.atleast_2d(X), return_std=True, validate=False)[1][0]
            # New acquisition should not be higher than the old one, since the new one
            # corresponds to a model with more training points (though fake ones).
            # This may happen anyway bc numerical errors, e.g. when the correlation
            # length is really huge. Also, when alpha or the noise level for two cached
            # models are different. We can just ignore it.
            # (Sometimes, it's just ~1e6 relative differences, which is not worrying)
            acq_cond = min(acq_cond, self._acq_func(y, sigma_cond))
            self.log(
                level=4,
                msg=f"[pool.add] --> updated conditional acquisition: {self.acq[i_new]}")
            i_new_last = i_new
        # The final position is just a place-holder: don't save it if it falls there.
        if i_new >= len(self):
            self.log(level=4, msg="[pool.add] Discarded!")
            return
        self.log(level=4, msg=f"[pool.add] Final position: [{i_new + 1}] of {len(self)}")
        # Insert the new one in its place, and push the rest down one place.
        # We track the acq. value (but not the sigma), to retain the information about
        # whether each slot was empty (acq = -inf)
        # (but not for the acq values, which will be updated below)
        for pool, value in [(self.X, X), (self.y, y), (self.acq, acq_cond)]:
            pool[i_new + 1:] = pool[i_new:-1]
            pool[i_new] = value
        self.sigma[i_new] = sigma_cond  # the ones below will be discarded
        # For the sub-list below the new element: update acquisitions, cache and sort
        if i_new < len(self) - 1:
            # If not in the last position, we can safely assume that it has finite value,
            # since -inf's from conditional acq cannot climb.
            assert self.acq[i_new] > -np.inf
            self.cache_model(i_new)
            # Corner case: list not full: next element is -inf. Do not recompute acq.
            sigmas_cond = self.gpr[i_new].predict(
                self.X[i_new + 1:], return_std=True, validate=False)[1]
            # Again, cond acq cannot be higher then less cond one.
            # In particular, keep -inf if it was so
            self.acq[i_new + 1:] = np.clip(
                self._acq_func(self.y[i_new + 1:], sigmas_cond),
                a_min=None, a_max=self.acq[i_new + 1:])
        self.log(level=4, msg="[pool.add] Current unsorted pool:")
        self.log_pool(level=4)
        self.sort(i_new + 1)
        self.log(level=4, msg="[pool.add] The new pool, sorted:")
        self.log_pool(level=4)

    def cache_model(self, i):
        """
        Cache the GP model that contains the training set
        plus the pool points up to position ``i``, with predicted dummy y,
        keeping the GPR hyperparameters unchanged.
        """
        self.log(level=4, msg=f"[pool.cache] Caching model [{i + 1}]")
        self.gpr[i] = deepcopy(self._gpr)
        # NB: old code contains a loop to increasingly add noise during this "fit"
        #     if needed (doesn't matter too much in an augmented model)
        self.gpr[i].append_to_data(
            self.X[:i + 1], self.y[:i + 1], fit=False)

    def sort(self, i_start=0):
        """
        Sorts in descending order of acquisition function value, where the acq of the
        ``i``-th element is conditioned on the GPR model that includes the points with
        j<i with their predicted (mean) y.

        If ``i_start!=0`` is given, assumes the upper elements in the list are already
        sorted following this criterion, and in particular that all acquisition function
        values in ``self.acq`` from ``i_start`` down are already computed taking into
        account the points above ``i_start``.
        """
        self.log(
            level=4,
            msg=f"[pool.sort] Sorting the pool starting at [{i_start + 1}]",
        )
        for i_this in range(i_start, len(self)):
            # Find next best point, and move it to next slot
            # Indices `j` refer to the sublist from `i_this` down.
            j_max = np.argmax(self.acq[i_this:])
            i_new_max = i_this + j_max
            # If the max found was -inf, then fill all below with -inf
            if self.acq[i_new_max] == -np.inf:
                self.acq[i_this:] = -np.inf
                break
            if i_this != i_new_max:
                self.log(
                    level=4,
                    msg=f"[pool.sort] Swapping [{i_this + 1}] and [{i_new_max + 1}]",
                )
                self.X[[i_this, i_new_max]] = self.X[[i_new_max, i_this]]
                self.y[[i_this, i_new_max]] = self.y[[i_new_max, i_this]]
                self.acq[i_this] = self.acq[i_new_max]
            # Update model and recompute acq values below it
            if i_this < len(self) - 1:
                self.cache_model(i_this)
                sigmas_cond = self.gpr[i_this].predict(
                    self.X[i_this + 1:], return_std=True, validate=False)[1]
                self.acq[i_this + 1:] = np.clip(
                    self._acq_func(self.y[i_this + 1:], sigmas_cond),
                    a_min=None, a_max=self.acq[i_this + 1:])
