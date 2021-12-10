"""
Top level run file which constructs the loop for mapping a posterior
distribution and sample the GP to get chains.
"""

from gpry.mpi import mpi_comm, mpi_size, mpi_rank, is_main_process, get_random_state, \
    split_number_for_parallel_processes, multiple_processes
from gpry.gpr import GaussianProcessRegressor
from gpry.gp_acquisition import GP_Acquisition
from gpry.svm import SVM
from gpry.preprocessing import Normalize_bounds, Normalize_y
from gpry.tools import cobaya_gp_model_input, mcmc_info_from_run
import gpry.convergence as gpryconv
from cobaya.model import Model, get_model
from cobaya.output import get_output
from cobaya.sampler import get_sampler
from copy import deepcopy
import numpy as np
import warnings
import pickle
import os


def run(model, gp="RBF", gp_acquisition="Log_exp",
        convergence_criterion="ConvergenceCriterionGaussianMCMC",
        convergence_options=None, options={}, callback=None, verbose=1):
    """
    This function takes care of constructing the Bayesian quadrature/likelihood
    characterization loop. This is the easiest way to make use of the
    gpry algorithm. The minimum requirements for running this are a Cobaya
    prior object and a likelihood. Furthermore the details of the GP and
    and acquisition can be specified by the user.

    Parameters
    ----------

    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        Contains all information about the parameters in the likelihood and
        their priors as well as the likelihood itself. Cobaya is only used here
        as a wrapper to get the logposterior etc.

    gp : GaussianProcessRegressor, "RBF" or "Matern", optional (default="RBF")
        The GP used for interpolating the posterior. If None or "RBF" is given
        a GP with a constant kernel multiplied with an anisotropic RBF kernel
        and dynamic bounds is generated. The same kernel with a Matern 3/2
        kernel instead of a RBF is generated if "Matern" is passed. This might
        be useful if the posterior is not very smooth.
        Otherwise a custom GP regressor can be created and passed.

    gp_acquisition : GP_Acquisition, optional (default="Log_exp")
        The acquisition object. If None is given the Log_exp acquisition
        function is used (with the :math:`\zeta` value chosen automatically
        depending on the dimensionality of the prior) and the GP's X-values are
        preprocessed to be in the uniform hypercube before optimizing the
        acquistion function.

    convergence_criterion : Convergence_criterion, optional (default="KL")
        The convergence criterion. If None is given the KL-divergence between
        consecutive runs is calculated with an MCMC run and the run converges
        if KL<0.02 for two consecutive steps.

    convergence_option: optional parameters passed to the convergence criterion.

    options : dict, optional (default=None)
        A dict containing all options regarding the bayesian optimization loop.
        The available options are:

            * n_initial : Number of initial samples before starting the BO loop
              (default: 3*number of dimensions)
            * n_points_per_acq : Number of points which are aquired with
              Kriging believer for every acquisition step (default: equals the
              number of parallel processes)
            * max_points : Maximum number of points before the run fails
              (default: 1000)
            * max_init : Maximum number of points drawn at initialization
              before the run fails (default: 10 * number of dimensions * n_initial).
              If the run fails repeatadly at initialization try decreasing the volume
              of your prior.

    callback : str, optional (default=None)
        Path for storing callback information from which to resume in case the
        algorithm crashes. If None is given no callback is saved.

    Returns
    -------

    model : Cobaya model
        The model that was used to run the GP on (if running in parallel, needs to be
        passed for all processes).

    gp : GaussianProcessRegressor
        This can be used to call an MCMC sampler for getting marginalized
        properties. This is the most crucial component.

    gp_acquisition : GP_acquisition
        The acquisition object that was used for the active sampling procedure.

    convergence_criterion : Convergence_criterion
        The convergence criterion used for determining convergence. Depending
        on the criterion used this also contains the approximate covariance
        matrix of the posterior distribution which can be used by the MCMC
        sampler.

    options : dict
        The options dict used for the active sampling loop.
    """

    n_d = model.prior.d()
    if is_main_process:
        # Construct GP if it's not already constructed
        if isinstance(gp, str):
            prior_bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
            gpr = GaussianProcessRegressor(
                kernel=gp,
                n_restarts_optimizer=10,
                preprocessing_X=Normalize_bounds(prior_bounds),
                preprocessing_y=Normalize_y(),
                bounds=prior_bounds,
                verbose=verbose
            )

        elif isinstance(gp, GaussianProcessRegressor):
            gpr = gp
        else:
            raise TypeError("gp should be a GP regressor, 'RBF' or 'Matern', got "
                            "%s" % gp)

        # Construct the acquisition object if it's not already constructed
        if isinstance(gp_acquisition, str):
            prior_bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
            if gp_acquisition == "Log_exp":
                acquisition = GP_Acquisition(prior_bounds,
                                             acq_func="Log_exp",
                                             acq_optimizer="fmin_l_bfgs_b",
                                             n_restarts_optimizer=5,
                                             preprocessing_X=Normalize_bounds(
                                                 prior_bounds),
                                             verbose=verbose)
        elif isinstance(gp_acquisition, GP_Acquisition):
            acquisition = acquisition
        else:
            raise TypeError("gp_acquisition should be an Acquisition object or "
                            "'Log_exp', got %s" % gp_acquisition)

        # Construct the convergence criterion
        if isinstance(convergence_criterion, str):
            try:
                conv_class = getattr(gpryconv, convergence_criterion)
            except AttributeError:
                raise ValueError(
                    f"Unknown convergence criterion {convergence_criterion}. "
                    f"Available convergence criteria: {gpryconv.builtin_names()}")
            convergence = conv_class(model.prior, convergence_options or {})
        elif isinstance(convergence_criterion, gpryconv.ConvergenceCriterion):
            convergence = convergence_criterion
        else:
            raise TypeError("convergence_criterion should be a "
                            "Convergence_criterion object or KL, got %s"
                            % convergence_criterion)

        # Check if a callback exists already and if so resume from there
        callback_files = _check_callback(callback)
        if np.any(callback_files):
            if np.all(callback_files):
                print("#########################################")
                print("Callback found. Resuming from there...")
                print("If this behaviour is unintentional either")
                print("turn the callback option off or rename it")
                print("to a file which doesn't exist.")
                print("#########################################")

                model, gpr, acquisition, convergence, options = _read_callback(
                    callback)
            else:
                warnings.warn("Callback files were found but are incomplete. "
                              "Ignoring those files...")

        print("Model has been initialized")

        # Read in options for the run
        if options is None:
            options = {}
        n_initial = options.get("n_initial", 3 * n_d)
        max_points = options.get("max_points", 1000)
        max_init = options.get("max_init", 10 * n_d * n_initial)
        n_points_per_acq = options.get("n_points_per_acq", mpi_size)
        if n_points_per_acq < mpi_size:
            print("Warning: parallellisation not fully utilised! It is advised to make "
                  "n_points_per_acq equal to the number of MPI processes (default when "
                  "not specified.")

        # Sanity checks
        if n_initial >= max_points:
            raise ValueError("The number of initial samples needs to be "
                             "smaller than the maximum number of points")
        if n_initial <= 0:
            raise ValueError("The number of initial samples needs to be bigger "
                             "than 0")
    if multiple_processes:
        max_init, max_points, n_points_per_acq = mpi_comm.bcast(
            (max_init, max_points, n_points_per_acq) if is_main_process else None)
        convergence_is_MPI_aware = mpi_comm.bcast(
            convergence.is_MPI_aware if is_main_process else None)
        if convergence_is_MPI_aware:
            convergence = mpi_comm.bcast(convergence if is_main_process else None)
    else:
        convergence_is_MPI_aware = convergence.is_MPI_aware

    # Set MPI-aware random state
    random_state = get_random_state()

    # Define initial tranining set
    get_initial_sample(model, gpr, n_initial)
    if is_main_process:
        # Save callback
        _save_callback(callback, model, gpr, acquisition, convergence, options)
        print("Initial samples drawn, starting with Bayesian "
              "optimization loop.")
    if multiple_processes:
        n_finite = mpi_comm.bcast(len(gpr.y_train) if is_main_process else None)
    else:
        n_finite = len(gpr.y_train)

    # Run bayesian optimization loop
    n_iterations = 100000
    #int((max_points - n_finite) / n_points_per_acq)
    n_evals_per_acq_per_process = \
        split_number_for_parallel_processes(n_points_per_acq)
    n_evals_this_process = n_evals_per_acq_per_process[mpi_rank]
    i_evals_this_process = sum(n_evals_per_acq_per_process[:mpi_rank])
    it = 0
    n_left = max_points - n_finite
    for it in range(n_iterations):
        if is_main_process:
            print(f"+++ Iteration {it} ({n_left} points left) +++++++++")
            # Save old gp for convergence criterion
            old_gpr = deepcopy(gpr)
            # Get new point(s) from Bayesian optimization
            new_X, y_lies, acq_vals = acquisition.multi_optimization(
                gpr, n_points=n_points_per_acq)
        # Get logposterior value(s) for the acquired points (in parallel)
        if multiple_processes:
            new_X = mpi_comm.bcast(new_X if is_main_process else None)
        new_X_this_process = new_X[
            i_evals_this_process: i_evals_this_process + n_evals_this_process]
        new_y = np.empty(0)
        for x in new_X_this_process:
            new_y = np.append(new_y, model.logpost(x))
        # Collect (if parallel) and append to the current model
        if multiple_processes:
            all_new_y = mpi_comm.gather(new_y)
        else:
            all_new_y = [new_y]
        if is_main_process:
            new_y = np.concatenate(all_new_y)
            gpr.append_to_data(new_X, new_y, fit=True)
            n_left -= gpr.last_num_accepted
        # Calculate convergence and break if the run has converged
        if not convergence_is_MPI_aware:
            if is_main_process:
                try:
                    is_converged = convergence.is_converged(gpr, old_gpr)
                except gpryconv.ConvergenceCheckError:
                    is_converged = False
            if multiple_processes:
                is_converged = mpi_comm.bcast(is_converged if is_main_process else None)
        else:  # run by all processes
            # NB: this assumes that when the criterion fails,
            #     ALL processes raise ConvergenceCheckerror, not just rank 0
            if multiple_processes:
                gpr, old_gpr =  mpi_comm.bcast(
                    (gpr, old_gpr) if is_main_process else None)
            try:
                is_converged = convergence.is_converged(gpr, old_gpr)
            except gpryconv.ConvergenceCheckError as converr:
                is_converged = False
        if is_converged:
            break
        if is_main_process:
            if n_left <=0:
              break
            # Save
            _save_callback(callback, model, gpr, acquisition, convergence, options)

    # Save
    if is_main_process:
        _save_callback(callback, model, gpr, acquisition, convergence, options)

    if n_left <=0 and is_main_process:
        warnings.warn("The maximum number of points was reached before "
                      "convergence. Either increase max_points or try to "
                      "choose a smaller prior.")
    if it == n_iterations and is_main_process:
        warnings.warn("Not enough points were accepted before "
                      "reaching convergence/reaching the specified max_points.")

    # Now that the run has converged we can return the gp and all other
    # relevant quantities which can then be processed with an MCMC or other
    # sampler
    if multiple_processes:
        gpr, acquisition, convergence, options = mpi_comm.bcast(
            (gpr, acquisition, convergence, options) if is_main_process else None)
    return model, gpr, acquisition, convergence, options


def get_initial_sample(model, gpr, n_initial, max_init=None):
    """
    Draws an initial sample for the `gpr` GP model until it has a training set of size
    `n_initial`, counting only finite-target points ("finite" here meaning over the
    threshold of the SVM classifier, if present).

    Will fail is it needed to evaluate the target more than `max_init` times (defaults
    to 10 times the dimension of the problem times the number of initial samples
    requested).

    This function is MPI-aware.
    """
    n_d = model.prior.d()
    max_init = max_init or 10 * n_d * n_initial
    # Check if there's an SVM and if so read out it's threshold value
    # We will compare it against y - max(y)
    if isinstance(gpr.account_for_inf, SVM):
        # Grab the threshold from the internal SVM (the non-preprocessed one)
        gpr.account_for_inf.update_threshold(dimension=model.prior.d())
        is_finite = lambda y: gpr.account_for_inf.is_finite(y, y_is_preprocessed=False)
    else:
        is_finite = lambda y: np.isfinite(y)
    if is_main_process:
        # Check if the GP already contains points. If so they are reused.
        pretrained = 0
        if hasattr(gpr, "y_train"):
            if len(gpr.y_train) > 0:
                pretrained = len(gpr.y_train)
        n_still_needed = n_initial - pretrained
        n_to_sample_per_process = int(np.ceil(n_still_needed / mpi_size))
        # Arrays to store the initial sample
        X_init = np.empty((0, n_d))
        y_init = np.empty(0)
    if multiple_processes:
        n_to_sample_per_process = mpi_comm.bcast(
            n_to_sample_per_process if is_main_process else None)
    if n_to_sample_per_process == 0:  # Enough pre-training
        print("The number of pretrained points exceeds the number of initial samples")
        return
    n_iterations_before_giving_up = int(np.ceil(max_init / n_to_sample_per_process))
    # Initial samples loop. The initial samples are drawn from the prior
    # and according to the distribution of the prior.
    for i in range(n_iterations_before_giving_up):
        X_init_loop = np.empty((0, n_d))
        y_init_loop = np.empty(0)
        for j in range(n_to_sample_per_process):
            # Draw point from prior and evaluate logposterior at that point
            X = model.prior.reference()
            print(f"Evaluating true posterior at {X}")
            y = model.logpost(X)
            print(f"Got {y}")
            X_init_loop = np.append(X_init_loop, np.atleast_2d(X), axis=0)
            y_init_loop = np.append(y_init_loop, y)
        # Gather points and decide whether to break.
        if multiple_processes:
            all_points = mpi_comm.gather(X_init_loop)
            all_posts = mpi_comm.gather(y_init_loop)
        else:
            all_points = [X_init_loop]
            all_posts = [y_init_loop]
        if is_main_process:
            X_init = np.concatenate([X_init, np.concatenate(all_points)])
            y_init = np.concatenate([y_init, np.concatenate(all_posts)])
            # Only finite values contributes to the number of initial samples
            n_finite_new = sum(is_finite(y_init - max(y_init)))
            # Break loop if the desired number of initial samples is reached
            finished = (n_finite_new >= n_still_needed)
        if multiple_processes:
            finished = mpi_comm.bcast(finished if is_main_process else None)
        if finished:
            break
        else:
            # TODO: maybe re-fit SVM to shrink initial sample region
            pass
    print("Done getting initial training samples.")
    if is_main_process:
        # Append the initial samples to the gpr
        gpr.append_to_data(X_init, y_init)
        # Raise error if the number of initial samples hasn't been reached
        if not finished:
            raise RuntimeError("The desired number of finite initial "
                               "samples hasn't been reached. Try "
                               "increasing max_init or decreasing the "
                               "volume of the prior")


def mcmc(model_truth, gp, convergence=None, options=None, output=None):
    """
    This function is essentially just a wrapper for the Cobaya MCMC sampler
    (monte python) which runs an MCMC on the fitted GP regressor. It returns
    the chains which can then be used with GetDist to get the triangle plots or
    be postprocessed in any other way.
    The plotting is explained in the
    `Cobaya documentation <https://cobaya.readthedocs.io/en/latest/example_advanced.html#from-the-shell>`_.

    Parameters
    ----------

    model_truth : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        Contains all information about the parameters in the likelihood and
        their priors as well as the likelihood itself. The likelihood is not
        used actively/it is replaced by the gp regressor so it does not need to
        be correct and can be replaced by a dummy function. Alternatively a
        string containing a path with the location of a saved GP run (callback)
        can be provided (the same path that was used to save the callback
        in the ``run`` function).

    gp : GaussianProcessRegressor, which has been fit to data and returned from
        the ``run`` function.
        Alternatively a string containing a path with the
        location of a saved GP run (callback) can be provided (the same path
        that was used to save the callback in the ``run`` function).

    convergence : Convergence_criterion, optional
        The convergence criterion which has been used to fit the GP. This is
        used to extract the covariance matrix if it is available from the
        Convergence_criterion class. Alternatively a string containing a path
        with the location of a saved GP run (callback) can be provided (the
        same path that was used to save the callback in the ``run`` function).

    options: dict, optional
        Containing the options for the mcmc sampler
        defined in the "sampler" block of the Cobaya input. For more
        information see
        `here <https://cobaya.readthedocs.io/en/latest/sampler.html>`.

        .. note::
            If you specify any options here you need to define the whole
            "sampler" block. This leaves room for the possibility to also use
            other samplers which are built into Cobaya (i.e. PolyChord).

    output: path, optional
        The path where the output of the MCMC (chains) shall be stored.

    Returns
    -------

    updated_info : dict
        The (expanded) dictionary that was used to run the MCMC on the GP.

    sampler : Sampler instance
        The sampler instance contains the chains etc. and can be used for
        plotting etc.
    """

    # Check GP
    if isinstance(gp, GaussianProcessRegressor):
        if hasattr(gp, 'y_train'):
            gpr = gp
        else:
            warnings.warn("The provided GP hasn't been trained to data "
                          "before. This is likely unintentional...")
    elif isinstance(gp, str):
        _, gpr, _, _, _ = _read_callback(gp)
        if gpr is None:
            raise RuntimeError("Could not load the GP regressor from callback")
        if not hasattr(gpr, "y_train"):
            warnings.warn("The provided GP hasn't been trained to data "
                          "before. This is likely unintentional...")
    else:
        raise TypeError("The GP needs to be a gpry GP Regressor or a string "
                        "with a path to a callback file.")

    # Check model_truth
    if isinstance(model_truth, str):
        model_truth, _, _, _, _ = _read_callback(model_truth)
        if model_truth is None:
            raise RuntimeError("Could not load the model from callback")
    elif not isinstance(model_truth, Model):
        raise TypeError("model_truth needs to be a Cobaya model instance.")

    sampled_parameter_names = list(model_truth.parameterization.sampled_params())
    model_surrogate = get_model(
        cobaya_gp_model_input(model_truth.prior, gpr))

    # Check if options for the sampler are given else build the sampler
    if options is None:
        sampler_info = mcmc_info_from_run(model_surrogate, gp, convergence)
    else:
        sampler_info = options

    out = None
    if output is not None:
        out = get_output(prefix=output, resume=False, force=True)

    # Create the sampler
    sampler = get_sampler(sampler_info, model=model_surrogate, output=out)

    # Run the sampler on the GP
    print("Starting sampler")
    sampler.run()
    updated_info = model_surrogate.info()
    updated_info["sampler"] = {list(sampler_info)[0]: sampler.info()}

    return updated_info, sampler


def _save_callback(path, model, gp, gp_acquisition, convergence_criterion, options):
    """
    This function is used to save all relevant parts of the GP loop for reuse
    as callback in case the procedure crashes.
    This function saves 5 files as .pkl files which contain the instances
    of the different modules.
    The files can be loaded with the _read_callback function.

    Parameters
    ----------

    path : The path where the files shall be saved
        The files will be saved as *path* +(mod, gpr, acq, con, opt).pkl

    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_

    gp : GaussianProcessRegressor

    gp_acquisition : GP_Acquisition

    convergence_criterion : Convergence_criterion

    options : dict
    """
    if path is not None:
        try:
            with open(os.path.join(path, "mod.pkl"), 'wb') as f:
                # Save model as dict
                model_dict = model.info()
                pickle.dump(model_dict, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path, "gpr.pkl"), 'wb') as f:
                pickle.dump(gp, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path, "acq.pkl"), 'wb') as f:
                pickle.dump(gp_acquisition, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path, "con.pkl"), 'wb') as f:
                # Need to delete the prior object in convergence so it doesn't
                # do weird stuff while pickling
                convergence = deepcopy(convergence_criterion)
                convergence.prior = None
                pickle.dump(convergence, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path, "opt.pkl"), 'wb') as f:
                pickle.dump(options, f, pickle.HIGHEST_PROTOCOL)
        except Exception as excpt:
            raise RuntimeError("Couldn't save the callback. Check if the path "
                               "is correct and exists. Error message: " + str(excpt))


def _check_callback(path):
    """
    Checks if there are callback files in a specific location and if so if they
    are complete. Returns a list of bools.

    Parameters
    ----------

    path : The path where the files are located

    Returns
    -------

    A boolean array containing whether the files exist in the specified
    location in the following order:
    [model, gp, acquisition, convergence, options]
    """
    if path is not None:
        callback_files = [os.path.exists(os.path.join(path, "mod.pkl")),
                          os.path.exists(os.path.join(path, "gpr.pkl")),
                          os.path.exists(os.path.join(path, "acq.pkl")),
                          os.path.exists(os.path.join(path, "con.pkl")),
                          os.path.exists(os.path.join(path, "opt.pkl"))]
    else:
        callback_files = [False] * 5
    return callback_files


def _read_callback(path):
    """
    Loads callback files to be able to resume a run or save the results for
    further processing.

    Parameters
    ----------

    path : The path where the files are located

    Returns
    -------

    model, gp, acquisition, convergence, options.
    If any of the files does not exist or cannot be read the function will
    return None instead.
    """
    # Check if a file exists in the callback and if so resume from there.
    callback_files = _check_callback(path)
    # Read in callback
    with open(os.path.join(path, "mod.pkl"), 'rb') as i:
        model = pickle.load(i) if callback_files[0] else None
        # Convert model from dict to model object
        model = get_model(model)
    with open(os.path.join(path, "gpr.pkl"), 'rb') as i:
        gpr = pickle.load(i) if callback_files[1] else None
    with open(os.path.join(path, "acq.pkl"), 'rb') as i:
        acquisition = pickle.load(i) if callback_files[2] else None
    with open(os.path.join(path, "con.pkl"), 'rb') as i:
        if callback_files[3]:
            convergence = pickle.load(i)
            convergence.prior = model.prior
        else:
            convergence = None

    with open(os.path.join(path, "opt.pkl"), 'rb') as i:
        options = pickle.load(i) if callback_files[4] else None

    return model, gpr, acquisition, convergence, options
