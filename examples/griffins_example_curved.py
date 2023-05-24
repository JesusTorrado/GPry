"""
Example code for a simple GP Characterization of a likelihood.
"""

import os
from gpry.mpi import is_main_process
from gpry.io import create_path

# Path for saving plots, make sure it exists!
checkpoint = None # "output/curved"

# Building the likelihood
from scipy.stats import multivariate_normal
import numpy as np

# Create likelihood
def log_lkl(x_1, x_2):
    return  -(10*(0.45-x_1))**2./4. - (20*(x_2/4.-x_1**4.))**2.

# Construct model instance
info = {"likelihood": {"curved_degeneracy": log_lkl}}
info["params"] = {
    "x_1": {"prior": {"min": -0.5, "max": 1.5}},
    "x_2": {"prior": {"min": -0.5, "max": 2.}}
    }

# Define the model (containing the prior and the likelihood)
from cobaya.model import get_model
model = get_model(info)

#############################################################
# Plotting the likelihood
from cobaya.sampler import get_sampler
from cobaya.output import get_output
sampler = get_sampler({"polychord": {"num_repeats": "10d", "measure_speeds": False}}, model, get_output("images/truth_chains"))
sampler.run()

if is_main_process:
    gdsample_truth = sampler.products(to_getdist=True)["sample"]
    import getdist.plots as gdplt
    gdplot = gdplt.get_subplot_plotter()
    gdplot.triangle_plot(gdsample_truth, list(info["params"]), filled=True)
    gdplot.export("images/truth.png")

#############################################################

verbose = 3

from gpry.preprocessing import Normalize_bounds
from gpry.gp_acquisition import NORA, GPAcquisition
acquisition = NORA(
    model.prior.bounds(), acq_func="NonlinearLogExp",
    mc_every=model.prior.d(),
    preprocessing_X=Normalize_bounds(model.prior.bounds()),
    zeta_scaling=0.85, verbose=verbose)
#acquisition = GPAcquisition(
#    model.prior.bounds(), proposer=None, acq_func="LogExp",
#    acq_optimizer="fmin_l_bfgs_b",
#    n_restarts_optimizer=5 * model.prior.d(), n_repeats_propose=10,
#    preprocessing_X=Normalize_bounds(model.prior.bounds()),
#    zeta_scaling=0.85, verbose=verbose)
#acquisition = None

options = {"n_points_per_acq": 2}
from gpry.run import Runner
runner = Runner(model, checkpoint=checkpoint, load_checkpoint="overwrite",
####                account_for_inf=None,  # FOR NOW, DISABLE SVM!!!
                gp_acquisition=acquisition,
                seed=None, verbose=verbose, plots=False, options=options)

# Run the GP
runner.run()

# Run the MCMC and extract samples
if is_main_process:
    print(runner.gpr.alpha_)
updated_info, sampler = runner.generate_mc_sample()

# Plotting
#runner.plot_mc(updated_info, sampler)
runner.plot_distance_distribution(updated_info, sampler)

#exit()

# Validation
if is_main_process:
    from getdist.gaussian_mixtures import GaussianND
    from getdist.mcsamples import MCSamplesFromCobaya
    import getdist.plots as gdplt
    from gpry.plots import getdist_add_training
    gpr = runner.gpr
    gdsample_gp = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
    to_plot = [gdsample_truth, gdsample_gp]
    filled = [False, True]
    legend_labels = ['Truth', 'MC from GP']

    if acquisition is not None:
        poly_out = getattr(acquisition, "last_polychord_output", None)
        if poly_out is not None:
            paramnames = [(p, p) for p in info["params"]]
            poly_out.make_paramnames_files(paramnames)
            to_plot += [poly_out.posterior]
            filled += [True]
            legend_labels += ['Last Nested from GP']
    
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(to_plot, list(info["params"]),
                         filled=filled,
                         legend_labels=legend_labels)
    getdist_add_training(gdplot, model, gpr)
    gdplot.export("images/Comparison_triangle.png")
