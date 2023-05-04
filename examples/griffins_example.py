"""
Example code for a simple GP Characterization of a likelihood.
"""

import os
from gpry.mpi import is_main_process
from gpry.io import create_path

# Path for saving plots, make sure it exists!
checkpoint = None # "output/simple"

# Building the likelihood
from scipy.stats import multivariate_normal
import numpy as np

mean = [3, 2]
cov = np.array([[0.5, 0.4], [0.4, 1.5]]) / 5**2
rv = multivariate_normal(mean, cov)

def lkl(x, y):
    return np.log(rv.pdf(np.array([x, y]).T))

#############################################################
# Plotting the likelihood

if is_main_process:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    a = np.linspace(-10., 10., 200)
    b = np.linspace(-10., 10., 200)
    A, B = np.meshgrid(a, b)

    x = np.stack((A, B), axis=-1)
    xdim = x.shape
    x = x.reshape(-1, 2)
    Y = -1 * lkl(x[:, 0], x[:, 1])
    Y = Y.reshape(xdim[:-1])

    # Plot ground truth
    fig = plt.figure()
    im = plt.pcolor(A, B, Y, norm=LogNorm())
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    if is_main_process and checkpoint:
        create_path(os.path.join(checkpoint or "./", "images/"), verbose=False)
        plt.savefig(os.path.join(checkpoint or "./", "images/Ground_truth.png"), dpi=300)
    plt.close()

#############################################################

# Define the model (containing the prior and the likelihood)
from cobaya.model import get_model

info = {"likelihood": {"normal": lkl}}
info["params"] = {
    "x": {"prior": {"min": -10, "max": 10}},
    "y": {"prior": {"min": -10, "max": 10}}
    }

model = get_model(info)

verbose = 3

from gpry.preprocessing import Normalize_bounds
from gpry.gp_acquisition import Griffins, GPAcquisition
acquisition = Griffins(
    model.prior.bounds(), acq_func="NonlinearLogExp",
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
                seed=10, verbose=verbose, plots=False, options=options)

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
    gdsamples_gp = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
    gdsamples_truth = GaussianND(mean, cov, names=list(info["params"]))
    to_plot = [gdsamples_truth, gdsamples_gp]
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
    plt.savefig(os.path.join(checkpoint or "./", "images/Comparison_triangle.png"), dpi=300)
