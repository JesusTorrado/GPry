"""
Runs random gaussians and runs a KL after every iteration and saves the KL divergence
at every iteration. Also has options to make corner plots at every iteration.
Takes an argument when running from the console which defines the location of the checkpoint and all plots.
"""

import numpy as np
from time import time
import pickle
import sys

# GPry things needed for building the model
from gpry.preprocessing import Normalize_bounds
from gpry.gp_acquisition import Griffins
from gpry.tools import kl_norm
import matplotlib.pyplot as plt
from gpry.tools import kl_norm
from gpry.run import Runner
from gpry.io import create_path
from model_generator import Random_gaussian
from getdist.gaussian_mixtures import MixtureND

checkpoint_location = f"{sys.argv[1]}"
create_path(checkpoint_location)

# Options for the Gaussians
n_d = 2 # Number of dimensions
n_kb = 2 # Number of Kriging believer steps 
n_accepted_evals = 50 # Number of accepted steps before killing

# Options for plotting
plot_intermediate_contours = True
info_text_in_plot = True

# Options for the comparison MCMCs
rminusone = 0.005 
rminusonecl = 0.02

generator = Random_gaussian(ndim=n_d)
model = generator.get_model()

# Saves relevant parameters for the run
history = {
    "new_y" : [],
    "y_pred": [],
    "y_max": [],
    "n_tot": [],
    "n_acc": [],
    "KL": []
}

verbose = 3 # Verbosity of the BO loop

# Get dimensionality and prior bounds
dim = model.prior.d()
prior_bounds = model.prior.bounds()

def callback(runner):
    """
    Runs an MCMC on the current model at every iteration.
    """
    global checkpoint_location
    global generator
    global plot_intermediate_contours
    global info_text_in_plot

    n_total = runner.gpr.n_total
    n = runner.gpr.n
    true_mean = generator.mean
    true_cov = generator.cov
    paramnames = list(runner.model.parameterization.sampled_params())
    create_path(f"{checkpoint_location}/{n_total}_{n}")
    surr_info, sampler = runner.generate_mc_sample(
        sampler="mcmc", add_options={"Rminus1_stop": rminusone, "Rminus1_cl_stop":rminusonecl,
                             "max_tries": 10000, "covmat": true_cov, "covmat_params": paramnames},
        output=f"{checkpoint_location}/{n_total}_{n}/mc_samples")
    mc_mean = sampler.products()["sample"].mean()
    mc_cov = sampler.products()["sample"].cov()
    # Compute KL_truth and save it to progress table (hacky!)
    kl = np.mean([kl_norm(true_mean, true_cov, mc_mean, mc_cov),
             kl_norm(mc_mean, mc_cov, true_mean, true_cov)])

    # Save everything
    history["new_y"].append(runner.new_y)
    history["y_pred"].append(runner.y_pred)
    history["y_max"].append(runner.gpr.y_max)
    history["n_tot"].append(runner.gpr.n_total)
    history["n_acc"].append(runner.gpr.n)
    history["KL"].append(kl)

    with open(f"{checkpoint_location}/history.pkl", "wb") as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

    if plot_intermediate_contours:
        true_dist = MixtureND([true_mean], [true_cov], names=paramnames)
        gdplot = runner.plot_mc(surr_info, sampler, add_samples={"Truth": true_dist})
        if info_text_in_plot:
            n_d = runner.model.prior.d()
            info_text = ("$n_{tot}=%i$\n $n_{accepted}=%i$\n $d_{KL}=%.2e$"
                %(n_total, n, kl))
            ax = gdplot.get_axes(ax=(0, n_d-max(1,int(n_d/3.))))
            gdplot.add_text_left(info_text, x=0.2, y=0.5, ax=(0, n_d-max(1,int(n_d/3.)))) #, transform=ax.transAxes
            ax.axis('off')
        plt.savefig(f"{checkpoint_location}/{n_total}.pdf")
        plt.close()

prior_bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
acquisition = Griffins(
    model.prior.bounds(),
    mc_every=model.prior.d(),
    preprocessing_X=Normalize_bounds(model.prior.bounds()),
    use_prior_sample=False,
    zeta_scaling=0.85, verbose=verbose,
    tmpdir=f"{checkpoint_location}/tmpdir"
)

runner = Runner(model,
    gp_acquisition=acquisition,
    convergence_criterion="DontConverge",
    verbose=verbose, callback = callback, checkpoint=f"{checkpoint_location}",
    load_checkpoint="overwrite",
    options={'max_accepted':n_accepted_evals, 'max_points':10000, 'n_points_per_acq':n_kb})
runner.run()
