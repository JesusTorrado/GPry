"""
Runs Planck lite until convergence.
"""

import numpy as np
from time import time
import pickle
import sys

# GPry things needed for building the model
from gpry.gpr import GaussianProcessRegressor
from gpry.preprocessing import Normalize_y, Normalize_bounds
from gpry.gp_acquisition import Griffins
from gpry.tools import kl_norm
from gpry.svm import SVM
from cobaya.model import get_model
import matplotlib.pyplot as plt
from cobaya.model import get_model
from gpry.tools import kl_norm
from gpry.run import Runner
from gpry.io import create_path
from getdist.mcsamples import loadMCSamples
from cobaya.yaml import yaml_load

# Give option to turn on/off different parts of the code
n_r = sys.argv[1] # The run ID
rminusone = 0.007
rminusonecl = 0.05
drag = False
oversample_power = 0.4
proposal_scale = 1.9
n_accepted_evals = 1000 # Number of accepted steps before killing
plot_intermediate_contours = True
info_text_in_plot = True

checkpoint_location = f"{sys.argv[1]}"
yaml_file = open("planck_lite_omega_k.yaml")
#####################################
info = yaml_load(yaml_file)
model = get_model(info)
n_d = model.prior.d()

verbose = 4 # Verbosity of the BO loop

d = model.prior.d()
prior_bounds = model.prior.bounds()
normalize_bounds = Normalize_bounds(prior_bounds)

gpr = GaussianProcessRegressor(
    n_restarts_optimizer=10 + 2 * d,
    preprocessing_X=Normalize_bounds(prior_bounds),
    preprocessing_y=Normalize_y(),
    account_for_inf=SVM(preprocessing_X=Normalize_bounds(prior_bounds), threshold_sigma=20.),
    bounds=prior_bounds,
    verbose=verbose,
    noise_level=5e-1
)

prior_bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
acquisition = Griffins(
    model.prior.bounds(), acq_func="NonlinearLogExp",
    mc_every=model.prior.d(),
    preprocessing_X=Normalize_bounds(model.prior.bounds()),
    use_prior_sample=False,
    zeta_scaling=0.85, verbose=verbose,
    tmpdir=f"{checkpoint_location}/tmpdir")

runner = Runner(model, gpr=gpr, gp_acquisition=acquisition,
    verbose=verbose, checkpoint=f"{checkpoint_location}",
    load_checkpoint="overwrite", initial_proposer="reference",
    options={'max_accepted':n_accepted_evals, 'max_points':10000, 'n_points_per_acq':n_d})
runner.run()


paramnames = list(runner.model.parameterization.sampled_params())
surr_info, sampler = runner.generate_mc_sample(
        sampler="mcmc", add_options={
            "Rminus1_stop": rminusone, "Rminus1_cl_stop": rminusonecl, "drag": drag,
            "oversample_power": oversample_power, "proposal_scale": proposal_scale},
        output=f"{checkpoint_location}/gp_mc_samples")
gdplot = runner.plot_mc(surr_info, sampler)
plt.savefig(f"final_contours.pdf")
plt.close()
