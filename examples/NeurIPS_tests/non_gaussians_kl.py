"""
Runs the non-gaussian examples and runs a KL after every iteration and saves the KL divergence
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
from cobaya.run import run as cobaya_run
from getdist.mcsamples import MCSamplesFromCobaya
from gpry.run import Runner
from gpry.io import create_path
from model_generator import *

start_time = time()

# Budget of points 
n_accepted_evals = 80 # Number of accepted steps before killing

# Options for plotting
plot_intermediate_contours = True
info_text_in_plot = True


# Options for the MC samplers
use_sampler = "polychord" # Available: "MCMC" "polychord"

# For MCMC
rminusone = 0.005
rminusonecl = 0.02
# For Polychord
nlive = "50d"
num_repeats = "10d"
nprior = "50nlive"
precision_criterion = 0.0001

# For a precise run at the end
nlive_precise = "200d"

# Likelihoods used in this paper (see model_generator.py for details)
# Curved_degeneracy()
# Himmelblau(ndim=2)
# Himmelblau(ndim=4)
# Ring()


checkpoint_location = f"{sys.argv[1]}"
create_path(checkpoint_location)
generator = Curved_degeneracy()
model = generator.get_model()
n_d = model.prior.d()

history = {
    "new_y" : [],
    "y_pred": [],
    "y_max": [],
    "n_tot": [],
    "n_acc": [],
    "KL": [],
    "KL1": [],
    "KL2": [],
    "KL_gauss": [],
    "finished_run": False
}

verbose = 3 # Verbosity of the BO loop

# Get dimensionality and prior bounds
dim = model.prior.d()
prior_bounds = model.prior.bounds()

#############################
### RUN THE COMPARISON MC ###
#############################
if use_sampler == "MCMC":
    info_sampler = {"mcmc": {"Rminus1_stop": rminusone, "Rminus1_cl_stop":rminusonecl,
                             "max_tries": 10000}}
    info_sampler_precise = {"mcmc": {"Rminus1_stop": rminusone, "Rminus1_cl_stop":rminusonecl,
                            "max_tries": 10000}}
elif use_sampler == "polychord":
    info_sampler = {"polychord": {"nlive": nlive,
                                  "num_repeats": num_repeats,
                                  "nprior": nprior,
                                  "precision_criterion": precision_criterion}}
    info_sampler_precise = {"polychord": {"nlive": nlive_precise,
                                  "num_repeats": num_repeats,
                                  "nprior": nprior,
                                  "precision_criterion": precision_criterion}}
    create_path(f"{checkpoint_location}/polychord")

info = model.info()
info_run = info.copy()
info_run['sampler'] = info_sampler_precise
info_run['output'] = f"{checkpoint_location}/truth/mc_samples"
updated_info1, sampler1 = cobaya_run(info_run)
s_true = sampler1.products()["sample"]
gdsamples_true = MCSamplesFromCobaya(updated_info1, s_true)

def kl_div(s1, s2, runner):
    # calculate the full kl divergence between the GP and full model given
    # mcmc runs on both
    model = runner.model
    gpr = runner.gpr
    x_values1 = s1.data[s1.sampled_params]
    logp1 = s1['minuslogpost']
    logp1 = -logp1
    weights1 = s1['weight']
    y_values = []
    for i in range(0,len(x_values1), 256):
        y_values = np.concatenate([y_values,gpr.predict(x_values1[i:i+256])])
    logq1 = np.array(y_values)
    mask1 = np.isfinite(logq1)
    logp1 = logp1[mask1]
    logq1 = logq1[mask1]
    weights1 = weights1[mask1]

    x_values2 = s2.data[s2.sampled_params]
    x_values2 = x_values2.to_numpy()
    logp2 = s2['minuslogpost']
    logp2 = -logp2
    weights2 = s2['weight']
    y_values = []
    for i in range(len(x_values2)):
        in_dict = dict(zip(list(s2.sampled_params), x_values2[i]))
        y_values = np.append(y_values, model.logpost(in_dict))
    logq2 = np.array(y_values)
    mask2 = np.isfinite(logq2)
    logp2 = logp2[mask2]
    logq2 = logq2[mask2]
    print(logp2-logq2)
    weights2 = weights2[mask2]
    kl1_full = np.sum(weights1*(logp1-logq1))/np.sum(weights1)
    kl2_full = np.sum(weights2*(logp2-logq2))/np.sum(weights2)

    s1mean, s1cov = s1.mean(), s1.cov()
    s2mean, s2cov = s2.mean(), s2.cov()
    kl1_gauss = kl_norm(s1mean,s1cov, s2mean, s2cov)
    kl2_gauss = kl_norm(s2mean,s2cov, s1mean, s1cov)
    return kl1_full, kl2_full, np.max([kl1_gauss, kl2_gauss])


def callback(runner):
    """
    Runs an MCMC on the current model at every iteration.
    """
    global checkpoint_location
    global generator
    global plot_intermediate_contours
    global info_text_in_plot
    global s_true
    global gdsamples_true

    n_total = runner.gpr.n_total
    n = runner.gpr.n
    paramnames = list(runner.model.parameterization.sampled_params())
    create_path(f"{checkpoint_location}/{n_total}_{n}")
    if use_sampler == "MCMC":
        surr_info, sampler = runner.generate_mc_sample(
            sampler="mcmc", add_options={"Rminus1_stop": rminusone,
                "Rminus1_cl_stop":rminusonecl, "max_tries": 10000},
            output=f"{checkpoint_location}/{n_total}_{n}/mc_samples")
        # "covmat": true_cov, "covmat_params": paramnames,
    elif use_sampler == "polychord":
        surr_info, sampler = runner.generate_mc_sample(
            sampler="polychord", add_options=info_sampler["polychord"],
            output=f"{checkpoint_location}/{n_total}_{n}/mc_samples")
    s_gp = sampler.products()["sample"]

    kl1, kl2, kl_gauss = kl_div(s_true, s_gp, runner)

    # Save everything
    history["new_y"].append(runner.new_y)
    history["y_pred"].append(runner.y_pred)
    history["y_max"].append(runner.gpr.y_max)
    history["n_tot"].append(runner.gpr.n_total)
    history["n_acc"].append(runner.gpr.n)
    history["KL"].append(np.max([kl1, kl2]))
    history["KL1"].append(kl1)
    history["KL2"].append(kl2)
    history["KL_gauss"].append(kl_gauss)

    with open(f"{checkpoint_location}/history.pkl", "wb") as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

    if plot_intermediate_contours:
        gdplot = runner.plot_mc(surr_info, sampler, add_samples={"Truth": gdsamples_true})
        if info_text_in_plot:
            n_d = runner.model.prior.d()
            info_text = ("$n_{tot}=%i$\n $n_{accepted}=%i$\n $d_{KL}=%.2e$"
                %(n_total, n, kl1))
            ax = gdplot.get_axes(ax=(0, n_d-max(1,int(n_d/3.))))
            gdplot.add_text_left(info_text, x=0.2, y=0.5, ax=(0, n_d-max(1,int(n_d/3.)))) #, transform=ax.transAxes
            ax.axis('off')
        plt.savefig(f"{checkpoint_location}/{n_total}.pdf")
        plt.close()

prior_bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
acquisition = Griffins(
    model.prior.bounds(), acq_func="NonlinearLogExp",
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
    options={'max_accepted':n_accepted_evals, 'max_points':10000, 'n_points_per_acq':n_d,
             'zeta_scaling':0.85})
runner.sample_at = []
runner.run()

if use_sampler == "MCMC":
    surr_info, sampler = runner.generate_mc_sample(
        sampler="mcmc", add_options={"Rminus1_stop": rminusone,
            "Rminus1_cl_stop":rminusonecl, "max_tries": 10000},
        output=f"{checkpoint_location}/final/mc_samples")
elif use_sampler == "polychord":
    surr_info, sampler = runner.generate_mc_sample(
        sampler="polychord", add_options=info_sampler_precise["polychord"],
        output=f"{checkpoint_location}/final/mc_samples")


s_gp = sampler.products()["sample"]
gdplot = runner.plot_mc(surr_info, sampler, add_samples={"Truth": gdsamples_true})
plt.savefig(f"{checkpoint_location}/final.pdf")

history["finished_run"] = True
with open(f"{checkpoint_location}/history.pkl", "wb") as f:
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

print("Everything finished...")
print("Total runtime:")
print(f"{time()-start_time} s")
