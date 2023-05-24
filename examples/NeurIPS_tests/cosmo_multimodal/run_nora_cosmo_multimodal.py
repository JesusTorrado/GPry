"""
Runs the multimodal cosmological example (see supp. material).
"""

import os
from cobaya.model import get_model
from getdist.mcsamples import MCSamplesFromCobaya, loadMCSamples
import getdist.plots as gdplt

from gpry.mpi import is_main_process
from gpry.preprocessing import Normalize_bounds
from gpry.gp_acquisition import Griffins, GPAcquisition
from gpry.convergence import CorrectCounter
from gpry.run import Runner
from gpry.plots import getdist_add_training


model = get_model("cosmo_multimodal_3d.yaml")

checkpoint = "output/cosmo_multimodal"
truth_chains = os.path.join(checkpoint, "truth")
gdsample_truth = loadMCSamples(truth_chains)

verbose = 3

acquisition = Griffins(
    model.prior.bounds(), acq_func="LogExp",
    mc_every=model.prior.d(),
    preprocessing_X=Normalize_bounds(model.prior.bounds()),
    zeta_scaling=0.85, verbose=verbose)
# acquisition = GPAcquisition(
#     model.prior.bounds(), proposer=None, acq_func="LogExp",
#     acq_optimizer="fmin_l_bfgs_b",
#     n_restarts_optimizer=5 * model.prior.d(), n_repeats_propose=10,
#     preprocessing_X=Normalize_bounds(model.prior.bounds()),
#     zeta_scaling=0.85, verbose=verbose)
# acquisition = None

options = {
    "n_points_per_acq": 2 * model.prior.d(),  # multi-modal: many local minima
    "fit_full_every": 20,  # no need to fit hyperparamenters too often
    "max_total": 2000,  # effectively never reached
}
convergence = CorrectCounter(model.prior, {})
runner = Runner(model, checkpoint=checkpoint, load_checkpoint="overwrite",
                gp_acquisition=acquisition, convergence_criterion=convergence,
                seed=None, verbose=verbose, plots=False, options=options)

# Run the GP
runner.run()

updated_info, sampler = runner.generate_mc_sample(
    sampler="polychord",
    add_options={"nlive": "50d", "num_repeats": "10d", "nprior": "50nlive"}
)

# Plotting
if is_main_process:
    gpr = runner.gpr
    gdsample_gp = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
    to_plot = [gdsample_gp]
    filled = [True]
    legend_labels = ['MC from GP']
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(to_plot, list(model.parameterization.sampled_params()),
                         filled=filled, legend_labels=legend_labels)
    getdist_add_training(gdplot, model, gpr)
    gdplot.export(os.path.join(checkpoint, "images/Comparison_triangle.png"))
