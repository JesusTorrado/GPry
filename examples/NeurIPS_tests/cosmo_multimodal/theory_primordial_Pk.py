"""
Adapted from Cobaya's paper (https://github.com/CobayaSampler/paper_cosmo_demo), under
arXiv license (http://arxiv.org/licenses/nonexclusive-distrib/1.0/).
"""

import numpy as np
from cobaya.theory import Theory


def feature_power_spectrum(As, ns, A, l, phi,
                           kmin=1e-6, kmax=10, # generous, for transfer integrals
                           k_pivot=0.05, n_samples_wavelength=20):
    """
    Creates the primordial scalar power spectrum as a power law plus an oscillatory
    feature of given amplitude A, wavelength l and phase phi:

        Delta P/P = A * sin(2 pi (k/l + phi))

    The characteristic delta_k is determined by the number of samples per oscillation
    n_samples_wavalength (default: 20).

    Returns a sample of k, P(k)
    """
    # Ensure thin enough sampling at low-k
    delta_k = min(0.0005, l / n_samples_wavelength)
    ks = np.arange(kmin, kmax, delta_k)
    power_law = lambda k: As * (k / k_pivot) ** (ns - 1)
    DeltaP_over_P = lambda k: A * np.sin(2 * np.pi * (k / l + phi))
    Pks = power_law(ks) * (1 + DeltaP_over_P(ks))
    return ks, Pks


class FeaturePrimordialPk(Theory):
    """
    Theory class producing a slow-roll-like power spectrum with a linearly-oscillatory
    feature on top.
    """

    params = {"As": None, "ns": None,
              "amplitude": None, "wavelength": None, "phase": None}
    n_samples_wavelength = 20
    k_pivot = 0.05

    def calculate(self, state, want_derived=True, **params_values_dict):
        As, ns, amplitude, wavelength, phase = \
            [params_values_dict[p] for p in
             ["As", "ns", "amplitude", "wavelength", "phase"]]
        ks, Pks = feature_power_spectrum(
            As, ns, amplitude, wavelength, phase, kmin=1e-6, kmax=10,
            k_pivot=self.k_pivot, n_samples_wavelength=self.n_samples_wavelength)
        state['primordial_scalar_pk'] = {'k': ks, 'Pk': Pks, 'log_regular': False}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']
