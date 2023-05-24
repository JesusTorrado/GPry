 
# Running this example

## Install

cobaya-install multimodal_cosmo_fixed.yaml

## Run multimodal fiducial cases

### Multimodal test: fixed LCDM universe with a 3-parameter sinusoidal feature in the primordial power spectrum

Ground truth: with 32 physical single-thread cores in this example:

    mpirun -n 30 -x OMP_NUM_THREADS=1 cobaya-run cosmo_multimodal_3d.yaml --force

NORA:

With 32 physical single-thread cores in this example:

    mpirun -n 8 -x OMP_NUM_THREADS=4 python3 run_nora_cosmo_multimodal.py
