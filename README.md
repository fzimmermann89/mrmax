<h1 align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/fzimmermann89/mrmax/refs/heads/master/docs/source/_static/logo_white.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/fzimmermann89/mrmax/refs/heads/master/docs/source/_static/logo.svg">
  <img src="https://raw.githubusercontent.com/fzimmermann89/mrmax/refs/heads/master/docs/source/_static/logo.svg" alt="MRmax logo" width="50%">
</picture>

</h1><br>

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Coverage Bagde](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ckolbPTB/48e334a10caf60e6708d7c712e56d241/raw/coverage.json)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14509598.svg)](https://doi.org/10.5281/zenodo.14509598)

MR image reconstruction and processing package in JAX.


- **Source code:** <https://github.com/fzimmermann89/mrmax>
- **Documentation:** <https://fzimmermann89.github.io/mrmax/>
- **Bug reports:** <https://github.com/fzimmermann89/mrmax/issues>
- **Try it out:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fzimmermann89/mrmax)


## Main features
- **Maximized professionalism** Max is even better than pro!
- **Jax** based
- **ISMRMRD support** MRmax supports [ismrmrd-format](https://ismrmrd.readthedocs.io/en/latest/) for MR raw data.
- **Pytrees** ❤️All data containers utilize pytrees.
- **Cartesian and non-Cartesian trajectories** MRmax can reconstruct data obtained with Cartesian and non-Cartesian (e.g. radial, spiral...) sapling schemes . MRmax automatically detects if FFT or nuFFT is required to reconstruct the k-space data.
- **Pulseq support** If the data acquisition was carried out using a [pulseq-based](http://pulseq.github.io/) sequence, the seq-file can be provided to MRmax and the used trajectory is automatically calculated.
- **Signal models** A range of different MR signal models are implemented (e.g. T1 recovery, WASABI).
- **Regularized image reconstruction** Regularized image reconstruction algorithms including Wavelet-based compressed sensing or total variation regularized image reconstruction are available.

## Examples

In the following, we show some code snippets to highlight the use of MRmax. Each code snippet only shows the main steps. A complete working notebook can be found in the provided link.

### Simple reconstruction

Read the data and trajectory and reconstruct an image by applying a density compensation function and then the adjoint of the Fourier operator and the adjoint of the coil sensitivity operator.

```python
# Read the trajectory from the ISMRMRD file
trajectory = mrmax.data.traj_calculators.KTrajectoryIsmrmrd()
# Load in the Data from the ISMRMRD file
kdata = mrmax.data.KData.from_file(data_file.name, trajectory)
# Perform the reconstruction
reconstruction = mrmax.algorithms.reconstruction.DirectReconstruction(kdata)
img = reconstruction(kdata)
```

Full example: <https://github.com/fzimmermann89/mrmax/blob/main/examples/scripts/direct_reconstruction.py>

### Estimate quantitative parameters

Quantitative parameter maps can be obtained by creating a functional to be minimized and calling a non-linear solver such as ADAM.

```python
# Define signal model
model = MagnitudeOp() @ InversionRecovery(ti=idata_multi_ti.header.ti)
# Define loss function and combine with signal model
mse = MSE(idata_multi_ti.data.abs())
functional = mse @ model
[...]
# Run optimization
params_result = adam(functional, [m0_start, t1_start], n_iterations=n_iterations, learning_rate=learning_rate)
```

Full example: <https://github.com/fzimmermann89/mrmax/blob/main/examples/scripts/qmri_sg_challenge_2024_t1.py>

### Pulseq support

The trajectory can be calculated directly from a provided pulseq-file.

```python
# Read raw data and calculate trajectory using KTrajectoryPulseq
kdata = KData.from_file(data_file.name, KTrajectoryPulseq(seq_path=seq_file.name))
```

Full example: <https://github.com/fzimmermann89/mrmax/blob/main/examples/scripts/comparison_trajectory_calculators.py>

### Contributin
This code is partially based on mrpro (copyright Physikalisch-Technische Bundesanstalt, MIT licensed)