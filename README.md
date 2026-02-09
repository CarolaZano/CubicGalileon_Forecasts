# Cubic Galileon Forecasts

This repository contains code to perform LSST 3x2pt data forecasts for the cubic Galileon gravity model using the Cubic Galileon emulator: https://nesar.github.io/CubicGalileonEmu/.

The forecasts are currently set up for LSST Y1 only (with SRD covariance, https://github.com/CosmoLike/DESC_SRD). The setup uses CCL, which allows to modify the background evolution for Cubic Galileon as well as the power spectrum.

The emulator was built on a LHS over the following parameters:

$$
        0.275 \leq \Omega_{m0} = \Omega_{c0} + \Omega_{b0} \leq 0.33 
$$
$$
        0.85 \leq n_s \leq 1.1 
$$
$$
         1.45 \times 10^{-9} \leq A_s \leq 3.3 \times 10^{-9} 
$$
$$
        0.61 \leq h \leq 0.73
$$
$$
        0.02 \leq f_\phi \leq 1
$$

Where $f_\phi$ is the characteristic parameter for the given Cubic Galileon gravity model (see in prep paper for more details), defined as the fraction of the dark energy component due to the scalar field to the total dark energy ${\Large (} f_\phi = \frac{\Omega_{\phi 0}}{\Omega_{\phi 0} + \Omega_{\Lambda 0}}{\Large )}$.

## Repository Structure
```
.
├── Figures/                     # Figures and plots
├── HiCOLA_background/           # Background cosmology scripts
├── Validation_data/             # Reference training + validation datasets
├── ini_files/                   # Configuration and settings files - change these to change the forecasts!
├── parameters/                  # Survey parameters - hardcoded
├── .gitignore
├── CubicGalileonForecasts.py    # Main forecast driver script
├── CubicGalileonFunctions.py    # Model functions & helpers
├── Get_Data_3x2pt_fsigma8_CubicGalileon.py  # Data loader for observables
├── Likelihood_functions.py      # Old code, not used
├── binning.py                   # Binning
├── srd_redshift_distributions.py# Redshift distributions
├── RunMCMC*.ipynb               # Jupyter notebooks showing the full code for defining functions (with examples/tests) and running the inference pipeline
├── Read_emcee*.ipynb            # Jupyter notebooks to read the output of the mcmc chains from a folder mcmc/ (they are not being loaded on gitgub at the moment)
├── Validation.ipynb             # Validation plots at z=0 for the Cubic Galileon emulator
└── README.md
```

## Setup
1. Clone the repository

```bash
git clone https://github.com/CarolaZano/CubicGalileon_Forecasts.git
cd CubicGalileon_Forecasts
```

2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
# Still need to write this up
```

4. Configure parameters

Edit the files in ini_files/ and parameters/ to set your model setup and survey specifications.

5. Run a basic forecast

python CubicGalileonForecasts.py
