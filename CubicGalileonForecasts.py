""" Create forecasts for the cubic galileon model. """

# inputs: params.yaml file (if type = 1, that gives the parameters of the data vector)
#         NOTE: cosmo_validation_edges.txt file (if type = 0, that gives the parameters of the data vector)
#               Boost_validation_edges.txt file (if type = 0, that gives the Boost for the simulated data vector)
#               z_k_validation_edges.txt file (if type = 0, that gives the redshifts and k values for the simulated data vector)
#         config_run.yaml file (that gives the info on the survey, the likelihood, the sampler, and the output chain name)

# outputs: chain file with the forecasted constraints on the parameters
##########################################################################
import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

from CubicGalileonFunctions import *

# Generic
#import pandas as pd
import numpy as np
import scipy
from itertools import islice, cycle
import math
import os
import sys
from scipy.integrate import odeint
#from joblib import Parallel, delayed
import itertools
from importlib import reload
from functools import lru_cache
import scipy.integrate
from scipy.interpolate import interpn
from scipy.interpolate import CubicSpline
from scipy.stats import norm
import gc
from datetime import timedelta

# cosmology
import pyccl as ccl
from astropy.io import fits
import yaml
import sacc
import time

# SRD Binning
import srd_redshift_distributions as srd
import binning

# Data Visualization
import matplotlib.pyplot as plt
#from tabulate import tabulate
from matplotlib.colors import LogNorm
#import seaborn as sns

# Parallelising 
from multiprocessing import Pool
import multiprocessing

# MCMC
from nautilus import Prior, Sampler
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
#from chainconsumer import ChainConsumer, Chain, make_sample
from IPython.display import display, Math
from multiprocessing import Pool
#from tqdm import tqdm

# HiCOLA background
from HiCOLA.Frontend import expression_builder as eb
import HiCOLA.Frontend.numerical_solver as ns
from HiCOLA.Frontend.read_parameters import read_in_parameters
import sympy as sym


# Cubic Galileon emu and background
from CubicGalileonEmu.load import *
from CubicGalileonEmu.viz import *
from CubicGalileonEmu.pca import *
from CubicGalileonEmu.gp import *
from CubicGalileonEmu.emu import *
from CubicGalileonEmu.mcmc import *


if_train_all = False ## Re-train all the models. Time-consuming. 
if_mcmc_all = False  ## Full MCMC run. Time-consuming. 
if_savefig = False

from configobj import ConfigObj
import subprocess

#################################################################
# 1. Mock redshift distribution
# Define the redshift interval and forecast years
redshift_range = np.linspace(0.01, 3.5, 500)
forecast_years = ["1", "10"]  # Assuming integers are appropriate

# Create a dictionary to store the redshift distributions
# for each forecast year and galaxy sample
redshift_distribution = {
    "sources": {},
    "lenses": {}
}

for year in forecast_years:
    source_dist = srd.SRDRedshiftDistributions(redshift_range, 
                                               galaxy_sample="source_sample",
                                               forecast_year=year)
    lens_dist = srd.SRDRedshiftDistributions(redshift_range, 
                                             galaxy_sample="lens_sample",
                                             forecast_year=year)

    redshift_distribution["sources"][year] = source_dist.get_redshift_distribution(normalised=True,
                                                                                   save_file=False)
    redshift_distribution["lenses"][year] = lens_dist.get_redshift_distribution(normalised=True,
                                                                                save_file=False)

# Uncomment to check if the dictionary is populated correctly
# print(redshift_distribution["sources"].keys())


bins = {
    "sources": {},
    "lenses": {}
}

# Perform the binning procedure
for year in forecast_years:
    bins["sources"][year] = binning.Binning(redshift_range, 
                                            redshift_distribution["sources"][year],
                                            year).source_bins(normalised=True,
                                                              save_file=False)
    bins["lenses"][year] = binning.Binning(redshift_range, 
                                           redshift_distribution["lenses"][year],
                                           year).lens_bins(normalised=True,
                                                           save_file=False)


#(5, 256)
Binned_distribution_lens = [list(bins["lenses"]["1"].items())[0][1]]
for i in range(4):
    Binned_distribution_lens = np.append(Binned_distribution_lens,\
               [list(bins["lenses"]["1"].items())[i+1][1]], axis=0)

Binned_distribution_source = [list(bins["sources"]["1"].items())[0][1]]
for i in range(4):
    Binned_distribution_source = np.append(Binned_distribution_source,\
               [list(bins["sources"]["1"].items())[i+1][1]], axis=0)

z = redshift_range

#################################################################
# get filename from command line
config_file = sys.argv[1]

# read the config file
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

############# GET SIMULATED DATAVECTOR #########################
params_filename = config['data']['params']

with open(params_filename, 'r') as f:
    params = yaml.safe_load(f)

Bias_distribution_fiducial = np.array([params['b1']*np.ones(len(z)),
                             params['b2']*np.ones(len(z)),
                             params['b3']*np.ones(len(z)),
                             params['b4']*np.ones(len(z)),
                             params['b5']*np.ones(len(z))]
)
cosmo_fid = ccl.Cosmology(Omega_c = 0.269619, 
                          Omega_b = 0.050041,
                          h = 0.6688,
                          n_s = 0.9626,
                          A_s = 2.092e-9)


# if type = 0, read the cosmo file that gives the parameters of the data vector

# define ell and C_ell shapes -- will depend on the data

ell_min_mockdata = 20
ell_max_mockdata = 15000

# define quantities for binning of ell -- will depend on the data

ell_bin_num_mockdata = 20

if config['data']['type'] == 0:
    print("Reading cosmo file to get the parameters of the data vector...")
    txt = config['data']['cosmo_file']
    sim_index = config['data']['index']
    hcube_val_edges  = np.loadtxt(txt).T
    f_phi_val_edges = hcube_val_edges[4][sim_index]
    h_val_edges = hcube_val_edges[3][sim_index]
    Omega_m_val_edges = hcube_val_edges[0][sim_index]
    n_s_val_edges = hcube_val_edges[1][sim_index]
    A_s_val_edges = hcube_val_edges[2][sim_index]

    
    # Load the saved array - gives Boost(i = sample point, z, k)
    Bk_arr_val_edges = np.load(config['data']['data_file'])
    # extract data from text file - gives z and k arrays
    txt_arr_val_edges = np.loadtxt(config['data']['z_k_file'])

    z_arr_val_edges = np.array(txt_arr_val_edges.T[0][np.isfinite(txt_arr_val_edges.T[0])])
    k_arr_val_edges = np.array(txt_arr_val_edges.T[1])*h_val_edges

    Bk_CuGal_cosmo_funct =  scipy.interpolate.RectBivariateSpline(z_arr_val_edges, k_arr_val_edges, Bk_arr_val_edges[sim_index])

    # Define cosmology -- our "universe cosmology"

    cosmo_universe = ccl.Cosmology(Omega_c = Omega_m_val_edges - 0.0223/h_val_edges**2,
                                Omega_b = 0.0223/h_val_edges**2,
                                h = h_val_edges,
                                n_s = n_s_val_edges,
                                A_s = A_s_val_edges)

    cosmo_universe_linear = ccl.Cosmology(Omega_c = Omega_m_val_edges - 0.0223/h_val_edges**2,
                                Omega_b = 0.0223/h_val_edges**2,
                                h = h_val_edges,
                                n_s = n_s_val_edges,
                                A_s = A_s_val_edges,
                                matter_power_spectrum='linear')


    f_phi_universe = f_phi_val_edges
    a_setup_universe, UE_setup_universe, coupling_setup_universe = CuGal_initialize(f_phi_universe, cosmo_universe)

    P_delta2D_GR_lin_universe = Get_Pk2D_obj_kk_GR_lin(cosmo_universe)
    P_delta2D_GR_nl_universe = Get_Pk2D_obj_kk_GR_nl(cosmo_universe)

    # Get Get mock 3x2pt data
    ells_SRD = np.loadtxt("ell-values")


    """Get mock C(ell) data"""

    ## LENSING - LENSING

    binned_ell = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_source)

    # find C_ell for non-linear matter power spectrum
    mockdata = Cell_CuGal_Validation(binned_ell,a_setup_universe, UE_setup_universe, coupling_setup_universe, f_phi_universe, cosmo_universe, 
                        z , Binned_distribution_source,Binned_distribution_lens,
                        Bias_distribution_fiducial, P_delta2D_GR_nl_universe, Bk_CuGal_cosmo_funct, tracer1_type="k", tracer2_type="k")

    ell_kk_mockdata = mockdata[0]
    D_kk_mockdata = mockdata[1]
    D_kk_mockdata = (np.array(D_kk_mockdata)).flatten()

    ## CLUSTERING - LENSING

    binned_ell = bin_ell_delk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_source,Binned_distribution_lens)

    # find C_ell for non-linear matter power spectrum
    mockdata = Cell_CuGal_Validation(binned_ell,a_setup_universe, UE_setup_universe, coupling_setup_universe, f_phi_universe,
                        cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,\
                        Bias_distribution_fiducial,P_delta2D_GR_nl_universe, Bk_CuGal_cosmo_funct, tracer1_type="k", tracer2_type="g")

    ell_delk_mockdata = mockdata[0]
    D_delk_mockdata = mockdata[1]
    D_delk_mockdata = (np.array(D_delk_mockdata)).flatten()

    ## CLUSTERING - CLUSTERING
    binned_ell = bin_ell_deldel(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_lens)

    # find C_ell for non-linear matter power spectrum
    mockdata = Cell_CuGal_Validation(binned_ell,a_setup_universe, UE_setup_universe, coupling_setup_universe, f_phi_universe,
                        cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,\
                        Bias_distribution_fiducial, P_delta2D_GR_nl_universe, Bk_CuGal_cosmo_funct, tracer1_type="g", tracer2_type="g")

    ell_deldel_mockdata = mockdata[0]
    D_deldel_mockdata = mockdata[1]
    D_deldel_mockdata = (np.array(D_deldel_mockdata)).flatten()


    ell_mockdata = np.append(np.append(ell_kk_mockdata, ell_delk_mockdata), ell_deldel_mockdata)
    D_mockdata = np.append(np.append(D_kk_mockdata, D_delk_mockdata), D_deldel_mockdata)

    del mockdata




# if type = 1, read the params file that gives the parameters of the data vector
elif config['data']['type'] == 1:
    print("Reading params file to get the parameters of the data vector...")
    params_filename = config['data']['params']
    with open(params_filename, 'r') as f:
        params = yaml.safe_load(f)
    Omega_m = params['Omega_m']
    Omega_b = params['Omega_b']
    h = params['h']
    n_s = params['n_s']
    A_s = params['A_s']
    f_phi_universe = params['f_phi']

    # Define cosmology -- our "universe cosmology"

    cosmo_universe = ccl.Cosmology(Omega_c = Omega_m - Omega_b, 
                            Omega_b = Omega_b,
                            h = h,
                            n_s = n_s,
                            A_s = A_s)
    cosmo_universe_linear = ccl.Cosmology(Omega_c = Omega_m - Omega_b, 
                            Omega_b = Omega_b,
                            h = h,
                            n_s = n_s,
                            A_s = A_s,
                            matter_power_spectrum='linear')

    a_setup_universe, UE_setup_universe, coupling_setup_universe = CuGal_initialize(f_phi_universe, cosmo_universe)

    P_delta2D_GR_lin_universe = Get_Pk2D_obj_kk_GR_lin(cosmo_universe)
    P_delta2D_GR_nl_universe = Get_Pk2D_obj_kk_GR_nl(cosmo_universe)


    """Get mock C(ell) data"""

    ## LENSING - LENSING

    binned_ell = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_source)

    # find C_ell for non-linear matter power spectrum
    mockdata = Cell_CuGal(binned_ell,a_setup_universe, UE_setup_universe, coupling_setup_universe, f_phi_universe, cosmo_universe, 
                        z , Binned_distribution_source,Binned_distribution_lens,
                        Bias_distribution_fiducial, P_delta2D_GR_nl_universe, tracer1_type="k", tracer2_type="k")

    ell_kk_mockdata = mockdata[0]
    D_kk_mockdata = mockdata[1]
    D_kk_mockdata = (np.array(D_kk_mockdata)).flatten()

    ## CLUSTERING - LENSING

    binned_ell = bin_ell_delk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_source,Binned_distribution_lens)

    # find C_ell for non-linear matter power spectrum
    mockdata = Cell_CuGal(binned_ell,a_setup_universe, UE_setup_universe, coupling_setup_universe, f_phi_universe,
                        cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,\
                        Bias_distribution_fiducial,P_delta2D_GR_nl_universe, tracer1_type="k", tracer2_type="g")

    ell_delk_mockdata = mockdata[0]
    D_delk_mockdata = mockdata[1]
    D_delk_mockdata = (np.array(D_delk_mockdata)).flatten()

    ## CLUSTERING - CLUSTERING
    binned_ell = bin_ell_deldel(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_lens)

    # find C_ell for non-linear matter power spectrum
    mockdata = Cell_CuGal(binned_ell,a_setup_universe, UE_setup_universe, coupling_setup_universe, f_phi_universe,
                        cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,\
                        Bias_distribution_fiducial, P_delta2D_GR_nl_universe, tracer1_type="g", tracer2_type="g")

    ell_deldel_mockdata = mockdata[0]
    D_deldel_mockdata = mockdata[1]
    D_deldel_mockdata = (np.array(D_deldel_mockdata)).flatten()


    ell_mockdata = np.append(np.append(ell_kk_mockdata, ell_delk_mockdata), ell_deldel_mockdata)
    D_mockdata = np.append(np.append(D_kk_mockdata, D_delk_mockdata), D_deldel_mockdata)

    del mockdata

else:
    raise ValueError("Invalid data type specified in config file. Must be 0 or 1.")



############# GET COVARIANCE MATRIX #########################
"""Get SRD covariance matrix"""

# covariance for shear bin combinations, in order: z11, z12, z13,..., z15, z22, z23,...z55

########## Get full covariance (gauss only) ##########

covfile = np.genfromtxt("/global/u2/c/carolazn/CuGal_Emu_project_mcmc/Y1_3x2pt_clusterN_clusterWL_cov")
print(covfile.shape)

shear_SRD = np.zeros((705,705))
ell_test_SRD = np.zeros(705)

for i in range(0,covfile.shape[0]):
    shear_SRD[int(covfile[i,0]),int(covfile[i,1])] = covfile[i,8]+covfile[i,9] # non-gauss
    shear_SRD[int(covfile[i,1]),int(covfile[i,0])] = covfile[i,8]+covfile[i,9] # non-gauss
    if int(covfile[i,0]) == int(covfile[i,1]):
        ell_test_SRD[int(covfile[i,0])] = covfile[i,2]

del covfile
print(shear_SRD.shape)

SRD_compare = shear_SRD[:540,:540].copy()

"""Get mock C(ell) data"""

## LENSING - LENSING

binned_ell = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_source)

# find C_ell for non-linear matter power spectrum
mockdata = Cell_GR(binned_ell, \
                cosmo_fid, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,\
                tracer1_type="k", tracer2_type="k")

ell_kk_mockdata = mockdata[0]
D_kk_mockdata_test = mockdata[1]
D_kk_mockdata_test = (np.array(D_kk_mockdata_test)).flatten()

print(D_kk_mockdata_test.shape)
del mockdata

k_max = config['specs']['scale_cuts']['max_GC'] # in h/Mpc
ell_cut = config['specs']['scale_cuts']['max_WL'] # in ell

# apply scale cuts
newdat = scale_cuts(cosmo_fid, ell_mockdata,D_mockdata, D_kk_mockdata_test, SRD_compare, k_max, ell_cut)

gauss_invcov_rotated = np.linalg.pinv(SRD_compare)

for i in range(len(newdat)):
    gauss_invcov_rotated[newdat[i]] = np.zeros(len(gauss_invcov_rotated[0]))
    gauss_invcov_rotated[:,newdat[i]] = np.zeros(len(gauss_invcov_rotated[0]))

# WITHOUT NOISE
C_ell_data_mock = [D_mockdata, ell_mockdata, z,  Binned_distribution_source,\
                   Binned_distribution_lens, ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata]


######################## LIKELIHOOD ################################

def log_likelihood(theta_dict, Data, invcovmat):
    theta = [float(theta_dict[key]) for key in ["Omega_m", "f_phi", "1e9As", "h", "ns", "Omega_b", "b1", "b2", "b3", "b4", "b5"]]

    Omega_m, f_phi, A_s1e9, h, n_s, Omega_b, b1, b2, b3, b4, b5 = theta 
    Bias_distribution = np.array([b1*np.ones(len(z)),
                             b2*np.ones(len(z)),
                             b3*np.ones(len(z)),
                             b4*np.ones(len(z)),
                             b5*np.ones(len(z))])
    #h = cosmo_universe["h"]
    #A_s = cosmo_universe["A_s"]
    A_s = A_s1e9*1e-9
    #n_s = cosmo_universe["n_s"]
    Omega_c = Omega_m - Omega_b

    cosmoMCMCstep = ccl.Cosmology(Omega_c = Omega_c, 
                      Omega_b = Omega_b,
                      h = h,
                      n_s = n_s,
                      A_s = A_s)
    return loglikelihood(Data, cosmoMCMCstep, f_phi, invcovmat, Bias_distribution)



#### Get Planck priors #####
sampler_Planck_arr = np.load("/global/homes/c/carolazn/CuGal_Emu_project_mcmc/Prior_Planck_arr.npy")
mu_prior = [cosmo_universe['n_s'], cosmo_universe["Omega_b"]*cosmo_universe["h"]**2]
cov_prior = np.cov(sampler_Planck_arr.T)

with open(config['data']['params'], 'r') as f:
        params = yaml.safe_load(f)

# add additional bespoke priors if needed
def log_prior(theta_dict):
    theta = [float(theta_dict[key]) for key in ["Omega_m", "f_phi", "1e9As", "h", "ns", "Omega_b", "b1", "b2", "b3", "b4", "b5"]]
    Omega_c, f_phi, A_s1e9, h, n_s, Omega_b, b1, b2, b3, b4, b5 = theta 
    priors = params['priors']
    if not priors['Planck_prior']:
        return 0.0
    gauss_funct = scipy.stats.multivariate_normal(mu_prior, cov_prior)
    
    return gauss_funct.logpdf([n_s, Omega_b*h**2])

# add standard priors
prior = Prior()
params_list = ["Omega_m", "f_phi", "1e9As", "h", "ns", "Omega_b", "b1", "b2", "b3", "b4", "b5"]
for par_i in params_list:
    prior.add_parameter(par_i, dist=(params['priors'][par_i][0], params['priors'][par_i][1]))


def log_probability(theta_dict):
    lp = log_prior(theta_dict)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta_dict, C_ell_data_mock, gauss_invcov_rotated)


################## RUN MCMC ########################
# Set the random seed for reproducibility
np.random.seed(10)

"""
# Initialize the walkers
Omega_m_est = 0.3
h_est = 0.68
A_s1e9_est = 3
n_s_est = 0.96
f_phi_est = 0.6
Omega_b_est = 0.0502
b1_est = Bias_distribution_fiducial[0][0] 
b2_est = Bias_distribution_fiducial[1][0]
b3_est = Bias_distribution_fiducial[2][0]
b4_est = Bias_distribution_fiducial[3][0]
b5_est = Bias_distribution_fiducial[4][0]

n_steps = 15000
chain_len = 100
converged = False
nwalkers = 40

# Initialize the walkers
pos = [Omega_m_est, f_phi_est, A_s1e9_est, h_est, n_s_est, Omega_b_est,b1_est,b2_est,b3_est,b4_est,b5_est] \
+ np.append(np.append(1e-3 * np.random.randn(nwalkers, 4), 1e-5*np.random.randn(nwalkers, 2), axis = 1), \
            1e-3 * np.random.randn(nwalkers, 5), axis = 1)

nwalkers, ndim = pos.shape
print(nwalkers, ndim)
"""
# Create the output directory and set up the HDF5 backend
mcmc_dir = "/global/homes/c/carolazn/CuGal_Emu_project_mcmc/mcmc"
#filename = mcmc_dir + "/ " + config['output']['chain_name'] + ".h5"
#backend = emcee.backends.HDFBackend(filename)

# Optionally reset the backend if starting a new run
#backend.reset(nwalkers, ndim)

print("Running MCMC...")
# Ensure the directories exist
os.makedirs('mcmc', exist_ok=True)
os.makedirs('mcmc/hdf5', exist_ok=True)

def main():
    sampler = Sampler(prior, log_probability, 
                      filepath= mcmc_dir + "/" + config['output']['chain_name'] +'.hdf5', resume=config['sampler']['mcmc']['resume'], n_live=config['sampler']['mcmc']['n_live'], pool=Pool(multiprocessing.cpu_count()))
    start = time.time()
    sampler.run(verbose=bool(config['sampler']['mcmc']['verbose']), discard_exploration=True, n_eff=float(config['sampler']['mcmc']['n_eff']))
    log_z = sampler.evidence()
    points, log_w, log_l = sampler.posterior()
    finish = time.time()
    chain_time = finish-start

    # save header information with Obervable = 3x2pt, LSST Y1, and infromation about scale cuts, parameter values and priors, datavector type/file/index

    header = "# Observable = 3x2pt, LSST Y1 \n" + "# Scale cuts: k_max = {k_max} h/Mpc, ell_max = {ell_cut} \n".format(k_max=k_max, ell_cut=ell_cut) + \
    "# Parameter values and priors: \n" + "# Omega_m = {Omega_m}, f_phi = {f_phi}, A_s = {A_s}, h = {h}, n_s = {n_s}, wb = {wb}, b1 = {b1}, b2 = {b2}, b3 = {b3}, b4 = {b4}, b5 = {b5} \n".format(
        Omega_m=cosmo_universe["Omega_m"], f_phi=f_phi_universe, A_s=cosmo_universe["A_s"], h=cosmo_universe["h"], n_s=cosmo_universe["n_s"], wb=cosmo_universe["Omega_b"], b1=Bias_distribution_fiducial[0][0], b2=Bias_distribution_fiducial[1][0], b3=Bias_distribution_fiducial[2][0], b4=Bias_distribution_fiducial[3][0], b5=Bias_distribution_fiducial[4][0]) + \
    "# Priors: \n" + "# Omega_m: {Omega_m_prior} \n".format(Omega_m_prior=str(params['priors']['Omega_m'])) + \
    "# f_phi: {f_phi_prior} \n".format(f_phi_prior=str(params['priors']['f_phi'])) + \
    "# A_s: {A_s_prior} \n".format(A_s_prior=str(params['priors']['1e9As'])) + \
    "# h: {h_prior} \n".format(h_prior=str(params['priors']['h'])) + \
    "# n_s: {n_s_prior} \n".format(n_s_prior=str(params['priors']['ns'])) + \
    "# Omega_b: {Omega_b_prior} \n".format(Omega_b_prior=str(params['priors']['Omega_b'])) + \
    "# b1: {b1_prior} \n".format(b1_prior=str(params['priors']['b1'])) + \
    "# b2: {b2_prior} \n".format(b2_prior=str(params['priors']['b2'])) + \
    "# b3: {b3_prior} \n".format(b3_prior=str(params['priors']['b3'])) + \
    "# b4: {b4_prior} \n".format(b4_prior=str(params['priors']['b4'])) + \
    "# b5: {b5_prior} \n".format(b5_prior=str(params['priors']['b5'])) + "Planck priors on n_s and wb: {Planck_prior} \n".format(Planck_prior=str(params['priors']['Planck_prior'])) + \
    "# Datavector type: {datavector_type} \n".format(datavector_type="from params file" if config['data']['type'] == 1 else "from cosmo file") + \
    "# Datavector file/index (if from cosmo file): {datavector_info} \n".format(datavector_info="file: " + config['data']['cosmo_file'] + ", index: " + str(config['data']['index']) if config['data']['type'] == 0 else "N/A")

    np.savetxt("chains/chain_"+config['output']['chain_name']+".txt", np.c_[points, log_w, log_l], header=header, footer='log_Z = {log_z};  chain_time = {chain_time} (--> {chain_time_hms} hh:mm:ss)'.format(log_z=log_z, chain_time=chain_time, chain_time_hms=timedelta(seconds=chain_time)))
    

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all pools are properly closed
        multiprocessing.active_children()


"""
with Pool(5) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend, pool=pool)
    pos = sampler.get_last_sample() if backend.iteration > 0 else pos

    while not converged:
        gc.collect(generation=0)
        sampler.run_mcmc(pos, chain_len, progress=True, store=True)
        
        # Clear references in the worker pool
        pool.close()
        pool.join()
        del pool  # Ensure the pool object is removed
        gc.collect()  # Collect any lingering garbage
        pool = Pool(5)
        sampler.pool = pool
        pos = sampler.get_last_sample() if backend.iteration > 0 else pos
        
        # Check convergence
        try:
            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * 100 < sampler.iteration)
        except emcee.autocorr.AutocorrError:
            pass
"""


