
"""Python functions to compute forecasts for the cubic Galileon model."""

##########################################################################
import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

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
import gc

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

# Loading CuGal files

Bk_all, Bk_all_smooth, k_all, z_all = load_boost_data()
Bk_lin_all, _, _ = load_boost_data_lin()
p_all = load_params()

## Data prep
z_index = 1

y_vals = Bk_all[:, z_index, :]

# y_ind = np.arange(0, y_vals.shape[1])
y_ind = k_all

# Load validation data

Bk_all_val, Bk_lin_all_val, _, _ = load_boost_data(LIBRARY_BK_FILE_VAL, LIBRARY_ZK_FILE_VAL)
target_vals = Bk_all_val[:, z_index, :]
input_params = load_params(LIBRARY_PARAM_FILE_VAL)

train_indices = [i for i in np.arange(49)] # if i not in test_indices]

p_all_train = p_all[train_indices]
y_vals_train = Bk_all[:, z_index, :][train_indices]

sepia_data = sepia_data_format(p_all_train, y_vals_train, y_ind)
model_filename = '/global/homes/c/carolazn/CubicGalileonEmu/CubicGalileonEmu/model/multivariate_model_z_index' + str(z_index) 

#sepia_model = do_pca(sepia_data, exp_variance=0.95)
#sepia_model = do_gp_train(sepia_model, model_filename)
#plot_train_diagnostics(sepia_model)

if if_train_all:
    
    do_gp_train_multiple(model_dir='/global/homes/c/carolazn/CubicGalileonEmu/CubicGalileonEmu/model/', 
                        p_train_all = p_all[train_indices],
                        y_vals_all = Bk_all_smooth[train_indices],
                        y_ind_all = k_all,
                        z_index_range=range(49))


sepia_model_list, sepia_data_list = load_model_multiple(model_dir='/global/homes/c/carolazn/CubicGalileonEmu/CubicGalileonEmu/model/', 
                                        p_train_all=p_all[train_indices],
                                        y_vals_all=Bk_all_smooth[train_indices],
                                        y_ind_all=k_all,
                                        z_index_range=range(50))


################# 1. Background LCDM and Cubic Galileon evolution ################

"""Define tracker functions"""

# If we are on the tracker, we will get the following constraints equations

# For today formalism 
def k_1T(Omg_m, Omg_r, f_phi):
    return 6*f_phi*(Omg_m + Omg_r - 1)

def wr_funct(T_CMB):
    return 4.48150052e-7*T_CMB**4 *(1+ 3.044*7/8 * (4/11)**(4/3))

T_CMB = 2.72548 # K
print(wr_funct(T_CMB))

"""Background in GR"""

# dimensionless hubble parameter in GR
def E_LCDM(cosmoMCMCStep, a):
    Omg_r = cosmoMCMCStep["Omega_g"]*(1+ 3.046*7/8 * (4/11)**(7/8))
    return np.sqrt(cosmoMCMCStep["Omega_m"]/a**3 +Omg_r/a**4 + (1 - cosmoMCMCStep["Omega_m"] - Omg_r))

# deriv. of E wrt scale factor, GR
def dEda(cosmo, a):
    Omg_r = cosmo["Omega_g"]*(1+ 3.046*7/8 * (4/11)**(7/8))
    E_val = E(cosmo, a)
    
    return (-3*cosmo["Omega_m"]/a**4 -4*Omg_r/a**5)/2/E_val


def initialize_Horndeski():
    """
    Generates and stores the Horndeski functions needed for simulations.
    This should be run once at the start.
    """
    global lambdified_functions  # Store in a global variable for later use

    to_exec = eb.declare_symbols()
    
    # Create a dictionary for execution
    local_dict = {"sym": sym}
    
    # Execute symbol declarations
    exec(to_exec, globals(), local_dict)  

    # Store declared symbols into globals()
    globals().update(local_dict)

    # Generate symbolic Horndeski functions
    lambdified_functions = lambda K, G3, G4, symbol_list, mass_ratio_list: eb.create_Horndeski(
        K, G3, G4, symbol_list, mass_ratio_list
    )

    print("Horndeski functions initialized.")

# Call this once at the start
initialize_Horndeski()


K = sym.Symbol('X')*sym.Symbol('k_1')#read_out_dict['K']
G3 = sym.Symbol('X')*sym.Symbol('g_31')#read_out_dict['G3']
G4 = 0.5#read_out_dict['G4']
symbol_list = ['k_1', 'g_31']# read_out_dict['symbol_list']
mass_ratio_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]#read_out_dict['mass_ratio_list']

# Generate the Horndeski functions
functions_CuGal_HiCOLA = lambdified_functions(K, G3, G4, symbol_list, mass_ratio_list)


def run_Horndeski_simulation(f_phi, cosmo):
    """
    Given the Horndeski and numerical ini files, solves the system and returns:
    - a_arr (scale factor array)
    - UE_arr (Hubble parameter)
    - coupling_factor_arr (Horndeski coupling factor)
    """
    Omg_L = (1. - f_phi)*(1. - cosmo['Omega_g']*(1+ 3.046*7/8 * (4/11)**(7/8)) - cosmo['Omega_m'])
    k1_track = k_1T(cosmo['Omega_m'], cosmo['Omega_g']*(1+ 3.046*7/8 * (4/11)**(7/8)), f_phi)

    # Read the parameters from the ini files
    read_out_dict = {'model_name': 'horndeski_model', 
                     'cosmo_name': 'run_X', 
                     'output_directory': '/global/homes/c/carolazn/CuGal_Emu_project_mcmc/HiCOLA_background/Output_validation', 
                     'K': sym.Symbol('X')*sym.Symbol('k_1'), 
                     'G3': sym.Symbol('X')*sym.Symbol('g_31'), 
                     'G4': 0.500000000000000, 
                     'mass_ratio_list': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                     'symbol_list': ['k_1', 'g_31'], 
                     'closure_declaration': ['parameters', 1], 
                     'simulation_parameters': [1000, 1000.0, False, 0.0, False], 
                     'threshold_value': 0.0, 
                     'GR_flag': False, 
                     'cosmological_parameters': [cosmo['Omega_g']*(1+ 3.046*7/8 * (4/11)**(7/8)), cosmo['Omega_m'], Omg_L], 
                     'Horndeski_parameters': [k1_track, 0.9], 'initial_conditions': [1.0, 1.0]}

    # Update from funct, defined further up
    read_out_dict.update(functions_CuGal_HiCOLA)
    
    # Ensure E, phiprime, omegar, omegam are accessible from globals()
    E = globals().get('E')
    phiprime = globals().get('phiprime')
    omegar = globals().get('omegar')
    omegam = globals().get('omegam')

    # Define 'odeint_parameter_symbols'
    odeint_parameter_symbols = [E, phiprime, omegar, omegam]
    read_out_dict.update({'odeint_parameter_symbols': odeint_parameter_symbols})
    # Run the solver
    background_quantities = ns.run_solver(read_out_dict)

    # Extract required arrays
    a_arr = background_quantities['a']
    UE_arr = background_quantities['Hubble']
    coupling_factor_arr = background_quantities['coupling_factor']

    return a_arr, UE_arr, coupling_factor_arr


def CuGal_initialize(f_phi, cosmo):
    if f_phi == 0.0:
        a_arr = np.logspace(0,-3,1000)
        return a_arr, E_LCDM(cosmo, a_arr), np.ones(1000)
    else:
        a_arr, UE_arr, coupling_factor_arr = run_Horndeski_simulation(f_phi,cosmo)
        return a_arr, UE_arr, coupling_factor_arr

def E_CuGal(a_arr, UE_arr, a):
    spl = CubicSpline(a_arr[::-1], UE_arr[::-1])
    return spl(a)

# mu(k,a) = mu(a) in CuGal (modified gravity parametrization parameter)
def mu_CuGal(a_arr, coupling_factor_arr, a):
    spl = CubicSpline(a_arr[::-1], coupling_factor_arr[::-1])
    return spl(a) + 1.0


############## 2. Define functions to get various (P(k)) from Emulators, etc. #################

def Get_Pk2D_obj_kk_GR_nl(cosmo):
   
    ########### Functions for linear matter power spectrum multiplied by Sigma**2 ###########        
    def pk_funcSigma2(k, a):
        return ccl.nonlin_matter_power(cosmo, k=k, a=a)

    return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma2, is_logp=False)

def Get_Pk2D_obj_kk_GR_lin(cosmo):
   
    ########### Functions for linear matter power spectrum multiplied by Sigma**2 ###########        
    def pk_funcSigma2(k, a):
        return ccl.linear_matter_power(cosmo, k=k, a=a)

    return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma2, is_logp=False)


# NL matter power spectra in fR
def P_k_NL_CuGal(GR_pk2D_obj, f_phi, cosmo, k, a):
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float or array) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    
    output Pk_fR (array) -> Nonlinear matter power spectrum for Hu-Sawicki fR gravity, units (Mpc)^3
    """
    if isinstance(a, (float, int)):  # Single scale factor case
        input_params_and_redshift = np.append(
            np.array([cosmo["Omega_m"], cosmo["n_s"], 1e9 * cosmo["A_s"], cosmo["h"], f_phi]),
            1.0 / a - 1.0
        )
        bk_target, err_target = emu_redshift(input_params_and_redshift[np.newaxis, :], sepia_model_list,sepia_data_list, z_all)
        interp_func = scipy.interpolate.interp1d(k_all * cosmo["h"], bk_target.flatten(), kind='linear', fill_value="extrapolate")
        pkratio_CuGal = interp_func(k)
        
    else:
        bk_target = []
        z_range = 1.0 / a - 1.0  # Array of redshift values

        # Loop over each redshift value
        for z_val in z_range:
            input_params_and_redshift = np.append(
                np.array([cosmo["Omega_m"], cosmo["n_s"], 1e9 * cosmo["A_s"], cosmo["h"], f_phi]),
                z_val
            )
            bk_target_i, _ = emu_redshift(input_params_and_redshift[np.newaxis, :], sepia_model_list,sepia_data_list, z_all)
            bk_target.append(bk_target_i.flatten()) 
    
        # Convert list to array with shape (len(a), len(k_all))
        bk_target = np.array(bk_target)

        
        # Interpolating each row in bk_target over k
        pkratio_CuGal = np.array([
            scipy.interpolate.interp1d(k_all * cosmo["h"], bk_row, kind='linear', fill_value="extrapolate")(k) 
            for bk_row in bk_target
        ])
    

    Pk_ccl = GR_pk2D_obj.__call__(k, a=a) # units (Mpc)^3
    Pk = pkratio_CuGal*Pk_ccl
    
    return Pk



"""Linear matter power spectra CuGal"""

def P_k_CuGal_lin(GR_pk2D_obj,f_phi, cosmo, k, a):
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    
    output Pk_fR (array) -> Nonlinear matter power spectrum for Hu-Sawicki fR gravity, units (Mpc)^3
    """

    input_params_and_redshift = np.append(np.array([cosmo["Omega_m"],cosmo["n_s"],1e9*cosmo["A_s"],cosmo["h"],f_phi]) , 1.0/a -1.0)     
    bk_target, err_target = emu_redshift(input_params_and_redshift[np.newaxis, :], sepia_model_list,sepia_data_list, z_all)
    
    pkratio_CuGal = bk_target[0]

    Pk_ccl = GR_pk2D_obj.__call__(k, a=a) # units (Mpc)^3
    Pk = pkratio_CuGal*Pk_ccl

    return Pk



def sigma_8_CuGal(GR_pk2D_obj,f_phi, cosmo, a_array):
    k_val = np.logspace(-4, 3, 3000)
    sigma_8_vals = []

    for a in a_array:
        P_k_vals = P_k_CuGal_lin(GR_pk2D_obj,f_phi, cosmo, k_val, a)
        j1_vals = 3 * scipy.special.spherical_jn(1, k_val * 8 / cosmo["h"], derivative=False) / (k_val * 8 / cosmo["h"])
        integrand = k_val**2 * P_k_vals * j1_vals**2
        integral_val = scipy.integrate.trapz(integrand, x=k_val)
        sigma_8_val = np.sqrt(integral_val / (2 * np.pi**2))
        sigma_8_vals.append(sigma_8_val)
    
    return np.array(sigma_8_vals)

def solverGrowth_CuGal(y,a,a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo):
    E_val = E_CuGal(a_arr, UE_arr, a)
    D , a3EdDda = y
    
    mu = mu_CuGal(a_arr, coupling_factor_arr, a)
    
    ydot = [a3EdDda / (E_val*a**3), 3*cosmo["Omega_m"]*D*(mu)/(2*E_val*a**2)]
    return ydot
    
def fsigma8_CuGal(GR_pk2D_obj,a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo, a):
    
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    """
    
    
    a_solver = np.linspace(1/50,1,100)
    Soln = odeint(solverGrowth_CuGal, [a_solver[0], (E_CuGal(a_arr, UE_arr, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(a_arr, UE_arr, coupling_factor_arr,f_phi, cosmo), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    a3EdDda = Soln.T[1]

    f_CuGal_interp = a3EdDda/a_solver**2 / Delta / E_CuGal(a_arr, UE_arr, a_solver)
    
    f_CuGal = np.interp(a, a_solver, f_CuGal_interp)

    return f_CuGal * sigma_8_CuGal(GR_pk2D_obj,f_phi,cosmo, a)

def growthfactor_CuGal(a_arr, UE_arr, coupling_factor_arr,f_phi, cosmo, a):
        
    a_solver = np.linspace(1/50,1,100)
    Soln = odeint(solverGrowth_CuGal, [a_solver[0], (E_CuGal(a_arr, UE_arr, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(a_arr, UE_arr, coupling_factor_arr,f_phi, cosmo), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    return np.interp(a,a_solver,Delta/Delta[-1])

def growthrate_CuGal(a_arr, UE_arr, coupling_factor_arr,f_phi, cosmo, a):
        
    a_solver = np.linspace(1/50,1,100)
    Soln = odeint(solverGrowth_CuGal, [a_solver[0], (E_CuGal(a_arr, UE_arr, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(a_arr, UE_arr, coupling_factor_arr,f_phi, cosmo), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    a3EdDda = Soln.T[1]

    f_CuGal_interp = a3EdDda/a_solver**2 / Delta / E_CuGal(a_arr, UE_arr, a_solver)
    
    f_CuGal = np.interp(a, a_solver, f_CuGal_interp)

    return f_CuGal



################# 3. Getting C(ell) functions #################

"""Get n_zbins logarithmically spaced ell bins (total n of ell bins = ell_bin_num)"""
def bin_ell_kk(ell_min, ell_max, ell_bin_num, Binned_distribution):
    # define quantities for binning in ell
    n_zbins = int(((len(Binned_distribution)+1)*len(Binned_distribution))/2)
    ell_binned_limits = np.linspace(np.log10(ell_min),np.log10(ell_max),num=ell_bin_num + 1)
    bin_edge1 = ell_binned_limits[:-1]
    bin_edge2 = ell_binned_limits[1:]
    ell_binned = 10**((bin_edge1 + bin_edge2) / 2)

    # Repeat ell_binned over all redshift bins, so that len(ell_binned)=len(C_ell_array)
    ell_binned = np.repeat([ell_binned], repeats=n_zbins, axis=0)
    
    #ell_binned = list(islice(cycle(ell_binned), int(ell_bin_num*((len(Binned_distribution)+1)*len(Binned_distribution))/2)))
    return ell_binned
    
"""Get n_zbins logarithmically spaced ell bins (total n of ell bins = ell_bin_num)"""
def bin_ell_delk(ell_min, ell_max, ell_bin_num,Binned_distribution_s, Binned_distribution_l):
    # define quantities for binning in ell
    n_zbins = 0
    for j in range(len(Binned_distribution_l)):
        for k in range(len(Binned_distribution_s)):
            if k - 1 > j or (k == 4 and j == 3):
                n_zbins += 1
    
    ell_binned_limits = np.linspace(np.log10(ell_min),np.log10(ell_max),num=ell_bin_num + 1)
    bin_edge1 = ell_binned_limits[:-1]
    bin_edge2 = ell_binned_limits[1:]
    ell_binned = 10**((bin_edge1 + bin_edge2) / 2)

    # Repeat ell_binned over all redshift bins, so that len(ell_binned)=len(C_ell_array)
    ell_binned = np.repeat([ell_binned], repeats=n_zbins, axis=0)
    
    #ell_binned = list(islice(cycle(ell_binned), int(ell_bin_num*((len(Binned_distribution)+1)*len(Binned_distribution))/2)))
    return ell_binned

"""Get n_zbins logarithmically spaced ell bins (total n of ell bins = ell_bin_num)"""
def bin_ell_deldel(ell_min, ell_max, ell_bin_num, Binned_distribution):
    # define quantities for binning in ell
    n_zbins = len(Binned_distribution)
    ell_binned_limits = np.linspace(np.log10(ell_min),np.log10(ell_max),num=ell_bin_num + 1)
    bin_edge1 = ell_binned_limits[:-1]
    bin_edge2 = ell_binned_limits[1:]
    ell_binned = 10**((bin_edge1 + bin_edge2) / 2)

    # Repeat ell_binned over all redshift bins, so that len(ell_binned)=len(C_ell_array)
    ell_binned = np.repeat([ell_binned], repeats=n_zbins, axis=0)
    
    #ell_binned = list(islice(cycle(ell_binned), int(ell_bin_num*((len(Binned_distribution)+1)*len(Binned_distribution))/2)))
    return ell_binned

def bins(ell_min, ell_max, ell_bin_num):

    # define quantities for binning in ell
    ell_binned_limits = np.linspace(np.log10(ell_min),np.log10(ell_max),num=ell_bin_num + 1)
    bin_edge1 = ell_binned_limits[:-1]
    bin_edge2 = ell_binned_limits[1:]
    ell_binned = 10**((bin_edge1 + bin_edge2) / 2)
    # Repeat ell_binned over all redshift bins, so that len(ell_binned)=len(C_ell_array)
    return ell_binned


############### for GR ######################

"""Functions to find Cell given a Pdelta_2D ccl object  - GR"""

# A: Function for cosmic shear angular power spectrum (lensing-lensing C_ell) from a given P_delta2D_S
def C_ell_arr_kk_GR(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution):
    C_ell_array = []
    n_zbins = int(((len(Binned_distribution_s)+1)*len(Binned_distribution_s))/2)
    # how far along z binning we are
    idx = 0
    # at what z bin we start calculating Cell
    start_idx = n_zbins - len(ell_binned)

    for j in range(len(Binned_distribution_s)):
        tracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z, Binned_distribution_s[j]))
        for k in range(len(Binned_distribution_s)):
            if k >= j:
                if start_idx <= idx:
                    tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z, Binned_distribution_s[k]))
                    C_ell = ccl.angular_cl(cosmo, tracer1, tracer2, ell_binned[idx - start_idx])
                    C_ell_array.append([C_ell])
                    idx += 1
                else:
                    idx += 1
    return C_ell_array

# B: Function for galaxy-galaxy lensing angular power spectrum (clustering-lensing C_ell) from a given P_delta2D_S
def C_ell_arr_delk_GR(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution):
    C_ell_array = []
    
    n_zbins = 0
    for j in range(len(Binned_distribution_l)):
        for k in range(len(Binned_distribution_s)):
            if k - 1 > j or (k == 4 and j == 3):
                n_zbins += 1
                
    # how far along z binning we are
    idx = 0
    # at what z bin we start calculating Cell
    start_idx = n_zbins - len(ell_binned)

    for j in range(len(Binned_distribution_l)):
        tracer1 = ccl.NumberCountsTracer(cosmo, dndz=(z, Binned_distribution_l[j]), bias=(z, Bias_distribution[j]), has_rsd=False)
        for k in range(len(Binned_distribution_s)):
            if k - 1 > j or (k == 4 and j == 3):
                if start_idx <= idx:
                    tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z, Binned_distribution_s[k]))
                    C_ell = ccl.angular_cl(cosmo, tracer1, tracer2, ell_binned[idx - start_idx])
                    C_ell_array.append([C_ell])
                    idx += 1
                else:
                    idx += 1
    return C_ell_array

# C: Function for galaxy-galaxy clustering angular power spectrum (clustering-clustering C_ell) from a given P_delta2D_S
def C_ell_arr_deldel_GR(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution):
    C_ell_array = []
    n_zbins = len(Binned_distribution_l)
    # how far along z binning we are
    idx = 0
    # at what z bin we start calculating Cell
    start_idx = n_zbins - len(ell_binned)

    for j in range(len(Binned_distribution_l)):
        tracer1 = ccl.NumberCountsTracer(cosmo, dndz=(z, Binned_distribution_l[j]), bias=(z, Bias_distribution[j]), has_rsd=False)
        for k in range(len(Binned_distribution_l)):
            if k == j:
                if start_idx <= idx:
                    tracer2 = ccl.NumberCountsTracer(cosmo, dndz=(z, Binned_distribution_l[k]), bias=(z, Bias_distribution[k]), has_rsd=False)
                    C_ell = ccl.angular_cl(cosmo, tracer1, tracer2, ell_binned[idx - start_idx])
                    C_ell_array.append([C_ell])
                    idx += 1
                else:
                    idx += 1
    return C_ell_array


def Cell_GR(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution,
         tracer1_type="k", 
         tracer2_type="k"):
    """
    Finds C^{i,j}(ell) for {i,j} redshift bins.
    tracer_type = "k", "g"
    linear = True, False
    if tracer1_type = "k" and tracer2_type = "k", shape-shape angular power spectrum
    if tracer1_type = "k" and tracer2_type = "g", galaxy-galaxy lensing angular power spectrum
    if tracer1_type = "g" and tracer2_type = "g", pos-pos angular power spectrum
    if linear=True, use linear matter power spectrum to compute the angular one, otherwise use the non-linear
    input:
        ell_binned: array of ell bins for the full C{ij}(ell) range (for all i and j), with scale cuts included
        cosmo: ccl cosmology object
        redshift z: numpy.array with dim:N
        Binned_distribution_s: numpy.array with dim:(N,M) (M = no. source z bins)
        Binned_distribution_l: numpy.array with dim:(N,L) (L = no. lens z bins)
        Bias_distribution: numpy.array with dim:(N,L) (galaxy bias)
    returns:
        ell bins: numpy.array (dim = dim C_ell)
        C_ell: numpy.array
    """

    ops = {
        ("k" , "k"): C_ell_arr_kk_GR,
        ("k" , "g"): C_ell_arr_delk_GR, 
        ("g" , "k"): C_ell_arr_delk_GR,
        ("g" , "g"): C_ell_arr_deldel_GR
    }

    def invalid_op2():
        raise ValueError('invalid tracer selected.')
    ########## Find Cell ##########

    C_ell_array_funct = ops.get((tracer1_type, tracer2_type), invalid_op2)
    C_ell_array = C_ell_array_funct(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution)

    return np.array(list(itertools.chain(*ell_binned))), C_ell_array


############## for CuGal ############

"""Functions to find Cell for Cubic Galileon given a Pdelta_2D ccl object"""


"""Comoving radial distance in Cubic Galileon"""
def comoving_radial_dist_CuGal(a_arr, UE_arr, cosmo, a_array):
    c = 3e5  # Speed of light in km/s
    
    # Define the redshift integral range
    #z_integral = np.linspace(1/a_array.min() - 1, 0, int(1e4))  # Use the minimum value of `a_array`
    x_integral = np.linspace(np.log(a_array.min()), 0, int(2e2)) # Compute the scale factor over the range
    a_integral = np.exp(x_integral)
    z_integral = 1/a_integral - 1
    
    # Calculate E(a_integral) only once over the entire range
    E_val = E_CuGal(a_arr, UE_arr, a_integral)

    # Define the integrand for each value of a
    integrand = c / (a_integral * E_val * cosmo["H0"])
    
    # Now integrate over the entire range for each value in `a_array`
    results = []
    for a_iter in a_array:
        z_lower_bound = 1/a_iter - 1  # Adjust the upper bound of the integration
        mask = (z_integral <= z_lower_bound)  # Mask the integrand for the valid integration range
        
        # Perform integration for the valid portion of the integrand
        integral = scipy.integrate.simpson(integrand[mask], x_integral[mask])
        results.append(integral)

    return np.array(results)


def Cell_CuGal(ell_binned, a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo_GR, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution,
         GR_pk2D_obj, tracer1_type="k", 
         tracer2_type="k"):
    # Define the scale factor array
    a_array = np.logspace(np.log10(1/14), 0, 50)
    
    # Compute chi using the comoving radial distance function
    chi_array = comoving_radial_dist_CuGal(a_arr, UE_arr, cosmo_GR, a_array)
    
    # Compute h_over_h0 using the Hubble expansion rate function
    h_over_h0_array = E_CuGal(a_arr, UE_arr, a_array)
    
    # Create the background dictionary
    background_dict = {
        'a': a_array,
        'chi': chi_array,
        'h_over_h0': h_over_h0_array
    }

    growthfact_array = growthfactor_CuGal(a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo_GR, a_array)
    growthrate_array = growthrate_CuGal(a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo_GR, a_array)
    
    # Create the growth dictionary
    growth_dict = {
        'a': a_array,
        'growth_factor': growthfact_array,
        'growth_rate': growthrate_array
    }
    k_array = np.logspace(-4,3,100)

    """
    # Split a_array based on the condition a < 1/12
    a_threshold = 1/12
    a_GR = a_array[a_array < a_threshold]
    a_MG = a_array[a_array >= a_threshold]

    # Compute P(k, a) for a < 1/12 using P_k_GR
    Pk_GR_array = np.array([ccl.power.nonlin_power(cosmo_GR, k_array, a) for a in a_GR])
    """
    # Compute P(k, a) for a >= 1/12 using P_k_NL_CuGal
    Pk_NL_array = P_k_NL_CuGal(GR_pk2D_obj,f_phi,cosmo_GR, k_array, a_array) #np.array([P_k_NL_CuGal(f_phi,cosmo_GR, k_array, a) for a in a_array])

    # Combine the results back into a single array
    #Pk_NL_array = np.vstack((Pk_GR_array, Pk_NL_CuGal_array))

    Pk_NL_dict = {
        'a': a_array,
        'k': k_array,
        'delta_matter:delta_matter': Pk_NL_array,
    }
    
    cosmo = ccl.cosmology.CosmologyCalculator(Omega_c = cosmo_GR["Omega_c"],
                                              Omega_b = cosmo_GR["Omega_b"],
                                              h = cosmo_GR["h"],
                                              n_s = cosmo_GR["n_s"],
                                              A_s = cosmo_GR["A_s"],
                                              background = background_dict,
                                              growth = growth_dict,
                                              pk_nonlin = Pk_NL_dict)
    
    ops = {
        ("k" , "k"): C_ell_arr_kk_GR,
        ("k" , "g"): C_ell_arr_delk_GR, 
        ("g" , "k"): C_ell_arr_delk_GR,
        ("g" , "g"): C_ell_arr_deldel_GR
    }

    def invalid_op2():
        raise ValueError('invalid tracer selected.')
    ########## Find Cell ##########

    C_ell_array_funct = ops.get((tracer1_type, tracer2_type), invalid_op2)
    C_ell_array = C_ell_array_funct(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution)

    return np.array(list(itertools.chain(*ell_binned))), C_ell_array


def Cell_CuGal(ell_binned, a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo_GR, z, 
               Binned_distribution_s, Binned_distribution_l, Bias_distribution,
               GR_pk2D_obj, tracer1_type="k", tracer2_type="k"):
    # Define the scale factor array
    a_array = np.logspace(np.log10(1/14), 0, 50)
    
    # Compute chi using the comoving radial distance function
    chi_array = comoving_radial_dist_CuGal(a_arr, UE_arr, cosmo_GR, a_array)
    
    # Compute h_over_h0 using the Hubble expansion rate function
    h_over_h0_array = E_CuGal(a_arr, UE_arr, a_array)
    
    # Create the background dictionary
    background_dict = {
        'a': a_array,
        'chi': chi_array,
        'h_over_h0': h_over_h0_array
    }

    growthfact_array = growthfactor_CuGal(a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo_GR, a_array)
    growthrate_array = growthrate_CuGal(a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo_GR, a_array)
    
    # Create the growth dictionary
    growth_dict = {
        'a': a_array,
        'growth_factor': growthfact_array,
        'growth_rate': growthrate_array
    }

    k_array = np.logspace(-4, 3, 100)
    
    # Compute P(k, a) separately for different tracer combinations
    mu_cugal_val = np.repeat(mu_CuGal(a_arr, coupling_factor_arr, a_array)[:, np.newaxis], 100, axis=1)
    Pk_NL_kk = mu_cugal_val**2 * P_k_NL_CuGal(GR_pk2D_obj,f_phi,cosmo_GR, k_array, a_array)
    Pk_NL_kg = mu_cugal_val * P_k_NL_CuGal(GR_pk2D_obj,f_phi,cosmo_GR, k_array, a_array)
    Pk_NL_gg = P_k_NL_CuGal(GR_pk2D_obj,f_phi,cosmo_GR, k_array, a_array)

    # Dictionary to store the correct P(k, a) choice
    Pk_NL_dict_map = {
        ("k", "k"): Pk_NL_kk,
        ("k", "g"): Pk_NL_kg,
        ("g", "k"): Pk_NL_kg,
        ("g", "g"): Pk_NL_gg
    }

    # Select the correct Pk_NL array
    Pk_NL_selected = Pk_NL_dict_map.get((tracer1_type, tracer2_type))
    if Pk_NL_selected is None:
        raise ValueError(f"Invalid tracer combination: ({tracer1_type}, {tracer2_type})")

    Pk_NL_dict = {
        'a': a_array,
        'k': k_array,
        'delta_matter:delta_matter': Pk_NL_selected,
    }
    
    # Create the cosmology object
    cosmo = ccl.cosmology.CosmologyCalculator(
        Omega_c=cosmo_GR["Omega_c"],
        Omega_b=cosmo_GR["Omega_b"],
        h=cosmo_GR["h"],
        n_s=cosmo_GR["n_s"],
        A_s=cosmo_GR["A_s"],
        background=background_dict,
        growth=growth_dict,
        pk_nonlin=Pk_NL_dict
    )
    
    # Mapping of tracer combinations to their respective power spectra
    ops = {
        ("k", "k"): C_ell_arr_kk_GR,
        ("k", "g"): C_ell_arr_delk_GR, 
        ("g", "k"): C_ell_arr_delk_GR,
        ("g", "g"): C_ell_arr_deldel_GR
    }

    def invalid_op2():
        raise ValueError('Invalid tracer selection.')

    ########## Compute C_ell ##########

    C_ell_array_funct = ops.get((tracer1_type, tracer2_type), invalid_op2)
    C_ell_array = C_ell_array_funct(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l, Bias_distribution)

    return np.array(list(itertools.chain(*ell_binned))), C_ell_array

### Add a function that lets you use the specific P(k,a) for validation instead of the one from the emulator

def Cell_CuGal_Validation(ell_binned, a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo_GR, z, 
               Binned_distribution_s, Binned_distribution_l, Bias_distribution, 
               GR_pk2D_obj, Bk_CuGal_cosmo_funct, tracer1_type="k", tracer2_type="k"):
    # Define the scale factor array
    a_array = np.logspace(np.log10(1/14), 0, 50)
    
    # Compute chi using the comoving radial distance function
    chi_array = comoving_radial_dist_CuGal(a_arr, UE_arr, cosmo_GR, a_array)
    
    # Compute h_over_h0 using the Hubble expansion rate function
    h_over_h0_array = E_CuGal(a_arr, UE_arr, a_array)
    
    # Create the background dictionary
    background_dict = {
        'a': a_array,
        'chi': chi_array,
        'h_over_h0': h_over_h0_array
    }

    growthfact_array = growthfactor_CuGal(a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo_GR, a_array)
    growthrate_array = growthrate_CuGal(a_arr, UE_arr, coupling_factor_arr, f_phi, cosmo_GR, a_array)
    
    # Create the growth dictionary
    growth_dict = {
        'a': a_array,
        'growth_factor': growthfact_array,
        'growth_rate': growthrate_array
    }

    k_array = np.logspace(-4, 3, 100)

    Pk_ccl = GR_pk2D_obj.__call__(k_array, a=a_array)

    # Compute P(k, a) separately for different tracer combinations
    mu_cugal_val = np.repeat(mu_CuGal(a_arr, coupling_factor_arr, a_array)[:, np.newaxis], 100, axis=1)
    Pk_NL_kk = mu_cugal_val**2 * Bk_CuGal_cosmo_funct(1/a_array[::-1] - 1.0, k_array)[::-1, :] * Pk_ccl
    Pk_NL_kg = mu_cugal_val * Bk_CuGal_cosmo_funct(1/a_array[::-1] - 1.0, k_array)[::-1, :] * Pk_ccl
    Pk_NL_gg = Bk_CuGal_cosmo_funct(1/a_array[::-1] - 1.0, k_array)[::-1, :] * Pk_ccl

    # Dictionary to store the correct P(k, a) choice
    Pk_NL_dict_map = {
        ("k", "k"): Pk_NL_kk,
        ("k", "g"): Pk_NL_kg,
        ("g", "k"): Pk_NL_kg,
        ("g", "g"): Pk_NL_gg
    }

    # Select the correct Pk_NL array
    Pk_NL_selected = Pk_NL_dict_map.get((tracer1_type, tracer2_type))
    if Pk_NL_selected is None:
        raise ValueError(f"Invalid tracer combination: ({tracer1_type}, {tracer2_type})")

    Pk_NL_dict = {
        'a': a_array,
        'k': k_array,
        'delta_matter:delta_matter': Pk_NL_selected,
    }
    
    # Create the cosmology object
    cosmo = ccl.cosmology.CosmologyCalculator(
        Omega_c=cosmo_GR["Omega_c"],
        Omega_b=cosmo_GR["Omega_b"],
        h=cosmo_GR["h"],
        n_s=cosmo_GR["n_s"],
        A_s=cosmo_GR["A_s"],
        background=background_dict,
        growth=growth_dict,
        pk_nonlin=Pk_NL_dict
    )
    
    # Mapping of tracer combinations to their respective power spectra
    ops = {
        ("k", "k"): C_ell_arr_kk_GR,
        ("k", "g"): C_ell_arr_delk_GR, 
        ("g", "k"): C_ell_arr_delk_GR,
        ("g", "g"): C_ell_arr_deldel_GR
    }

    def invalid_op2():
        raise ValueError('Invalid tracer selection.')

    ########## Compute C_ell ##########

    C_ell_array_funct = ops.get((tracer1_type, tracer2_type), invalid_op2)
    C_ell_array = C_ell_array_funct(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l, Bias_distribution)

    return np.array(list(itertools.chain(*ell_binned))), C_ell_array




# log likelihood - only baryonic cuts applied
def loglikelihood(Data, cosmo, f_phi, InvCovmat, Bias_distribution):
    #start = time.time()
        
    # Extract 3x2pt data vector
    D_data, ell_mockdata, z, Binned_distribution_s,Binned_distribution_l,\
                   ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata = Data

    # Initialize CuGal stuff
    a_setup_mcmc, UE_setup_mcmc, coupling_setup_mcmc = CuGal_initialize(f_phi, cosmo)
    P_delta2D_GR_lin_mcmc = Get_Pk2D_obj_kk_GR_lin(cosmo)
    P_delta2D_GR_nl_mcmc = Get_Pk2D_obj_kk_GR_nl(cosmo)
    # shape-shape
    binned_ell_kk = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_s)

    # shape-pos
    binned_ell_delk = bin_ell_delk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, \
                              Binned_distribution_s,Binned_distribution_l)

    # pos-pos
    binned_ell_deldel = bin_ell_deldel(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_l)

    
    ########## Get theoretical data vector for single MCMC step - linear , muSigmaparam ##########
    # shape-shape
    D_theory_kk = np.array(Cell_CuGal(binned_ell_kk,a_setup_mcmc, UE_setup_mcmc, coupling_setup_mcmc,f_phi,cosmo, z , Binned_distribution_s,Binned_distribution_l,\
                      Bias_distribution,P_delta2D_GR_nl_mcmc, tracer1_type="k", tracer2_type="k")[1]).flatten()
   
    # shape-pos
    
    D_theory_delk = np.array(Cell_CuGal(binned_ell_delk,a_setup_mcmc, UE_setup_mcmc, coupling_setup_mcmc,f_phi,cosmo, z , Binned_distribution_s,Binned_distribution_l,\
                      Bias_distribution,P_delta2D_GR_nl_mcmc, tracer1_type="g", tracer2_type="k")[1]).flatten()

    # pos-pos

    D_theory_deldel = np.array(Cell_CuGal(binned_ell_deldel,a_setup_mcmc, UE_setup_mcmc, coupling_setup_mcmc,f_phi,cosmo, z , Binned_distribution_s,Binned_distribution_l,\
                      Bias_distribution,P_delta2D_GR_nl_mcmc, tracer1_type="g", tracer2_type="g")[1]).flatten()


    D_theory = np.append(np.append(D_theory_kk, D_theory_delk), D_theory_deldel)
    
    Diff = (D_data - D_theory)

    #print("time = ", time.time() - start)
    #### fsigma8 ####
    #Diff_fsigma8 = fsigma_8_dataset - fsigma8_CuGal(P_delta2D_GR_lin_mcmc,a_setup_mcmc, UE_setup_mcmc, coupling_setup_mcmc,f_phi,cosmo, 1/(z_fsigma8+1))
    #loglik_fsigma8 = -0.5*(np.matmul(np.matmul(Diff_fsigma8,invcovariance_fsigma8),Diff_fsigma8))

    return -0.5*(np.matmul(np.matmul(Diff,InvCovmat),Diff)) #+ loglik_fsigma8 

def cov2corr(cov):
    """
    Convert a covariance matrix into a correlation matrix
    input:
        cov: numpy.array with dim:(N,N)
    returns:
        corr: numpy.array with dim:(N,N)
    """
    sig = np.sqrt(cov.diagonal())
    return cov/np.outer(sig, sig)



"""Get list of lists rather than 1d array for non-uniform ell spacing"""
def ell_arrayfromlist(list):
    list_new = [[]]
    idx = 0
    for i in range(len(list)):
        if list[(i+1) % (len(list))] <= list[i]:
            list_new[idx].append(list[i])
            idx += 1
            list_new.append([])
        else:
            list_new[idx].append(list[i])
    del list_new[-1]
    return list_new


def scale_cuts(cosmo, ell, dvec_full, dvec_shear, cov_full, k_max, ell_cut):
    """ 
    Modified function from Danielle.
    Applies scale cuts from max k and ell values
    """

    # Save originals for index comparison
    dvec_full_in = dvec_full.copy()
    dvec_shear_in = dvec_shear.copy()
    cov_in = cov_full.copy()

    #### first cuts - clustering ######

    delk_z_array = np.array([0.30,0.30,0.30,0.50,0.50,0.70,0.70])
    deldel_z_array = np.array([0.30,0.50,0.70,0.90,1.10])
    z_array = np.append(delk_z_array, deldel_z_array)

    chi = ccl.background.comoving_radial_distance(cosmo, 1/(z_array+1))
    ellmax = k_max * chi - 0.5

    starting_index = len(dvec_shear)
    len_ell_ranges = int((len(cov_full[0]) - len(dvec_shear)) / len(z_array))
    idx_count = 0

    for j in range(len(z_array)):
        for i in range(len_ell_ranges):
            if ell[starting_index + i] >= ellmax[j]:
                cut_ind = starting_index + j*len_ell_ranges + i - idx_count
                cov_full = np.delete(np.delete(cov_full, cut_ind, axis=0), cut_ind, axis=1)
                dvec_full = np.delete(dvec_full, cut_ind)
                idx_count += 1

    #### second cuts - lensing (NEW: hard ell cut) ######
    # Indices (within lensing block) to remove
    lensing_inds_to_cut = np.where(ell[:len(dvec_shear)] > ell_cut)[0]

    # Apply cuts
    dvec_shear = np.delete(dvec_shear, lensing_inds_to_cut)
    dvec_full  = np.delete(dvec_full,  lensing_inds_to_cut)

    cov_full = np.delete(np.delete(cov_full,
                                   lensing_inds_to_cut, axis=0),
                                   lensing_inds_to_cut, axis=1)

    #### get excluded indices ######

    ex_inds = [i for i in range(len(dvec_full_in))
               if dvec_full_in[i] not in dvec_full]

    print('ex_inds=', ex_inds)
    return ex_inds
