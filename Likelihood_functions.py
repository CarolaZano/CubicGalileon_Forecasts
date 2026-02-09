""" import useful functions """

# Generic
import numpy as np
import scipy
import os
import sys
from scipy.integrate import odeint
import itertools
from functools import lru_cache
import scipy.integrate
from scipy.interpolate import RectBivariateSpline

# cosmology
import pyccl as ccl
import time

# SRD Binning
import srd_redshift_distributions as srd
import binning

# MCMC
import emcee

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

###################################################################
################ INITIALISE CUBIC GALILEON EMULATOR ###############
###################################################################

# Loading CuGal files
Bk_all, k_all, z_all = load_boost_data()
p_all = load_params()

# Initialise CuGal emulator.
z_index = 1
train_indices = [i for i in  np.arange(49)] # if i not in test_indices]
p_all_train = p_all[train_indices]
y_vals_train = Bk_all[:, z_index, :][train_indices]
y_ind = k_all
sepia_data = sepia_data_format(p_all_train, y_vals_train, y_ind)

model_filename = '/home/c2042999/CubicGalileonEmu/CubicGalileonEmu/model/multivariate_model_z_index' + str(z_index) 
sepia_model = do_pca(sepia_data, exp_variance=0.95)
sepia_model = do_gp_train(sepia_model, model_filename)
plot_train_diagnostics(sepia_model)

sepia_model_list = load_model_multiple(model_dir='/home/c2042999/CubicGalileonEmu/CubicGalileonEmu/model/', 
                                        p_train_all=p_all[train_indices],
                                        y_vals_all=Bk_all[train_indices],
                                        y_ind_all=k_all,
                                        z_index_range=range(49), 
                                        sepia_model_i=sepia_model)


###################################################################
######## CUBIC GALILEON BACKGROUND AND GROWTH FUNCTIONS ###########
###################################################################
# If we are on the tracker, we will get the following constraints 
# equations for today formalism 
def k_1T(Omg_m, Omg_r, f_phi):
    return 6*f_phi*(Omg_m + Omg_r - 1)

def wr_funct(T_CMB):
    return 4.48150052e-7*T_CMB**4 *(1+ 3.044*7/8 * (4/11)**(4/3))

T_CMB = 2.72548 # K
print(wr_funct(T_CMB))

# ~~~~ Define useful functions for getting background from HiCOLA ~~~~

def hypercube_arrays(txt):
    """
    Takes in txt file with (omega_m,n_s,A_s,h,f_phi)
    returns two arrays, one for numerical and one for horndeski array
    # numerical - [h , wr , wb , wc, n_s , A_s]
    # horndeski - [f_phi, H0, k_1, g_31]
    """
    hcube = np.loadtxt(txt).T

    wb = 0.0223*np.ones(len(hcube[0]))
    wr = wr_funct(T_CMB)*np.ones(len(hcube[0]))
    
    n_s = hcube[1]
    A_s = hcube[2]
    h = hcube[3]
    wm = hcube[0]*h**2


    arr_num = [h, wr, wb, wm - wb, n_s, A_s]
    
    return arr_num

def Update_ini(f_phi, arr_num, horn_ini, num_ini):
    """
    Updates files num_ini and horn_ini to include the parameters (numerical and MG) from 
    f_phi and
    arr_num = h , wr , wb , wc, n_s , A_s
    """
    ## Horndeski params
    h , wr , wb , wc, n_s , A_s = arr_num
    config_h = ConfigObj(horn_ini)
    
    k1_track = k_1T((wc + wb)/h**2, wr/h**2, f_phi)

    keys_h = config_h.keys()
    
    config_h[keys_h[1]] = '/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/Backend_input_parameter_files/Output_validation_edges'
    config_h[keys_h[0]] = 'horndeski_model'
    
    config_h[keys_h[2]] = 'k_1*X'
    config_h[keys_h[3]] = 'g_31*X'
    config_h[keys_h[4]] = '0.5'

    config_h[keys_h[7]] = str(k1_track)
    config_h[keys_h[8]] = str("1")
    config_h[keys_h[9]] = str(f_phi)
    config_h[keys_h[10]] = str("1")
    config_h[keys_h[11]] = str("1")
    config_h[keys_h[20]] = str("False")
    config_h[keys_h[22]] = str("1")
    
    ## Numerical params
    config_n = ConfigObj(num_ini)
    keys_n = config_n.keys()

    config_n[keys_n[0]] = 'run_X'
    
    config_n[keys_n[6]] = str(h)
    config_n[keys_n[7]] = str(wr)
    config_n[keys_n[8]] = str(wb)
    config_n[keys_n[9]] = str(wc)
    config_h.filename = '/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/horndeski_today.ini'
    config_n.filename = '/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/numerical_today.ini'
    config_h.write(); config_n.write()

def Update_ini_GR(f_phi, arr_num, horn_ini, num_ini):
    """
    Updates files num_ini and horn_ini to include the parameters (numerical and MG) from 
    f_phi and
    arr_num = h , wr , wb , wc, n_s , A_s
    """
    ## Horndeski params
    h , wr , wb , wc, n_s , A_s = arr_num
    config_h = ConfigObj(horn_ini)
    
    k1_track = k_1T((wc + wb)/h**2, wr/h**2, f_phi)

    keys_h = config_h.keys()
    
    config_h[keys_h[1]] = '/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/Backend_input_parameter_files/Output_validation_edges'
    config_h[keys_h[0]] = 'horndeski_model'
    
    config_h[keys_h[2]] = 'k_1*X'
    config_h[keys_h[3]] = 'g_31*X'
    config_h[keys_h[4]] = '0.5'

    config_h[keys_h[7]] = str(k1_track)
    config_h[keys_h[8]] = str("1")
    config_h[keys_h[9]] = str(f_phi)
    config_h[keys_h[10]] = str("1")
    config_h[keys_h[11]] = str("1")
    config_h[keys_h[20]] = str("False")
    config_h[keys_h[22]] = str("1")
    
    ## Numerical params
    config_n = ConfigObj(num_ini)
    keys_n = config_n.keys()

    config_n[keys_n[0]] = 'run_X_GR'
    
    config_n[keys_n[6]] = str(h)
    config_n[keys_n[7]] = str(wr)
    config_n[keys_n[8]] = str(wb)
    config_n[keys_n[9]] = str(wc)
    config_h.filename = '/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/horndeski_today.ini'
    config_n.filename = '/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/numerical_today.ini'
    config_h.write(); config_n.write()

def initialize_cosmology_CuGal(cosmo, MGparams, a):
    H0rc, fR0, n, f_phi, mu, Sigma = MGparams
    cosmo_arr = [cosmo["h"] , cosmo["Omega_g"]*cosmo["h"]**2 , cosmo["Omega_b"]*cosmo["h"]**2 , cosmo["Omega_c"]*cosmo["h"]**2, cosmo["n_s"] , cosmo["A_s"]]
    Update_ini(f_phi, cosmo_arr, '/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/horndeski_parameters.ini', '/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/numerical_parameters.ini')
    command = 'cd; source HiCOLA_env/3.7.12/bin/activate; cd /home/c2042999/HiCOLA_env/Hi-COLA; python3 -m HiCOLA.Frontend.generate_simulation_input /home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/horndeski_today.ini /home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/numerical_today.ini'
    subprocess.run(['bash', '-c', 'cd /home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/Backend_input_parameter_files/Output_validation_edges; rm horndeski_model_run_X_force.txt; rm horndeski_model_run_X_expansion.txt'])
    subprocess.run(['bash', '-c', command])

    Update_ini_GR(0.0, cosmo_arr, '/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/horndeski_parameters.ini', '/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/numerical_parameters.ini')
    command = 'cd; source HiCOLA_env/3.7.12/bin/activate; cd /home/c2042999/HiCOLA_env/Hi-COLA; python3 -m HiCOLA.Frontend.generate_simulation_input /home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/horndeski_today.ini /home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/numerical_today.ini'
    subprocess.run(['bash', '-c', 'cd /home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/Backend_input_parameter_files/Output_validation_edges; rm horndeski_model_run_X_GR_force.txt; rm horndeski_model_run_X_GR_expansion.txt'])
    subprocess.run(['bash', '-c', command])
    return


def E_CuGal(cosmo, a):
    
    expansion_today = np.loadtxt("/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/Backend_input_parameter_files/Output_validation_edges/horndeski_model_run_X_expansion.txt")
    expansion_today_GR = np.loadtxt("/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/Backend_input_parameter_files/Output_validation_edges/horndeski_model_run_X_GR_expansion.txt")

    a_today = expansion_today.T[0]
    E_a_today = expansion_today.T[1]
    E_a_today_GR = expansion_today_GR.T[1]
    
    E_a_GR = E(cosmo, a)
    return np.interp(a, a_today, E_a_today/E_a_today_GR)*E_a_GR

# mu(k,a) = mu(a) in CuGal (modified gravity parametrization parameter)
def mu_CuGal(MGparams, a):
    H0rc, fR0, n, f_phi, mu, Sigma = MGparams
    if f_phi == 0:
        return np.ones(len(a))
    else:
        force_today = np.loadtxt("/home/c2042999/HiCOLA_env/Cubic_galileon_emu-ICGweek/Backend_input_parameter_files/Output_validation_edges/horndeski_model_run_X_force.txt")
    
        a_today = force_today.T[0]
        coupling = force_today.T[2]
        return np.interp(a, a_today, coupling + 1)

###################################################################
############### MATTER POWER SPECTRUM FUNCTIONS ###################
###################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~ Non-Linear ~~~~~~~~~~~~~~~~~~~~~~~~~~~

# NL matter power spectrum in Cubic Galileon

def P_k_NL_CuGal(cosmo, MGparams, k, a):
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output Pk_fR (array) -> Nonlinear matter power spectrum for Hu-Sawicki fR gravity, units (Mpc)^3
    """
    H0rc, fR0, n, f_phi, mu, Sigma = MGparams

    input_params_and_redshift = np.append(np.array([cosmo["Omega_m"],cosmo["n_s"],1e9*cosmo["A_s"],cosmo["h"],f_phi]) , 1.0/a -1.0)     
    bk_target, err_target = emu_redshift(input_params_and_redshift[np.newaxis, :], sepia_model_list, z_all)

    pkratio_CuGal = np.interp(k, k_all*cosmo["h"], bk_target.reshape(len(k_all)))
    
    Pk_ccl = ccl.power.nonlin_power(cosmo, k, a=a) # units (Mpc)^3
    Pk = pkratio_CuGal*Pk_ccl

    return Pk

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Linear ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Linear matter power spectrum in Cubic Galileon
def P_k_CuGal_lin(cosmo, MGparams, k, a):
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output Pk_fR (array) -> Nonlinear matter power spectrum for Hu-Sawicki fR gravity, units (Mpc)^3
    """
    H0rc, fR0, n, f_phi, mu, Sigma = MGparams

    input_params_and_redshift = np.append(np.array([cosmo["Omega_m"],cosmo["n_s"],1e9*cosmo["A_s"],cosmo["h"],f_phi]) , 1.0/a -1.0)     
    bk_target, err_target = emu_redshift(input_params_and_redshift[np.newaxis, :], sepia_model_list, z_all)
    
    pkratio_CuGal = bk_target[0]

    Pk_ccl = ccl.power.linear_matter_power(cosmo, k, a=a) # units (Mpc)^3
    Pk = pkratio_CuGal*Pk_ccl

    return Pk

###################################################################
################### SIGMA8 AND F FUNCTIONS ########################
###################################################################

def sigma_8_CuGal(cosmo, MGparams, a_array):
    k_val = np.logspace(-4, 3, 3000)
    sigma_8_vals = []

    for a in a_array:
        P_k_vals = P_k_CuGal_lin(cosmo, MGparams, k_val, a)
        j1_vals = 3 * scipy.special.spherical_jn(1, k_val * 8 / cosmo["h"], derivative=False) / (k_val * 8 / cosmo["h"])
        integrand = k_val**2 * P_k_vals * j1_vals**2
        integral_val = scipy.integrate.trapz(integrand, x=k_val)
        sigma_8_val = np.sqrt(integral_val / (2 * np.pi**2))
        sigma_8_vals.append(sigma_8_val)
    
    return np.array(sigma_8_vals)

def solverGrowth_CuGal(y,a,cosmo, MGparams):
    E_val = E_CuGal(cosmo, a)
    D , a3EdDda = y
    
    mu = mu_CuGal(MGparams, a)
    
    ydot = [a3EdDda / (E_val*a**3), 3*cosmo["Omega_m"]*D*(mu)/(2*E_val*a**2)]
    return ydot
    
def fsigma8_CuGal(cosmoMCMCStep, MGparams, a):
    
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output P_k_musigma (array) -> linear matter power spectrum for mu sigma param, units (Mpc)^3
    """
    
    H0rc, fR0, n, f_phi, mu, Sigma = MGparams
    
    a_solver = np.linspace(1/50,1,100)
    Soln = odeint(solverGrowth_CuGal, [a_solver[0], (E_CuGal(cosmoMCMCStep, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(cosmoMCMCStep,MGparams), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    a3EdDda = Soln.T[1]

    f_CuGal_interp = a3EdDda/a_solver**2 / Delta / E_CuGal(cosmoMCMCStep, a_solver)
    
    f_CuGal = np.interp(a, a_solver, f_CuGal_interp)

    k_val = np.logspace(-4,3,3000)
    return f_CuGal * sigma_8_CuGal(cosmoMCMCStep, MGparams, a)
    
###################################################################
################### ANGULAR P(k) FUNCTIONS ########################
###################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Binning ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cell ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""Functions to find Cell for Cubic Galileon given a Pdelta_2D ccl object"""


#### NOTE: NEED TO INITIALIZE BACKGROUND FILES FIRST FOR A GIVEN f_phi AND cosmo ######

"""Comoving radial distance in Cubic Galileon"""
def comoving_radial_dist_CuGal(cosmo, a_array):
    c = 3e5  # Speed of light in km/s
    
    # Define the redshift integral range
    #z_integral = np.linspace(1/a_array.min() - 1, 0, int(1e4))  # Use the minimum value of `a_array`
    x_integral = np.linspace(np.log(a_array.min()), 0, int(5e2)) # Compute the scale factor over the range
    a_integral = np.exp(x_integral)
    z_integral = 1/a_integral - 1
    
    # Calculate E(a_integral) only once over the entire range
    E_val = E_CuGal(cosmo, a_integral)

    # Define the integrand for each value of a
    integrand = c / (a_integral * E_val * cosmo["H0"])
    
    # Now integrate over the entire range for each value in `a_array`
    results = []
    for a_iter in a_array:
        z_lower_bound = 1/a_iter - 1  # Adjust the upper bound of the integration
        mask = (z_integral <= z_lower_bound)  # Mask the integrand for the valid integration range
        
        # Perform integration for the valid portion of the integrand
        integral = scipy.integrate.simps(integrand[mask], x_integral[mask])
        results.append(integral)

    return np.array(results)

"""radial kernel - shear"""
def q_kappa_CuGal(i, cosmo, MGparams, z, Binned_distribution_s):
    c = 3e5
    zmax = 4
    a = np.logspace(np.log10(1/(1+zmax)),np.log10(1), int(5e2)) 
    z_var = 1 / a - 1  # Find useful functs of z
    
    # Precompute comoving radial distance for a grid of 'a' values (for zp and ap)
    ap_grid = np.logspace(np.log10(1 / (1 + zmax + 1)), np.log10(1), int(5e2))   # Grid of 'ap' values
    chip_grid = comoving_radial_dist_CuGal(cosmo, ap_grid)  # Compute comoving distance once for the grid

    # Create an interpolation function for comoving radial distance
    chip_interp = scipy.interpolate.interp1d(ap_grid, chip_grid, kind='cubic', fill_value=0.0)

    # Precompute comoving radial distance for 'a' values
    chiz = comoving_radial_dist_CuGal(cosmo, a)  # units of Mpc

    q_kappa_array = []

    for k in range(len(z_var)):
        zp = np.linspace(z_var[k] + 1e-3,zmax,int(1e3))                                # variable we integrate over
        ap = 1 / (1 + zp)

        # Use the precomputed interpolation for comoving radial distance
        chip = chip_interp(ap)
        
        # Interpolate n_p
        n_p = np.interp(zp, z, Binned_distribution_s[i])

        # Compute the integrand using np.where for vectorized conditional calculation
        integrand = n_p * chiz[k] * (chip - chiz[k]) / (a[k] * chip)

        # Integrate using simpson's rule
        q_k = scipy.integrate.simps(integrand, zp) * 3 * cosmo['Omega_m'] * (cosmo['H0']**2) / (2 * c * c)
        q_kappa_array.append(q_k)

    return np.array(q_kappa_array).reshape(len(q_kappa_array))
    
"""radial kernel - galaxy clustering"""
def q_delta_CuGal(i, cosmo,MGparams, z, Binned_distribution_l,Bias_distribution):
    
    # takes in the cosmology
    # finds radial kernel for the n_{i}th bin of the lens redshift distribution
    c=3e5
    zmax = 4
    a = np.logspace(np.log10(1/(1+zmax)),np.log10(1), int(5e2)) 
    z_var = 1/a -1 #np.linspace(0.001, 4., int(1e2))                              # find useful functs of z
    #a = 1/(1 + z_var)
    
    chiz = comoving_radial_dist_CuGal(cosmo, a)                  # units of Mpc
    dzdchi = E_CuGal(cosmo, a) * cosmo['H0'] / c
    n_p = np.interp(z_var, z , Binned_distribution_l[i])
    
    q_del = dzdchi * np.interp(z_var, z ,Bias_distribution[i]) * n_p

    return np.array(q_del).reshape(len(q_del))
    
def findC_ell_kk_CuGal(i,j, cosmo,MGparams, z, ell, Binned_distribution_s, Binned_distribution_l,Bias_distribution):

    c = 3e5
    zmax = 4
    a = np.logspace(np.log10(1/(1+zmax)),np.log10(1), int(5e2)) 
    z_var = 1/a -1 #np.linspace(0.001, 4., int(1e2))                              # find useful functs of z

    chiz = comoving_radial_dist_CuGal(cosmo, a) + 1e-3
    q_k_val_i = q_kappa_CuGal(i, cosmo,MGparams, z, Binned_distribution_s)          # find tomographic lens efficiency
    q_k_val_j = q_kappa_CuGal(j, cosmo,MGparams, z, Binned_distribution_s)          # find tomographic lens efficiency

    # In Cubic Galileon, Sigma = mu
    Sigma_val = mu_CuGal(MGparams, a)
    
    integrand_term = -Sigma_val**2 * q_k_val_i*q_k_val_j/E_CuGal(cosmo, a)/chiz**2/cosmo['H0'] * c

    C_ell = []
    for k in range(len(ell)):
        Pk = []
        for m in range(len(a)):
            Pk.append(P_k_NL_CuGal(cosmo,MGparams, (ell[k] + 0.5)/chiz[m] , a[m]))
        
        integrand = integrand_term*np.array(Pk).reshape(len(a))
        C_ell.append([scipy.integrate.simps(integrand, z_var)])
        
    return np.array(C_ell)

def findC_ell_delk_CuGal(i,j, cosmo,MGparams, z, ell, Binned_distribution_s, Binned_distribution_l,Bias_distribution):

    c = 3e5
    zmax = 4
    a = np.logspace(np.log10(1/(1+zmax)),np.log10(1), int(5e2)) 
    z_var = 1/a -1 #np.linspace(0.001, 4., int(1e2))                              # find useful functs of z
    #a = 1/(1 + z_var)
    chiz = comoving_radial_dist_CuGal(cosmo, a) + 1e-3
    q_k_val_i = q_kappa_CuGal(i, cosmo, MGparams,z, Binned_distribution_s)          # find tomographic lens efficiency
    q_k_val_j = q_delta_CuGal(j, cosmo,MGparams, z, Binned_distribution_l,Bias_distribution)          # find tomographic lens efficiency

    # In Cubic Galileon, Sigma = mu
    Sigma_val = mu_CuGal(MGparams, a)
    
    integrand_term = -Sigma_val * q_k_val_i*q_k_val_j/E_CuGal(cosmo, a)/chiz**2/cosmo['H0'] * c
    
    C_ell = []
    for k in range(len(ell)):
        Pk = []
        for m in range(len(a)):
            Pk.append(P_k_NL_CuGal(cosmo,MGparams, (ell[k] + 0.5)/chiz[m] , a[m]))
        
        integrand = integrand_term*np.array(Pk).reshape(len(a))
        C_ell.append([scipy.integrate.simps(integrand, z_var)])
    
    return np.array(C_ell)

def findC_ell_deldel_CuGal(i,j, cosmo,MGparams, z, ell, Binned_distribution_s, Binned_distribution_l,Bias_distribution):

    c = 3e5
    zmax = 4
    a = np.logspace(np.log10(1/(1+zmax)),np.log10(1), int(5e2)) 
    z_var = 1/a -1 #np.linspace(0.001, 4., int(1e2))                              # find useful functs of z
    #a = 1/(1 + z_var)
    chiz = comoving_radial_dist_CuGal(cosmo, a) + 1e-3
    q_k_val_i = q_delta_CuGal(i, cosmo,MGparams, z, Binned_distribution_l,Bias_distribution)          # find tomographic lens efficiency
    q_k_val_j = q_delta_CuGal(j, cosmo,MGparams, z, Binned_distribution_l,Bias_distribution)          # find tomographic lens efficiency
    
    integrand_term = -q_k_val_i*q_k_val_j/E_CuGal(cosmo, a)/chiz**2/cosmo['H0'] * c

    C_ell = []
    for k in range(len(ell)):
        Pk = []
        for m in range(len(a)):
            Pk.append(P_k_NL_CuGal(cosmo,MGparams, (ell[k] + 0.5)/chiz[m] , a[m]))
        
        integrand = integrand_term*np.array(Pk).reshape(len(a))
        C_ell.append([scipy.integrate.simps(integrand, z_var)])

    return np.array(C_ell)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pk_2D object ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Get_Pk2D_obj(cosmo, MGParams,linear=False,gravity_model="GR"):
    """
    Finds Get_Pk2D object
    linear = True, False
    gravity theory = "GR", "nDGP", "f(R)", "muSigma"
    if linear=True, use linear matter power spectrum to compute the angular one, otherwise use the non-linear
    input:
        ell_binned: array of ell bins for the full C{ij}(ell) range (for all i and j), with scale cuts included
        cosmo: ccl cosmology object
        redshift z: numpy.array with dim:N
        Binned_distribution_s: numpy.array with dim:(N,M) (M = no. source z bins)
        Binned_distribution_l: numpy.array with dim:(N,L) (L = no. lens z bins)
        Bias_distribution: numpy.array with dim:(N,L) (galaxy bias)
        pk_F: function (cosmo, MGParams, k,a) for k (in 1/Mpc), returns matter power spectrum (in Mpc^3)
        MGParams: 
    returns:
        ell bins: numpy.array (dim = dim C_ell)
        C_ell: numpy.array
    """


    ########### Functions for non-linear matter power spectrum ###########
    def pk_func_nDGP_NL(k, a):
        z = 1 / a - 1
        if z > 2:
            return ccl.nonlin_matter_power(cosmo, k=k, a=a)
        
        # Determine the index range for k
        k_min = 0.0156606 * cosmo["h"]
        k_max = 4.99465 * cosmo["h"]
        
        idx_min = np.searchsorted(k, k_min)
        idx_max = np.searchsorted(k, k_max, side='right')
        
        k_allowed = k[idx_min:idx_max]
        
        # Calculate power spectra for different k ranges
        pk_lin_start = P_k_nDGP_lin(cosmo, MGParams, k[:idx_min], a)
        pk_nl_mid = P_k_NL_nDGP(cosmo, MGParams, k_allowed, a)
        pk_nl_end = ccl.nonlin_matter_power(cosmo, k=k[idx_max:], a=a)
        
        pk = np.concatenate((pk_lin_start, pk_nl_mid, pk_nl_end), axis = 0)
        
        return pk
    
    def pk_func_fR_NL(k, a):        
        z = 1 / a - 1
        
        if a < 0.3333:
            return ccl.nonlin_matter_power(cosmo, k=k, a=a)
        
        # Determine the index range for k
        idx_min = np.argmin(np.abs(k - (emu_fR.kbins[0])*cosmo["h"])) + 1
        idx_max = np.argmin(np.abs(k - (emu_fR.kbins[-1])*cosmo["h"])) -1

        k_min = k[idx_min]
        k_max = k[idx_max]
        k_allowed = k[idx_min:idx_max]
        
        # Calculate power spectra for different k ranges
        interp_funct_fR = scipy.interpolate.PchipInterpolator([1e-3,k[idx_min]], \
                [1.0,P_k_NL_fR(cosmo, MGParams, k[idx_min], a)[0]/ccl.nonlin_matter_power(cosmo,k[idx_min],a=a)])

        pk_lin_start = interp_funct_fR(k[:idx_min]) * ccl.nonlin_matter_power(cosmo, k=k[:idx_min], a=a)
        pk_nl_mid = P_k_NL_fR(cosmo, MGParams, k_allowed, a)
        pk_nl_end = ccl.nonlin_matter_power(cosmo, k=k[idx_max:], a=a)
        
        pk = np.concatenate((pk_lin_start, pk_nl_mid, pk_nl_end), axis = 0)
        
        return pk
    
    def pk_func_muSigma_NL(k, a):
        raise Exception('there is no non-linear power spectrum available for muSigma parametrization.')
        
    ########### Functions for linear matter power spectrum multiplied by Sigma**2 ###########
    def pk_func_nDGP_lin(k, a):

        # condition on z
        return P_k_nDGP_lin(cosmo, MGParams, k, a)
        
    def pk_func_fR_lin(k, a):
        
        # condition on z
        return P_k_fR_lin(cosmo, MGParams, k, a)
        
    def pk_func_muSigma_lin(k, a):
        # condition on z
        return P_k_musigma(cosmo, MGParams, k, a)

    def invalid_op(k, a):
        raise Exception("Invalid gravity model entered or Linear must be True or False.")

    ops = {
        ("nDGP" , False): pk_func_nDGP_NL, 
        ("f(R)" , False): pk_func_fR_NL, 
        ("muSigma" , False): pk_func_muSigma_NL,
        ("nDGP" , True): pk_func_nDGP_lin, 
        ("f(R)" , True): pk_func_fR_lin, 
        ("muSigma" , True): pk_func_muSigma_lin
    }
    
    ########### Find matter power spectrum multiplied by Sigma**2 ###########
    pk_func = ops.get((gravity_model, linear), invalid_op)

    return ccl.pk2d.Pk2D.from_function(pkfunc=pk_func, is_logp=False)

########### Functions for NL P(k) multiplied by Sigma - only for Sigma diff 1, so MuSigma param only ###########

def Get_Pk2D_obj_delk_musigma(cosmo, MGParams):
   
    ########### Functions for linear matter power spectrum multiplied by Sigma ###########        
    def pk_funcSigma_muSigma_lin(k, a):
        return sigma_lin_param(MGParams, cosmo,a)*P_k_musigma(cosmo, MGParams, k, a)

    return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma_muSigma_lin, is_logp=False)


def Get_Pk2D_obj_kk_musigma(cosmo, MGParams):
   
    ########### Functions for linear matter power spectrum multiplied by Sigma**2 ###########        
    def pk_funcSigma2_muSigma_lin(k, a):
        return sigma_lin_param(MGParams, cosmo, a)**2 * P_k_musigma(cosmo, MGParams, k, a)

    return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma2_muSigma_lin, is_logp=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pk_2D object for ESS-C ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Get_Pk2D_obj_ESS_C(cosmo, MGParams, linear=False):
   
    ########### Functions for matter power spectrum multiplied by Sigma**2 ###########        
    def pk_funcSigma_ESS_C(k, a):
        if linear == False:
            return P_k_NL_ESS_C(cosmo, k, a)
        elif linear == True:
            return P_k_ESS_C_lin(cosmo, k, a)

    return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma_ESS_C, is_logp=False)
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Wrapper Cell function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def Cell(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution, MGParams,P_delta2D_S,
         tracer1_type="k", 
         tracer2_type="k"):
    """
    Finds C^{i,j}(ell) for {i,j} redshift bins.
    tracer_type = "k", "g"
    linear = True, False
    gravity theory = "GR", "nDGP", "f(R)", "muSigma"
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
        pk_F: function (cosmo, MGParams, k,a) for k (in 1/Mpc), returns matter power spectrum (in Mpc^3)
        MGParams: 
    returns:
        ell bins: numpy.array (dim = dim C_ell)
        C_ell: numpy.array
    """

    ops = {
        ("k" , "k"): C_ell_arr_kk,
        ("k" , "g"): C_ell_arr_delk, 
        ("g" , "k"): C_ell_arr_delk,
        ("g" , "g"): C_ell_arr_deldel
    }

    def invalid_op2():
        raise ValueError('invalid tracer selected.')
    ########## Find Cell ##########

    C_ell_array_funct = ops.get((tracer1_type, tracer2_type), invalid_op2)
    C_ell_array = C_ell_array_funct(P_delta2D_S, ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution)

    return np.array(list(itertools.chain(*ell_binned))), C_ell_array

        

def Cell_GR(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution, MGParams,
         tracer1_type="k", 
         tracer2_type="k"):
    """
    Finds C^{i,j}(ell) for {i,j} redshift bins.
    tracer_type = "k", "g"
    linear = True, False
    gravity theory = "GR", "nDGP", "f(R)", "muSigma"
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
        pk_F: function (cosmo, MGParams, k,a) for k (in 1/Mpc), returns matter power spectrum (in Mpc^3)
        MGParams: 
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

###################################################################
############## Functions to find fsigma8, from Danielle  ##########
###################################################################

def f_frac_err(cosmo, MGparams,b, zeff, nbar, V):
    """ Get the fractional error on the growth rate.
    params is a dictionary of required cosmological parameters.
    zeff is the effective redshift of the sample.
    nbar is the number density of the sample in units of (h/Mpc)^3.
    V is the volume of the survey in (Mpc/h)^3. """
    
    # Set up a k and mu vector at which to do the integrals
    # (Result depends on kmax chosen, see White et al. 2008)
	
    k = np.logspace(-3, -1, 400)
    mu = np.linspace(-1, 1., 200)
        
    # Get the inverse covariance value at each k and mu
    invcov = Pobs_covinv(cosmo, MGparams,b, k, mu, zeff, nbar)
    
    # Get the derivative of the observed z-space
    # P(k) with respect to b and f at each k and mu
    # (linear theory)
    dPdf = diff_P_f(cosmo, MGparams,b, k, mu, zeff)
    
    # Do the integration in k in each case
    int_in_k_ff = [scipy.integrate.simps(k**2 * dPdf[mi] * invcov[mi] * dPdf[mi], k) for mi in range(len(mu))]
	
    # And in mu.
    int_in_mu_ff = scipy.integrate.simps(np.asarray(int_in_k_ff), mu)
	
    Fisher_ff = np.zeros((2,2)) # order is b then f
    
    # Add necessary factors of volume (Mpc/h)^3 and pi etc
    ff= V * int_in_mu_ff / (2. * np.pi**2)
    err_f = np.sqrt(1./ff)
    
    # Now use this to construct the error on f:
    f_sigma8_val = fsigma8_ESS_C(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]
    f_fid = f_sigma8_val/fsigma8_ESS_C(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]
    
    frac_err_f = err_f / f_fid
    
    return frac_err_f, f_sigma8_val
    
def diff_P_f(cosmo, MGparams,b, k, mu, zeff):
    """ Calculate the derivative of the redshift space power spectrum
    wrt linear growth rate f at each k and mu
    params: dictionary of cosmological parameters
    k: list or array of wavenumbers
    mu: list of array of angles
    lens: lens sample label
    """
		
    # Get the linear power spectrum at the effective z of the lens sample
    Pklin = P_k_ESS_C_lin(cosmo, k,  1./ (1. + zeff))
    
    # Get the derivative at each mu / k
    f = fsigma8_ESS_C(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]/sigma_8_ESS_C(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]
    dPdf = [2. * (b + mu[mi]**2*f) * mu[mi]**2 * Pklin  for mi in range(len(mu))]
	
    return dPdf
    
def Pobs_covinv(cosmo, MGparams,b, k, mu, zeff, nbar):
    """ Get the inverse covariance of the redshift space observed power 
    spectrum at a list of k and mu (cosine of angle to line of sight) vals.
    params: dictionary of cosmological parameters
    k: list or array of wavenumbers
    mu: list of array of angles
    lens: lens sample label """	
 
	
    # Get the linear power spectrum at the effective z of the lens sample
    Pklin = P_k_ESS_C_lin(cosmo, k,  1./ (1. + zeff))
	
    # Get the redshift space galaxy power spectrum in linear theory
    f = fsigma8_ESS_C(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]/sigma_8_ESS_C(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]
    Pgg = [(b + f * mu[mi]**2)**2 * Pklin for mi in range(len(mu))]
	
    # Get the covariance matrix at each k and mu
    cov = [ 2. * (Pgg[mi]**2 + 2 * Pgg[mi] / nbar + 1. / nbar**2) for mi in range(len(mu))]
	
    #Pgg_arr = np.zeros((len(k), len(mu)))
    #for mi in range(len(mu)):
    #	Pgg_arr[:, mi] = Pgg[mi]
	
    # Get the inverse at each k and mu
    invcov = [[1./ cov[mi][ki] for ki in range(len(k))] for mi in range(len(mu))]
			
    return invcov
    
###################################################################
################### FUNCTIONS FOR SCALE CUTS  #####################
###################################################################

def linear_scale_cuts_v2(dvec_nl, dvec_lin, cov):
    """ 
    Function from Danielle.
    Gets the scales (and vector indices) which are excluded if we
    are only keeping linear scales. We define linear scales such that 
    chi^2_{nl - lin) <=1.
	
    This is a version that is hopefully more reliable when data are highly correlated.
	
    dvec_nl: data vector from nonlinear theory 
    dvec_lin: data vector from linear theory
    cov: data covariance. """
	
    # Make a copy of these initial input things before they are changed,
    # so we can compare and get the indices
    dvec_nl_in = dvec_nl; dvec_lin_in = dvec_lin; cov_in = cov;
	
    # Check that data vector and covariance matrices have consistent dimensions.
    if ( (len(dvec_nl)!=len(dvec_lin)) or (len(dvec_nl)!=len(cov[:,0])) or (len(dvec_nl)!=len(cov[0,:])) ):
        raise(ValueError, "in linear_scale_cuts: inconsistent shapes of data vectors and / or covariance matrix.")
		
    while(True):
		
        # Get an array of all the individual elements which would go into 
        # getting chi2
        #sum_terms = np.zeros((len(dvec_nl), len(dvec_nl)))
        #for i in range(0,len(dvec_nl)):
        #    for j in range(0,len(dvec_nl)):
        #        sum_terms[i,j] = (dvec_nl[i] - dvec_lin[i]) * inv_cov[i,j] * (dvec_nl[j] - dvec_lin[j])
				
        #print("sum_terms=", sum_terms)
        #print("chi2=", np.sum(sum_terms))
        # Check if chi2<=1		
        
        # Get the chi2 in the case where you cut each data point
        # and then actually cut the one that reduces the chi2
        # the most
        chi2_temp = np.zeros(len(dvec_nl))
        for i in range(len(dvec_nl)):
            delta_dvec = np.delete(dvec_nl, i) - np.delete(dvec_lin, i)
            cov_cut = np.delete(np.delete(cov,i, axis=0), i, axis=1)
            inv_cov_cut = np.linalg.pinv(cov_cut)
            chi2_temp[i] = np.dot(delta_dvec, np.dot(inv_cov_cut, delta_dvec))
            #sum_temp[i] = np.sum(np.delete(np.delete(sum_terms, i, axis=0), i, axis=1))
        print('chi2_temp=', chi2_temp)
            
        #Find the index of data point that is cut to produce the smallest chi2:
        ind_min = np.argmin(chi2_temp)
        print('ind_min=', ind_min)
            
        # Cut that element
        dvec_nl = np.delete(dvec_nl, ind_min)
        dvec_lin = np.delete(dvec_lin, ind_min)
        cov = np.delete( np.delete(cov, ind_min, axis=0), ind_min, axis=1)
            
        if (chi2_temp[ind_min]<=1.0):
            break
				
    # Now we should have the final data vector with the appropriate elements cut.
    # Use this to get the rp indices and scales we should cut.

    ex_inds = [i for i in range(len(dvec_nl_in)) if dvec_nl_in[i] not in dvec_nl]
    print('ex_inds=', ex_inds)
	
    return ex_inds

def baryonic_scale_cuts_v2(cosmo, ell, dvec_full, dvec_shear, dvec_kmax, cov_full):
    """ 
    Modified function from Danielle.
    Gets the scales (and vector indices) which are excluded if we
    are only keeping non-baryonic scales. We define these scales such that 
    chi^2_{baryonic - DMO) <=1.
    dvec_full: full data vector from baryonic theory 
    dvec_shear: shear data vector from baryonic theory 
    dvec_kmax: shear data vector from DMO theory
    cov_full: full data covariance, shear components come first
    derived    cov: shear data covariance. """
    
    # Make a copy of these initial input things before they are changed,
    # so we can compare and get the indices
    dvec_full_in = dvec_full; dvec_shear_in = dvec_shear; dvec_kmax_in = dvec_kmax; cov_in = cov_full;
	
	#### first cuts - clustering ######
    
    # for galaxy-galaxy lensing (size=91) bins=(02 , 03 , 04 , 13 , 14 , 24 , 34)
    # for galaxy-galaxy (size=65) bins=(00 , 11 , 22 , 33 , 44)
    
    k_max = 0.3 # h/Mpc
    delk_z_array = np.array([0.30,0.30,0.30,0.50,0.50,0.70,0.70])
    deldel_z_array = np.array([0.30,0.50,0.70,0.90,1.10])
    z_array = np.append(delk_z_array, deldel_z_array)
    chi = ccl.background.comoving_radial_distance(cosmo, 1/(z_array+1))
    ellmax = k_max * chi - 0.5
    
    starting_index = len(dvec_shear)
    len_ell_ranges = int((len(cov_full[0]) - len(dvec_shear))/len(z_array))
    idx_count = 0
    for j in range(len(z_array)):
        for i in range(len_ell_ranges):
            if ell[starting_index + i] >= ellmax[j]:
                cov_full = np.delete(np.delete(cov_full, starting_index + j*len_ell_ranges + i - idx_count, axis=0), starting_index + j*len_ell_ranges + i - idx_count, axis=1)
                dvec_full = np.delete(dvec_full, starting_index + j*len_ell_ranges + i - idx_count)
                idx_count +=1
    
    #### second cuts - lensing ######
    cov = cov_full[:len(dvec_shear), :len(dvec_shear)]
    
    while(True):
		
        # Get an array of all the individual elements which would go into 
        # getting chi2
        #sum_terms = np.zeros((len(dvec_shear), len(dvec_shear)))
        #for i in range(0,len(dvec_shear)):
        #    for j in range(0,len(dvec_shear)):
        #        sum_terms[i,j] = (dvec_shear[i] - dvec_kmax[i]) * inv_cov[i,j] * (dvec_shear[j] - dvec_kmax[j])
				
        #print("sum_terms=", sum_terms)
        #print("chi2=", np.sum(sum_terms))
        # Check if chi2<=1		
        
        # Get the chi2 in the case where you cut each data point
        # and then actually cut the one that reduces the chi2
        # the most
        chi2_temp = np.zeros(len(dvec_shear))
        for i in range(len(dvec_shear)):
            delta_dvec = np.delete(dvec_shear, i) - np.delete(dvec_kmax, i)
            cov_cut = np.delete(np.delete(cov,i, axis=0), i, axis=1)
            inv_cov_cut = np.linalg.pinv(cov_cut)
            chi2_temp[i] = np.dot(delta_dvec, np.dot(inv_cov_cut, delta_dvec))
            #sum_temp[i] = np.sum(np.delete(np.delete(sum_terms, i, axis=0), i, axis=1))
        print('chi2_temp=', chi2_temp)
            
        #Find the index of data point that is cut to produce the smallest chi2:
        ind_min = np.argmin(chi2_temp)
            
        # Cut that element
        dvec_shear = np.delete(dvec_shear, ind_min)
        dvec_kmax = np.delete(dvec_kmax, ind_min)
        cov = np.delete(np.delete(cov, ind_min, axis=0), ind_min, axis=1)
        dvec_full = np.delete(dvec_full, ind_min)

        if (chi2_temp[ind_min]<=1.0):
            break
				
    # Now we should have the final data vector with the appropriate elements cut.
    # Use this to get the rp indices and scales we should cut.
    cov_full[:len(dvec_shear), :len(dvec_shear)] = cov
    
    ex_inds = [i for i in range(len(dvec_full_in)) if dvec_full_in[i] not in dvec_full]
    print('ex_inds=', ex_inds)
	
    return ex_inds

"""Define functions to apply k_max cuts"""
def Get_Pk2D_obj_kmax(cosmo, MGParams,linear=False,gravity_model="GR"):
    """
    Finds Get_Pk2D object
    linear = True, False
    gravity theory = "GR", "nDGP", "f(R)", "muSigma"
    if linear=True, use linear matter power spectrum to compute the angular one, otherwise use the non-linear
    input:
        ell_binned: array of ell bins for the full C{ij}(ell) range (for all i and j), with scale cuts included
        cosmo: ccl cosmology object
        redshift z: numpy.array with dim:N
        Binned_distribution_s: numpy.array with dim:(N,M) (M = no. source z bins)
        Binned_distribution_l: numpy.array with dim:(N,L) (L = no. lens z bins)
        Bias_distribution: numpy.array with dim:(N,L) (galaxy bias)
        pk_F: function (cosmo, MGParams, k,a) for k (in 1/Mpc), returns matter power spectrum (in Mpc^3)
        MGParams: 
    returns:
        ell bins: numpy.array (dim = dim C_ell)
        C_ell: numpy.array
    """


    ########### Functions for non-linear matter power spectrum multiplied by Sigma**2 ###########
    def pk_funcSigma2_GR_NL(k, a):
        # Determine the index range for k
        k_max = 0.3 # from https://arxiv.org/pdf/2212.09345
        
        idx_max = np.searchsorted(k, k_max, side='right')
                
        # Calculate power spectra for different k ranges
        pk_nl_start = ccl.nonlin_matter_power(cosmo, k=k[:idx_max], a=a)
        pk_nl_end = np.zeros(len(k[idx_max:]))
        
        pk = np.concatenate((pk_nl_start, pk_nl_end), axis = 0)
        
        return pk
       
    ########### Functions for linear matter power spectrum multiplied by Sigma**2 ###########
    def pk_funcSigma2_GR_lin(k, a):
        # Determine the index range for k
        k_max = 1
        
        idx_max = np.searchsorted(k, k_max, side='right')
                
        # Calculate power spectra for different k ranges
        pk_nl_start = ccl.linear_matter_power(cosmo, k=k[:idx_max], a=a)
        pk_nl_end = np.zeros(len(k[idx_max:]))
        
        pk = np.concatenate((pk_nl_start, pk_nl_end), axis = 0)
        
        return pk

    def invalid_op(k, a):
        raise Exception("Invalid gravity model entered or Linear must be True or False.")

    ops = {
        ("GR" , False): pk_funcSigma2_GR_NL,
        ("GR" , True): pk_funcSigma2_GR_lin
    }
    
    ########### Find matter power spectrum multiplied by Sigma**2 ###########
    pk_funcSigma2 = ops.get((gravity_model, linear), invalid_op)

    return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma2, is_logp=False)

###################################################################
############### LIKELIHOOD FUNCTIONS  #############################
###################################################################

# log likelihood - no PCA cuts
def loglikelihood_noscalecut(Data, cosmo, MGparams, InvCovmat, Bias_distribution, data_fsigma8):
    #start = time.time()
    
    # Extract fsigma8 data vector
    z_fsigma8, fsigma_8_dataset, invcovariance_fsigma8 = data_fsigma8
    
    # Extract 3x2pt data vector
    D_data, ell_mockdata, z, Binned_distribution_s,Binned_distribution_l,\
                   ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata = Data

    # shape-shape
    binned_ell_kk = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_s)

    # shape-pos
    binned_ell_delk = bin_ell_delk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, \
                              Binned_distribution_s,Binned_distribution_l)

    # pos-pos
    binned_ell_deldel = bin_ell_deldel(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_l)

    # Precompute Pk2D objects
    P_delta2D_muSigma_kk = Get_Pk2D_obj_kk_musigma(cosmo, MGparams)
    P_delta2D_muSigma_delk = Get_Pk2D_obj_delk_musigma(cosmo, MGparams)
    P_delta2D_muSigma_deldel =   Get_Pk2D_obj(cosmo, MGparams, linear=True, gravity_model="muSigma")
    
    ########## Get theoretical data vector for single MCMC step - linear , muSigmaparam ##########
    # shape-shape
    D_theory_kk = np.array(Cell(binned_ell_kk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                P_delta2D_muSigma_kk, tracer1_type="k", tracer2_type="k")[1]).flatten()
    # shape-pos
    
    D_theory_delk = np.array(Cell(binned_ell_delk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                P_delta2D_muSigma_delk, tracer1_type="g", tracer2_type="k")[1]).flatten()
    # pos-pos

    D_theory_deldel = np.array(Cell(binned_ell_deldel, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                P_delta2D_muSigma_deldel, tracer1_type="g", tracer2_type="g")[1]).flatten()

    D_theory = np.append(np.append(D_theory_kk, D_theory_delk), D_theory_deldel)
    
    Diff = (D_data - D_theory)

    #print("time = ", time.time() - start)

    #### fsigma8 ####
    Diff_fsigma8 = fsigma_8_dataset - fsigma8_musigma(cosmo, MGparams, 1/(z_fsigma8+1))
    loglik_fsigma8 = -0.5*(np.matmul(np.matmul(Diff_fsigma8,invcovariance_fsigma8),Diff_fsigma8))

    return -0.5*(np.matmul(np.matmul(Diff,InvCovmat),Diff)) + loglik_fsigma8 
    
# log likelihood with cut data
# P_k_sim = P_k_sim_mock
# Data = C_ell_data_mock
# Covmat = gauss_cov
def loglikelihood(Data, cosmo,cosmo_linear, MGparams, L_ch_inv, Bias_distribution, data_fsigma8):
    
    #start = time.time()
    z_fsigma8, fsigma_8_dataset, invcovariance_fsigma8 = data_fsigma8

    # Extract real data vector
    D_data, ell_mockdata, z, Binned_distribution_s,Binned_distribution_l,\
                ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata = Data

    # Do binning
    binned_ell_kk = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_s)
    binned_ell_delk = bin_ell_delk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, \
                              Binned_distribution_s,Binned_distribution_l)
    binned_ell_deldel = bin_ell_deldel(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_l)

    # Precompute Pk2D objects
    P_delta2D_muSigma_kk = Get_Pk2D_obj_kk_musigma(cosmo, MGparams)
    P_delta2D_muSigma_delk = Get_Pk2D_obj_delk_musigma(cosmo, MGparams)
    P_delta2D_muSigma_deldel =   Get_Pk2D_obj(cosmo, MGparams, linear=True, gravity_model="muSigma")
    
    P_delta2D_nDGP_lin = Get_Pk2D_obj(cosmo, MGparams, linear=True, gravity_model="nDGP")
    P_delta2D_nDGP_nl = Get_Pk2D_obj(cosmo, MGparams, linear=False, gravity_model="nDGP")
    P_delta2D_fR_lin = Get_Pk2D_obj(cosmo, MGparams, linear=True, gravity_model="f(R)")
    P_delta2D_fR_nl = Get_Pk2D_obj(cosmo, MGparams, linear=False, gravity_model="f(R)")
    
    ########## Get theoretical data vector for single MCMC step - linear , muSigmaparam ##########
    # shape-shape
    D_theory_kk = (np.array(Cell(binned_ell_kk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_muSigma_kk,\
                tracer1_type="k", tracer2_type="k")[1])).flatten()
    # shape-pos
    D_theory_delk = (np.array(Cell(binned_ell_delk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_muSigma_delk,\
                tracer1_type="k", tracer2_type="g")[1])).flatten()
    # pos-pos
    D_theory_deldel = (np.array(Cell(binned_ell_deldel, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_muSigma_deldel,\
                tracer1_type="g", tracer2_type="g")[1])).flatten()

    D_theory = np.concatenate((D_theory_kk, D_theory_delk, D_theory_deldel), axis=0)
    
    Diff = (D_data - D_theory)
    
    # Find Choleski scaled data vector
    Diff_ch = np.array(np.matmul(L_ch_inv, Diff.T))[0]

    ########## GET DATA FOR DIFFERENCE MATRIX ##########
    """MG1: nDGP"""
    # A: find C_ell for non-linear matter power spectrum
    # shape-shape
    B1_kk = (np.array(Cell(binned_ell_kk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_nDGP_nl,\
                tracer1_type="k", tracer2_type="k")[1])).flatten()
    # shape-pos
    B1_delk = (np.array(Cell(binned_ell_delk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_nDGP_nl,\
                tracer1_type="g", tracer2_type="k")[1])).flatten()
    # pos-pos
    B1_deldel = (np.array(Cell(binned_ell_deldel, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_nDGP_nl,\
                tracer1_type="g", tracer2_type="g")[1])).flatten()

    B1 = np.concatenate((B1_kk, B1_delk, B1_deldel), axis=0)
       
    # B: find C_ell for linear matter power spectrum
    # shape-shape
    M1_kk = (np.array(Cell(binned_ell_kk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_nDGP_lin,\
                tracer1_type="k", tracer2_type="k")[1])).flatten()
    # shape-pos
    M1_delk = (np.array(Cell(binned_ell_delk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_nDGP_lin,\
                tracer1_type="g", tracer2_type="k")[1])).flatten()
    # pos-pos
    M1_deldel = (np.array(Cell(binned_ell_deldel, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_nDGP_lin,\
                tracer1_type="g", tracer2_type="g")[1])).flatten()

    M1 = np.concatenate((M1_kk, M1_delk, M1_deldel), axis=0)

    """MG2: f(R)""" 
        
    # A: find C_ell for non-linear matter power spectrum
    # shape-shape 
    B3_kk = (np.array(Cell(binned_ell_kk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_fR_nl,\
                tracer1_type="k", tracer2_type="k")[1])).flatten()
    # shape-pos
    B3_delk = (np.array(Cell(binned_ell_delk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_fR_nl,\
                tracer1_type="g", tracer2_type="k")[1])).flatten()
    # pos-pos
    B3_deldel = (np.array(Cell(binned_ell_deldel, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_fR_nl,\
                tracer1_type="g", tracer2_type="g")[1])).flatten()

    B3 = np.concatenate((B3_kk, B3_delk, B3_deldel), axis=0)
       
    # B: find C_ell for linear matter power spectrum
    # shape-shape
    M3_kk = (np.array(Cell(binned_ell_kk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_fR_lin,\
                tracer1_type="k", tracer2_type="k")[1])).flatten()
    # shape-pos
    M3_delk = (np.array(Cell(binned_ell_delk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_fR_lin,\
                tracer1_type="g", tracer2_type="k")[1])).flatten()
    # pos-pos
    M3_deldel = (np.array(Cell(binned_ell_deldel, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,P_delta2D_fR_lin,\
                tracer1_type="g", tracer2_type="g")[1])).flatten()

    M3 = np.concatenate((M3_kk, M3_delk, M3_deldel), axis=0)
    
    """GR"""
    # shape-shape
    B2_kk = (np.array(Cell_GR(binned_ell_kk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                tracer1_type="k", tracer2_type="k")[1])).flatten()
    # shape-pos
    B2_delk = (np.array(Cell_GR(binned_ell_delk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                tracer1_type="g", tracer2_type="k")[1])).flatten()
    # pos-pos
    B2_deldel = (np.array(Cell_GR(binned_ell_deldel, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                tracer1_type="g", tracer2_type="g")[1])).flatten()

    B2 = np.concatenate((B2_kk, B2_delk, B2_deldel), axis=0)
       
    # B: find C_ell for linear matter power spectrum
    # shape-shape
    M2_kk = (np.array(Cell_GR(binned_ell_kk, \
                cosmo_linear, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                tracer1_type="k", tracer2_type="k")[1])).flatten()
    # shape-pos
    M2_delk = (np.array(Cell_GR(binned_ell_delk, \
                cosmo_linear, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                tracer1_type="g", tracer2_type="k")[1])).flatten()
    # pos-pos
    M2_deldel = (np.array(Cell_GR(binned_ell_deldel, \
                cosmo_linear, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                tracer1_type="g", tracer2_type="g")[1])).flatten()

    M2 = np.concatenate((M2_kk, M2_delk, M2_deldel), axis=0)

    ### COMBINE
    
    B_data =np.array([B1,B2,B3])
    M_data =np.array([M1,M2,M3])

    # EXTRACT PCA MATRIX
    Usvd = findPCA(M_data, B_data, L_ch_inv)

    # Cut data vector (choleski cov. matrix = I)
    Diff_cut = np.matmul(Usvd[len(M_data):], Diff_ch.T)
    
    #print("time = ", time.time() - start)
    
    #### fsigma8 ####
    Diff_fsigma8 = fsigma_8_dataset - fsigma8_musigma(cosmo, MGparams, 1/(z_fsigma8+1))
    loglik_fsigma8 = -0.5*(np.matmul(np.matmul(Diff_fsigma8,invcovariance_fsigma8),Diff_fsigma8))
    
    return -0.5*(np.matmul(Diff_cut.T,Diff_cut)) + loglik_fsigma8


###################################################################
############### MISCELLANEOUS Functions for code  #################
###################################################################

"""Returns binned ell values (to check we match SRD)"""
def bins(ell_min, ell_max, ell_bin_num):

    # define quantities for binning in ell
    ell_binned_limits = np.linspace(np.log10(ell_min),np.log10(ell_max),num=ell_bin_num + 1)
    bin_edge1 = ell_binned_limits[:-1]
    bin_edge2 = ell_binned_limits[1:]
    ell_binned = 10**((bin_edge1 + bin_edge2) / 2)
    # Repeat ell_binned over all redshift bins, so that len(ell_binned)=len(C_ell_array)
    return ell_binned

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

# Perform PCA with numpy.linalg.svd - find rotation matrix
def findPCA(M_data, B_data, L_ch_inv):
    Delta = np.array(np.matmul(L_ch_inv, (B_data - M_data).T).T)
    Usvd, s, vh = np.linalg.svd(Delta.T, full_matrices=True)
    Usvd = Usvd.T
    return Usvd