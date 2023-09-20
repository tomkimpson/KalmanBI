

import numpy as np
#from numba import njit

import logging 
from gravitational_waves import gw_psr_terms,gw_earth_terms,null_model
import sys


def H_function(dim_x):
    H_function = np.zeros((dim_x,dim_x))
    H_function[1][1] = 1.0 # only select the nu term
    return H_function


# #These functions have to be outside the class to enable JIT compilation
# #Bit ugly, but works from a performance standpoint
# """
# The diagonal F matrix as a vector
# """
# @njit(fastmath=True)
# def F_function(gamma,dt):
#     return np.exp(-gamma*dt)

# """
# The diagonal Q matrix as a vector
# """
# @njit(fastmath=True)
# def Q_function(gamma,sigma_p,dt):
#     value = sigma_p**2 * (1. - np.exp(-2.0*gamma* dt)) / (2.0 * gamma)
#     return value 
    
# """
# The R matrix as a scalar
# """
# @njit(fastmath=True)
# def R_function(sigma_m):
#     return sigma_m**2
    
