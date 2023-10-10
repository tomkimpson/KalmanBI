

import numpy as np 
import logging

logging.basicConfig()
logging.getLogger(name="KalmanGW").setLevel(logging.INFO)

"""
Class of parameters which define the system
"""
class SystemParameters:


    def __init__(self,
                 NF=np.float64,    # the number format of the arguments
                 T = 10,           # how long to integrate for in years
                 cadence=7,        # the interval between observations
                 n =   3,          # pulsar braking index
                 σp = 1e-25,       # process noise standard deviation
                 σm = 1e-11,       #measurement noise standard deviation
                 seed = 1234,      # this is the noise seed. It is used for sdeint and gaussian measurement noise
                 ν0 = [1.5,-5e-14,5e-27], #nu(t_0), \dot{nu(t_0)}, \ddot{nu(t_0)} Canonical values from https://arxiv.org/pdf/2305.09079.pdf
                 γ = [1e-13,1e-13,1e-6] #gamma, \dot{gamma}, \ddot{gamma}. Canonical values 
                 ): 

        logging.info("Welcome to the Kalman Filter Nested Sampler for estimating pulsar braking indices")

        self.NF = NF 
        self.T = NF(T) 
        self.cadence = NF(cadence)
        self.n = NF(n)
        self.σp = NF(σp)
        self.σm = NF(σm)
        self.seed = seed
        self.ν0 = ν0
        self.γ = γ


        self.dt      = self.cadence * 24*3600 #from days to seconds
        end_seconds  = self.T* 365*24*3600    # from years to second
        self.t       = np.arange(0,end_seconds,self.dt)


        
    


