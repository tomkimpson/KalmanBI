

import sdeint
import numpy as np 

from gravitational_waves import gw_earth_terms,gw_psr_terms
import logging
from model import H_function

#i.e. nu_{em}(t)
def ν_em(ν0,τ,n,t): #arguments are nu_{\rm em}(t_0), \dot{nu}_{\rm em}(t_0), n_{\rm pl}, t
    exponent = -(n-1)**(-1)
    return ν0*(1+ t/τ)**exponent


# i.e. \dot{nu}_em(t), first derivative 
# copied from mathematica since I am too lazy do differentiate by hand
#See the notebook in notebooks/mathematica
def ν_em_dot(ν0,τ,n,t): 

    exponent = -1 - (1/(-1+n))
    return -ν0*(1+ t/τ)**exponent /((-1+n)*τ)

#second derivative
def ν_em_doubledot(ν0,τ,n,t): 

    prefix = -1 - (1/(-1+n))
    exponent = -2 - (1/(-1+n))

    return -prefix*ν0*(1+ t/τ)**exponent /((-1+n)*τ**2)


def _calculate_tau(n,ν0,ν0_dot):
    return -ν0/((n-1)*ν0_dot)


class SyntheticData:
    
    def __init__(self,P):


        #Load some PTA related quantities
        t = P.t

        #Pulsar parameters
        γ= P.γ  
        σp= P.σp 
        ν0 = P.ν0 
        n = P.n

        #Precalculate tau
        τ = _calculate_tau(P.n,ν0[0],ν0[1])

        #Random seeding
        generator = np.random.default_rng(P.seed)


        #Integrate the state equation
        def f(x,t):
            
            #Equation 3 of Vargas & Melatos
            A = np.array([[0,  1,    0,    0],
                          [0, -γ[0], 1,    0],
                          [0, 0,    -γ[1], 1],
                          [0, 0,     0,   -γ[2]]
                          ])

            #Get the EM evolution and its derivatives
            νem           = ν_em(ν0[0],τ,n,t)
            νem_dot       = ν_em_dot(ν0[0],τ,n,t)
            νem_doubledot = ν_em_doubledot(ν0[0],τ,n,t)
            νem_tripledot = 0.0 #i.e. νem_doubledot is a constant
  
    
            #Equation 4 
            E = np.array([0,γ[0]*νem,γ[1]*νem_dot,νem_tripledot+γ[1]*νem_doubledot])

            return A.dot(x) + E
        def g(x,t):
            Σ = np.zeros((len(x),len(x)))
            Σ[-1,-1] = σp
            return Σ

        x0 = np.array([0.0,ν0[0],ν0[1],ν0[2]])
        self.state = sdeint.itoint(f,g,x0,t,generator=generator)

        
            
        # #The measured frequency, no noise
        # self.f_measured_clean= (1.0-X_factor)*self.intrinsic_frequency - X_factor*pulsars.ephemeris
        
        Hmatrix = H_function(len(x0))


        
        out_product = np.sum(np.dot(self.state, Hmatrix), axis=1) 
        assert np.all(out_product == self.state[:,1])

        measurement_noise = generator.normal(0, P.σm,out_product.shape) # Measurement noise. Seeded


        self.observation = np.sum(np.dot(self.state, Hmatrix), axis=1)  + measurement_noise

