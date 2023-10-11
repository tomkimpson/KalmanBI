from logging import raiseExceptions
from pprint import pp
from signal import pause

from pyparsing import PositionToken
from models import construct_QUKFReduced
from numpy.linalg import inv, det, pinv, solve, slogdet
from scipy.integrate import odeint, ode
import matplotlib.pyplot as plt
import sdeint
import scipy.linalg as la
import scipy.optimize as optimize
import numpy as np
#import torch
#import torchdiffeq

#from numba import jit
from scipy.integrate import odeint
#from numba.types import float64, int64
#from numba.typed import List


################################


################################

#@jit(nopython=True)
def f(t, x, g_nudd):

    # The chances of there being a typo below are high. 
    # It would be good to get fresh eyes over it.

    vem0 = 1.5051430406
    vem1 = -5.1380792001e-14

    g_nu = 1e-13
    g_nud = 3e-13
    
    nu = x[0]
    nud = x[1]
    nudd = x[2]
    n=3
    b = n-1

    nstar = 1/b

    tau = -vem0/vem1*nstar

    time_sum = t + tau
    time_ratio = t/tau
    term1 = (1+time_ratio)**(-nstar)

    nu_em = vem0*term1
    nu_emd = -vem0*term1/time_sum*nstar
    nu_emdd = vem0*term1/time_sum**2*(nstar + nstar**2)
    nu_emddd = -vem0*term1/time_sum**3*(2*nstar + 3*nstar**2 + nstar**3)

    a1 = nu - nu_em
    a2 = nud - nu_emd 
    a3 = nudd - nu_emdd

    return np.array([-(g_nu*a1) + nud, -(g_nud*a2) + nudd, -(g_nudd*a3) + nu_emddd])

#@jit(nopython=True)
def rk4_step(func, x, t, dt, *args):
    k1 = dt * func(t, x, *args)
    k2 = dt * func(t + dt / 2, x + k1 / 2, *args)
    k3 = dt * func(t + dt / 2, x + k2 / 2, *args)
    k4 = dt * func(t + dt, x + k3, *args)

    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

#@jit(nopython=True)
def propagate_sigma_points(SigmaPoints, params, Deltat):
    g_nudd = params 

    PropagatedSigmaPoints = np.empty_like(SigmaPoints)
    for i, x in enumerate(SigmaPoints):
        PropagatedSigmaPoints[i] = rk4_step(f, x, 0, Deltat, g_nudd)

    return PropagatedSigmaPoints

################################

#@jit(nopython=True)
def Predict(Wc, Wm, PropagatedSigmaPoints, nstates, Q):
    Xp = np.zeros((nstates, 1))
    for i in range(len(Wm)):
        Xp += Wm[i] * PropagatedSigmaPoints[i].reshape(nstates, 1)
        
    SigmaPointDiff = np.zeros((len(PropagatedSigmaPoints), nstates))
    for i, point in enumerate(PropagatedSigmaPoints):
        SigmaPointDiff[i] = point - Xp.ravel()

    Pp = np.zeros((nstates, nstates))
    for i, diff in enumerate(SigmaPointDiff):
        Pp += Wc[i] * np.outer(diff, diff)
    Pp += Q

    return Xp, Pp

#@jit(nopython=True)
def Update(Xp, Pp, Wc, PropagatedSigmaPoints, Observation, WeightedMeas, MeasurementDiff, RMeas, nmeas):

    Inn = Observation - WeightedMeas

    print("Update")
    print("the size of RMeas is",RMeas.shape)
    print(nmeas)

    SigmaPointDiff = PropagatedSigmaPoints - Xp.T

    PredMeasurementCov = np.zeros((nmeas,nmeas))
    for i in range(len(Wc)):
        PredMeasurementCov += Wc[i] * np.outer((MeasurementDiff.T)[i], (MeasurementDiff.T)[i])
    PredMeasurementCov += RMeas

    CrossCovariance = np.zeros((len(Xp), nmeas))
    for i in range(len(Wc)):
        CrossCovariance += Wc[i] * np.outer(SigmaPointDiff[i], (MeasurementDiff.T)[i])


    kalman_gain = np.linalg.solve(PredMeasurementCov.T, CrossCovariance.T).T

    X = Xp + np.dot(kalman_gain, Inn.reshape(nmeas,1))

    P = Pp - np.dot(kalman_gain, np.dot(PredMeasurementCov, kalman_gain.T))

    sign, log_det = np.linalg.slogdet(PredMeasurementCov)
    ll = -0.5 * (log_det + np.dot(Inn.T, np.linalg.solve(PredMeasurementCov, Inn))
                 + np.log(2 * np.pi))

    return X, P, ll

class KalmanFilterUpdateUKFOPtim(object):
    def __init__(self, ObsRaw, R, nx, ny):
        #self.ObsRaw = ObsRaw
        
        
        self.Obs = ObsRaw




        self.RMeas = R
        #self.t0 = tinit
        self.nstates = nx
        self.nmeas = ny
        #self.gammas = gammas 
        #self.sigmas = sigmas
        self.ll = 0

        self.Q = np.zeros((len(self.Obs),self.nstates,self.nstates))
        self.MeasurementNoise = np.zeros((len(self.Obs),self.nstates,self.nstates))

    def Predict(self,Q):
        self.Xp, self.Pp = Predict(self.Wc, self.Wm, self.PropagatedSigmaPoints, self.nstates, Q)

    def Update(self, Observation, MeasurementNoise):
        print("The size of the measurement noise is: ",MeasurementNoise.shape)
        self.X, self.P, ll = Update(self.Xp, self.Pp, self.Wc, self.PropagatedSigmaPoints,
                             Observation, self.WeightedMeas, self.MeasurementDiff, MeasurementNoise, self.nmeas)

        self.ll += ll

    def PropagateScaledReduced(self, params, Observation):
        Deltat = Observation
        # nstar, nstar2, g_nu, g_nud, g_nudd, vem0, vem1

        params_tuple = (params['g_nudd'])
        self.PropagatedSigmaPoints = propagate_sigma_points(self.SigmaPoints, params_tuple, Deltat)

    def SigmaPointMeasurementsScaledReduced(self, params):

        SP_Meas = np.column_stack((self.PropagatedSigmaPoints[:, 0]))

        # SP_Meas = np.column_stack((self.PropagatedSigmaPoints[:, 0],
        #                            self.PropagatedSigmaPoints[:, 1]))

        # SP_Meas = np.column_stack((self.PropagatedSigmaPoints[:, 0],
        #                            self.PropagatedSigmaPoints[:, 1],
        #                            self.PropagatedSigmaPoints[:, 2]))

        self.WeightedMeas = np.einsum('i,ij->j', self.Wm, SP_Meas.T)

        self.MeasurementDiff = SP_Meas - self.WeightedMeas

    def CalculateWeights(self):
        L = self.nstates
        alpha = 3e-2
        beta =  2
        kappa = 3 - self.nstates
        # kappa = 1

        # Compute sigma point weights
        lambda_ = alpha**2 * (self.nstates + kappa) - self.nstates

        self.Wm = np.concatenate(([lambda_/(self.nstates+lambda_)], 
                             (0.5/(self.nstates+lambda_))*np.ones(2*self.nstates)), axis=None)
        
        self.Wc = np.concatenate(([lambda_/(self.nstates+lambda_)+(1-alpha**2+beta)], 
                             (0.5/(self.nstates+lambda_))*np.ones(2*self.nstates)), axis=None)

        self.gamma = np.sqrt(self.nstates + lambda_)
        # self.Wc = np.full(2*L + 1,  1. / (2*(L + lmbda)))
        # self.Wm = np.full(2*L + 1,  1. / (2*(L + lmbda)))

    def CalculateSigmaPoints(self, X, P):

        epsilon = 0.0# 1e-18
        Pos_definite_Check= 0.5*(P + P.T) + epsilon*np.eye(len(X))
        
        U = la.cholesky(Pos_definite_Check).T # sqrt
        sigma_points = np.zeros((2*self.nstates + 1, self.nstates))
        sigma_points[0] = X
        for i in range(self.nstates):
            sigma_points[i+1] = X + self.gamma*U[:, i]
            sigma_points[self.nstates+i+1] = X - self.gamma*U[:, i]

        self.SigmaPoints = sigma_points
                
    def Get_MeasuremenNoise(self, params):
        self.MeasurementNoise[:,0,0] = self.RMeas[0,0]
        # self.MeasurementNoise[:,1,1] = self.RMeas[1,1]
        # self.MeasurementNoise[:,2,2] = self.RMeas[2,2]

        print("Got measurement nouse. Whats its shape?")
        print(self.MeasurementNoise.shape)


    def ll_on_data(self, params, returnstates=False):
        self.ll = 0

        self.X = np.array([1.5051430406, -5.1380792001e-14, 5.23e-27])

        self.P = np.eye(self.nstates)
        # Remember these are variances, e.g. squared values.
        self.P[0][0] = 1e-20
        self.P[1][1] = 1e-36
        self.P[2][2] = 1e-54

        NObs = len(self.Obs)
        if returnstates:
            xx = np.zeros((NObs,self.nstates))
            px = np.zeros((NObs,self.nstates))

        self.CalculateWeights()

        self.Q = construct_QUKFReduced(params, self.Obs[:,0])
        self.Get_MeasuremenNoise(params)
        
        i = 0
        for step, Obs in enumerate(self.Obs):

            self.CalculateSigmaPoints(self.X.squeeze(), self.P)

            self.PropagateScaledReduced(params, Obs[0])
            self.Predict(self.Q[step,:,:])

            if (Obs[1]==0):
                self.X = self.Xp 
                self.P = self.Pp
            else:
                
                self.CalculateSigmaPoints(self.Xp.squeeze(), self.Pp)
                self.PropagatedSigmaPoints = self.SigmaPoints

                self.SigmaPointMeasurementsScaledReduced(params)
                self.Update(Obs[1:self.nmeas+1], self.MeasurementNoise[step,:,:])
                if returnstates:
                    xx[i,:] = self.X.squeeze()
                    px[i,:] = np.diag(self.P)
                    i+=1

        if returnstates:
            return xx, px, self.ll
        else:
            return self.ll




