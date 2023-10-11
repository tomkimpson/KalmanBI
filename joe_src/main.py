# import numpy as np 
# import matplotlib.pyplot as plt
# import pandas as pd

# from UKF_Optim import KalmanFilterUpdateUKFOPtim
# from sample import KalmanLikelihood
# from corner import corner
# import bilby



import numpy as np 
import logging 


from DataGenerator import GetData

def RunKF():
    logger = logging.getLogger().setLevel(logging.INFO)


    #Define all parameters 
    x0                             = [1.5051430406, -5.1380792001e-14, 5.23e-27]        #initial conditions of nu, nu dot, nu double dot for generating synthetic data
    γ                              = [1e-13, 3e-13, 1e-6]                                    #the gammas 
    n                              = 3                                                     #pulsar braking index
    σp                             = 2.5e-32                                               #process noise sigma which acts on nu double dot
    σm                             = 1e-11                                                 #measurement noise sigma which acts on the state nu to give us our observations
    seed                           = 1234

    #Define the observation period
    NObs = 1000
    day = 86400
    num_days = 1.5e3
    t   = np.linspace(0,num_days*day,NObs) #Nobs observations over num_days
    logging.info(f"The interval between observations is {t[1] - t[0]} seconds or {(t[1] - t[0])/day} days ")

    #Generate some synthetic data
    logging.info(f"Generating synthetic data")
    states,obseravations = GetData(t, x0, γ, n, σp, σm, seed)





















#     Data = pd.read_csv('Data/Data.csv', sep=',')
# #   print(Data)

#     OutputDirectory = 'Results'
#     fID = 'test'

#     ObsData = np.asarray(Data)
#     ObsScaled = ObsData[:,0:2]
#     tinit = 0
#     nstates = 3
#     nmeas = 1

#     g_nu = 1e-13 
#     g_nud = 3e-13
#     g_nudd = 1e-6

#     n = 3

#     b = 1/(n-1)

#     nstar = b

#     t = np.linspace(0,len(ObsData[:,0]),len(ObsData[:,0]))

#     Sigmanudd = 2.5e-32

#     # Measurement noise MeasR defined below. Remember, we are dealing with variances. Hence 
#     # the entries of MeasR are squared. 
#     MeasR = np.zeros((nmeas, nmeas))
#     MeasR[0,0] = 1e-22

#     # gammas/sigmas below are brought forward from a different project. They don't actually do anything 
#     # in the present code.

#     gammas = [] 
#     sigmas = []

#     model = KalmanFilterUpdateUKFOPtim(ObsData, ObsScaled, MeasR, tinit, nstates, nmeas, gammas, sigmas)

#     likelihood = KalmanLikelihood(model)

#     # # # Set priors for recovery
#     limL = 1e-3
#     limU = 1e3
#     print("TRUTH", nstar, Sigmanudd, g_nudd)
#     priors = bilby.core.prior.PriorDict()
#     priors['Sigma'] = bilby.core.prior.LogUniform(limL*Sigmanudd, limU*Sigmanudd, 'Sigma')
#     priors['g_nudd'] = bilby.core.prior.LogUniform(limL*g_nudd, limU*g_nudd, 'g_nudd')

#     # # # Do the parameter estimation

#     result = bilby.run_sampler(likelihood, priors, 
#                         sampler='dynesty', sample='rwalk', walks=15, npoints=250,
#                         resume=True, outdir=OutputDirectory, npool = 6, dlogz=0.1,
#                         label=fID, check_point_plot=False, plot=True)
    
#     # result = bilby.result.read_in_result(filename='Results/test_result.json')

#     range = [(limL*Sigmanudd, limU*Sigmanudd),
#              (limL*g_nudd, limU*g_nudd)]
    
#     truth = [Sigmanudd, g_nudd]
#     print("truth", truth)
#     truth = np.log10(truth)
#     # range = np.log10(range)

#     samples = result.posterior.to_numpy()[:, :2].copy()
#     samples[:, 0:2] = np.log10(samples[:, 0:2])
#     fig = corner(samples, 
#                  color='b',
#                 #  truths = truth,
#                 #  range = range,
#                 #  smooth=True, 
#                 #  smooth1d=True, 
#                  levels=[1-np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9/2)])
#     plt.show()


#     param_dict = { "Sigma": 10**samples[-1,0], 
#                   "g_nu": g_nu, 
#                   "g_nud": g_nud, 
#                   "g_nudd": 10**samples[-1,1]}
    
#     # param_dict = {"Sigma": Sigmanudd, 
#     #               "g_nu":  g_nu, 
#     #               "g_nud": g_nud, 
#     #               "g_nudd": g_nudd}
    
#     print(param_dict)
#     xx, px, ll = model.ll_on_data(params = param_dict, returnstates = True)

#     day = 86400
#     NObs = 1000
#     t = np.linspace(0,1.5e3*day,NObs)
    
#     plt.figure(0)
#     plt.plot(t, xx[:,0], 'r')
#     plt.plot(t,ObsData[:,1], alpha=0.2)
#     plt.figure(1)
#     plt.plot(t, xx[:,1], 'r')
#     plt.plot(t, ObsData[:,2], alpha=0.2)
#     plt.figure(2)
#     plt.plot(t, xx[:,2], 'r')
#     plt.plot(t,ObsData[:,3])
#     plt.figure(3)
#     plt.plot(t, xx[:,0]*xx[:,2]/xx[:,1]**2, 'r')
#     # plt.plot(t,ObsData[:,3])
#     plt.show()

#     print((xx[:,0]*xx[:,2]/xx[:,1]**2).mean())

#     plt.figure(3)
#     plt.plot(t, xx[:,0]-ObsData[:,1], 'ro')
#     plt.figure(4)
#     plt.plot(t, xx[:,1]-ObsData[:,2], 'ro')
#     plt.figure(5)
#     plt.plot(t, xx[:,2]-ObsData[:,3], 'ro')
#     plt.show()

if __name__=="__main__":
    RunKF()

