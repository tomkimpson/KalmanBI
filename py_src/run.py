


from system_parameters import SystemParameters

from synthetic_data import SyntheticData

#from kalman_filter import KalmanFilter
#from bilby_wrapper import BilbySampler
#from priors import priors_dict,bilby_priors_dict
import logging 
import sys

def bilby_inference_run(arg_name,seed):

    logger = logging.getLogger().setLevel(logging.INFO)
    #Setup the system
    P   = SystemParameters(seed=seed) # define the system parameters as a dict. Todo: make this a class
                                # setup the PTA
    data = SyntheticData(P)                            # generate some synthetic data





    



    # #Define the model 
    # model = LinearModel(P)

    # #Initialise the Kalman filter
    # KF = KalmanFilter(model,data.f_measured,PTA)

    # #Run the KF once with the correct parameters.
    # #This allows JIT precompile
    # optimal_parameters = priors_dict(PTA,P)
    # model_likelihood = KF.likelihood(optimal_parameters)
    # logging.info(f"Ideal likelihood given optimal parameters = {model_likelihood}")
    
    # #Bilby
    # init_parameters, priors = bilby_priors_dict(PTA,P)
   

    # logging.info("Testing KF using parameters sampled from prior")
    # params = priors.sample(1)
    # model_likelihood = KF.likelihood(params)
    # logging.info(f"Non -ideal likelihood for randomly sampled parameters = {model_likelihood}")

    
    # # #Now run the Bilby sampler
    # BilbySampler(KF,init_parameters,priors,label=arg_name,outdir="../data/nested_sampling/")
    # logging.info("The run has completed OK")

