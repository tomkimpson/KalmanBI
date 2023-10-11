import numpy as np


def construct_QUKFReduced(params, Observation):
    sigma = params['Sigma']
    g_nudd = params['g_nudd']

    Q_list = np.zeros((len(Observation), 3, 3))
    dts = Observation
    S = sigma**2

    Q_list[:,1,1] = dts**3*S/3.
    Q_list[:,1,2] = 1/6*dts**2*(3. - 2*g_nudd*dts)*S
    Q_list[:,2,1] = Q_list[:,1,2].copy()
    Q_list[:,2,2] = S*(1+(-1+g_nudd*dts)**3)/(3*g_nudd)
    return Q_list

