import numpy as np

def construct_Q(params, Observation):

    Q_list = np.zeros((4, 4))
    dts = Observation['Deltat']
    g = params['gamma_omega']
    s = params['gamma_S']
    q = params['gamma_Q']
    # n = params['gamma_eta']
    sigma_Q = params['sigma_Q']
    sigma_S = params['sigma_S']
    # sigma_eta = params['sigma_eta']
    a = (3*g/5)
    S = sigma_S**2
    Q = sigma_Q**2
    # N = sigma_eta**2

    Q_list[0,0] = Q*a**2/(g-q)**2 * ((1-np.exp(-2*g*dts))/(2*g) - 2*(1-np.exp(-(g+q)*dts))/(g+q) + (1-np.exp(-2*q*dts))/(2*q)) + S*a**2/(g-s)**2 * ((1-np.exp(-2*g*dts))/(2*g) - 2*(1-np.exp(-(g+s)*dts))/(g+s) + (1-np.exp(-2*s*dts))/(2*s))
    Q_list[0,1] = Q*a/(g-q) * ((1-np.exp(-(g+q)*dts))/(g+q) - (1-np.exp(-2*q*dts))/(2*q))
    Q_list[0,2] = -S*a/(g-s) * ((1-np.exp(-(g+s)*dts))/(g+s) - (1-np.exp(-2*s*dts))/(2*s))
    Q_list[0,3] = 0
    Q_list[1,0] = Q_list[0,1].copy()
    Q_list[1,1] = Q*(1-np.exp(-2*q*dts))/(2*q)
    Q_list[1,2] = 0
    Q_list[1,3] = 0
    Q_list[2,0] = Q_list[0,2].copy()
    Q_list[2,1] = 0
    Q_list[2,2] = S*(1-np.exp(-2*s*dts))/(2*s)
    Q_list[2,3] = 0
    Q_list[3,0] = 0
    Q_list[3,1] = 0
    Q_list[3,2] = 0
    # Q_list[3,3] = N*(1-np.exp(-2*n*dts))/(2*n)

    return Q_list

def construct_QReduced(params, Observation):

    Q_list = np.zeros((3,3))

    dts = Observation['Deltat']

    # g = params['gamma_omega']
    g = params['beta2']
    s = params['gamma_S']
    q = params['gamma_Q']

    sigma_Q = params['sigma_Q']
    sigma_S = params['sigma_S']

    a = (3*g/5)

    Q = sigma_Q**2
    S = sigma_S**2

    Q_list[0,0] = Q*a**2/(g-q)**2 * ((1-np.exp(-2*g*dts))/(2*g) - 2*(1-np.exp(-(g+q)*dts))/(g+q) + (1-np.exp(-2*q*dts))/(2*q)) + S*a**2/(g-s)**2 * ((1-np.exp(-2*g*dts))/(2*g) - 2*(1-np.exp(-(g+s)*dts))/(g+s) + (1-np.exp(-2*s*dts))/(2*s))
    Q_list[0,1] = Q*a/(g-q) * ((1-np.exp(-(g+q)*dts))/(g+q) - (1-np.exp(-2*q*dts))/(2*q))
    Q_list[0,2] = -S*a/(g-s) * ((1-np.exp(-(g+s)*dts))/(g+s) - (1-np.exp(-2*s*dts))/(2*s))
    Q_list[1,0] = Q_list[0,1].copy()
    Q_list[1,1] = Q*(1-np.exp(-2*q*dts))/(2*q)
    Q_list[1,2] = 0
    Q_list[2,0] = Q_list[0,2].copy()
    Q_list[2,1] = 0
    Q_list[2,2] = S*(1-np.exp(-2*s*dts))/(2*s)


    return Q_list


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

def construct_QUKF(params, Observation):
    
    b1 = params['beta1']
    b2 = params['beta2']
    gamma_Q = params['gamma_Q']
    gamma_S = params['gamma_S']
    gamma_eta = params['gamma_eta']

    sigma_Q = params['sigma_Q']
    sigma_S = params['sigma_S']
    sigma_eta = params['sigma_eta']

    Q_list = np.zeros((4, 4))
    dts = Observation['Deltat']
    s = gamma_S
    q = gamma_Q
    n = gamma_eta
    S = sigma_S**2
    Q = sigma_Q**2
    N = sigma_eta**2

    g = b2

    A = 6/5*b1 - 9/5*b2
    B = 1/5*b1 - 4/5*b2


    Q_list[0,0] = Q*A**2/(g-q)**2 * ((1-np.exp(-2*g*dts))/(2*g) - 2*(1-np.exp(-(g+q)*dts))/(g+q) + (1-np.exp(-2*q*dts))/(2*q)) + S*B**2/(g-s)**2 * ((1-np.exp(-2*g*dts))/(2*g) - 2*(1-np.exp(-(g+s)*dts))/(g+s) + (1-np.exp(-2*s*dts))/(2*s))
    Q_list[0,1] = Q*A/(g-q) * ((1-np.exp(-(g+q)*dts))/(g+q) - (1-np.exp(-2*q*dts))/(2*q))
    Q_list[0,2] = -S*B/(g-s) * ((1-np.exp(-(g+s)*dts))/(g+s) - (1-np.exp(-2*s*dts))/(2*s))
    Q_list[0,3] = 0
    Q_list[1,0] = Q_list[0,1].copy()
    Q_list[1,1] = Q*(1-np.exp(-2*q*dts))/(2*q)
    Q_list[1,2] = 0
    Q_list[1,3] = 0
    Q_list[2,0] = Q_list[0,2].copy()
    Q_list[2,1] = 0
    Q_list[2,2] = S*(1-np.exp(-2*s*dts))/(2*s)
    Q_list[2,3] = 0
    Q_list[3,0] = 0
    Q_list[3,1] = 0
    Q_list[3,2] = 0
    Q_list[3,3] = N*(1-np.exp(-2*n*dts))/(2*n)

    return Q_list

def construct_transition(gamma_omega, gamma_Q,
    gamma_S, gamma_eta, Observation):

    F_list = np.zeros((4,4))

    dts = Observation['Deltat']
    g = gamma_omega
    s = gamma_S
    q = gamma_Q
    n = gamma_eta
    a = (3*g/5)

    F_list[0,0] = np.exp(-g*dts)
    F_list[0,1] = a*(np.exp(-g*dts)-np.exp(-q*dts))/(g-q)
    F_list[0,2] = a*(np.exp(-s*dts)-np.exp(-g*dts))/(g-s)
    F_list[0,3] = 0
    F_list[1,0] = 0
    F_list[1,1] = np.exp(-q*dts)
    F_list[1,2] = 0
    F_list[1,3] = 0
    F_list[2,0] = 0
    F_list[2,1] = 0
    F_list[2,2] = np.exp(-s*dts)
    F_list[2,3] = 0
    F_list[3,0] = 0
    F_list[3,1] = 0
    F_list[3,2] = 0
    # F_list[3,3] = np.exp(-n*dts)

    return F_list

