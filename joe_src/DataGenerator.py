import numpy as np
import pandas as pd
import sdeint
import matplotlib.pyplot as plt

def GetData(t, tau, n, x0, g_nu, 
            g_nud, g_nudd, sigma):
    
    def f(x, t):

        vem0 = np.array([1.5051430406, -5.1380792001e-14, 5.23e-27])

        nu = x[0]
        nud = x[1]
        nudd = x[2]

        b = n-1
        time_sum = t + tau
        time_ratio = t/tau
        term1 = (1+time_ratio)**(-1/b)

        nu_em = vem0[0]*term1
        nu_emd = -vem0[0]*term1/b/time_sum
        nu_emdd = vem0[0]*n*term1/b**2/time_sum**2
        nu_emddd = -vem0[0]*n*(2*n-1)*term1/b**3/time_sum**3


        a1 = nu - nu_em
        a2 = nud - nu_emd 
        a3 = nudd - nu_emdd

        a = np.asarray([-g_nu*a1 + nud, -g_nud*a2 + nudd, -g_nudd*a3 + nu_emddd])
        return a

    def g(x, t):
        return np.diag([0., 0., sigma])


    states = sdeint.itoint(f, g, x0, t)

    nu_D = states[:,0]
    nud_D = states[:,1]
    nudd_D = states[:,2]

    nu_data =  nu_D  + np.random.randn(t.size) * np.sqrt(1e-22)
    nud_data = nud_D
    nudd_data = nudd_D


    Obs = np.zeros((len(t),4))
    Obs[0,0] = 0
    for i in range(0, len(Obs[:,0])-1):
        Obs[i+1,0] = t[i+1]-t[i]
    Obs[:,1] = nu_data
    Obs[:,2] = nud_data
    Obs[:,3] = nudd_data

    plt.figure(0)
    plt.plot(t,nu_data)
    plt.figure(1)
    plt.plot(t,nud_data)
    plt.figure(2)
    plt.plot(t,nudd_data)
    plt.show()

    return Obs


x0 = np.array([1.5051430406, -5.1380792001e-14, 5.23e-27])
g_nu = 1e-13 
g_nud = 3e-13
g_nudd = 1e-6

sigma = 2.5e-32

day = 86400
NObs = 1000
t = np.linspace(0,1.5e3*day,NObs)

n = 3
tau = -x0[0]/x0[1]/(n-1.) 

Obs = GetData(t, tau, n, x0, g_nu, 
            g_nud, g_nudd, sigma)

df = pd.DataFrame({"dt": Obs[:,0], "nu": Obs[:,1], "nudot": Obs[:,2], "nuddot": Obs[:,3]})

df.to_csv('Data/Data.csv', index=False)



