import numpy as np
import sdeint



def GetData(t, x0, γ, n, σp, σm, seed):
    

    #Random seeding
    generator = np.random.default_rng(seed)

    tau = -x0[0]/x0[1]/(n-1.) 
    vem0 = x0
    g_nu,g_nud, g_nudd = γ

    def f(x, t):

        
        nu,nud,nudd = x[0],x[1],x[2]

        b = n-1
        time_sum = t + tau
        time_ratio = t/tau
        term1 = (1+time_ratio)**(-1/b)

        #Assuming these equations are correct for now. Unchecked
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
        return np.diag([0., 0., σp])



    #Integrate to get the state variables
    states = sdeint.itoint(f, g, x0, t,generator=generator)

    #Add some mean zero noise to nu to get our observation
    measurement_noise = generator.normal(0,σm,states[:,0].shape) # Measurement noise. Seeded
    observations = states[:,0] + measurement_noise


    return states,observations


