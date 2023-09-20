
import matplotlib.pyplot as plt 
import numpy as np 
import json
import pandas as pd 
import corner
import scienceplots # noqa: F401
from scipy import interpolate
import warnings
import random
from parse import * 
warnings.filterwarnings("error")
plt.style.use('science')

def plot_states_and_observation(t,states,observation):
    
    plt.style.use('science')

    tplot = t / (365*24*3600)
   
    h,w = 12,8
    rows = 5
    cols = 1
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=True)

    for i,ax in enumerate(axes):
        ax.plot(tplot,states[:,i])

        if i ==3:
            break
    axes[-1].plot(tplot,observation)
    

    fs=20
    axes[-1].set_xlabel('t [years]', fontsize=fs)

    axes[0].set_ylabel(r'$\Phi$ [rad]', fontsize=fs)
    axes[1].set_ylabel(r'$\nu$ [Hz]', fontsize=fs)
    #axes[2].set_ylabel(r'$\dot{f}$ [s^{-2}]', fontsize=fs)
    #axes[3].set_ylabel(r'$\ddot{f}$ [s^{-3}]', fontsize=fs)
    axes[4].set_ylabel(r'$\nu_{\rm m}$ [Hz]', fontsize=fs) 


    for ax in axes:
        ax.yaxis.set_tick_params(labelsize=fs-4)
   

    plt.subplots_adjust(wspace=0.1, hspace=0.1)


    plt.show()
