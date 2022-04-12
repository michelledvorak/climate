#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import random
import pandas as pd

import time
start_time = time.time()

import fair
from fair.forward2 import fair_scm
from fair.forward import fair_scm
from fair.forward3 import fair_scm
from fair.ancil import natural
from fair.ancil import natural_2


# In[2]:


from fair.SSPs_mine import ssp119, ssp126, ssp245, ssp370, ssp370_lowNTCF, ssp434, ssp460, ssp585

nt = 2101-1765 # up to 2100

# In[3]:


natural = fair.ancil.natural.Emissions.emissions[:nt]
#natural_2 = fair.ancil.natural_2.Emissions.emissions[:nt]


# In[4]:


SSP_119 = ssp119.Emissions_119.emissions[0:nt,:]
SSP_126 = ssp126.Emissions_126.emissions[0:nt,:]
SSP_245 = ssp245.Emissions_245.emissions[0:nt,:]
SSP_370 = ssp370.Emissions_370.emissions[0:nt,:]
SSP_370_lowNTCF = ssp370_lowNTCF.Emissions_370_lowNTCF.emissions[0:nt,:]
SSP_434 = ssp434.Emissions_434.emissions[0:nt,:]
SSP_460 = ssp460.Emissions_460.emissions[0:nt,:]
SSP_585 = ssp585.Emissions_585.emissions[0:nt,:]

SSP_list = [SSP_126, SSP_370, SSP_245, SSP_585]
SSP_names = ['126', '370', '245', '585']


# In[5]:
# Upload posteriors

ECS = np.load('../FAIR-master/remote_runs_NOx/ECS_post_remote.npy')
lamda = np.load('../FAIR-master/remote_runs_NOx/lamda_post_remote.npy')
gamma = np.load('../FAIR-master/remote_runs_NOx/gamma_post_remote.npy')
epsilon = np.load('../FAIR-master/remote_runs_NOx/epsilon_post_remote.npy')
scale = np.load('../FAIR-master/remote_runs_NOx/scaling_post_remote.npy')
Cml = np.load('../FAIR-master/remote_runs_NOx/Cml_post_remote.npy')
Cdeep = np.load('../FAIR-master/remote_runs_NOx/Cdeep_post_remote.npy')
r0 = np.load('../FAIR-master/remote_runs_NOx/r0_post.npy')
rc = np.load('../FAIR-master/remote_runs_NOx/rc_post.npy')
rt = np.load('../FAIR-master/remote_runs_NOx/rt_post.npy')

# In[7]:


nc = len(lamda)
print('The constrained ensemble has %s members' %nc)

sy = 2020-1765 #start year

#return to 1765 emissions levels in 2020
from collections import defaultdict

emissions = defaultdict(list)

for SSP, name in zip(SSP_list, SSP_names):
    
    for j in range(60):

        emissions1 = np.zeros((nt, 40))
        emissions1[:sy+j,:] = SSP[0:sy+j,:]
        emissions1[sy+j:,0] = SSP[sy+j:,0]
        emissions1[sy+j:,1:5] = 0 # 0 emissions of CO2, CH4 and N2O

        for i in range(5,40):
            emissions1[sy+j:,i] = SSP[0,i] # 1765 levels of all the rest
        
        emissions[name].append(emissions1)

from fair.ancil import cmip6_volcanic
volcanic = cmip6_volcanic.Forcing.volcanic[:nt]

years = len(emissions['119']) # number of shut-off years

print('The simulation will run for %s shut-off years' %years)
print('The first shut-off year is: %s' %sy)

# In[40]:

# Run the shut-off experiment (6723 x 60 x 8)

for name in SSP_names[1:]:
    print('Running full shut-off experiment for %s' %(name))
    T_alloff = np.zeros((nt,nc, 60))

    for i in range(nc):

        for k in range(years):        
            _, _, T_alloff[:,i,k], _ = fair.forward3.fair_scm(emissions=emissions[name][k], 
                                          eps = epsilon[i], #mean feedback parameters to be consistent with scaling factors
                                          lam = (3.71*scale[i,0])/ECS[i],
                                          gam = gamma[i],
                                          Cml = Cml[i] * 31363200,
                                          Cdeep = Cdeep[i] * 31363200,
                                          r0 = r0[i],
                                          rc = rc[i],
                                          rt = rt[i],
                                          scale = scale[i,:],
    #                                      F2x = 3.71*scale_norm[i,0], removed due to redunancy
                                          F_solar=np.zeros(nt),
                                          F_volcanic=0.6*volcanic,
                                          natural=natural,
                                          scaleAerosolAR5=False)

    #         if T_370[:,i,k].max() > 2:
    #             committed_370[i] = k
    T_alloff = T_alloff - T_alloff[1850-1765:1900-1765,:,:].mean(axis=0)            
    #             break

    np.save('../FAIR-master/remote_runs_NOx/T_%s_alloff.npy' %(name), T_alloff)

for name, SSP in zip(SSP_names, SSP_list):
    print('Running no cessation experiment for %s' %(name))
    F = np.zeros((nt,13,nc))

    for i in range(nc):
   
        _, F[:,:,i], _, _ = fair.forward3.fair_scm(emissions=SSP, 
                                      eps = epsilon[i], #mean feedback parameters to be consistent with scaling factors
                                      lam = (3.71*scale[i,0])/ECS[i],
                                      gam = gamma[i],
                                      Cml = Cml[i] * 31363200,
                                      Cdeep = Cdeep[i] * 31363200,
                                      r0 = r0[i],
                                      rc = rc[i],
                                      rt = rt[i],
                                      scale = scale[i,:],
#                                      F2x = 3.71*scale_norm[i,0], removed due to redunancy
                                      F_solar=np.zeros(nt),
                                      F_volcanic=0.6*volcanic,
                                      natural=natural,
                                      scaleAerosolAR5=False)

    #         if T_370[:,i,k].max() > 2:
    #             committed_370[i] = k
 #   T = T - T[1850-1765:1900-1765,:].mean(axis=0)            
    #             break

    np.save('../FAIR-master/remote_runs_NOx/F_%s.npy' %(name), F)

end_time = time.time()
print(f"The execution time is: {(end_time-start_time)/60} minutes")

