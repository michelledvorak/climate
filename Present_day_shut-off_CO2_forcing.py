#!/usr/bin/env python
# coding: utf-8

# Use constrained ensemble to estimate committed warming today. Cessation of all emissions + CO2 only.

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

from fair.SSPs_mine import ssp245

nt = 2201-1765 # up to 2100

# In[3]:


natural = fair.ancil.natural.Emissions.emissions[:nt]
SSP_245 = ssp245.Emissions_245.emissions[0:nt,:]
#natural_2 = fair.ancil.natural_2.Emissions.emissions[:nt]

#run all three ensembles: original, standard (should be almost the same), and inflated
# all ensembles use natural, not natural_2
# figure out why standard + inflated warm so much more in CO2-only case (looks like an issue with GHG emissions)

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

sy = 2021-1765 #start year
nc = len(lamda)

# Pull all non-CO2 forcing from the most recent forcing timeseries

F = np.load('../FAIR-master/remote_runs_NOx/F_245.npy')
other_F = np.sum(F[:,1:,:], axis=1)

constant_F = np.zeros((nt,nc))

for i in range(nc):
    constant_F[:sy,i] = other_F[:sy,i]
    constant_F[sy:,i] = other_F[sy,i] # all non-CO2 forcing is held at 2020 levels
    
CO2_off = np.zeros(nt)
CO2_off[:sy] = SSP_245[:sy,1]
CO2_off[sy:] = 0 # CO2 emissions are zeroed at (beg. of) 2020

CO2_1765 = np.zeros(nt)
CO2_1765[:sy] = SSP_245[:sy,1]
CO2_1765[sy:] = SSP_245[0,1] # CO2 emissions are zeroed at (beg. of) 2020

CO2_pi = np.zeros(nt)
CO2_pi[:sy] = SSP_245[:sy,1]
CO2_pi[sy:] = SSP_245[1850-1765:1900-1765,1].mean(axis=0) # CO2 emissions are zeroed at (beg. of) 2020

print('The constrained ensemble has %s members' %nc)
#to be consistent with the large ensemble run, start the shut-off experiment last year (2020) 
# i.e., index of 0 = shut-off year 0 = shut-off 2020
print('other_rf has shape: '+ str(constant_F.shape))

print('CO2 forcing is of shape: ' + str(CO2_off.shape))

print('median other_rf array is: %s' %(np.percentile(constant_F, 50, axis=1)))

from fair.ancil import cmip6_volcanic
from fair.ancil import cmip6_solar
volcanic = cmip6_volcanic.Forcing.volcanic[:nt]

print('The first shut-off year is: %s' %(sy+1765))
print()
print('Running the standard ensemble for a shut-off of only CO2 today, all other forcing fixed')

print('NOx emissions in 2020 are: ' + str(SSP_245[2021-1765,8]))
print('NOx forcing in 2020 is: ' + str(F[2021-1765,8,:].mean()))

# T_245_CO2_1765 = np.zeros((nt,nc))

# for i in range(nc):
#     _, _, T_245_CO2_1765[:,i], _ = fair.forward3.fair_scm(useMultigas=False, emissions=CO2_1765, 
#                                   eps = epsilon[i], #mean feedback parameters to be consistent with scaling factors
#                                   lam = (3.71*scale[i,0])/ECS[i],
#                                   gam = gamma[i],
#                                   Cml = Cml[i] * 31363200,
#                                   Cdeep = Cdeep[i] * 31363200,
#                                   r0 = r0[i],
#                                   rc = rc[i],
#                                   rt = rt[i],
#                                   other_rf = constant_F[:,i],
#                                   scale = scale[i,0],
# #                                      F2x = 3.71*scale_norm[i,0], removed due to redunancy
#                                   F_solar=np.zeros(nt),
#                                   F_volcanic=0.6*volcanic,
#                                   natural=natural,
#                                   scaleAerosolAR5=False)

# #         if T_370[:,i,k].max() > 2:
# #             committed_370[i] = k
# T_245_CO2_1765 = T_245_CO2_1765 - T_245_CO2_1765[1850-1765:1900-1765,:].mean(axis=0)    

# np.save('../FAIR-master/remote_runs_NOx/Sensitivity_tests/T_245_CO2_1765', T_245_CO2_1765)

# T_245_CO2_pi = np.zeros((nt,nc))

# for i in range(nc):
#     _, _, T_245_CO2_pi[:,i], _ = fair.forward3.fair_scm(useMultigas=False, emissions=CO2_pi, 
#                                   eps = epsilon[i], #mean feedback parameters to be consistent with scaling factors
#                                   lam = (3.71*scale[i,0])/ECS[i],
#                                   gam = gamma[i],
#                                   Cml = Cml[i] * 31363200,
#                                   Cdeep = Cdeep[i] * 31363200,
#                                   r0 = r0[i],
#                                   rc = rc[i],
#                                   rt = rt[i],
#                                   other_rf = constant_F[:,i],
#                                   scale = scale[i,0],
# #                                      F2x = 3.71*scale_norm[i,0], removed due to redunancy
#                                   F_solar=np.zeros(nt),
#                                   F_volcanic=0.6*volcanic,
#                                   natural=natural,
#                                   scaleAerosolAR5=False)

# #         if T_370[:,i,k].max() > 2:
# #             committed_370[i] = k
# T_245_CO2_pi = T_245_CO2_pi - T_245_CO2_pi[1850-1765:1900-1765,:].mean(axis=0)    

# np.save('../FAIR-master/remote_runs_NOx/Sensitivity_tests/T_245_CO2_preind', T_245_CO2_pi)

# C_245_CO2 = np.zeros((nt,nc))
# F_245_CO2 = np.zeros((nt,nc))
# T_245_CO2 = np.zeros((nt,nc))
N_245_CO2 = np.zeros((nt,nc))

natural = np.zeros(fair.ancil.natural.Emissions.emissions[:nt].shape)

print('natural emissions in %s are: %s' %(sy, natural[2020-1765]))

for i in range(nc):
    _, _, _, N_245_CO2[:,i] = fair.forward3.fair_scm(useMultigas=False, emissions=CO2_off, 
                                  eps = epsilon[i], #mean feedback parameters to be consistent with scaling factors
                                  lam = (3.71*scale[i,0])/ECS[i],
                                  gam = gamma[i],
                                  Cml = Cml[i] * 31363200,
                                  Cdeep = Cdeep[i] * 31363200,
                                  r0 = r0[i],
                                  rc = rc[i],
                                  rt = rt[i],
                                  other_rf = constant_F[:,i],
                                  scale = scale[i,0],
#                                      F2x = 3.71*scale_norm[i,0], removed due to redunancy
                                  F_solar=np.zeros(nt),
                                  F_volcanic=0.6*volcanic,
                                  natural=natural,
                                  scaleAerosolAR5=False)

#         if T_370[:,i,k].max() > 2:
#             committed_370[i] = k
# T_245_CO2 = T_245_CO2 - T_245_CO2[1850-1765:1900-1765,:].mean(axis=0)    

# np.save('../FAIR-master/remote_runs_NOx/Sensitivity_tests/CO2/C_245_CO2_long', C_245_CO2)
# np.save('../FAIR-master/remote_runs_NOx/Sensitivity_tests/CO2/F_245_CO2_long', F_245_CO2)
# np.save('../FAIR-master/remote_runs_NOx/Sensitivity_tests/CO2/T_245_CO2_long', T_245_CO2)
np.save('../FAIR-master/remote_runs_NOx/Sensitivity_tests/CO2/N_245_CO2_long', N_245_CO2)

end_time = time.time()
print(f"The execution time is: {(end_time-start_time)/60} minutes")