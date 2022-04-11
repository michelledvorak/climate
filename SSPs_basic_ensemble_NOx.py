#!/usr/bin/env python
# coding: utf-8

# Running World Without Us scenarios for SSP 585 with AR6 scaling factors; generating large ensemble; constraining results; identifying parameter relationships that arise

# In[1]:

import time
start_time = time.time()
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import random
import pandas as pd

import fair
from matplotlib import pyplot as plt
from fair.forward2 import fair_scm
from fair.forward import fair_scm
from fair.forward3 import fair_scm

# In[2]:


from fair.SSPs import ssp119, ssp126, ssp245, ssp370, ssp434, ssp585
nt = 2020-1765

# In[3]:


natural = fair.ancil.natural.Emissions.emissions[0:nt,:]


# In[5]:


SSP_119 = ssp119.Emissions_119.emissions[:nt,:]
SSP_126 = ssp126.Emissions_126.emissions[:nt,:]
SSP_434 = ssp434.Emissions_434.emissions[:nt,:]
SSP_245 = ssp245.Emissions_245.emissions[:nt,:]
SSP_370 = ssp370.Emissions_370.emissions[:nt,:]
SSP_585 = ssp585.Emissions_585.emissions[:nt,:]

SSP_list = [SSP_119, SSP_126, SSP_434, SSP_245, SSP_370, SSP_585]


# Create AR6 scaling factors by running FaIR with SSP historical data with NO AR5 scaling

# In[6]:

print('Using FaIR stock natural background emissions')

C585, F585, T585, _ = fair.forward3.fair_scm(emissions=SSP_585, # mean values for lam, eps, gam Geo v2!
                                             lam=1.18, 
                                             eps=1.28, 
                                             gam=0.67, 
                                             scaleAerosolAR5=False,
                                             natural=natural)
#                                              scaleHistoricalAR5=True)


# In[7]:


F_1765 = [F585[0,0], F585[0,1], F585[0,2], F585[0,8]]
F_1880 = [F585[115,0], F585[115,1], F585[115,2], F585[115,8]]
F_2016 = [F585[251,0], F585[251,1], F585[251,2], F585[251,8]]
F_2018 = [F585[253,0], F585[253,1], F585[253,2], F585[253,8]]

F_2018[0] - F_1765[0]


# World Without Us Scenarios: return to pre-industrial (1765) emissions levels for primary gases

# In[8]:



# In[9]:


#2021 = year 256
#2018 = year 253

# #return to 1765 emissions levels

# sy = 2020-1765 #start year

# emissions = np.zeros((nt, 40))
# emissions[:sy,:] = SSP_245[0:sy,:]
# emissions[sy:,0] = SSP_245[sy:,0]
# emissions[sy:,1:5] = 0

# for i in range(5,40):
#     emissions[sy:,i] = SSP_245[0,i]


# In[10]:


import scipy as sci

# AR6 reported 90% confidence interval

CO2_95 = np.array([1.89, 2.15, 2.41])
CH4_95 = np.array([0.43, 0.54, 0.65])
N2O_95 = np.array([0.16, 0.19, 0.22])
aerosols_95 = np.array([-0.4, -1.1, -2.0])

lamda = np.array([0.4, 1.18, 3.3]) # update low and high ends with CMIP6 (low: 0.4 to 3.0)
# 0.570, 1.18, 1.79
#epsilon = np.array([0.868, 1.28, 1.69]) #original
epsilon = np.array([0.434, 1.28, 2.54]) # 50% expansion
#gamma = np.array([0.423, 0.67, 0.918]) #original
gamma = np.array([0.212, 0.67, 1.38]) # 50% expansion

# calculate standard deviation, s, from C.I. and mean:

CO2_s = (CO2_95[2] - CO2_95[1])/1.65
CH4_s = (CH4_95[2] - CH4_95[1])/1.65
N2O_s = (N2O_95[2] - N2O_95[1])/1.65
aerosols_s = np.abs((aerosols_95[2] - aerosols_95[1])/1.65+(aerosols_95[1] - aerosols_95[0])/1.65)/2

# standard deviation for lambda, epsilon and gamma from Geo v2

lam_s = 0.4727 # 0.3697, 0.4848
# eps_s = 0.2485 #original
eps_s = 0.375 # 50% expansion
# gam_s = 0.1503 #original
gam_s = 0.225 # 50% expansion

lnsigma = np.log(1+(eps_s/epsilon[1])**2)
lnmu = np.log(epsilon[1])-0.5*(lnsigma**2)

# Generate Gaussian distributions of CO2, methane, nitrous oxide and aerosol radiative forcing using AR6 means and s.d.; do the same for climate feedback parameters using Geoffrey v.2

# In[11]:

ne = 300000 #size of ensemble

from scipy import stats

CO2_norm = stats.norm.rvs(size=ne, loc=CO2_95[1], scale=CO2_s, random_state=3970)
CH4_norm = stats.norm.rvs(size=ne, loc=CH4_95[1], scale=CH4_s, random_state=53060)
N2O_norm = stats.norm.rvs(size=ne, loc=N2O_95[1], scale=N2O_s, random_state=1532)
aerosols_norm = stats.norm.rvs(size=ne, loc=aerosols_95[1], scale=aerosols_s, random_state=888955)

np.random.seed(2001)
aerosols_norm = np.random.uniform(low=-2.2, high=-0.1, size=ne)

r0 = stats.norm.rvs(size=ne, loc=33.8, scale=3.38, random_state=41000)
rc = stats.norm.rvs(size=ne, loc=0.019, scale=0.0019, random_state=42000)
rt = stats.norm.rvs(size=ne, loc=4.165, scale=0.4165, random_state=45000)

Cml_norm = stats.norm.rvs(size=ne, loc=8.2, scale=1.35, random_state=25400) # 50% expansion of s.d.
Cdeep_norm = stats.norm.rvs(size=ne, loc=109, scale=78, random_state=21000) # 50% expansion of s.d.

CO2_scale = CO2_norm/(F_2018[0] - F_1765[0])
CH4_scale = CH4_norm/(F_2018[1] - F_1765[1])
N2O_scale = N2O_norm/(F_2018[2] - F_1765[2])
aerosols_scale = aerosols_norm/(F_2018[3] - F_1765[3])

scale_norm = np.ones((ne, 13))

scale_norm[:, 0] = CO2_scale
scale_norm[:, 1] = CH4_scale
scale_norm[:, 2] = N2O_scale
scale_norm[:, 8] = aerosols_scale

np.random.seed(50121)
lam_uniform = np.random.uniform(low=0.4, high=3.3, size=ne)
ECS_uniform = np.random.uniform(low=1, high=6, size=ne)
#eps_uniform = np.random.uniform(low=0.8, high= 2.5, size=1000)
#gam_uniform = np.random.uniform(low=0.1, high=1.5, size=100)

# lam_norm = np.random.lognormal(0.17, 0.3697, ne)
# eps_norm = stats.norm.rvs(size=1000, loc=epsilon[1], scale=eps_s, random_state=52000)
eps_lognorm = np.random.lognormal(lnmu,lnsigma, ne)
gam_norm = stats.norm.rvs(size=ne, loc=gamma[1], scale=gam_s, random_state=63990)
   
#eps_replace = np.random.uniform(low = 0.78, high=1.8, size=ne)
gam_replace = stats.norm.rvs(size=ne, loc=gamma[1], scale=gam_s, random_state=63340)
#Cdeep_replace = np.random.uniform(low=20, high=280, size=ne)
Cdeep_replace = stats.norm.rvs(size=ne, loc=109, scale=78, random_state=39494)
    
for i,j in zip(range(len(gam_norm)), range(len(gam_replace))):
    
    while gam_norm[i] < 0.1:
        if gam_replace[j] > 0.1:
            gam_norm[i] = gam_replace[j]
        else:
            j += 1
        
for i,j in zip(range(len(Cdeep_norm)), range(len(Cdeep_replace))):
    
    while Cdeep_norm[i] < 10:
        if Cdeep_replace[j] > 10:
            Cdeep_norm[i] = Cdeep_replace[j]
        else:
            j += 1
# In[22]:

# pull up volcanic forcing time-series

from fair.ancil import cmip6_volcanic
volcanic = cmip6_volcanic.Forcing.volcanic[:nt]
print('volcanic forcing array is of shape: ' + str(volcanic.shape))
print('The ensemble is of size: %s' %(ne))


# generate the ensemble

T = np.zeros((nt,ne))
F = np.zeros((nt,13,ne))
C = np.zeros((nt,31,ne))
N = np.zeros((nt,ne))
T_mean = np.zeros(nt,)
T_std = np.zeros(nt,)

for i in range(ne):
    C[:,:,i], F[:,:,i], T[:,i], N[:,i] = fair.forward3.fair_scm(emissions=SSP_245, 
                                      eps = eps_lognorm[i], #mean feedback parameters to be consistent with scaling factors
                                      lam = (3.71*scale_norm[i,0])/ECS_uniform[i],
#                                      lam = lam_uniform[i],
                                      gam = gam_norm[i],
                                      Cml = Cml_norm[i]*31363200,
                                      Cdeep = Cdeep_norm[i]*31363200,
                                      r0 = r0[i],
                                      rc = rc[i],
                                      rt = rt[i],                   
                                      scale = scale_norm[i,:],
#                                      F2x = 3.71*scale_norm[i,0], removed due to redunancy
                                      natural=natural,
                                      F_volcanic=0.6*volcanic,
                                      scaleAerosolAR5=False)

T = T - T[1850-1765:1900-1765,:].mean(axis=0)
#F = F - F[1850-1765:1900-1765,:,:].mean(axis=0)

# Bayesian updating step that constrains temperature, heat uptake and radiative forcing by observational means

# In[25]:

# updated 21 May 2021

constrained_2 = np.zeros(ne, dtype=bool)

def constrain(T_model, N_model, F_model, C_model, sigma_T, sigma_N, sigma_F):
    return np.sqrt(((T_model-1.03)/sigma_T)**2 + 
                   #((T_model_2-0.7)/sigma_T_2)**2 +
                   ((N_model-0.59)/sigma_N)**2 +
                   ((F_model-2.20)/sigma_F)**2) < 1.645 and 393.98 < C_model < 397.98

for i in range(ne):
    # we use observed trends from 1850-1900 and 2006-2018, plus 1970-1980 and 2008-2018
    T_model = T[2006-1765:2019-1765,i].mean()
    #T_model_2 = T[2008-1765:2018-1765,i].mean() - T[1970-1765:1980-1765,i].mean()
    N_model = N[2006-1765:2019-1765,i].mean() - N[1850-1765:1900-1765,i].mean()
    F_model = np.sum(F[2006-1765:2019-1765,:,i], axis=1).mean() - np.sum(F[1850-1765:1900-1765,:,i], axis=1).mean()   
    C_model = C[2006-1765:2019-1765,0,i].mean(axis=0)
    constrained_2[i] = constrain(T_model,
                                 #T_model_2,
                                 N_model, 
                                 F_model,
                                 C_model,
                                 sigma_T=(0.20/1.645),
                                 sigma_N=(0.35/1.645), 
                                 sigma_F=(0.7/1.645))

print('%d ensemble members passed observational constraint' % np.sum(constrained_2))
end_time = time.time()
print('The execution time is: %s minutes' %((end_time-start_time)/60))

# Compare the priors (distributions of each variable) with the posteriors for forcing variables derived from the Bayesian method

# In[176]:

nc = np.sum(constrained_2)
scale_dist = np.ones((nc,13))

scale_dist[:,0] = scale_norm[constrained_2,0]
scale_dist[:,1] = scale_norm[constrained_2,1]
scale_dist[:,2] = scale_norm[constrained_2,2]
scale_dist[:,8] = scale_norm[constrained_2,8]


# In[177]:

lam = (3.71*scale_norm[:,0])/ECS_uniform
lam_dist = lam[constrained_2]

gam_dist = gam_norm[constrained_2]
#gam_dist = gam_uniform[constrained_2]

#eps_dist = eps_uniform[constrained_2]
eps_dist = eps_lognorm[constrained_2]

Cml_dist = Cml_norm[constrained_2]

Cdeep_dist = Cdeep_norm[constrained_2]


# convert lambda and gamma to ECS and TCR

TCR_c = (3.71*scale_norm[constrained_2,0])/(lam_dist + eps_dist*gam_dist)
TCR = (3.71*scale_norm[:,0])/(lam + eps_lognorm*gam_norm)             

ECS_c = ECS_uniform[constrained_2]

aero_c = F[253,8,constrained_2]

# In[181]:
np.save('../FAIR-master/remote_runs_NOx/ECS_prior_remote.npy', ECS_uniform)
np.save('../FAIR-master/remote_runs_NOx/ECS_post_remote.npy', ECS_c)

np.save('../FAIR-master/remote_runs_NOx/TCR_prior_remote.npy', TCR)
np.save('../FAIR-master/remote_runs_NOx/TCR_post_remote.npy', TCR_c)

np.save('../FAIR-master/remote_runs_NOx/scaling_prior_remote.npy', scale_norm)
np.save('../FAIR-master/remote_runs_NOx/scaling_post_remote.npy', scale_dist)

np.save('../FAIR-master/remote_runs_NOx/lambda_prior_remote.npy', lam)
np.save('../FAIR-master/remote_runs_NOx/lamda_post_remote.npy', lam_dist)

np.save('../FAIR-master/remote_runs_NOx/gamma_prior_remote.npy', gam_norm)
np.save('../FAIR-master/remote_runs_NOx/gamma_post_remote.npy', gam_dist)

np.save('../FAIR-master/remote_runs_NOx/epsilon_prior_remote.npy', eps_lognorm)
np.save('../FAIR-master/remote_runs_NOx/epsilon_post_remote.npy', eps_dist)

np.save('../FAIR-master/remote_runs_NOx/Cml_prior_remote.npy', Cml_norm)
np.save('../FAIR-master/remote_runs_NOx/Cml_post_remote.npy', Cml_dist)

np.save('../FAIR-master/remote_runs_NOx/Cdeep_prior_remote.npy', Cdeep_norm)
np.save('../FAIR-master/remote_runs_NOx/Cdeep_post_remote.npy', Cdeep_dist)

np.save('../FAIR-master/remote_runs_NOx/Temperature_prior_remote.npy', T)
np.save('../FAIR-master/remote_runs_NOx/Temperature_post_remote.npy', T[:,constrained_2])

np.save('../FAIR-master/remote_runs_NOx/Forcings_prior_remote.npy', F)
np.save('../FAIR-master/remote_runs_NOx/Forcings_post_remote.npy', F[:,:,constrained_2])

np.save('../FAIR-master/remote_runs_NOx/Concentrations_post_remote.npy', C[:,:,constrained_2])
np.save('../FAIR-master/remote_runs_NOx/r0_post.npy', r0[constrained_2])
np.save('../FAIR-master/remote_runs_NOx/rc_post.npy', rc[constrained_2])
np.save('../FAIR-master/remote_runs_NOx/rt_post.npy', rt[constrained_2])

# In[182]:


df_array = np.ndarray((16,3))
df_array[0,0] = np.percentile(ECS_uniform, 5)
df_array[0,1] = np.percentile(ECS_uniform, 50)
df_array[0,2] = np.percentile(ECS_uniform, 95)

df_array[1,0] = np.percentile(ECS_c, 5)
df_array[1,1] = np.percentile(ECS_c, 50)
df_array[1,2] = np.percentile(ECS_c, 95)

df_array[2,0] = np.percentile(TCR, 5)
df_array[2,1] = np.percentile(TCR, 50)
df_array[2,2] = np.percentile(TCR, 95)

df_array[3,0] = np.percentile(TCR_c, 5)
df_array[3,1] = np.percentile(TCR_c, 50)
df_array[3,2] = np.percentile(TCR_c, 95)

df_array[4,0] = np.percentile(lam, 5)
df_array[4,1] = np.percentile(lam, 50)
df_array[4,2] = np.percentile(lam, 95)

df_array[5,0] = np.percentile(lam_dist, 5)
df_array[5,1] = np.percentile(lam_dist, 50)
df_array[5,2] = np.percentile(lam_dist, 95)

df_array[6,0] = np.percentile(eps_lognorm, 5)
df_array[6,1] = np.percentile(eps_lognorm, 50)
df_array[6,2] = np.percentile(eps_lognorm, 95)

df_array[7,0] = np.percentile(eps_dist, 5)
df_array[7,1] = np.percentile(eps_dist, 50)
df_array[7,2] = np.percentile(eps_dist, 95)

df_array[8,0] = np.percentile(gam_norm, 5)
df_array[8,1] = np.percentile(gam_norm, 50)
df_array[8,2] = np.percentile(gam_norm, 95)

df_array[9,0] = np.percentile(gam_dist, 5)
df_array[9,1] = np.percentile(gam_dist, 50)
df_array[9,2] = np.percentile(gam_dist, 95)

df_array[10,0] = np.percentile(aerosols_norm, 5)
df_array[10,1] = np.percentile(aerosols_norm, 50)
df_array[10,2] = np.percentile(aerosols_norm, 95)

df_array[11,0] = np.percentile(aero_c, 5)
df_array[11,1] = np.percentile(aero_c, 50)
df_array[11,2] = np.percentile(aero_c, 95)

df_array[12,0] = np.percentile(Cml_norm, 5)
df_array[12,1] = np.percentile(Cml_norm, 50)
df_array[12,2] = np.percentile(Cml_norm, 95)

df_array[13,0] = np.percentile(Cml_norm[constrained_2], 5)
df_array[13,1] = np.percentile(Cml_norm[constrained_2], 50)
df_array[13,2] = np.percentile(Cml_norm[constrained_2], 95)

df_array[14,0] = np.percentile(Cdeep_norm, 5)
df_array[14,1] = np.percentile(Cdeep_norm, 50)
df_array[14,2] = np.percentile(Cdeep_norm, 95)

df_array[15,0] = np.percentile(Cdeep_norm[constrained_2], 5)
df_array[15,1] = np.percentile(Cdeep_norm[constrained_2], 50)
df_array[15,2] = np.percentile(Cdeep_norm[constrained_2], 95)

posteriors = pd.DataFrame(df_array, index = ['ECS prior','ECS post','TCR prior','TCR post',
                                             'lambda prior','lambda post',
                                             'epsilon prior', 'epsilon post',
                                             'gamma prior', 'gamma post',
                                             'aerosol forcing prior', 'aerosol forcing post',
                                            'Cml prior','Cml post',
                                             'Cdeep prior', 'Cdeep post'],
                          columns=['5th', '50th', '95th'])

posteriors.round(decimals=3)

posteriors.to_csv('../FAIR-master/remote_runs_NOx/Posteriors_remote.csv')


