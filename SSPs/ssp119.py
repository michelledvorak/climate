# load in SSP 119 emissions data

import numpy as np
from fair.SSPs.ssps import SSPs

class Emissions_119:
    emissions_init = SSPs()
    emissions, _, _ , _, _, _, _, _ = emissions_init.get_SSPs()
    year      = emissions[:,0]
    co2_fossil= emissions[:,1]
    co2_land  = emissions[:,2]
    co2       = np.sum(emissions[:,1:3],axis=1)
    ch4       = emissions[:,3]
    n2o       = emissions[:,4]
    sox       = emissions[:,5]
    co        = emissions[:,6]
    nmvoc     = emissions[:,7]
    nox       = emissions[:,8]
    bc        = emissions[:,9]
    oc        = emissions[:,10]
    nh3       = emissions[:,11]
    cf4       = emissions[:,12]
    c2f6      = emissions[:,13]
    c6f14     = emissions[:,14]
    hfc23     = emissions[:,15]
    hfc32     = emissions[:,16]
    hfc43_10  = emissions[:,17]
    hfc125    = emissions[:,18]
    hfc134a   = emissions[:,19]
    hfc143a   = emissions[:,20]
    hfc227ea  = emissions[:,21]
    hfc245fa  = emissions[:,22]
    sf6       = emissions[:,23]
    cfc11     = emissions[:,24]
    cfc12     = emissions[:,25]
    cfc113    = emissions[:,26]
    cfc114    = emissions[:,27]
    cfc115    = emissions[:,28]
    carb_tet  = emissions[:,29]
    mcf       = emissions[:,30]
    hcfc22    = emissions[:,31]
    hcfc141b  = emissions[:,32]
    hcfc142b  = emissions[:,33]
    halon1211 = emissions[:,34]
    halon1202 = emissions[:,35]
    halon1301 = emissions[:,36]
    halon2402 = emissions[:,37]
    ch3br     = emissions[:,38]
    ch3cl     = emissions[:,39]
