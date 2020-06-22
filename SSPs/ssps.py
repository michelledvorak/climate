# convenience module for loading in ALL SSPs

import numpy as np
import pandas as pd
import os

ssps_emissions_filename = os.path.join(
    os.path.dirname(__file__), 'data/rcmip-emissions-annual-means-v4-0-0.csv')

class SSPs:
    def __init__(self):
        self._ssps = pd.read_csv(ssps_emissions_filename)
        
    def get_SSPs(self):
        SSPs_scenario = self._ssps['Scenario'].str.contains('ssp')
        SSPs_only = self._ssps[SSPs_scenario]

        SSPs_region = SSPs_only['Region'].str.contains('R5.2')
        SSPs_world = SSPs_only[~SSPs_region]

        sums_only = SSPs_world.loc[:,'Variable'].str.contains('MAGICC')
        SSPs = SSPs_world[~sums_only]

        del SSPs['Region']
        del SSPs['Activity_Id']
        del SSPs['Mip_Era']

        SSP_119 = SSPs.loc[:,'Scenario'].str.contains('ssp119')
        SSP_119 = SSPs[SSP_119]

        SSP_126 = SSPs.loc[:,'Scenario'].str.contains('ssp126')
        SSP_126 = SSPs[SSP_126]

        SSP_245 = SSPs.loc[:,'Scenario'].str.contains('ssp245')
        SSP_245 = SSPs[SSP_245]

        SSP_370 = SSPs.loc[:,'Scenario'].str.contains('ssp370')
        SSP_370 = SSPs[SSP_370]

        SSP_434 = SSPs.loc[:,'Scenario'].str.contains('ssp434')
        SSP_434 = SSPs[SSP_434]

        SSP_460 = SSPs.loc[:,'Scenario'].str.contains('ssp460')
        SSP_460 = SSPs[SSP_460]

        SSP_534_os = SSPs.loc[:,'Scenario'].str.contains('ssp534-over')
        SSP_534_os = SSPs[SSP_534_os]

        SSP_585 = SSPs.loc[:,'Scenario'].str.contains('ssp585')
        SSP_585 = SSPs[SSP_585]

        model_list = [SSP_119, SSP_126, SSP_245, SSP_370, SSP_434, SSP_460, SSP_534_os, SSP_585]
        empty_list = []
        empty_list_2 = []

        for model in model_list:
            model = model.groupby('Variable').mean()
            model = model.T
            model.loc[:,'Year'] = model.index
            model.loc[:,'Year'] = model.loc[:,'Year'].astype(int)
            #     model.loc[:,'Year2'] = model.loc[:,'Year']
            model.drop(model[model.Year < 1765].index, inplace=True)
            #     model.drop(model[model.Year > 2160].index, inplace=True)
            #     model.loc[:,'Year'] = pd.to_datetime(model['Year'], format='%Y')
            model.index = model['Year']
            #     model = model.resample('Y').mean()
            model = model.interpolate(method ='linear', limit_direction ='forward')
            empty_list.append(model)

            for col_name in model.columns:
                empty_list_2.append(col_name)

        SSP_119 = empty_list[0]
        SSP_126 = empty_list[1]
        SSP_245 = empty_list[2]
        SSP_370 = empty_list[3]
        SSP_434 = empty_list[4]
        SSP_460 = empty_list[5]
        SSP_534_os = empty_list[6]
        SSP_585 = empty_list[7]
        
        nt = len(SSP_119)
        
        emissions_119 = np.zeros((nt,40))
        emissions_126 = np.zeros((nt,40))
        emissions_245 = np.zeros((nt,40))
        emissions_370 = np.zeros((nt,40))
        emissions_434 = np.zeros((nt,40))
        emissions_460 = np.zeros((nt,40))
        emissions_534_os = np.zeros((nt,40))
        emissions_585 = np.zeros((nt,40))

        array_list = [emissions_119, 
              emissions_126, 
              emissions_245, 
              emissions_370, 
              emissions_434, 
              emissions_460, 
              emissions_534_os, 
              emissions_585]

        SSP_list = [SSP_119, 
            SSP_126, 
            SSP_245, 
            SSP_370, 
            SSP_434, 
            SSP_460, 
            SSP_534_os, 
            SSP_585]

        empty_list = []

        for array, SSP in zip(array_list, SSP_list):
            array[:,0] = SSP['Year']
            array[:,1] = (SSP['Emissions|CO2'])/1000/3.67 # MtCO2 to GtC
            array[:,3] = SSP['Emissions|CH4']
            array[:,4] = (SSP['Emissions|N2O'])/1000/1.57 #ktN2O to MtN
            array[:,5] = (SSP['Emissions|Sulfur'])/1.998 #MtSO2 to MtS
            array[:,6] = SSP['Emissions|CO']
            array[:,7] = SSP['Emissions|VOC']
            array[:,9] = SSP['Emissions|BC']
            array[:,10] = SSP['Emissions|OC']
            array[:,11] = SSP['Emissions|NH3']/1.217 #MtNH3 to MtN
            array[:,12] = SSP['Emissions|F-Gases|PFC|CF4']
            array[:,13] = SSP['Emissions|F-Gases|PFC|C2F6']
            array[:,14] = SSP['Emissions|F-Gases|PFC|C6F14']
            array[:,15] = SSP['Emissions|F-Gases|HFC|HFC23']
            array[:,16] = SSP['Emissions|F-Gases|HFC|HFC32']
            array[:,17] = SSP['Emissions|F-Gases|HFC|HFC4310mee']
            array[:,18] = SSP['Emissions|F-Gases|HFC|HFC125']
            array[:,19] = SSP['Emissions|F-Gases|HFC|HFC134a']
            array[:,20] = SSP['Emissions|F-Gases|HFC|HFC143a']
            array[:,21] = SSP['Emissions|F-Gases|HFC|HFC227ea']
            array[:,22] = SSP['Emissions|F-Gases|HFC|HFC245fa']
            array[:,23] = SSP['Emissions|F-Gases|SF6']
            array[:,24] = SSP['Emissions|Montreal Gases|CFC|CFC11']
            array[:,25] = SSP['Emissions|Montreal Gases|CFC|CFC12']
            array[:,26] = SSP['Emissions|Montreal Gases|CFC|CFC113']
            array[:,27] = SSP['Emissions|Montreal Gases|CFC|CFC114']
            array[:,28] = SSP['Emissions|Montreal Gases|CFC|CFC115']
            array[:,29] = SSP['Emissions|Montreal Gases|CCl4']
            array[:,30] = SSP['Emissions|Montreal Gases|CH3CCl3']
            array[:,31] = SSP['Emissions|Montreal Gases|HCFC22']
            array[:,32] = SSP['Emissions|Montreal Gases|HCFC141b']
            array[:,33] = SSP['Emissions|Montreal Gases|HCFC142b']
            array[:,34] = SSP['Emissions|Montreal Gases|Halon1211']
            array[:,35] = SSP['Emissions|Montreal Gases|Halon1202']
            array[:,36] = SSP['Emissions|Montreal Gases|Halon1301']
            array[:,37] = SSP['Emissions|Montreal Gases|Halon2402']
            array[:,38] = SSP['Emissions|Montreal Gases|CH3Br']
            array[:,39] = SSP['Emissions|Montreal Gases|CH3Cl']

            empty_list.append(array)

        emissions_119 = empty_list[0]
        emissions_126 = empty_list[1]
        emissions_245 = empty_list[2]
        emissions_370 = empty_list[3]
        emissions_434 = empty_list[4]
        emissions_460 = empty_list[5]
        emissions_534_os = empty_list[6]
        emissions_585 = empty_list[7]

        return emissions_119, emissions_126, emissions_245, emissions_370, emissions_434, emissions_460, emissions_534_os, emissions_585

    