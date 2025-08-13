#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

def get_sim_incidence_shape(site: str, agebin: str) -> pd.DataFrame:
    """
    Compute the normalized monthly incidence “shape” from simulation outputs.
    
    """
    sim_cases = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"ClinicalIncidence_monthly.csv"))
    # filter to age of interest
    sim_cases = sim_cases[sim_cases['agebin']==agebin]
    # get mean population and clinical cases by month, year, and Sample_ID
    sim_cases['Inc'] = sim_cases['Cases'] / sim_cases['Pop']
    sim_cases=sim_cases.merge(sim_cases.groupby(['Sample_ID','Year'])['Inc'].agg(np.nanmax).reset_index(name='max_simincd'), on=['Sample_ID',"Year"],how='left').reset_index()
    sim_cases['norm_simincd'] = sim_cases.apply(lambda row: (row['Inc'] / row['max_simincd']), axis=1)
    #print(sim_cases)
    # get mean normalized incidence by month/sample_id (across years)
    sim_cases = sim_cases.groupby(['Sample_ID', 'month'])['norm_simincd'].agg(np.nanmean).reset_index()
    
      
    return shape_df
