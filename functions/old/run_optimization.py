# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:11:09 2020

@author: vivijac14771
"""

#%%

import numpy  as np
import pandas as pd
from datetime import datetime

from optimizationClasses2 import singleZoneOptimizer


#%%

def run_optimization(optimal_parameters,
                     opt_settings,
                     forecast,
                     setpoint,
                     loc_settings):
    
    n_hours = opt_settings['nhours_horizon']
    
    # Initialize dataframe with optimal scheduling
    optvals = pd.DataFrame() 
    optvals.columns = list('q_hp','T_hs','q_hc')
    optvals.date = pd.date_range(start   = datetime.today(), 
                                 periods = 4*n_hours,
                                 freq    = '15min',
                                 tz      = loc_settings['tz'])
    optvals.set_index(optvals['date'])

    # Read calibrated parameters
    C_m     = optimal_parameters[0]
    H_tr_em = optimal_parameters[1]
    H_tr_is = optimal_parameters[2]
    H_tr_ms = optimal_parameters[3]
    H_tr_w  = optimal_parameters[4]   
    H_ve    = optimal_parameters[5]
    k_conv  = optimal_parameters[6]  
    
#    # Run optimization
#    opt = singleZoneOptimizer(optimal_parameters, opt_settings)   # calibration object
#    
#    # Thermal comfort
#    theta_min, theta_max =
#    
#    # 
#    
#    # Initial conditions
#    theta_m0 = 
#    
#    # Weather conditions and PV production (forecasts)
#    theta_b = 
#    solar_input =
#    w_pv = 
#    
#    optvals = opt.update(w_pv, phi_int, solar_input, theta_b, theta_m0, theta_min, theta_max, 'h')
    
    return optvals