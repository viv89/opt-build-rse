# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:10:03 2021

@author: Laboratorio
"""


# for debugging only
cstate = estimateCurrentState(logs, hvac_data, opt_settings, 
                              loc_settings, cali_settings, building_properties)
fdata = update_future(date, setpoint_table, forecast_table, pv_obj, pv_data, opt_settings, hvac_data, logs)

fdata, opt = run_optimization(optpars, opt_settings, build_data, hvac_data, 
                              cstate, fdata, fdata_old, loc_settings) 