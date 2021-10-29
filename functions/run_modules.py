# -*- coding: utf-8 -*-
"""
Created on Jul 9 2020

@author: vivian
"""
#%%
#import os
#import schedule
#import pvlib
import numpy  as np
import pandas as pd
#from datetime                     import datetime
from modules.solarClasses         import solarProcessor
from modules.calibrationClasses   import calipso, calicop
from modules.optimizationClasses  import singleZoneOptimizer

import pvlib
from pvlib import pvsystem
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain

#%% Calibration

def run_calibration(loc_settings,
                    cali_settings,
                    building_properties,
                    logs,
                    *arg):
    
    # Date range of logged data for calibration
    stop_date  = logs.index[-1]    
    cali_steps = int(cali_settings['n_days']*24*(3600/cali_settings['tau']))
    if cali_steps > len(logs.index):
        cali_steps = len(logs.index)
    start_date = logs.index[-cali_steps]
    
    date_rng = pd.date_range(start = start_date, 
                             end  = stop_date, 
                             freq  ='15min', 
                             tz = loc_settings['tz'])
    bdf = logs.loc[start_date:stop_date]
    
    # Correct bdf with missing values
    missing_dates = date_rng.difference(bdf.index) # find missing dates
    nanarray = np.empty((len(missing_dates),bdf.shape[1])) # initialize empty array with correct size
    nanarray[:] = np.nan  # fills array with nans
    bdf = pd.concat([bdf, pd.DataFrame(nanarray, columns=bdf.columns,  index = missing_dates)]) # appends rows with nans on missing dates
    bdf = bdf.sort_index()   # sorts dataframe by date
    bdf = bdf.fillna(bdf.mean()) # replaces nan values with mean of entire column    
                                                    
    # Build input dataframes
    bdf_ext = bdf[['Te','ghi_1','riserva','dhi']]
    bdf_ext.columns = ['Te','ghi','dni','dhi']

    # Dataframe of logged data                                                      
    bdf_air = bdf[['Ti_A','Ti_B','Ti_C','Ti_D','relay_QA1','relay_QB1','relay_QC1','relay_QD1']]
    
    bdf_power = bdf[['Wel_od','q_fc']]
    bdf_power['Wel_od'] = bdf[['Wel_od']]*1000 # kW --> W    
    bdf_power['q_fc']   = bdf[['q_fc']]*1000   # kW --> W (negativo d'estate, positivo d'inverno)
    bdf_power.columns = ['W_el','q_fc']

    # solar process
    sp = solarProcessor(loc_settings, date_rng, bdf_ext)
    bdf_irrad = sp.vardata.surface_irradiance   # W/m2
    bdf_irrad = bdf_irrad.loc[start_date:stop_date]
        
    # Initialize calibration object with last data logs
    cs = calipso(building_properties, bdf_ext, bdf_air, bdf_power, bdf_irrad, cali_settings)   # calibration object
    test_rng  = cs.history.index[cs.calset.trainSteps+1:]
    # Test building model with parameters from last calibration on new data    
    try:
        lastpars = arg[0]  
        Ti_calc_opt_lastpars, _ = cs.runsim(lastpars, cs.history, cs.calset.t_m0, cali_settings['tau'])
        rmse_test_lastpars      = cs.rmse(Ti_calc_opt_lastpars[:,0], cs.history['Ti_meas_avg'].loc[test_rng].values)
        # Calculate new parameters (update calibration)    
        cs.update()
        # Use new parameters only in case of better performance on new dataset
        if rmse_test_lastpars < cs.info.rmse_test:
            cs.info.optpars   = lastpars
            cs.info.rmse_test = rmse_test_lastpars
    except:
        cs.update()



    # -------------------------------------------------------------------------- 
        # SPOSTA TUTTA QUESTA PARTE DENTRO UNA FUNCTION DENTRO CALIPSO
    # --------------------------------------------------------------------------
        

#    cs.history['Ti_calc_nom'], _                         = cs.runsim(cs.calset.x_nom, cs.history, cs.calset.t_0, cali_settings['tau'])
#    cs.history['Ti_calc_opt'], cs.history['Tm_calc_opt'] = cs.runsim(cs.info.optpars, cs.history, cs.calset.t_0, cali_settings['tau'])   
#    cs.history['phi_sol'] = cs.info.optpars[7]*(cs.auxdata.k0_s*cs.history['I_tot_gla'] + cs.auxdata.k1_s/cs.auxdata.A_walls*cs.history['I_tot_opa'] - 3*cs.auxdata.k1_s) 
#    cs.history['phi_int'] = cs.info.optpars[8]
    
    T_calc_nom = cs.runsim(cs.calset.x_nom, cs.history, cs.calset.t_0, cali_settings['tau'])
    T_calc_opt  = cs.runsim(cs.info.optpars, cs.history, cs.calset.t_0, cali_settings['tau'])   
    
    cs.history['Ti_calc_nom'] = T_calc_nom[:,0]          
    cs.history['Ti_calc_opt'] = T_calc_opt[:,0]    
    
#    df_train.columns = colnames
#    df_train['Ti_gb_nom'] = T_calc_nom[0:cs.calset.trainSteps,0]
#    df_train['Ti_gb_opt'] = T_calc_opt[0:cs.calset.trainSteps,0]
#    
#    df_test = df_test.iloc[1:,:]
#    df_test.columns = colnames
#    df_test['Ti_gb_nom']  = T_calc_nom[cs.calset.trainSteps:,0]
#    df_test['Ti_gb_opt']  = T_calc_opt[cs.calset.trainSteps:,0] 
#    
#        # KPIs
#    rmse_nom_train = np.sqrt(mean_squared_error(df_train['Ti_gb_nom'], df_train['Ti_meas_avg']))
#    rmse_nom_test  = np.sqrt(mean_squared_error(df_test['Ti_gb_nom'], df_test['Ti_meas_avg']))
#    rmse_train   = np.sqrt(mean_squared_error(df_train['Ti_gb_opt'], df_train['Ti_meas_avg'])) #cs.info.rmse_train
#    rmse_test    = np.sqrt(mean_squared_error(df_test['Ti_gb_opt'], df_test['Ti_meas_avg']))   #cs.info.rmse_test
#    r2_nom_train = r2_score(df_train['Ti_gb_nom'], df_train['Ti_meas_avg']) 
#    r2_nom_test  = r2_score(df_test['Ti_gb_nom'], df_test['Ti_meas_avg'])
#    r2_train     = r2_score(df_train['Ti_gb_opt'], df_train['Ti_meas_avg']) #cs.info.r2_train
#    r2_test      = r2_score(df_test['Ti_gb_opt'], df_test['Ti_meas_avg'])   #cs.info.r2_test
#    ratio        = np.divide(cs.info.optpars,cs.calset.x_nom)
#    kpis = [rmse_nom_train,rmse_nom_test,rmse_train,rmse_test,
#            r2_nom_train,  r2_nom_test,  r2_train  ,r2_test]

    return cs


#%% Subroutine that runs MILP optimization for heat pump scheduling

def run_optimization(optimal_parameters,
                     opt_settings,
                     build_data, 
                     hvac_data, 
                     cstate,
                     fdata,
                     fdata_old,
                     loc_settings):
    

    # Read calibrated parameters and settings and initialise optimisation object 
    opt = singleZoneOptimizer(optimal_parameters, opt_settings, build_data, hvac_data)   
    
    # Thermal comfort
    delta_theta = opt_settings['delta_theta']
    fdata['theta_min'] = fdata['Tset'] - delta_theta
    fdata['theta_max'] = fdata['Tset'] + delta_theta   
    
    # Put dhi and dni to 0 (same boundary condition of calibration)
    fdata['dhi'] = 0
    fdata['dni'] = 0

#    # Process solar radiation forecasts (irradiance on vertical surfaces)
    date_rng = fdata.index
    sp = solarProcessor(loc_settings, date_rng, fdata)
    bdf_irrad = sp.vardata.surface_irradiance   # W/m2
    fdata['I_tot_gla'] = np.sum(np.multiply(bdf_irrad, build_data['glazed_area']),axis=1).values       
    fdata['I_tot_opa'] = np.sum(np.multiply(bdf_irrad, build_data['opaque_area']),axis=1).values
    
    # Store last solution in fdata
    mask_last_solution = (fdata_old.index >= fdata.index[0]) & (fdata_old.index <= fdata.index[-1])
    fdata['phi_hc_old'] = fdata_old['phi_hc_opt'].loc[mask_last_solution]
    fdata['phi_hp_old'] = fdata_old['phi_hp_opt'].loc[mask_last_solution]
    fdata = fdata.fillna(0)
    
    # Run optimization with updated forecasts and boundary conditions
    opt.update(fdata, cstate, hvac_data['season'])
    
    # Write results into dataframe
    fdata['phi_hp_opt']   = opt.tseries.phi_hp
    fdata['phi_hc_opt']   = opt.tseries.phi_hc
    fdata['w_hp']         = opt.tseries.w_hp
    fdata['theta_i_opt']  = opt.tseries.theta_i
    fdata['theta_hs_opt'] = opt.tseries.theta_hs
    fdata['u_hp_opt']     = opt.tseries.u_hp
    fdata['x_su_opt']     = opt.tseries.x_su
    fdata['u_hc_opt']     = opt.tseries.u_hc
    fdata['phi_sol']      = opt.tseries.phi_sol
#    fdata['acr']          = opt.data.H_ve/opt.data.h_ve
    
    return fdata, opt


#%% Run calibration of heat pump COP correlation

def run_calicop(logs, hvac_data, fdata):
    # Initialize calibration process
    cc = calicop(logs, hvac_data)
    # Season/Operating mode
    season = hvac_data['season']
    # Correlations update
#    if (cc.info.len_hist100 >= hvac_data['datasize_qmax']) & (cc.info.len_hist3 >= hvac_data['datasize_cop']):   
    try:          
        # Update regression coefficients for HP heating/cooling capacity
        cc.update_qmax()
        # Update regression coefficients for part-load modulation
#        cc.update_qmod()
        # Update regression coefficients for COP correlation
        cc.update_cop()  
        # Store datasize
        hvac_data['datasize_qmax'] = cc.info.len_hist100
        hvac_data['datasize_cop'] = cc.info.len_hist3
        # Save coefficients
        if season == 'h':
            hvac_data['params_hp_heat_qmax'] = cc.calpars.coeff_qmax  # >0
#            hvac_data['params_hp_heat_qmod'] = cc.calpars.coeff_q     # >0
            hvac_data['params_hp_heat_fcop'] = cc.calpars.coeff_w     # cop>0
        else:
            hvac_data['params_hp_cool_qmax'] = cc.calpars.coeff_qmax # <0
#            hvac_data['params_hp_cool_qmod'] = cc.calpars.coeff_q    # <0
            hvac_data['params_hp_cool_fcop'] = cc.calpars.coeff_w    # cop>0 
        # Print regression metrics    
        print('R2 score for HP capacity correlation:', cc.info.score_qmax)
#        print('R2 score for HP partial load correlation:', cc.info.score_q)    
        print('R2 score for COP correlation:', cc.info.score_w)     
        # Overwrite 
        
    except:
        print('Heat pump capacity correlation not updated')
        print('COP and partial load correlations not updated')
    return cc, hvac_data
    


#%% Create PV object and data 

def run_pvlib(pv_properties, loc_settings):   

    sandia_modules = pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvsystem.retrieve_sam('cecinverter')
    
    module_name     = pv_properties['module_type']
    inverter_name   = pv_properties['inverter_type']
    number_modules  = pv_properties['number_modules']
    surface_tilt    = pv_properties['surface_tilt']
    surface_azimuth = pv_properties['surface_azimuth']
    
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    pv_data = {'module_params'  : sandia_modules[module_name],
               'inverter_params': sapm_inverters[inverter_name],
               'number_modules' : number_modules,
               'surface_tilt'   : surface_tilt,
               'surface_azimuth': surface_azimuth} # pvlib uses 0=North, 90=East, 180=South, 270=West convention
    
#    pv_sys = PVSystem(surface_tilt        = pv_data['surface_tilt'], 
#                      surface_azimuth     = pv_data['surface_azimuth'],
#                      module_parameters   = pv_data['module_params'], 
#                      inverter_parameters = pv_data['inverter_params'],
#                      albedo = 0.2) 
    
    mount = FixedMount(surface_tilt    = pv_data['surface_tilt'], 
                       surface_azimuth = pv_data['surface_azimuth'])
    array = Array(mount=mount,
                  module_parameters=pv_data['module_params'],
                  temperature_model_parameters=temperature_model_parameters)
    pv_sys = PVSystem(arrays=[array], 
                      inverter_parameters=pv_data['inverter_params'])
    
    location = Location(loc_settings['lat'], 
                        loc_settings['lon'], 
                        tz=loc_settings['tz'], 
                        altitude=loc_settings['alt'], 
                        name=loc_settings['city'])
    
    pv_obj = ModelChain(pv_sys, location) # creates a PV model object (including solar processing)
   
    return pv_data, pv_obj



   




























