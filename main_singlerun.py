# -*- coding: utf-8 -*-
"""
Created on Jul 9 2020

#@author: jviv
#"""

#%%
import os
#import pandas as pd
import numpy as np
from functions.run_modules  import run_calibration, run_optimization, run_calicop, run_pvlib
from functions.io           import read_logs, read_fcst, read_settings, read_setpoint, update_future, estimateCurrentState, write_output, data_unpickle, data_pickle
import matplotlib.pyplot    as plt
from matplotlib.dates       import DateFormatter
#from matplotlib import rc
from pvlib import pvsystem
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
import timeit
from datetime import datetime
from modules.calibrationClasses import calihvac


#%% Read/import settings
loc_settings, cali_settings, opt_settings, building_properties, hvac_properties, pv_properties = read_settings()

hvac_data  = data_unpickle(os.getcwd() + '/tmp/hvac_data') 
fdata_old  = data_unpickle(os.getcwd() + '/tmp/output/fdata')

#%%
season = 'h' 
date  = datetime(2021,10,27,7,0,0)  
#date  = datetime.today()

hvac_data['season'] = season 
hvac_data['theta_hs_min'] = 20
hvac_data['theta_hs_max'] = 50
#hvac_properties = hvac_data

fdata_old['Tset'] = 22
fdata_old['phi_hc_opt'] = 0
fdata_old['phi_hp_opt'] = 0

#%% PV characteristics

pv_data, pv_obj = run_pvlib(pv_properties, loc_settings)


#%% Create datasets

# commentare le 3 righe  e scommentare le successive se eseguito test in laboratorio 
logs_folder = 'C:/Codici/Python/SEM/data/logs/winter'                        
fcst_folder = 'C:/Codici/Python/SEM/data/weather/'       
user_folder = 'C:/Codici/Python/SEM/data/user/' 
     
#logs_folder = 'D:\ML\SW\ScambioDati'  
#fcst_folder = 'D:\PrevisioniMeteo\DatiMeteo'
#user_folder = 'D:\ML\SW\ScambioDati'

ndays = 20

logs           = read_logs(date, logs_folder, ndays, '15min', loc_settings)
setpoint_table = read_setpoint(user_folder)
forecast_table = read_fcst(date, fcst_folder, loc_settings)

#%% HVAC system calibration

logs_5min = read_logs(datetime.today(), logs_folder, 2, '5min', loc_settings) #last 2 weeks
ch = calihvac(logs_5min, hvac_data)

try: 
    ch.update_fc(hvac_data, fdata_old)
    hvac_data['params_hc'] = [ch.calpars.coeff_fc[0],ch.calpars.coeff_fc[1]]
    print('Fancoil power correlation updated with R2 score  {:.3f}'.format(ch.info.score_fc))
except:
    print('Fancoil power correlation not updated')     


#%% Get time
start = timeit.default_timer()

#%% Calibration of parameters

try:
    optpars = data_unpickle(os.getcwd() + '/tmp/optpars')
    cs = run_calibration(loc_settings, 
                         cali_settings,
                         building_properties,
                         logs,
                         optpars)
except:
    cs = run_calibration(loc_settings, 
                         cali_settings,
                         building_properties,
                         logs)

# Save optimal parameters
optpars = cs.info.optpars
data_pickle(optpars, os.getcwd() + '/tmp/optpars')

build_data = vars(cs.auxdata)
hist       = cs.history
build_data, hist, optpars = vars(cs.auxdata), cs.history, cs.info.optpars

## HVAC system (run_calicop) inside update future


#%% Compute simulation time
stop = timeit.default_timer()
caliTime = stop - start
print('Time for calibration: {:.2f}'.format(caliTime), ' seconds')

#%% Update setpoint schedule and forecasts

cstate = estimateCurrentState(logs, hvac_data, opt_settings, 
                              loc_settings, cali_settings, building_properties)
fdata = update_future(date, setpoint_table, forecast_table, pv_obj, pv_data, opt_settings, hvac_data, logs)


#%% Optimization

startopt = timeit.default_timer()
    
fdata, opt = run_optimization(optpars, opt_settings, build_data, hvac_data, 
                              cstate, fdata, fdata_old, loc_settings) 

output = write_output(fdata, season)

stopopt = timeit.default_timer()
optTime = stopopt - startopt
print('Time for optimization: {:.2f}'.format(optTime), ' seconds')


#%% Plot calibration results

#trainvals = int(cali_settings['n_days']*24*cali_settings['training_part']*3600/cali_settings['tau'])
#testvals = int(cali_settings['n_days']*24*(1-cali_settings['training_part'])*3600/cali_settings['tau'])
#rmse_train_nom = cs.rmse(hist['Ti_meas_avg'].values[:trainvals], hist['Ti_calc_nom'].values[:trainvals])
#rmse_train_opt = cs.rmse(hist['Ti_meas_avg'].values[:trainvals], hist['Ti_calc_opt'].values[:trainvals])
#rmse_test_nom  = cs.rmse(hist['Ti_meas_avg'].values[-testvals:], hist['Ti_calc_nom'].values[-testvals:])
#rmse_test_opt  = cs.rmse(hist['Ti_meas_avg'].values[-testvals:], hist['Ti_calc_opt'].values[-testvals:])
formatter = DateFormatter('%d/%m %H:%M')
fig = plt.figure(figsize=(13,9))
ax = fig.add_subplot(111)
plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
#dates = hist.index[:trainvals] 
#dates = hist.index[-testvals:] 
dates = hist.index[:]
ax.plot(dates, hist['Ti_calc_nom'].loc[dates],         label= 'Nominal')
ax.plot(dates, hist['Ti_calc_opt'].loc[dates],         label= 'Calibrated')   
ax.plot(dates, hist['Ti_meas_avg'].loc[dates], 'k--',  label= 'Measured avg.')
ax.plot(dates, hist['Text'].loc[dates]       , 'g',    label= 'Outdoor temp.')
ax.set(ylabel='Indoor air temperature (°C)')
ax.grid(which='major')
#ax.set_ylim(16,26)
legend = ax.legend()
plt.rcParams.update({'font.size': 18})
#folder, name = logs_folder.split('/')

#ax.text(dates[15],24, '(b)', fontsize=22, ha='center')
plt.xticks(rotation='30')
plt.show()
#fig.savefig('fig_cal_winter_' + str(date)[:10] +'.png')
#print('RMSE over training set with nominal parameters = % 4.2f K' %rmse_train_nom) 
#print('RMSE over training set with calibrated parameters = % 4.2f K' %rmse_train_opt)
##print('RMSE over testing set with nominal parameters = % 4.2f K' %rmse_test_nom) 
#print('RMSE over testing set with calibrated parameters = % 4.2f K' %rmse_test_opt)
##physical_sense_vote = 5 - abs(np.log10(np.divide(cs.info.optpars, cs.calset.x_nom))).mean()  # 5 = great, 0 = bad
##print('Vote to physical sense of the solution = % 4.1f/5' %physical_sense_vote) 


#%% Plot optimization results

fig,ax1 = plt.subplots(1,figsize=(12,8))

ax1.plot(fdata.index,fdata.theta_i_opt, "b-",lw=1.5, label=r"$\theta^{i,opt}$")
ax1.plot(fdata.index,fdata.Te,          "b--",lw=1.5, label=r"$\theta^{e}$")
ax1.plot(fdata.index,fdata.theta_hs_opt,"b:",lw=1.5, label=r"$\theta^{hs}$")
plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
plt.rcParams.update({'font.size': 18})
plt.xticks(rotation='30')
ax1.fill_between(fdata.index,fdata.theta_max,np.ones(len(fdata.index))*50,color='red', alpha='0.25')
ax1.fill_between(fdata.index,np.ones(len(fdata.index))*0,fdata.theta_min,color='red', alpha='0.25')
ax1.grid(which='major')
#ax1.set_ylim(4,44)
ax1.set_xlabel(r"Time", size=18)
ax1.set_ylabel(r"Temperature ($^o$C)", size=18)
ax1.set(title='Optimization results')
ax1.legend(fancybox=True, framealpha=0.5,loc = "upper left",prop={'size': 16})
ax2 = ax1.twinx() 
ax2.step(fdata.index, fdata.w_hp, 'k',  label= r"$w^{hp}$")
ax2.plot(fdata.index, fdata.W_pv, 'y',  label= r"$w^{pv}$")
ax2.step(fdata.index, fdata.phi_hc_opt, 'g',  label= r"$\phi^{hc}$")
ax2.plot(fdata.index, fdata.phi_sol, 'k:',  label= r"$\phi^{sol}$")
ax2.set_ylabel(r"Power [W]")
#ax2.set_ylim(0,1600)
#ax2.grid(which='minor')
ax2.legend(fancybox=True, framealpha=0.5,loc = "upper right",prop={'size': 16})

plt.show()
#fig.savefig('output/fig_opt_summer.png')

#%%

#logs['n_aspiratori'] = -(logs['relay_QA1']+logs['relay_QB1']+logs['relay_QC1']+logs['relay_QC2'])
#logs['Ti_meas_avg']  =  (logs['Ti_A']+logs['Ti_B']+logs['Ti_C']+logs['Ti_D'])/4
#fig = plt.figure(figsize=(18,9),dpi=80)
##fig, [[ax11, ax12], [ax21, ax22],[ax31, ax32], [ax41, ax42]] = plt.subplots(nrows = 4, ncols = 2)
#ax = fig.add_subplot(111)
#plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
##dates = hist.index[:trainvals] 
#dates = hist.index[-testvals:] 
##dates = hist.index[:]
#logs['Wel_od'] = -logs['QSS_Gen'] - logs['Wel_hp']
#ax.plot(logs.index, logs['q_fc'],         label= 'q_fancoils [kW]')
#ax.plot(logs.index, logs['Wel_od'],       label= 'W_el_tot [kW]')
#ax.plot(logs.index, logs['Te'],           label= 'T_ext [°C]')
#ax.plot(logs.index, logs['n_aspiratori'], label= 'n_aspiratori [-]')
#ax.plot(logs.index, logs['Ti_meas_avg'],  label= 'T_int [°C]')
#ax2 = ax.twinx()
#ax2.plot(logs.index, logs['ghi_1'], 'y', label= 'GHI [W/m2]')
#ax2.set_ylim(0,1600)
#ax.grid(which='major')
#ax.legend(fancybox=True, framealpha=0.5,loc = "upper right",prop={'size': 16})
#plt.xticks(rotation='30')
#fig.savefig('output/fig_cal_boundary_conditions.png')

#%%
#date_yesterday = datetime(2020,12,22,17,39,0) 
#fdata.to_csv('analisi/fdata vs logs/fdata_20210117.csv',sep=';')
#logs.loc[date_yesterday:].to_csv('analisi/fdata vs logs/logs_20201223.csv',sep=';')
#output.to_csv('output/ML_20210117.csv',sep=';')
#print('Output file successfully printed in output folder')

#logs.to_csv('analisi/logs_tot.csv',sep=';')
#



