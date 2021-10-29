# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:15:24 2020

@author: vivijac14771
"""

# */ -- import libraries --------------------------------------------------/* #

import pandas as pd
import numpy as np
import glob   as gl
from datetime import datetime
import pytz
import os
import pickle
import copy
from functions.run_modules import run_calicop
from modules.solarClasses         import solarProcessor
from modules.calibrationClasses   import calipso


def read_logs(date, subfolder, ndays, sampletime, loc_settings):
    colnames = ('Te','RHe', 'ghi_1','ghi_2','dhi','vw','riserva',
                'Ti_A','Tmr_A','Ti_B','Tmr_B','Ti_C','Tmr_C','Ti_D','Tmr_D',
                'RHi_B','Tfloor','Twall_W','Twall_S','riserva1',
                'Thp_in','Thp_out','m_hp',
                'Tfc_su','Tfc_ret','m_fc',
                'Tfc_su2','Tfc_ret2',
                'Tst_high','Tst_low','riserva2',
                'Tdhw_cold','Tdhw_hot','Tdhw_mix','m_dhw',
                'Thwt','m_net','Trec_in','Trec_out',
                'ctrl_pompa_V',
                'riserva4','riserva5','riserva6','riserva7','riserva-last',
                'relay_QA1','relay_QA2','relay_QB1','relay_QB2',
                'relay_QC1','relay_QC2','relay_QD1','relay_QD2',
                'q_fc','q_fc2','q_hp','q_dhw','q_rec',
                'QA1','QA2','QA3','QA4','QA5','QA6',
                'QB1','QB2','QB3','QB4','QB5','QB6',
                'QC1','QC2','QC3','QC4','QC5','QC6',
                'QD1','QD2','QD3','QD4',
                'Wel_hp','Wel_dhw',
                'QSSE_C2','QSSE_C3','QSSE4-C12','QSSE-C13','QSS_Gen',
                'Wel_od','p_cond','p_evapor', 'port_ut', 'eev','rps',
                't_serb', 'Tiut', 'Tuut','t_a_est', 't_sc_co', 't_asp_c',
                'r100', 'r100p', 'spu', 'q_spv', 'sm','naa', 'cua',
                'sc', 'bldc', 'defrost')
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') #set datetime format of imported data
    colnameslist = list(colnames)
    filepaths = gl.glob(subfolder + '/*.txt') 
    nfiles = 3*ndays
    fpaths = sorted(filepaths)[-nfiles:]
    dfr = pd.DataFrame()
    dfr1 = dfr    
    for fname in fpaths:
        try:
            df0 = pd.read_csv(fname, header=1, index_col=0, sep = '\t', date_parser = dateparse)
            numcols = len(df0.columns)
            df0.columns = colnameslist[:numcols]
            df0 = df0.fillna(method = 'bfill')
            try:
                dfr = df0.resample(sampletime).mean()
                dfr1 = dfr1.append(dfr)[dfr.columns.tolist()]
#               dfr1.to_csv(subfolder + '/logs_resampled.csv')
            except:
                print('Cannot resample data: too small sample in {file}'.format(file = fname))                           
        except:  
            try: 
                df0 = pd.read_csv(fname, header=1, index_col=0, sep = '\t', date_parser = dateparse, skiprows = 2)
                numcols = len(df0.columns)
                df0.columns = colnameslist[:numcols]
                df0 = df0.fillna(method = 'bfill')
                try:
                    dfr = df0.resample(sampletime).mean()
                    dfr1 = dfr1.append(dfr)[dfr.columns.tolist()]
#                   dfr1.to_csv(subfolder + '/logs_resampled.csv')
                except:
                    print('Cannot resample data: too small sample in {file}'.format(file = fname)) 
            except: 
                print('Found empty file: {file}'.format(file = fname))
    dfr1 = dfr1[~dfr1.index.duplicated(keep='first')]  # eliminates doubles            
    dfr1 = dfr1[dfr1.index > dfr1.index[-1] - pd.to_timedelta(ndays, unit='D')]       
    dfr1 = dfr1.tz_localize(loc_settings['tz'], ambiguous=True, nonexistent='shift_backward')   # assignes timezone to index   
    rome = pytz.timezone(loc_settings['tz'])
    local_date = rome.localize(date, is_dst = True)
    mask = (dfr1.index <= local_date)  
    dfr1 = dfr1.loc[mask]  
    print('Logs read at '+str(date)[:19])
    return dfr1                                           


def read_fcst(dateTime, subfolder, loc_settings):
#    fileID00 = '0000'
    colnames = ('data_run','fcnum','Te', 'RHe', 'Patm','ghi','dhi','dni','prec','vw','vdir')
    colnameslist = list(colnames)
    try:
        datestring = dateTime.strftime('%Y%m%d')
        filepaths = gl.glob(subfolder + '/*' + datestring +'*' + '.csv')      
        fpath = sorted(filepaths)[-1]  
    except:
        try:
            dayForecast = dateTime.day - 1
            dateTime = dateTime.replace(day = dayForecast)
            datestring = dateTime.strftime('%Y%m%d')
            filepaths = gl.glob(subfolder + '/*' + datestring +'*' + '.csv')      
            fpath = sorted(filepaths)[-1] 
            print('Alarm: Unavailable weather forecasts in last 24 hours')
        except:
            dayForecast = dateTime.day - 2
            dateTime = dateTime.replace(day = dayForecast)
            datestring = dateTime.strftime('%Y%m%d')
            filepaths = gl.glob(subfolder + '/*' + datestring +'*' + '.csv')      
            fpath = sorted(filepaths)[-1]   
            print('FatalError: Unavailable weather forecasts in last 2 days')           
    dfc = pd.DataFrame()    
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    dfc = pd.read_csv(fpath, header=0, index_col=2, sep = ';', date_parser = dateparse)  #pd.read_excel(filepath, header=0, index_col=2)
    dfc.columns = colnameslist
    dfc = dfc.fillna(method = 'bfill')
    # Upsampling: da 1 ora a 15 minuti
    dfc = dfc.resample('15min').interpolate(method='linear') #.bfill() 
    # conversione da UTC a CET
    dfc1 = dfc.tz_localize('utc').tz_convert(loc_settings['tz'])
    return dfc1


def read_setpoint(subfolder):
#    filepath = subfolder + '/setpoint_table.xls'
#    dfu = pd.read_excel(filepath, header=1, index_col=0)
    fpath = subfolder + '/DatiTermostato_inverno1.csv'
    dfu = pd.read_csv(fpath, header=0, index_col=0, skiprows = 0, sep = ';')
    # shift sunday to the end
    dom = dfu['dom']
    dfu.drop(labels=['dom'], axis=1,inplace = True)
    dfu.insert(6, 'dom', dom)
    dfu.columns = list(['LUN','MAR','MER','GIO','VEN','SAB','DOM'])
    return dfu


def update_future(dateTime, dfu, dfc, pv_obj, pv_data, settings, hvac_data, logs):     
    # Update weather forecasts
    rome = pytz.timezone('Europe/Rome')
    date = rome.localize(dateTime)
    nsteps = settings['hourly_steps']*settings['nhours_horizon']    
    dfc1 = dfc[dfc.index > date][:nsteps]
    # Update setpoint schedule    
    weekDays = list(['LUN','MAR','MER','GIO','VEN','SAB','DOM'])
    weekDayNum = date.weekday()
    weekDayToday = weekDays[weekDayNum]
    if weekDayNum == 6:
        weekDayTomorrow = weekDays[0]
    else:
        weekDayTomorrow = weekDays[weekDayNum+1]
    su = pd.concat([dfu[weekDayToday],dfu[weekDayTomorrow]])
    
    date_in = date.replace(second = 0, minute = 0, hour = 0)
    try:
        date_fin = date.replace(second = 0, minute = 45, hour = 23, day = dateTime.day + 1)
    except:
        date_fin = date.replace(second = 0, minute = 45, hour = 23, day = 1, month = dateTime.month + 1)
    date_rng = pd.date_range(start = date_in, end = date_fin, periods = 192)
    su.index = date_rng
    su1 = su[su.index > date][:nsteps]
    try:
        dfc1['Tset'] = su1.values[:nsteps]
    except:
        short_horizon = len(dfc1.index)
        dfc1['Tset'] = su1.values[:short_horizon]  
        print('Update forecast to cover whole optimization horizon!')             
    fdata = dfc1
    # Update heat pump correlation and maximum heat flow rate
    cc, hvac_data  = run_calicop(logs, hvac_data, fdata)
    data_pickle(hvac_data, os.getcwd() + '/tmp/hvac_data')
    # Run HVAC model calibration
    H = len(fdata.index)
    Xa = np.zeros([H,2])
    Xa[:,0] = fdata['Te'].values
    Xa[:,1] = 42*np.ones([H,]) #fdata['theta_hs_opt'].values + 3
    if hvac_data['season'] == 'h':
        fdata['Q_hp_max'] = cc.heatPumpCapacity(Xa, hvac_data['params_hp_heat_qmax'])*1000
        fdata['Q_hp_max'] = hvac_data['Q_hp_nom_heat']
    else:
        fdata['Q_hp_max'] = cc.heatPumpCapacity(Xa, hvac_data['params_hp_cool_qmax'])*1000
        fdata['Q_hp_max'] = -hvac_data['Q_hp_nom_cool']
    # PV data
    number_of_modules = pv_data['number_modules']
#    pv_obj.run_model(times=fdata.index, weather=fdata[['ghi', 'dni', 'dhi', 'Te', 'vw']])
#    fdata['W_pv'] = np.array(number_of_modules*pv_obj.ac)
    loc = pv_obj.location
    weather = loc.get_clearsky(fdata.index)
    pv_obj.run_model(weather)
    fdata['W_pv'] = np.array(number_of_modules*pv_obj.results.ac)
    
    return fdata


def estimateCurrentState(logs, hvac_data, opt_settings, loc_settings, cali_settings, building_properties):    
    # Set boundary conditions in previous 24 hours
    bdf_ext, bdf_air, bdf_power, bdf_irrad = boundaryConditionsPreProcess(logs, loc_settings, cali_settings, building_properties)
    # Initialize calibration to build the hist dataframe (not run)
    cs = calipso(building_properties, bdf_ext, bdf_air, bdf_power, bdf_irrad, cali_settings)   # calibration object
    # Import optimal parameters from last calibration
    optpars = data_unpickle(os.getcwd() + '/tmp/optpars')
    # Calculated Ti and Tm through building RC model based on boundary conditions of previous 24 hours
    T_calc_opt = cs.runsim(optpars, cs.history, cs.calset.t_0, cali_settings['tau'])  
    cs.history['Ti_calc_opt'] = T_calc_opt[:,0]          
    cs.history['Tm_calc_opt'] = T_calc_opt[:,2]       
#    L = opt_settings['observer_gain']
    Ti0 = (logs['Ti_A'][-1]+logs['Ti_B'][-1]+logs['Ti_C'][-1]+logs['Ti_D'][-1])/4
    Tm0 = cs.history['Tm_calc_opt'][-1] + opt_settings['observer_gain']*(cs.history['Ti_meas_avg'][-1] - cs.history['Ti_calc_opt'][-1]) #+ 0.5*(Ti0 - cs.history['Ti_meas_avg'][-1])
    lastlogs = 2*4 #ultime due ore
    if hvac_data['season'] == 'h':
        Q_hp_nom = hvac_data['Q_hp_nom_heat']/1000
        logs['plf'] = logs['q_hp'].loc[:]/Q_hp_nom 
        Thp_out_on = logs['Thp_out'].loc[abs(logs['plf']) > 0.1]
    else:
        Q_hp_nom = hvac_data['Q_hp_nom_cool']/1000
        logs['plf'] = logs['q_hp'].loc[:]/Q_hp_nom 
        Thp_out_on = logs['Thp_out'].loc[abs(logs['plf']) > 0.1]       
    Thp_out_avg = np.mean(Thp_out_on[-lastlogs:])    
    Ths0 = (logs['Tst_high'][-1] + logs['Tst_low'][-1])/2 # measured       
    u_hp0 = int(logs['q_hp'][-1]>1)
    cstate = {'Tm_0'       : Tm0,           # building thermal mass
              'Ths_0'      : Ths0,          # thermal storage tank
              'Thp_out_avg': Thp_out_avg,
              'u_hp0'      : u_hp0,
              'Ti_0'       : Ti0}        
    return cstate


def boundaryConditionsPreProcess(logs, loc_settings, cali_settings, building_properties):
    
    # Date range of logged data for calibration
    stop_date  = logs.index[-1]    
    nsteps = int(3600/cali_settings['tau'])
    start_date = logs.index[-24*nsteps]
    
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
    
    return bdf_ext, bdf_air, bdf_power, bdf_irrad


def write_output(fdata, season):
    
    H = len(fdata.index)    
#    fdata['Q_hp_max']     = np.zeros([H,1])
    fdata['freq']         = np.zeros([H,1])
    fdata['potenza_perc'] = np.zeros([H,1])
    fdata['Tset_utenza']  = np.zeros([H,1])
    fdata['m_fc_calc']    = np.zeros([H,1])
    fdata['pompa_fancoil']  = np.zeros([H,1])    
#    theta_e               = fdata['Te'].values
#    theta_hs_opt          = fdata['theta_hs_opt'].values
    phi_hp_opt   = fdata['phi_hp_opt'].values
    phi_hc_opt   = fdata['phi_hc_opt'].values       
    Q_hp_max     = fdata['Q_hp_max'].values
    fdata['a_1'] = np.zeros([H,1])
    fdata['a_2'] = np.ones([H,1])
    fdata['a_3'] = np.ones([H,1])
    fdata['a_4'] = np.zeros([H,1]) 
    freq                  = np.zeros([H,1])
    potenza_perc          = np.zeros([H,1])
    Tset_utenza           = np.zeros([H,1])
    deT_fc                = np.zeros([H,1])
    m_fc_calc             = np.zeros([H,1])
    pompa_fancoil         = np.zeros([H,1])
    q_fc                  = fdata['phi_hc_opt'].values/1000
#    T_st                  = fdata['theta_hs_opt'].values + 2
    deT_fc                = 5.0   # -12+0.44*T_st    
    m_fc_min, m_fc_max    = 500, 1810    
    if season == 'h':
        Tset_on = 60
        Tset_off = 20  
        m_fc_calc = np.divide(q_fc,(4.183*deT_fc))*3600
    elif season == 'c':
        Tset_on = 5
        Tset_off = 30
        m_fc_calc = np.divide(-q_fc,(4.183*deT_fc))*3600    
    deltaTset = abs(Tset_on-Tset_off)
    for i in range(H):
        if abs(phi_hp_opt[i])< 0.1:
            phi_hp_opt[i] = 0
        if abs(phi_hp_opt[i]) > 0:
            freq[i]         = 60*(phi_hp_opt[i]/Q_hp_max[i])
            potenza_perc[i] = 2.243*freq[i] - 34.26
#            potenza_perc[i] = 2.4986*freq[i] - 52.16
            Tset_utenza[i]  = Tset_on
        else:
            freq[i]         = 0
            potenza_perc[i] = 0
            Tset_utenza[i]  = Tset_off
        if abs(phi_hc_opt[i])< 0.1:
            phi_hc_opt[i] = 0        
        if abs(phi_hc_opt[i]) > 0:
            if m_fc_calc[i] < m_fc_min:
                m_fc_calc[i] = m_fc_min
            elif m_fc_calc[i] > m_fc_max:
                m_fc_calc[i] = m_fc_max
            pompa_fancoil[i] = -2.4351 + 0.0069*m_fc_calc[i]
        else:
            pompa_fancoil[i] = 0
        if potenza_perc[i] < 0:
            potenza_perc[i] = 0            
    for i in range(H):
        if i == 0:
            if (Tset_utenza[i] == Tset_on) & (potenza_perc[i]<30):
                potenza_perc[i] = 30
        if (i>0) & (i<H-1):
            x = Tset_utenza[i]-Tset_utenza[i-1]
            y = Tset_utenza[i+1]-Tset_utenza[i]
            if ((x-y)  < -deltaTset):
                Tset_utenza[i] = Tset_on                    
#    lonely_times = fdata.loc[(fdata['phi_hp_opt']==0) & (fdata['phi_hp_opt_prev']>0) & (fdata['phi_hp_opt_next']>0)]    
    fdata['phi_hp_opt']    = phi_hp_opt
    fdata['freq']          = freq
    fdata['potenza_perc']  = potenza_perc.astype(int)
    fdata['Tset_utenza']   = Tset_utenza
    fdata['m_fc_calc']     = m_fc_calc
    fdata['pompa_fancoil'] = pompa_fancoil.astype(int)   
    # select columns and print without timezone        
    output = fdata[['potenza_perc','Tset_utenza','pompa_fancoil',
                    'a_1','a_2','a_3','a_4']].tz_localize(None)  
    
    return output


def read_settings():
    # Find files in settings folder
    loc_fname        = gl.glob(os.getcwd() + '/settings/loc_settings.txt')[0] 
    cali_fname       = gl.glob(os.getcwd() + '/settings/cali_settings.txt')[0] 
    opt_fname        = gl.glob(os.getcwd() + '/settings/opt_settings.txt')[0] 
    bui_props_fname  = gl.glob(os.getcwd() + '/settings/building_properties.txt')[0] 
    hvac_props_fname = gl.glob(os.getcwd() + '/settings/hvac_properties.txt')[0] 
    pv_props_fname   = gl.glob(os.getcwd() + '/settings/pv_properties.txt')[0] 
    # Read settings files as dicts
    loc_settings  = eval(open(loc_fname).read())
    cali_settings = eval(open(cali_fname).read())    
    opt_settings = eval(open(opt_fname).read())
    building_properties = eval(open(bui_props_fname).read())  
    hvac_properties = eval(open(hvac_props_fname).read())  
    pv_properties = eval(open(pv_props_fname).read())   
    # Creatte temporary file from HVAC properties
    data_pickle(hvac_properties, os.getcwd() + '/tmp/hvac_data') 
    return loc_settings, cali_settings, opt_settings, building_properties, hvac_properties, pv_properties

def write_info(arg, date):
    '''Append given text as a new line at the end of file'''
    infofile = os.getcwd() + '/infofile.txt'
    info     = data_unpickle(os.getcwd() + '/tmp/info')
    # Open the file in append & read mode ('a+')
    with open(infofile, 'a+') as file_object:
        if arg == 'sem_start':
            start_msg = '\n Smart Energy Manager started at ' + str(date)[:19]
            file_object.write(start_msg + '\n')            
            # Append text at the end of file
        elif arg =='sem_log':
            file_object.write('At time ' + str(date)[:19] +': ')
            for message in info['log_messages'][:]:
                file_object.write(message)
                file_object.write('\n')
        elif arg =='sem_cal': 
            file_object.write('At time ' + str(date)[:19] +': ')
            for message in info['cali_messages'][:]:
                file_object.write(message)
                file_object.write('\n') 
        elif arg =='sem_opt': 
            file_object.write('At time ' + str(date)[:19] +': ')
            for message in info['opt_messages'][:]:
                file_object.write(message)
                file_object.write('\n')    
    return
        
#def write_info(info_folderpath):
#    infofile = info_folderpath + '/infofile.txt'
#    info     = data_unpickle(os.getcwd() + '/tmp/info')
#    newlines = info['messages']
#    with open(infofile, 'a') as f:
#        for newline in newlines:
#            f.writelines('\n'.join(newline))
#    return


#%% Pickling and unpickling data
    
def data_pickle(x, filename):
    
    y = copy.deepcopy(x)
    # Operations on input data (x)
    #...
    #...
    # Opening external output file to store data
    outfile = open(filename,'wb')
    # Storing data into output file
    pickle.dump(y,outfile)
    # Closing output file
    outfile.close()
    return


def data_unpickle(filename):
    infile = open(filename,'rb')
    x = pickle.load(infile)
    infile.close()
    return x




