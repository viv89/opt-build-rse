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


def read_logs(subfolder, ndays, loc_settings):
    colnames = ('Te','RHe', 'ghi_1','ghi_2','dhi','vw','riserva',
                'Ti_A','Tmr_A','Ti_B','Tmr_B','Ti_C','Tmr_C','Ti_D','Tmr_D',
                'RHi_B','Tfloor','Twall_W','Twall_S','riserva1',
                'Thp_in','Thp_out','m_hp',
                'Tfc_su','Tfc_ret','m_fc',
                'Tfc_su2','Tfc_ret2',
                'Tst_high','Tst_low','riserva2',
                'Tdhw_cold','Tdhw_hot','Tdhw_mix','m_dhw',
                'Thwt','m_net','Trec_in','Trec_out',
                'riserva3','riserva4','riserva5','riserva6','riserva7','riserva-last',
                'relay_QA1','relay_QA2','relay_QB1','relay_QB2',
                'relay_QC1','relay_QC2','relay_QD1','relay_QD2',
                'q_fc','q_fc2','q_hp','q_dhw','q_rec',
                'QA1','QA2','QA3','QA4','QA5','QA6',
                'QB1','QB2','QB3','QB4','QB5','QB6',
                'QC1','QC2','QC3','QC4','QC5','QC6',
                'QD1','QD2','QD3','QD4',
                'Wel_hp','Wel_dhw',
                'QSSE_C2','QSSE_C3','QSSE4-C12','QSSE-C13','QSS_Gen',
                'Wel_od')
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') #set datetime format of imported data
    colnameslist = list(colnames)
    filepaths = gl.glob(subfolder + '/*.txt')
    fpaths = filepaths[-ndays:]    
    dfr = pd.DataFrame(columns=colnames)
    dfr1 = dfr    
    for fname in fpaths:
        df0 = pd.read_csv(fname, header=1, index_col=0, sep = '\t', date_parser = dateparse)
        df0.columns = colnameslist
        df0 = df0.fillna(method = 'bfill')
        dfr = df0.resample('15min').mean()
        dfr1 = dfr1.append(dfr)
#    dfr1.to_csv(subfolder + '/logs_resampled.csv') 
    dfr1 = dfr1[~dfr1.index.duplicated(keep='first')]  # eliminates doubles
    dfr1 = dfr1.tz_localize(loc_settings['tz'])   # assignes timezone to index
    return dfr1
                                            


def read_fcst(subfolder, loc_settings):
    # commentare e scommentare riga successiva se eseguito test in laboratorio 
    dateTime = datetime(2020,7,24,19,19,37)
#    dateTime = datetime.today()
    if dateTime.hour<12:
        dayForecast = dateTime.day - 1
        fileID = '0000'
        dateTime.replace(day = dayForecast)
    else:
        fileID = '1200'
        
        # qui inserire anche prime 12 ore della previsione precedente (cioè da file 0000)
        
        
    colnames = ('data_run','fcnum','Te', 'RHe', 'Patm','ghi','dhi','dni','prec','vw','vdir')
    colnameslist = list(colnames)
    datestring = dateTime.strftime('%Y%m%d')
    filepath = subfolder + '/SSE-PCN00-meteo-' + datestring + fileID + '.csv' 
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    dfc = pd.read_csv(filepath, header=0, index_col=2, sep = ';', date_parser = dateparse)  #pd.read_excel(filepath, header=0, index_col=2)
    dfc.columns = colnameslist
    dfc = dfc.fillna(method = 'bfill')
    dfc1 = dfc.resample('15min').bfill()  
    # qua bisogna fare la conversione da UTC a CET
    return dfc1



def read_setpoint(subfolder):

    filepath = subfolder + '/setpoint_table.xls'
    dfu = pd.read_excel(filepath, header=1, index_col=0)

    return dfu


def update_future(dfu, dfc, settings):
    
    dateTime = datetime(2020,7,24,19,19,37)
#    dateTime = datetime.today()
    # setpoint
    weekDayList = list(['LUN','MAR','MER','GIO','VEN','SAB','DOM'])
    weekDayNum = dateTime.weekday()
    weekDayToday = weekDayList[weekDayNum]
    if weekDayNum == 6:
        weekDayTomorrow = weekDayList[0]
    else:
        weekDayTomorrow = weekDayList[weekDayNum+1]

    nsteps = nhours*hsteps
    fdata = pd.DataFrame()
    n_hours = settings['nhours_horizon']
    h_steps = settings['hourly_steps']
    
   
    # forecasts


    return fdata

#def clean_data(logs, cali_settings, loc_settings):
#    # Date range of logged data for calibration
#    stop_date  = logs.index[-1]    
#    cali_steps = int(cali_settings['n_days']*24*(3600/cali_settings['tau']))
#    if cali_steps > len(logs.index):
#        cali_steps = len(logs.index)
#    start_date = logs.index[-cali_steps]    
#    date_rng = pd.date_range(start = start_date, 
#                             end  = stop_date, 
#                             freq  ='15min', 
#                             tz = loc_settings['tz'])
#    bdf = logs.loc[start_date:stop_date]    
#    # Correct bdf with missing values
#    missing_dates = date_rng.difference(bdf.index) # find missing dates
#    nanarray = np.empty((len(missing_dates),bdf.shape[1])) # initialize empty array with correct size
#    nanarray[:] = np.nan  # fills array with nans
#    bdf = pd.concat([bdf, pd.DataFrame(nanarray, columns=bdf.columns,  index = missing_dates)]) # appends rows with nans on missing dates
#    bdf = bdf.sort_index()   # sorts dataframe by date
#    bdf = bdf.fillna(bdf.mean()) # replaces nan values with mean of entire column    
#    return bdf   
