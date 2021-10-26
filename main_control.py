# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:21:46 2021

@author: vivijac14771
"""


def logs_reader_job(logs_folder, fcst_folder, user_folder, ndays, loc_settings):
    # Set today date or a random date 
    date = datetime.today()   #datetime(2021,1,28,12,0,0) - timedelta(days=randint(0, 20)) 
    info      = data_unpickle(os.getcwd() + '/tmp/info') 
    # Read and resamples logs
    info['log_messages'][0] = 'Reading logged data, setpoint schedules and updating forecasts..'        
    logs           = read_logs(date, logs_folder, ndays, '15min', loc_settings)
    forecast_table = read_fcst(date, fcst_folder, loc_settings)
    setpoint_table = read_setpoint(user_folder)
    info['log_messages'][1] = 'Logs reader job finished.'  
    # Store logs dataframe into temporary folder
    data_pickle(logs,           os.getcwd() + '/tmp/logs')
    data_pickle(forecast_table, os.getcwd() + '/tmp/forecasts')
    data_pickle(setpoint_table, os.getcwd() + '/tmp/setpoints') 
    data_pickle(info,           os.getcwd() + '/tmp/info')
    write_info('sem_log', date)
    return 

def model_calibration_job(loc_settings, cali_settings, building_properties, hvac_properties):
    # Set time
    date = datetime.today()   #
    # Load logs from temporary folder
    logs      = data_unpickle(os.getcwd() + '/tmp/logs')
    hvac_data = data_unpickle(os.getcwd() + '/tmp/hvac_data') 
    fdata_old = data_unpickle(os.getcwd() + '/tmp/output/fdata') 
    info      = data_unpickle(os.getcwd() + '/tmp/info') 
    info['cali_messages'][0] = 'Building parameters are being calibrated using data from the last {:d} days..'.format(cali_settings['n_days'])        
    # Run building model calibration         
    if (info['runs'] == 0):
        cs = run_calibration(loc_settings, cali_settings, building_properties, logs) 
    else:
        lastpars = data_unpickle(os.getcwd() + '/tmp/lastpars/lastpars')
        cs = run_calibration(loc_settings, cali_settings, building_properties, logs, lastpars)
    optpars           = cs.info.optpars     
    info['cali_test'] = cs.info.rmse_test
    info['cali_messages'][1] = 'Building parameters calibration ended with RMSE = {:.3f} K over testing period.'.format(info['cali_test'])        
    # Fancoils power calibration
    info['cali_messages'][2] = 'HVAC system parameters calibration..'
    logs_5min = read_logs(datetime.today(), logs_folder, 3, '5min', loc_settings) #last n days
    ch = calihvac(logs_5min, hvac_data)
    try: 
        ch.update_fc(hvac_data, fdata_old)
        hvac_data['params_hc'] = [ch.calpars.coeff_fc[0],ch.calpars.coeff_fc[1]]
        data_pickle(hvac_data, os.getcwd() + '/tmp/hvac_data') 
        info['cali_messages'][3] = 'Fancoil power correlation updated with R2 = {:.3f}'.format(ch.info.score_fc)
    except:
        info['cali_messages'][3] = 'Fancoil power correlation not updated'        
    # Store calibration results in temporary folder
    build_data, hist = vars(cs.auxdata), cs.history
    data_pickle(build_data, os.getcwd() + '/tmp/build_data')
    data_pickle(hist      , os.getcwd() + '/tmp/hist') 
    data_pickle(info      , os.getcwd() + '/tmp/info')  
    data_pickle(optpars, os.getcwd() + '/tmp/lastpars/lastpars')      
    write_info('sem_cal', date)         
    return 

def optimization_job(opt_settings, loc_settings, cali_settings, building_properties, out_folder):
    # Set today date or a random date 
    date = datetime.today()   #datetime(2021,1,28,12,0,0) - timedelta(days=randint(0, 20)) 
    datestr = str(date)[:16]
    datestr = datestr.replace(':','-').replace(' ','_')      
    # Load hist and logs from temporary folder
    info      = data_unpickle(os.getcwd() + '/tmp/info')
    logs      = data_unpickle(os.getcwd() + '/tmp/logs')
    # Load additional data from temporary folder
    build_data     = data_unpickle(os.getcwd() + '/tmp/build_data')
    hvac_data      = data_unpickle(os.getcwd() + '/tmp/hvac_data') 
    setpoint_table = data_unpickle(os.getcwd() + '/tmp/setpoints')
    forecast_table = data_unpickle(os.getcwd() + '/tmp/forecasts')
    optpars        = data_unpickle(os.getcwd() + '/tmp/optpars')   
    fdata_old      = data_unpickle(os.getcwd() + '/tmp/output/fdata')  
#    cc             = data_unpickle(os.getcwd() + '/tmp/cc')
    # Load pv data from settings folder
    pv_obj  = data_unpickle(os.getcwd() + '/settings/pv_obj')
    pv_data = data_unpickle(os.getcwd() + '/settings/pv_data')
    info['opt_messages'][0] = 'Estimating current state and updating boundary conditions over optimization horizon..'
    # Set current state to be used in the optimization
    cstate = estimateCurrentState(logs, hvac_data, opt_settings, 
                                  loc_settings, cali_settings, building_properties)
    # Read future data as boundary conditions
    fdata  = update_future(date, setpoint_table, forecast_table, 
                           pv_obj, pv_data, opt_settings, hvac_data, logs)  
    # Heat pump calibration and maximum capacity over optimization horizon
    run_count = info['runs']
    info['opt_messages'][1] = 'Running optimization n. {:d}..'.format(run_count)
    fdata, opt = run_optimization(optpars, opt_settings, build_data, hvac_data, 
                                  cstate, fdata, fdata_old, loc_settings) 
    info['opt_messages'][2] = 'Minimum value of objective function = {:.3f}'.format(opt.model.objVal)
    info['runs'] = run_count + 1
    output     = write_output(fdata, hvac_data['season'])
    # Store optimization output in temporary folder
    data_pickle(cstate,    os.getcwd() + '/tmp/cstate')
#    data_pickle(hvac_data, os.getcwd() + '/tmp/hvac_data')
    data_pickle(fdata , os.getcwd() + '/tmp/output/fdata')
    data_pickle(output, os.getcwd() + '/tmp/output/output')
    data_pickle(info, os.getcwd() + '/tmp/info')
    # Write fdata and output dataframes into unique csv files inside output folder
    output.to_csv(out_folder + 'ML__' + datestr + '.csv', sep=';')
    fdata.to_csv(out_folder + 'fdata__' + datestr + '.csv', sep=';')  
    write_info('sem_opt', date)
    return

def save_ctrl_file_job(ctrl_folder):
    # Load control data file from temporary folder
    output = data_unpickle(os.getcwd() + '/tmp/output/output')
    # Write file for HVAC control into csv file inside control folder
    output.to_csv(ctrl_folder + '/ML.csv', sep=';')
    return    

def first_run_all_jobs(logs_folder, fcst_folder, user_folder, ndays, loc_settings, cali_settings, building_properties, hvac_properties,opt_settings, out_folder):
    # Inizialize info dict
    date = datetime.today()    
    info = {'log_messages'  : ['',''], 
            'cali_messages' : ['','','',''], 
            'opt_messages'  : ['','',''],
            'cali_test'     : [], 
            'runs': 0}
    data_pickle(info , os.getcwd() + '/tmp/info') 
    # First run of jobs
    write_info('sem_start', date)
    logs_reader_job(logs_folder, fcst_folder, user_folder, ndays, loc_settings)
    model_calibration_job(loc_settings, cali_settings, building_properties, hvac_properties)
    optimization_job(opt_settings, loc_settings, cali_settings, building_properties, out_folder)
    return

#def check_status():
#    print('Checking for manual interruptions..') #only with BlockingScheduler
#    return

def remove_all_jobs(sched):
    for j in sched.get_jobs():
        sched.remove_job(j.id)
    return

if __name__ == '__main__':
    
#    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from functions.run_modules  import run_calibration, run_optimization, run_pvlib
    from functions.io           import data_pickle, data_unpickle, read_logs, read_settings, read_fcst, read_setpoint, update_future, estimateCurrentState, write_output, write_info
    from modules.calibrationClasses import calihvac
    from datetime import datetime 
    import os
    
       
    # Input folders (logs, weather forecasts, setpoints)
    logs_folder = 'D:\ML\SW\ScambioDati'            #logs_folder = os.getcwd() +'/data/logs/winter/'
    fcst_folder = 'D:\PrevisioniMeteo\DatiMeteo'    #fcst_folder = os.getcwd() +'/data/forecasts/'
    user_folder = 'D:\ML\SW\ScambioDati'            #user_folder = os.getcwd() +'/data/user/'
    
    # Output folders
    out_folder  = 'D:\ML_PD\SEM_output/'
    ctrl_folder = 'D:\ML\SW\Input'
    
    # Read settings and properties of the building and technical systems
    loc_settings, cali_settings, opt_settings, building_properties, hvac_properties, pv_properties = read_settings()
    ndays = 10
    
    # All info for PV simulation are stored in pv_data and pv_obj
    pv_data, pv_obj = run_pvlib(pv_properties, loc_settings)
    
    # Run all jobs (initialization of the controller)
    first_run_all_jobs(logs_folder, fcst_folder, user_folder, ndays, loc_settings, 
                       cali_settings, building_properties, hvac_properties,opt_settings, out_folder)   
    
    # Initialize scheduler    
    sched = BackgroundScheduler()
    # Add jobs to scheduler
    sched.add_job(logs_reader_job, 
                  'cron', 
                  args    = [logs_folder, fcst_folder, user_folder, ndays, loc_settings], 
                  minute = '26, 56', # alle XX:56 ogni ora vengono letti i dati
                  id      = 'sem_log')
    sched.add_job(model_calibration_job, 
                  'cron', 
                  args = [loc_settings, cali_settings, building_properties, hvac_properties], 
                  minute = 57, #hour = '5,17' 
                  id   = 'sem_cal')
    sched.add_job(optimization_job, 
                  'cron', 
                  args = [opt_settings, loc_settings, cali_settings, building_properties, out_folder], 
                  minute = '29, 59',   # ogni ora alle XX:59 viene fatta l'ottimizzazione
                  id   = 'sem_opt')
    sched.add_job(save_ctrl_file_job, 
                  'cron', 
                  args = [ctrl_folder], 
                  minute = '0, 30',  # ogni ora alle XX:00 lo scheduling vien salvato nel file ML.csv che controlla l'impianto
                  id   = 'sem_ctr')
#    sched.start()

