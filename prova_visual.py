# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:30:20 2021

@author: Laboratorio
"""

import numpy as np
import pandas as pd
import glob   as gl
import matplotlib.pyplot as plt
import matplotlib.pylab  as pylab
from matplotlib.artist import Artist
from matplotlib.dates  import DateFormatter
import pytz
from sklearn.metrics import r2_score
from modules.calibrationClasses  import calipso
from functions.io  import read_logs, estimateCurrentState

#logs      = data_unpickle(os.getcwd() + '/tmp/logs')
#estimateCurrentState(logs, hvac_properties, opt_settings, loc_settings, cali_settings, building_properties)

#params = {'legend.fontsize': 12,
#          'figure.figsize' : (12, 8), 
#          'axes.labelsize' : 12,
#          'axes.titlesize' : 12,
#          'xtick.labelsize': 12,
#          'ytick.labelsize': 12}
#logs = read_logs(datetime.today(), logs_folder, 10, '15min', loc_settings)
# paths to fdata 
filepaths = gl.glob(out_folder + 'fdata__*.csv') 
# read last log file
logs = data_unpickle(os.getcwd() + '/tmp/logs')
hist = data_unpickle(os.getcwd() + '/tmp/hist')

past_steps, num_steps = 72,71
nf = min(len(filepaths), past_steps)
fpaths = sorted(filepaths)[-nf:]
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+02:00')

fd0 = pd.read_csv(fpaths[0], header=0, index_col=0, sep = ';', date_parser = dateparse)
fd0 = fd0.tz_localize(loc_settings['tz'], ambiguous=False)   # assignes timezone to index
date_init = fd0.index[0]
ld = logs[logs.index>date_init]

# Average indoor air temperature
ld['theta_i_meas'] = (ld['Ti_A'] + ld['Ti_B'] + ld['Ti_C'] + ld['Ti_D'])/4
# Average water temperature in the heat storage tank
ld['theta_hs_meas'] = (ld['Tst_high'] + ld['Tst_low'])/2

# Indoor air temperature: optimization vs logs
x1,y1  = fd0.index, fd0.theta_i_opt
x2, y2 = ld.index, ld.theta_i_meas
# Heat storage tank water temperature: optimization vs logs
x3, y3 = fd0.index, fd0.theta_hs_opt
x4, y4 = ld.index, ld.theta_hs_meas
# Heat flow rate from heat pump: optimization vs logs
x5, y5 = fd0.index, fd0.phi_hp_opt/1000
x6, y6 = ld.index, ld.q_hp
# Heat flow rate from fancoils: optimization vs logs
x7, y7 = fd0.index, -fd0.phi_hc_opt/1000
x8, y8 = ld.index, -ld.q_fc

# Initialize interactive plot
plt.ion()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.set_size_inches(15, 8)

line1, = ax1.plot(x1, y1, 'b-', lw = 1.0, label=r"$\theta^{i,opt}$") 
line2, = ax1.plot(x2, y2, 'b:', lw = 1.5, label=r"$\theta^{i,meas}$")

line3, = ax2.plot(x3, y3, 'r-', lw = 1.0, label=r"$\theta^{hs,opt}$") 
line4, = ax2.plot(x4, y4, 'r:', lw = 1.5, label=r"$\theta^{hs,meas}$")

line5, = ax3.plot(x5, y5, 'r-', lw = 1.0, label=r"$\phi^{hp,opt}$") 
line6, = ax3.plot(x6, y6, 'r:', lw = 1.5, label=r"$\phi^{hp,meas}$")

line7, = ax3.plot(x7, y7, 'b-', lw = 1.0, label=r"$\phi^{hc,opt}$") 
line8, = ax3.plot(x8, y8, 'b:', lw = 1.5, label=r"$\phi^{hc,meas}$")



formatter = DateFormatter('%d/%m %H:%M')
plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
plt.gcf().axes[1].xaxis.set_major_formatter(formatter)
plt.gcf().axes[2].xaxis.set_major_formatter(formatter)

plt.rc('grid', color='#397939', linewidth=1, linestyle='-')
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.ylabel('ylabel', fontsize=11)

ax1.set(ylabel='Indoor air emperature (°C)')
ax1.grid(which='major')
ax1.set_ylim([24, 30])
leg1 = ax1.legend()
ax2.set(ylabel='Water temperature (°C)')
ax2.grid(which='major')
leg2 = ax2.legend()
ax2.set_ylim([hvac_properties['theta_hs_min']-5, hvac_properties['theta_hs_max']+5])
ax3.set(ylabel='Heat flow rate (kW)')
ax3.grid(which='major')
ax3.set_ylim([-8, 6])
leg3 = ax3.legend()

res = np.empty((0,3), float) #pd.DataFrame(col_names = {'index','r2i','r2s'})
rdf = pd.DataFrame(columns=['time','r2_i','r2_hs','deT_0'])

title = ax1.text(0.5,1.1, "", bbox={'facecolor':'g', 'alpha':0.5, 'pad': 5}, transform=ax1.transAxes, ha='center')
#pylab.rcParams.update(params)

for i in range(1,num_steps):
    
    fpath = fpaths[i]
    launchtimestr = 'Optimization launched at ' + fpath[-20:-4].replace('_',' ')
    
    fd = pd.read_csv(fpath, header=0, index_col=0, sep = ';', date_parser = dateparse)   
    fd = fd.tz_localize(loc_settings['tz'], ambiguous=False)   # assignes timezone to index
    
    line1.set_xdata(fd.index)
    line1.set_ydata(fd.theta_i_opt)
    line3.set_xdata(fd.index)
    line3.set_ydata(fd.theta_hs_opt) 
    line5.set_xdata(fd.index)
    line5.set_ydata(fd.phi_hp_opt/1000) 
    line7.set_xdata(fd.index)
    line7.set_ydata(-fd.phi_hc_opt/1000) 
    title.set_text(launchtimestr)
       
    start_date = fd.index[0]
    end_date   = fd.index[-1]
  
    ax1.set_xlim([start_date, end_date])
    ax2.set_xlim([start_date, end_date])
    ax3.set_xlim([start_date, end_date])
    
    comfort_area = ax1.fill_between(fd.index, fd.theta_min, fd.theta_max,
                                   color='C1', alpha=0.2)
    pv_area2 = ax2.fill_between(fd.index, 0, 60, 
                                where = fd.W_pv > 50, 
                                color='C1', alpha=0.3)
    pv_area3 = ax3.fill_between(fd.index, 0, fd.W_pv/1000, 
                                where = fd.W_pv > 50, 
                                color='C1', alpha=0.3)
    
                  
    mask = (ld.index >= start_date) & (ld.index <= end_date)
    
    if len(mask)==len(fd.theta_i_opt.values):
        r2_theta_i  = r2_score(ld.theta_i_meas.loc[mask].values, fd.theta_i_opt.values) 
        r2_theta_hs = r2_score(ld.theta_hs_meas.loc[mask].values, fd.theta_hs_opt.values)  
        delta_theta_i0 = fd.theta_i_opt.values[0] - ld.theta_i_meas.loc[mask].values[0] 
        
        res = np.append(res, np.array([[r2_theta_i,r2_theta_hs, delta_theta_i0]]), axis=0)
        
        rdf_row = pd.DataFrame({'time': [start_date], 'i': [r2_theta_i], 'hs' : [r2_theta_hs]}) 
        rdf = rdf.append(rdf_row)
            
        testoi  = ax1.text(fd.index[5], 25, '{:.2f}'. format(r2_theta_i))
        testohs = ax2.text(fd.index[5], 45, '{:.2f}'. format(r2_theta_hs))
        
        Artist.set_visible(testoi, True)
        Artist.set_visible(testohs, True)
        
    fig.canvas.draw()
    fig.canvas.flush_events()
    
#    fig.savefig('figures/fig_opt_' + fpath[-20:-4] +'.png')
    
    plt.pause(1.0)
    comfort_area.remove()
    pv_area2.remove()
    pv_area3.remove()
    
    if len(mask)==len(fd.theta_i_opt.values):
        Artist.remove(testoi)
        Artist.remove(testohs)


rdf = rdf.set_index('time')
ld = logs[['spu','q_spv','q_hp','ctrl_pompa_V','q_fc']][-24*4:]



#%%

from sklearn.linear_model import LinearRegression

df = pd.DataFrame()
mask2 = (logs.index >= hist.index[0]) & (logs.index <= hist.index[-1])
df = logs[['Te',
           'Ti_A','Tmr_A','Ti_B','Tmr_B','Ti_C','Tmr_C','Ti_D','Tmr_D',
           'Tfloor','Twall_S','Twall_W']].loc[mask2]

# Linear regression to calculate COP
X_w = df.values   #'rps'
y_w = df[['f_cop']].values

reg_w = LinearRegression().fit(X_w, y_w)
coeff_w = [reg_w.intercept_,reg_w.coef_[0]]     
score_w = reg_w.score(X_w, y_w)

