# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:20:37 2020

@author: vivijac14771
"""

import numpy as np
import pandas as pd


def poly(T_hp_out, T_e, season):
    
    # Parameters of compressor polynomial curves 
    coeff_a = [6.081690780998,0.214949849110,-0.056650193911,0.003092910132,-0.001435948337,0.000495677688,0.000015563988,-0.000024948162,-0.000003927298,-0.000006518906]
    coeff_b = [0.381946813112,-0.009513196415,0.012570846323,-0.000174042685,0.000046315298,0.000145937054,0.000000031384,-0.000000151502,0.000003678081,0.000000581912]
   
    # Condensation and evaporation temperature
    if season == 'h':
        T_cd = T_hp_out - 3
        T_ev = T_e - 3
    else:
        T_cd = T_e + 3
        T_ev = T_hp_out - 3
       
#    a = 1.006 + 0.0069*T_e
#    b = 1.022 - 0.0036*T_e
    
#    a,b = 1, 1
    
    # Calculate polynomials
    t = [1, T_ev, T_cd, T_ev**2, T_ev*T_cd, T_cd**2, T_ev**3, T_cd*(T_ev**2), T_ev*(T_cd**2), T_cd**3]
    W_el = sum(np.multiply(coeff_b, t))*1000
    Q_ev_max = sum(np.multiply(coeff_a, t))*1000 
    Q_cd_max = Q_ev_max + W_el
       
#    if np.any(argv):
#        # Heat output
#        Q_cd = argv[0]
#        # Calculate part load factor for the compressor
#        q_cd_perc = Q_cd/Q_cd_max
#        res = [Q_cd_max, q_cd_perc]
#    else:
#        res = Q_cd_max
    res = [Q_ev_max, Q_cd_max]
    
    return res


def poly2(logs, T_hp_out, T_e, season):
    
#    coeff_c = [-3788.626232, -54.30124761, 241.0357154, 6.158790656, -0.002318474, -4.898041686, -0.067157101, -0.091974942, 0.017589256, 0.032066464]
#  
#    # Condensation and evaporation temperature
#    if season == 'h':
#        T_cd = T_hp_out + 3
#        T_ev = T_e - 3
#    else:
#        T_cd = T_e + 3
#        T_ev = T_hp_out - 3
#        
#    t = [1, T_ev, T_cd, T_ev**2, T_ev*T_cd, T_cd**2, T_ev**3, T_cd*(T_ev**2), T_ev*(T_cd**2), T_cd**3]    
#
#    Q_cd_max = sum(np.multiply(coeff_c, t))*1000
    
    Q_cd_max = (0.48*T_e - 1.3444)*1000

    
    return Q_cd_max


def write_output(fdata, hvac_data, cc):
    
    H = len(fdata.index)
    
    fdata['Q_hp_max']     = np.zeros([H,1])
    fdata['freq']         = np.zeros([H,1])
    fdata['potenza_perc'] = np.zeros([H,1])
    fdata['Tset_utenza']  = np.zeros([H,1])
    
#    theta_e               = fdata['Te'].values
#    theta_hs_opt          = fdata['theta_hs_opt'].values
    phi_hp_opt            = fdata['phi_hp_opt'].values
    
    Xa = np.zeros([H,2])
    Xa[:,0] = fdata['Te'].values
    Xa[:,1] = 42*np.ones([H,]) #fdata['theta_hs_opt'].values + 3
    fdata['Q_hp_max'] = cc.heatPumpCapacity(Xa, hvac_data['params_hp_heat_qmax'])*1000
    
    Q_hp_max              = fdata['Q_hp_max'].values
    freq                  = np.zeros([H,1])
    potenza_perc          = np.zeros([H,1])
    Tset_utenza           = np.zeros([H,1])
    
    for i in range(H):
        if phi_hp_opt[i] > 0:
            freq[i]         = 60*(phi_hp_opt[i]/Q_hp_max[i])
            potenza_perc[i] = 2.5*freq[i] - 50
            Tset_utenza[i]  = 60
        else:
            freq[i]         = 0
            potenza_perc[i] = 0
            Tset_utenza[i]  = 20
#    
#    fdata['Q_hp_max']     = Q_hp_max
    fdata['freq']         = freq.astype(int)
    fdata['potenza_perc'] = potenza_perc.astype(int)
    fdata['Tset_utenza']  = Tset_utenza
            
    output = fdata[['potenza_perc','Tset_utenza']]
    
    return output




