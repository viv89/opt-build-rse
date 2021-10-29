#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:52:32 2019

@author: nmazzi
"""

import gurobipy as gb
import numpy as np
#from functions.heatpumps    import poly

#%%
class obj(object):
    '''
        A small class which can have attributes set
    '''
    pass

class singleZoneOptimizer:
    
#        C_m     = x[0]
#        H_tr_em = x[1]
#        H_tr_is = x[2]
#        H_tr_ms = x[3]
#        H_tr_w  = x[4]   
#        H_ve    = x[5]
#        k_conv  = x[6]   
    
    
    def __init__(self, optimal_parameters, settings, build_data, hvac_data):

        self.data        = obj()
        self.vardata     = obj()
        self.variables   = obj()
        self.constraints = obj()
        self.output      = obj()
        self.tseries     = obj()
        self.info        = obj()

        
        # Load parameters
        self.data.hours_horizon = settings['nhours_horizon']
        self.data.hourly_steps  = settings['hourly_steps']
        self.data.T             = range(self.data.hours_horizon*self.data.hourly_steps);   T = self.data.T
        self.data.tau           = settings['timestep'];   tau = self.data.tau
        self.data.mode          = settings['mode']
        self.data.time_limit    = settings['time_limit']        
        self.data.pbuy          = settings['pbuy']/1000*(tau/3600)
        self.data.psell         = settings['psell']/1000*(tau/3600)
        self.data.blocked_hours = settings['blocked_hours']
        self.data.blocked_steps = int(self.data.blocked_hours*3600/tau)
        
        self.data.Cm     = optimal_parameters[0]/tau;      Cm     = self.data.Cm
        self.data.H_trem = optimal_parameters[1];          H_trem = self.data.H_trem
        self.data.H_tris = optimal_parameters[2];          H_tris = self.data.H_tris
        self.data.H_trms = optimal_parameters[3];          H_trms = self.data.H_trms
        self.data.H_trw  = optimal_parameters[4];          H_trw  = self.data.H_trw
        self.data.H_ve   = optimal_parameters[5];          H_ve   = self.data.H_ve 
        self.data.k_conv = optimal_parameters[6];          
        self.data.ks_gla = optimal_parameters[7];  
        self.data.ks_opa = optimal_parameters[8];  
        self.data.phi_0  = optimal_parameters[9]; 
        self.data.k_a    = optimal_parameters[10]; 
        self.data.k_s    = optimal_parameters[11]; 
        if len(optimal_parameters) == 12:
            self.data.Cs   = 0;                             Cs = self.data.Cs
            self.data.Ci   = 0;                             Ci = self.data.Ci
        elif len(optimal_parameters) == 13:
            self.data.Cs   = optimal_parameters[12]/tau;    Cs = self.data.Cs
            self.data.Ci   = 0;                             Ci = self.data.Ci
        elif len(optimal_parameters) == 14:
            self.data.Cs   = optimal_parameters[12]/tau;    Cs = self.data.Cs
            self.data.Ci   = optimal_parameters[13]/tau;    Ci = self.data.Ci
    
        self.data.Am        = build_data['A_m']        # Am = self.data.Am      
        self.data.A_floor   = build_data['A_floor']    # A_floor  = self.data.A_floor
        self.data.A_aw      = build_data['A_walls']
        self.data.ks_gla    = build_data['ks_gla']       # 0.6
        self.data.ks_opa    = build_data['ks_opa']       # 0.024 
        self.data.h_ve      = build_data['h_ve']        
        
#        scrivere i parametri in modo che sia COP = f(T_ext)  
        self.data.season       = hvac_data['season']
        self.data.Q_max_h      = hvac_data['Q_hp_nom_heat']                   
        self.data.coeff_qmax_h = hvac_data['params_hp_heat_qmax']     
        self.data.coeff_fcop_h = hvac_data['params_hp_heat_fcop']
        self.data.Q_max_c      = hvac_data['Q_hp_nom_cool']           
        self.data.coeff_qmax_c = hvac_data['params_hp_cool_qmax'] 
        self.data.coeff_fcop_c = hvac_data['params_hp_cool_fcop']
        self.data.V_hs         = hvac_data['V_hs']      # m3
        self.data.UA_hs        = hvac_data['UA_hs']     # W/K 
        self.data.theta_hs_min = hvac_data['theta_hs_min']    # °C
        self.data.theta_hs_max = hvac_data['theta_hs_max']    # °C   
        self.data.r_min        = hvac_data['r_min']           # -
        self.data.r_min_start  = hvac_data['r_min_start']     # -

        self.data.k_hc_0       = hvac_data['params_hc'][0][0]
        self.data.k_hc_1       = hvac_data['params_hc'][1][0] 
        self.data.k_hc_2       = hvac_data['params_hc'][1][1]
        self.data.theta_i_max  = 30
        self.data.Q_hc_min     = 1200 #W
        
        self.data.rho   = 970                    # kg/m3
        self.data.cpw   = 4183                   # J/(kg K)
        self.data.M_hs  = self.data.rho*self.data.V_hs*self.data.cpw  # J/K
        self.data.C_hs  = self.data.M_hs/tau 
        UA_hs           = self.data.UA_hs      # W/K
        C_hs            = self.data.C_hs  # W/K
       
        self.info.message = {}
        
        self.vardata.phi        = {}
        self.vardata.phi_sol    = {}
        self.vardata.theta_e    = {}
        self.vardata.theta_su   = {}
        self.vardata.Q_hp_max   = {}
        self.vardata.Q_hc_max   = {}
        self.vardata.f_cop      = {}

        self.model = gb.Model()
        self.model.Params.OutputFlag = False

        # Load variables
        self.variables.w_buy    = {};    w_buy = self.variables.w_buy
        self.variables.w_sell   = {};   w_sell = self.variables.w_sell
        self.variables.u_hp     = {};     u_hp = self.variables.u_hp
        self.variables.w_hp     = {};     w_hp = self.variables.w_hp
        self.variables.phi_hp   = {};   phi_hp = self.variables.phi_hp
        self.variables.Q_hp_mod = {}; Q_hp_mod = self.variables.Q_hp_mod
        self.variables.Q_hc_corr = {}; Q_hc_corr = self.variables.Q_hc_corr
        self.variables.theta_i  = {};  theta_i = self.variables.theta_i 
        self.variables.theta_s  = {};  theta_s = self.variables.theta_s  
        self.variables.theta_m  = {};  theta_m = self.variables.theta_m 
        self.variables.theta_hs = {}; theta_hs = self.variables.theta_hs 
        self.variables.u_hc     = {};     u_hc = self.variables.u_hc
        self.variables.phi_hc   = {};   phi_hc = self.variables.phi_hc
        self.variables.delta_up = {}; delta_up = self.variables.delta_up
        self.variables.delta_dw = {}; delta_dw = self.variables.delta_dw 
        self.variables.x_su     = {};     x_su = self.variables.x_su  
#        self.variables.acr     = {};     

        for t in T:
            self.variables.theta_i[t]  = self.model.addVar(lb = -gb.GRB.INFINITY)
            self.variables.theta_s[t]  = self.model.addVar(lb = -gb.GRB.INFINITY)
            self.variables.theta_m[t]  = self.model.addVar(lb = -gb.GRB.INFINITY)
#            self.variables.acr[t]     = self.model.addVar(lb = 0)
            self.variables.theta_hs[t] = self.model.addVar(lb = self.data.theta_hs_min , ub = self.data.theta_hs_max)            
            self.variables.delta_up[t] = self.model.addVar(lb = 0)
            self.variables.delta_dw[t] = self.model.addVar(lb = 0) 
            self.variables.w_buy[t]    = self.model.addVar(lb = 0)
            self.variables.w_sell[t]   = self.model.addVar(lb = 0)           
            self.variables.w_hp[t]     = self.model.addVar(lb = 0)
            self.variables.u_hp[t]     = self.model.addVar(lb = 0 , ub = 1, vtype = gb.GRB.BINARY)
            self.variables.u_hc[t]     = self.model.addVar(lb = 0 , ub = 1, vtype = gb.GRB.BINARY)
            self.variables.x_su[t]     = self.model.addVar(lb = 0 , ub = 1, vtype = gb.GRB.BINARY)
            self.variables.Q_hc_corr[t] = self.model.addVar(lb = 0) 
            if self.data.season == 'h':
                self.variables.phi_hp[t]   = self.model.addVar(lb = 0)
                self.variables.Q_hp_mod[t] = self.model.addVar(lb = 0)    
                self.variables.phi_hc[t]   = self.model.addVar(lb = 0)
            elif self.data.season == 'c':    
                self.variables.phi_hp[t]   = self.model.addVar(lb = -gb.GRB.INFINITY, ub = 0)
                self.variables.Q_hp_mod[t] = self.model.addVar(lb = -gb.GRB.INFINITY, ub = 0)
                self.variables.phi_hc[t]   = self.model.addVar(lb = -gb.GRB.INFINITY, ub = 0)

        self.model.update()
        
        # Load output
        self.output.u = np.zeros(len(T))

        # Objective
        self.model.setObjective(0, gb.GRB.MINIMIZE)

        # Constraints
        self.constraints.building_3a = {}
        self.constraints.building_3b = {}
        self.constraints.building_3c = {}
        self.constraints.storage_4a  = {}
        self.constraints.storage_4b  = {}
        self.constraints.heatpump_5a = {}
        self.constraints.heatpump_5b = {}
        self.constraints.heatpump_5c = {}
        self.constraints.heatpump_5d = {}
        self.constraints.heatpump_5e = {}
        self.constraints.heatpump_5f = {}
        self.constraints.heatpump_5g = {}
        self.constraints.heatpump_5h = {}
        self.constraints.elbalance_6 = {}
        self.constraints.comfort_7a  = {}
        self.constraints.comfort_7b  = {}
        self.constraints.fancoil_8a  = {}
        self.constraints.fancoil_8b  = {}
        self.constraints.fancoil_8c  = {}
        self.constraints.fancoil_8d  = {}

        for t in T:
            # Building energy balance
            if (t==min(T)):
                self.constraints.building_3a[t] = self.model.addConstr(-(H_ve+H_tris+Ci)*theta_i[t]+H_tris*theta_s[t]+self.data.k_conv*phi_hc[t], gb.GRB.EQUAL , 0)
                self.constraints.building_3b[t] = self.model.addConstr(H_tris*theta_i[t]-(H_tris+H_trw+H_trms+Cs)*theta_s[t]+H_trms*theta_m[t]+(1-self.data.k_conv)*phi_hc[t], gb.GRB.EQUAL , 0)
                self.constraints.building_3c[t] = self.model.addConstr(H_trms*theta_s[t]-(H_trms+H_trem+Cm)*theta_m[t], gb.GRB.EQUAL, 0)
            else:
                self.constraints.building_3a[t] = self.model.addConstr(-(H_ve+H_tris+Ci)*theta_i[t]+H_tris*theta_s[t]+self.data.k_conv*phi_hc[t]+Ci*theta_i[t-1], gb.GRB.EQUAL , 0)
                self.constraints.building_3b[t] = self.model.addConstr(H_tris*theta_i[t]-(H_tris+H_trw+H_trms+Cs)*theta_s[t]+H_trms*theta_m[t]+(1-self.data.k_conv)*phi_hc[t]+Cs*theta_s[t-1], gb.GRB.EQUAL , 0)
                self.constraints.building_3c[t] = self.model.addConstr(H_trms*theta_s[t]-(H_trms+H_trem+Cm)*theta_m[t]+Cm*theta_m[t-1], gb.GRB.EQUAL, 0)
            # Thermal storage energy balance
            if (t==min(T)):
                self.constraints.storage_4a[t]   = self.model.addConstr((C_hs+UA_hs)*theta_hs[t]-UA_hs*theta_i[t]-phi_hp[t]+phi_hc[t], gb.GRB.EQUAL, 0)               
            else:
                self.constraints.storage_4a[t]   = self.model.addConstr((C_hs+UA_hs)*theta_hs[t]-UA_hs*theta_i[t]-phi_hp[t]+phi_hc[t]-C_hs*theta_hs[t-1], gb.GRB.EQUAL, 0)
            # Temperature in the storage at the end of the horizon
            if self.data.season == 'h':
                self.constraints.storage_4b[0] = self.model.addConstr(theta_hs[0] - theta_hs[max(T)], gb.GRB.LESS_EQUAL, 0)
            elif self.data.season == 'c':
                self.constraints.storage_4b[0] = self.model.addConstr(theta_hs[max(T)] - theta_hs[0], gb.GRB.LESS_EQUAL, 0)            
            # Heat pump capacity and COP
            self.constraints.heatpump_5a[t]     = self.model.addConstr(phi_hp[t] - u_hp[t] - Q_hp_mod[t], gb.GRB.EQUAL, 0)   # incomplete: updated at line 228
            # Note: coefficients change in 5b and 5c for cooling season inside update function 
            self.constraints.heatpump_5b[t]     = self.model.addConstr(Q_hp_mod[t] - u_hp[t], gb.GRB.LESS_EQUAL, 0)
            self.constraints.heatpump_5c[t]     = self.model.addConstr(w_hp[t] + phi_hp[t], gb.GRB.EQUAL, 0)   # incomplete: updated at line 229
            # Definition of start-up (d,e,f)
            if t==min(T):
                self.constraints.heatpump_5d[t] = self.model.addConstr(u_hp[t] - x_su[t], gb.GRB.LESS_EQUAL, 0)
                self.constraints.heatpump_5f[t] = self.model.addConstr(x_su[t] , gb.GRB.LESS_EQUAL, 0)
            else:
                self.constraints.heatpump_5d[t] = self.model.addConstr(u_hp[t] - u_hp[t-1] - x_su[t], gb.GRB.LESS_EQUAL, 0)
                self.constraints.heatpump_5f[t] = self.model.addConstr(x_su[t] + u_hp[t-1], gb.GRB.LESS_EQUAL, 0)
            self.constraints.heatpump_5e[t]     = self.model.addConstr(x_su[t] - u_hp[t], gb.GRB.LESS_EQUAL, 0)           
            # Mimimum power at start-up
            if self.data.season == 'h':
                self.constraints.heatpump_5g[t] = self.model.addConstr(x_su[t] - phi_hp[t], gb.GRB.LESS_EQUAL, 0)  #incomplete: updated at line XXX           
            elif self.data.season == 'c':
                self.constraints.heatpump_5g[t] = self.model.addConstr(x_su[t] + phi_hp[t], gb.GRB.LESS_EQUAL, 0) 
            # Mimimum duration of heat pump operation (2 steps)
            if t < max(T):
                self.constraints.heatpump_5h[t] = self.model.addConstr(x_su[t] - u_hp[t+1], gb.GRB.LESS_EQUAL, 0)
            # Electrical energy balance
            self.constraints.elbalance_6[t]     = self.model.addConstr(w_hp[t] + w_sell[t] - w_buy[t], gb.GRB.EQUAL, 0)
            # Thermal comfort
            self.constraints.comfort_7a[t]      = self.model.addConstr(theta_i[t] - delta_up[t], gb.GRB.LESS_EQUAL, 1)
            self.constraints.comfort_7b[t]      = self.model.addConstr(theta_i[t] + delta_dw[t], gb.GRB.GREATER_EQUAL, 1)
            # Fancoils
            M = 10e4
            # Note that Q_hc_corr is always positive (absolute value of fancoil thermal power)
            if self.data.season == 'h':
                self.constraints.fancoil_8a[t]   = self.model.addConstr(Q_hc_corr[t] - self.data.k_hc_0 - self.data.k_hc_1*theta_hs[t] - self.data.k_hc_2*theta_i[t], gb.GRB.EQUAL, 0)
                self.constraints.fancoil_8b[t]   = self.model.addConstr(Q_hc_corr[t] - (1-u_hc[t])*M - phi_hc[t], gb.GRB.LESS_EQUAL, 0)
                self.constraints.fancoil_8c[t]   = self.model.addConstr(phi_hc[t] - Q_hc_corr[t], gb.GRB.LESS_EQUAL, 0)
                self.constraints.fancoil_8d[t]   = self.model.addConstr(phi_hc[t] - u_hc[t]*M, gb.GRB.LESS_EQUAL, 0) 
            elif self.data.season == 'c':
                self.constraints.fancoil_8a[t]   = self.model.addConstr(Q_hc_corr[t] - self.data.k_hc_0 - self.data.k_hc_1*theta_hs[t] - self.data.k_hc_2*theta_i[t], gb.GRB.EQUAL, 0)
                self.constraints.fancoil_8b[t]   = self.model.addConstr(Q_hc_corr[t] - (1-u_hc[t])*M + phi_hc[t], gb.GRB.LESS_EQUAL, 0)
                self.constraints.fancoil_8c[t]   = self.model.addConstr(-phi_hc[t] - Q_hc_corr[t], gb.GRB.LESS_EQUAL, 0)
                self.constraints.fancoil_8d[t]   = self.model.addConstr(-phi_hc[t] - u_hc[t]*M, gb.GRB.LESS_EQUAL, 0) 
                   
        # Load tseries
        self.tseries.phi_hc      = np.zeros(len(self.data.T))
        self.tseries.phi_sol     = np.zeros(len(self.data.T))
        self.tseries.w_hp        = np.zeros(len(self.data.T))
        self.tseries.phi_hp      = np.zeros(len(self.data.T))
        self.tseries.theta_i     = np.zeros(len(self.data.T))
        self.tseries.theta_hs    = np.zeros(len(self.data.T))
        self.tseries.u_hp        = np.zeros(len(self.data.T))
        self.tseries.x_su        = np.zeros(len(self.data.T))
        self.tseries.u_hc        = np.zeros(len(self.data.T))
        self.tseries.H_ve        = np.zeros(len(self.data.T))
        self.tseries.Q_hp_mod    = np.zeros(len(self.data.T))
        self.tseries.Q_hp_max    = np.zeros(len(self.data.T))
        self.tseries.Q_hc_corr   = np.zeros(len(self.data.T))
        
        # Load gurobi exit codes
        self.info.codes = {1: 'LOADED', 
                           2: 'OPTIMAL', 
                           3: 'INFEASIBLE', 
                           4: 'INF_OR_UNBD', 
                           5: 'UNBOUNDED', 
                           6: 'CUTOFF', 
                           7: 'ITERATION_LIMIT', 
                           8: 'NODE_LIMIT', 
                           9: 'TIME_LIMIT', 
                           10: 'SOLUTION_LIMIT', 
                           11: 'INTERRUPTED', 
                           12: 'NUMERIC', 
                           13: 'SUBOPTIMAL', 
                           14: 'INPROGRESS', 
                           15: 'USER_OBJ_LIMIT'}
        
    
    def gains2nodes(self, phi_int, I_opa, I_gla):
        ks_gla   = self.data.ks_gla
        k_a      = self.data.k_a
        k_s      = self.data.k_s
        phi_sol = np.zeros(len(self.data.T))
        for t in self.data.T:          
            self.vardata.phi_sol[t] = ks_gla*I_gla[t] #transmitted component only 
            # distribute solar and internal heat gains to temperature nodes
            self.vardata.phi[t,0] = 0.5*phi_int[t] + k_s*phi_sol[t]
            self.vardata.phi[t,1] = (1 - k_a)*(0.5*phi_int[t] + (1-k_s)*phi_sol[t])
            self.vardata.phi[t,2] = k_a*(0.5*phi_int[t] + (1-k_s)*phi_sol[t])
            
    def heatPump(self, t1, t2):
        # Maximum heat flow rate
        if self.data.season == 'h':
            c0, c = self.data.coeff_qmax_h[0], self.data.coeff_qmax_h[1]
            k0, k = self.data.coeff_fcop_h[0], self.data.coeff_fcop_h[1]
            Q_hp_nom = self.data.Q_max_h
        else:
            c0, c = self.data.coeff_qmax_c[0], self.data.coeff_qmax_c[1]
            k0, k = self.data.coeff_fcop_c[0], self.data.coeff_fcop_c[1] 
            Q_hp_nom = -self.data.Q_max_c
        if len(c) == 9:
            x = [t1, t2, t1**2, t2**2, t1*t2, t1**3, t2**3, (t1**2)*t2, t1*(t2**2)]
        elif len(c) == 5:
            x = [t1, t2, t1**2, t2**2, t1*t2]
        elif len(c) == 2:
            x = [t1, t2]
        else:
            print('Error: number of coefficient is not coherent with length of polynomials')
        # Maximum heat flow rate
        qmax  = c0 + np.sum(np.multiply(x,c))
        Q_hp_max = qmax[0]*1000  # converted kW (logs) --> W (optimization)
        # COP correlation
        fcop  = k0 + np.sum(np.multiply([t1, t2],k))
        f_cop = fcop[0] 
        # By-pass correlations
#        Q_hp_max = Q_hp_nom
#        f_cop = 1/3.0
        return Q_hp_max, f_cop
    

    def update(self, fdata, cstate, *argv):
               
        # Read input data
        self.vardata.I_opa     = fdata['I_tot_opa']
        self.vardata.I_gla     = fdata['I_tot_gla']
        self.vardata.theta_e   = fdata['Te'] + self.data.ks_opa/self.data.A_aw*self.vardata.I_opa #sol-air temp
        self.vardata.theta_su  = fdata['Te']    # to be updated with formula of heat recovery
        self.vardata.phi_int   = self.data.phi_0*np.ones([len(fdata.index),])      
        self.vardata.theta_max = fdata['theta_max']
        self.vardata.theta_min = fdata['theta_min']
        self.vardata.w_pv      = fdata['W_pv']
        self.vardata.phi_hc_old = fdata['phi_hc_old']
        self.vardata.phi_hp_old = fdata['phi_hp_old']
        self.vardata.theta_i0     = cstate['Tm_0'];  theta_i0 = self.vardata.theta_i0  # TO BE UPDATED IN estimateCurrentState
        self.vardata.theta_s0     = cstate['Tm_0'];  theta_s0 = self.vardata.theta_s0  # TO BE UPDATED IN estimateCurrentState
        self.vardata.theta_m0     = cstate['Tm_0'];  theta_m0 = self.vardata.theta_m0  # TO BE UPDATED IN estimateCurrentState
        self.vardata.theta_hs0    = cstate['Ths_0']; theta_hs0 = self.vardata.theta_hs0
        self.vardata.theta_hp_out = cstate['Thp_out_avg']
        self.vardata.u_hp0        = cstate['u_hp0']; u_hp0 = self.vardata.u_hp0
        
         # Distribute heat gains to nodes
        self.gains2nodes(self.vardata.phi_int, self.vardata.I_opa, self.vardata.I_gla)
        season = argv[0]
        
        self.model.update()        
        
        # Right hand side (rhs) terms of the constraints (equations)         
        for t in self.data.T:
            # Building energy balance (rhs)           
            
            
            if (t==min(self.data.T)):
                self.constraints.building_3a[t].rhs = -self.data.H_ve*self.vardata.theta_e[t]-self.vardata.phi[t,0]-self.data.Ci*theta_i0
                self.constraints.building_3b[t].rhs = -self.vardata.phi[t,1]-self.data.H_trw*self.vardata.theta_e[t]-self.data.Cs*theta_s0
                self.constraints.building_3c[t].rhs = -self.vardata.phi[t,2]-self.data.H_trem*self.vardata.theta_e[t]-self.data.Cm*theta_m0
            else:
                self.constraints.building_3a[t].rhs = -self.data.H_ve*self.vardata.theta_e[t]-self.vardata.phi[t,0]
                self.constraints.building_3b[t].rhs = -self.vardata.phi[t,1]-self.data.H_trw*self.vardata.theta_e[t]
                self.constraints.building_3c[t].rhs = -self.vardata.phi[t,2]-self.data.H_trem*self.vardata.theta_e[t]
            # Thermal storage energy balance (rhs)           
            if (t==min(self.data.T)):
                self.constraints.storage_4a[t].rhs = self.data.C_hs*theta_hs0                
            else:
                self.constraints.storage_4a[t].rhs = 0                
            # Heat pump  
            # calculate heat pump performance (max heat flow rate and COP)
            self.vardata.Q_hp_max[t], self.vardata.f_cop[t] = self.heatPump(self.vardata.theta_e[t], self.vardata.theta_hp_out)
            # Q_hp_max is positive in heating season and negative in the cooling season, f_cop=1/cop always positive
            self.model.chgCoeff(self.constraints.heatpump_5a[t], self.variables.u_hp[t], -self.data.r_min*self.vardata.Q_hp_max[t])
            if season == 'h':
                self.model.chgCoeff(self.constraints.heatpump_5b[t], self.variables.u_hp[t], -(1-self.data.r_min)*self.vardata.Q_hp_max[t]) 
                self.model.chgCoeff(self.constraints.heatpump_5c[t], self.variables.phi_hp[t], -self.vardata.f_cop[t])   
            elif season == 'c':
                self.model.chgCoeff(self.constraints.heatpump_5b[t], self.variables.Q_hp_mod[t], -1)
                self.model.chgCoeff(self.constraints.heatpump_5b[t], self.variables.u_hp[t], (1-self.data.r_min)*self.vardata.Q_hp_max[t])
                self.model.chgCoeff(self.constraints.heatpump_5c[t], self.variables.phi_hp[t], self.vardata.f_cop[t])            
            if (t==min(self.data.T)):
                self.constraints.heatpump_5d[t].rhs = u_hp0
                self.constraints.heatpump_5f[t].rhs = 1 - u_hp0
            else:
                self.constraints.heatpump_5d[t].rhs = 0
                self.constraints.heatpump_5f[t].rhs = 1
            if season == 'h':
                self.model.chgCoeff(self.constraints.heatpump_5g[t], self.variables.x_su[t], self.data.r_min_start*self.vardata.Q_hp_max[t])
            elif season == 'c':
                self.model.chgCoeff(self.constraints.heatpump_5g[t], self.variables.x_su[t], -self.data.r_min_start*self.vardata.Q_hp_max[t])
                
            # Electrical energy balance (rhs)
            self.constraints.elbalance_6[t].rhs  = self.vardata.w_pv[t]
            # Thermal comfort (rhs)
            self.constraints.comfort_7a[t].rhs = self.vardata.theta_max[t]
            self.constraints.comfort_7b[t].rhs = self.vardata.theta_min[t]
 
        # Objective
#        self.data.pgamma, self.data.ugamma = 10e-3, 10e-5
        self.data.pgamma, self.data.ugamma = self.objectiveCoefficients()
        try:
            self.model.setObjective(gb.quicksum(self.data.pbuy*self.variables.w_buy[t] - self.data.psell*self.variables.w_sell[t] + 
                                                self.data.pgamma*(self.variables.delta_up[t] + self.variables.delta_dw[t]) for t in self.data.T) + 
                                    gb.quicksum(self.data.ugamma*(self.variables.phi_hc[t] - self.vardata.phi_hc_old[t])**2  +   
                                                self.data.ugamma*(self.variables.phi_hp[t] - self.vardata.phi_hp_old[t])**2 for t in range(self.data.blocked_steps)), 
                                    gb.GRB.MINIMIZE)
        except:
            # Useful because last optimization is not always accessible (e.g. when last hours were done in thermostat mode)
            self.model.setObjective(gb.quicksum(self.data.pbuy*self.variables.w_buy[t] - self.data.psell*self.variables.w_sell[t] + 
                                    self.data.pgamma*(self.variables.delta_up[t] + self.variables.delta_dw[t]) for t in self.data.T), 
                                    gb.GRB.MINIMIZE)
                                                       
        # Launch optimization         
        try:
            # Set time limit
            self.model.Params.TimeLimit = self.data.time_limit 
            # set params for bilinear optimization
#            self.model.params.NonConvex = 2
            #  Launch optimization
            self.model.optimize()   
            # Evaluate objective function components
            self.info.obj1 = gb.quicksum(self.data.pbuy*self.variables.w_buy[t].x - self.data.psell*self.variables.w_sell[t].x  for t in self.data.T)
            self.info.obj2 = gb.quicksum(self.data.pgamma*(self.variables.delta_up[t].x + self.variables.delta_dw[t].x) for t in self.data.T)
            self.info.obj3 = gb.quicksum(self.data.ugamma*(self.variables.phi_hc[t].x - self.vardata.phi_hc_old[t])**2 + self.data.ugamma*(self.variables.phi_hp[t].x - self.vardata.phi_hp_old[t])**2 for t in range(self.data.blocked_steps))
            self.info.obj = self.info.obj1 + self.info.obj2 + self.info.obj3
            print('Minimum value of objective function ', '{:.3f}'.format(self.model.objVal))
#            print('First component of objective function (OBJ1):', '{:.3f}'.format(self.info.obj1))
#            print('First component of objective function (OBJ2):', '{:.3f}'.format(self.info.obj2))
#            print('First component of objective function (OBJ3):', '{:.3f}'.format(self.info.obj3))
        except:
            self.info.message = self.info.codes[self.model.status]
            print('Exit criterion: ' + self.info.message)
    
#        # Load updated variables in tseries object (for controller check)
        for t in self.data.T:
            self.tseries.phi_hc[t]   = self.variables.phi_hc[t].x
            self.tseries.phi_hp[t]   = self.variables.phi_hp[t].x
            self.tseries.w_hp[t]     = self.variables.w_hp[t].x
            self.tseries.theta_i[t]  = self.variables.theta_i[t].x
            self.tseries.theta_hs[t] = self.variables.theta_hs[t].x
#            self.tseries.H_ve[t]     = self.variables.H_ve[t].x
            self.tseries.u_hp[t]     = self.variables.u_hp[t].x
            self.tseries.x_su[t]     = self.variables.x_su[t].x
            self.tseries.u_hc[t]     = self.variables.u_hc[t].x
            self.tseries.phi_sol[t]  = self.vardata.phi_sol[t]
            self.tseries.Q_hp_max[t] = self.vardata.Q_hp_max[t]
            self.tseries.Q_hc_corr[t] = self.variables.Q_hc_corr[t].x
            
    def objectiveCoefficients(self):
        # Set the coefficients of the objective function
        # c2 = weight of thermal discomfort compared to costs
        # c3 = weight of stability compared to costs
        num_max_hrs_discomfort = 2.0
        deltaT_discomfort = 1.0
        Q_hc_old = sum(abs(self.vardata.phi_hc_old.values))
        Q_hp_old = sum(abs(self.vardata.phi_hp_old.values))
        Q_old = max(Q_hc_old, Q_hp_old)
        if Q_old == 0:
            Q_old = 2*self.data.Q_max_h
        f_cop_mean = sum(self.vardata.f_cop.values())/len(self.vardata.f_cop.values())
        Fmax_1 = self.data.pbuy*Q_old*f_cop_mean
        Fmax_2 = self.data.hourly_steps*num_max_hrs_discomfort*deltaT_discomfort
        Fmax_3 = np.sum((0.1*abs(self.vardata.phi_hc_old.values[0:self.data.blocked_steps-1]) + 
                         0.1*abs(self.vardata.phi_hp_old.values[0:self.data.blocked_steps-1]))**2)       
#        Alpha is the factor between different components of the O.F.
#        Example: 
#        alpha = 2.0 in comfort mode means that the value of discomfort is 2 times the value of the operating costs and viceversa
        alpha = 2.0
        # COMFORT-ORIENTED control mode
        if self.data.mode == 'comf':
            c2 = alpha*(Fmax_1/Fmax_2)
            if Fmax_3 >0:
                c3 = 1/alpha*(Fmax_1/Fmax_3)
            else:
                c3 = 0
        # ECONOMIC-ORIENTED control mode
        elif self.data.mode == 'eco':
            c2 = 1/alpha*(Fmax_1/Fmax_2)
            if Fmax_3 >0:
                c3 = (1/alpha**2)*(Fmax_1/Fmax_3)
            else:
                c3 = 0    
        return c2, c3
        
            



#%%
class cthermostat:
    def __init__(self,delta_theta):

        self.fixdata     = obj()
        self.vardata     = obj()
        self.output      = obj()
        self.tseries     = obj()

        # Load fixdata (heat pump parameters)
        self.fixdata.delta_theta = delta_theta  
        
        # Initialize vardata (time-dependent variables)
        self.vardata.state         = {}  
        self.vardata.theta_i       = {}  
        self.vardata.theta_set     = {}
        
        # Initialize output variables
        self.output.onoff     = {};
        
        # Initialize time-series
        self.tseries.onoff    = np.zeros(0)
      

    def update(self, state, theta_i, theta_set, season):

        # Update vardata (time-dependent variables)
        self.vardata.state     = state;     state     = self.vardata.state   
        self.vardata.theta_i   = theta_i;   theta_i   = self.vardata.theta_i
        self.vardata.theta_set = theta_set; theta_set  = self.vardata.theta_set
        
        self.vardata.theta_min = theta_set - self.fixdata.delta_theta  ;     theta_min = self.vardata.theta_min
        self.vardata.theta_max = theta_set + self.fixdata.delta_theta  ;     theta_max = self.vardata.theta_max
        
        if season == 'h':
            if (theta_i < theta_min):
                onoff = 1
            elif(theta_i > theta_max):
                onoff = 0
            else:
                if (state==1):
                    onoff = 1
                else:
                    onoff = 0
        elif season == 'c':
            if (theta_i < theta_min):
                onoff = 0
            elif(theta_i > theta_max):
                onoff = 1
            else:
                if (state==1):
                    onoff = 1
                else:
                    onoff = 0      

        self.output.onoff    = onoff     
      
        # Store values in time-series objects
#        self.tseries.time     = np.append(self.tseries.time    ,1+len(self.tseries.time))
        self.tseries.onoff  = np.append(self.tseries.onoff ,self.output.onoff)  