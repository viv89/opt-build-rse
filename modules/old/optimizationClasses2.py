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

        
        # Load sets
        self.data.T = range(settings['nhours_horizon']*settings['hourly_steps']);   T = self.data.T
        
        # Load output
        self.output.u = np.zeros(len(T))
        
        # Load parameters
        
        self.data.tau       = settings['timestep'];         tau = self.data.tau
        
        self.data.Cm        = optimal_parameters[0]/tau;      Cm       = self.data.Cm
        self.data.H_trem    = optimal_parameters[1];          H_trem   = self.data.H_trem
        self.data.H_tris    = optimal_parameters[2];          H_tris   = self.data.H_tris
        self.data.H_trms    = optimal_parameters[3];          H_trms   = self.data.H_trms
        self.data.H_trw     = optimal_parameters[4];          H_trw    = self.data.H_trw
        self.data.H_ve      = optimal_parameters[5];          H_ve     = self.data.H_ve   
#        self.data.k_conv    = optimal_parameters[6];          k_conv   = self.data.k_conv
    
        self.data.Am        = build_data.A_m        # Am = self.data.Am      
        self.data.A_floor   = build_data.A_floor    # A_floor  = self.data.A_floor
        self.data.A_aw      = build_data.A_walls
        self.data.k0_s      = build_data.k0_s       # 0.6
        self.data.k1_s      = build_data.k1_s       # 0.024 
        self.data.k2_s      = build_data.k2_s       # 0.190
        
#        scrivere i parametri in modo che sia COP = f(T_ext)  
        self.data.season       = hvac_data['season']
        self.data.Q_max_h      = hvac_data['Q_hp_nom_heat']                   
        self.data.coeff_qmax_h = hvac_data['params_hp_heat_qmax']     
        self.data.coeff_fcop_h = hvac_data['params_hp_heat_fcop']
#        self.data.Q_max_c      = hvac_data['Q_hp_nom_cool']           
#        self.data.coeff_qmax_c = hvac_data['params_hp_cool_qmax'] 
#        self.data.coeff_fcop_c = hvac_data['params_hp_cool_fcop']
        self.data.V_hs         = hvac_data['V_hs']      # m3
        self.data.UA_hs        = hvac_data['UA_hs']     # W/K 
        self.data.theta_hs_min = hvac_data['theta_hs_min']     # °C
        self.data.theta_hs_max = hvac_data['theta_hs_max']     # °C
        self.data.k_fc_c       = hvac_data['k_fancoil_c']     # W/K
        self.data.k_fc_h       = hvac_data['k_fancoil_h']     # W/K      
        self.data.r_min        = hvac_data['r_min']     # -
        self.data.r_min_start  = hvac_data['r_min_start']     # -
        
        self.data.rho       = 970                    # kg/m3
        self.data.cpw       = 4183                   # J/(kg K)
        self.data.M_hs      = self.data.rho*self.data.V_hs*self.data.cpw  # J/K
        self.data.C_hs  = self.data.M_hs/tau 
        UA_hs = self.data.UA_hs      # W/K
        C_hs = self.data.C_hs  # W/K
        
        self.data.pbuy   = settings['pbuy']/1000*(tau/3600)
        self.data.psell  = settings['psell']/1000*(tau/3600)
        self.data.pgamma = settings['pgamma']
        
        self.vardata.phi        = {}
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
        self.variables.theta_i  = {};  theta_i = self.variables.theta_i 
        self.variables.theta_s  = {};  theta_s = self.variables.theta_s  
        self.variables.theta_m  = {};  theta_m = self.variables.theta_m 
        self.variables.theta_hs = {}; theta_hs = self.variables.theta_hs 
        self.variables.u_hc     = {};     u_hc = self.variables.u_hc
        self.variables.phi_hc   = {};   phi_hc = self.variables.phi_hc  
        self.variables.delta_up = {}; delta_up = self.variables.delta_up
        self.variables.delta_dw = {}; delta_dw = self.variables.delta_dw 
        self.variables.x_su     = {};     x_su = self.variables.x_su          

        for t in T:
            self.variables.w_buy[t]    = self.model.addVar(lb = 0)
            self.variables.w_sell[t]   = self.model.addVar(lb = 0)
            self.variables.u_hp[t]     = self.model.addVar(lb = 0 , ub = 1, vtype = gb.GRB.BINARY)
            self.variables.w_hp[t]     = self.model.addVar(lb = -gb.GRB.INFINITY)
            self.variables.phi_hp[t]   = self.model.addVar(lb = -gb.GRB.INFINITY)
            self.variables.Q_hp_mod[t] = self.model.addVar(lb = 0)
            self.variables.theta_i[t]  = self.model.addVar(lb = -gb.GRB.INFINITY)
            self.variables.theta_s[t]  = self.model.addVar(lb = -gb.GRB.INFINITY)
            self.variables.theta_m[t]  = self.model.addVar(lb = -gb.GRB.INFINITY)
            self.variables.theta_hs[t] = self.model.addVar(lb = self.data.theta_hs_min , ub = self.data.theta_hs_max)
            self.variables.u_hc[t]     = self.model.addVar(lb = 0 , ub = 1)
            self.variables.phi_hc[t]   = self.model.addVar(lb = -gb.GRB.INFINITY)
            self.variables.delta_up[t] = self.model.addVar(lb = 0)
            self.variables.delta_dw[t] = self.model.addVar(lb = 0) 
            self.variables.x_su[t]     = self.model.addVar(lb = 0 , ub = 1, vtype = gb.GRB.BINARY)

        self.model.update()

        # Objective
        self.model.setObjective(0, gb.GRB.MINIMIZE)

        # Constraints
        self.constraints.building_3a = {}
        self.constraints.building_3b = {}
        self.constraints.building_3c = {}
        self.constraints.storage_4   = {}
        self.constraints.storage_4b  = {}
        self.constraints.heatpump_5a = {}
        self.constraints.heatpump_5b = {}
        self.constraints.heatpump_5c = {}
        self.constraints.heatpump_5d = {}
        self.constraints.heatpump_5e = {}
        self.constraints.heatpump_5f = {}
        self.constraints.elbalance_6 = {}
        self.constraints.comfort_7a  = {}
        self.constraints.comfort_7b  = {}
        self.constraints.fancoil_8a  = {}
        self.constraints.fancoil_8b  = {}
        

        for t in T:
            # Building energy balance
            self.constraints.building_3a[t]     = self.model.addConstr(-(H_ve+H_tris)*theta_i[t]+H_tris*theta_s[t]+phi_hc[t], gb.GRB.EQUAL , 0)
            self.constraints.building_3b[t]     = self.model.addConstr(H_tris*theta_i[t]-(H_tris+H_trw+H_trms)*theta_s[t]+H_trms*theta_m[t], gb.GRB.EQUAL , 0)
            if (t==min(T)):
                self.constraints.building_3c[t] = self.model.addConstr(H_trms*theta_s[t]-(H_trms+H_trem+Cm)*theta_m[t], gb.GRB.EQUAL, 0)
            else:
                self.constraints.building_3c[t] = self.model.addConstr(H_trms*theta_s[t]-(H_trms+H_trem+Cm)*theta_m[t]+Cm*theta_m[t-1], gb.GRB.EQUAL, 0)
            # Thermal storage energy balance
            if (t==min(T)):
                self.constraints.storage_4[t]   = self.model.addConstr((C_hs+UA_hs)*theta_hs[t]-UA_hs*theta_i[t]-phi_hp[t]+phi_hc[t], gb.GRB.EQUAL, 0)
                
            else:
                self.constraints.storage_4[t]   = self.model.addConstr((C_hs+UA_hs)*theta_hs[t]-UA_hs*theta_i[t]-phi_hp[t]+phi_hc[t]-C_hs*theta_hs[t-1], gb.GRB.EQUAL, 0)
           
            # Heat pump capacity and COP
            self.constraints.heatpump_5a[t]     = self.model.addConstr(phi_hp[t] - u_hp[t] - Q_hp_mod[t], gb.GRB.EQUAL, 0)   # incomplete: updated at line 228
            self.constraints.heatpump_5b[t]     = self.model.addConstr(Q_hp_mod[t] - u_hp[t], gb.GRB.LESS_EQUAL, 0)
            self.constraints.heatpump_5c[t]     = self.model.addConstr(w_hp[t] + phi_hp[t], gb.GRB.EQUAL, 0)   # incomplete: updated at line 229
            
            if (t==min(T)):
                self.constraints.heatpump_5d[t] = self.model.addConstr(u_hp[t] - x_su[t], gb.GRB.LESS_EQUAL, 0)
            else:
                self.constraints.heatpump_5d[t] = self.model.addConstr(u_hp[t] - u_hp[t-1] - x_su[t], gb.GRB.LESS_EQUAL, 0)
            self.constraints.heatpump_5e[t]     = self.model.addConstr(x_su[t] - u_hp[t], gb.GRB.LESS_EQUAL, 0)
            self.constraints.heatpump_5f[t]     = self.model.addConstr(x_su[t] - phi_hp[t], gb.GRB.LESS_EQUAL, 0)  #incomplete: updated at line XXX
            
            # Electrical energy balance
            self.constraints.elbalance_6[t]     = self.model.addConstr(w_hp[t] + w_sell[t] - w_buy[t], gb.GRB.EQUAL, 0)
            # Thermal comfort
            self.constraints.comfort_7a[t]      = self.model.addConstr(theta_i[t] - delta_up[t], gb.GRB.LESS_EQUAL, 1)
            self.constraints.comfort_7b[t]      = self.model.addConstr(theta_i[t] + delta_dw[t], gb.GRB.GREATER_EQUAL, 1)
            # Fan coils
            if self.data.season == 'h':
                self.constraints.fancoil_8a[t]   = self.model.addConstr(phi_hc[t] - u_hc[t], gb.GRB.EQUAL, 0)
                self.constraints.fancoil_8b[t]   = self.model.addConstr(phi_hc[t] - self.data.k_fc_h*theta_hs[t] + self.data.k_fc_h*theta_i[t], gb.GRB.LESS_EQUAL, 0)
            else:
                self.constraints.fancoil_8a[t]   = self.model.addConstr(phi_hc[t] - u_hc[t], gb.GRB.EQUAL, 0)
                self.constraints.fancoil_8b[t]   = self.model.addConstr(-phi_hc[t] - self.data.k_fc_c*theta_i[t] + self.data.k_fc_c*theta_hs[t], gb.GRB.LESS_EQUAL, 0)
        
        if self.data.season == 'h':
            self.constraints.storage_4b[0] = self.model.addConstr(theta_hs[0] - theta_hs[max(T)], gb.GRB.LESS_EQUAL, 0)
        else:
            self.constraints.storage_4b[0] = self.model.addConstr(theta_hs[max(T)] - theta_hs[0], gb.GRB.LESS_EQUAL, 0)
            
            
#        self.model.optimize()           
        timeLimit = 30
        try:
            oldSolutionLimit = self.model.Params.SolutionLimit
            self.model.Params.SolutionLimit = 1
            self.model.optimize()
            self.model.Params.TimeLimit = timeLimit - self.model.getAttr(gb.GRB.Attr.Runtime)
            self.model.Params.SolutionLimit = oldSolutionLimit - self.model.Params.SolutionLimit
            self.model.optimize()
        except (AttributeError, Exception) as e:
            print('Caught: ' + e.message)
        
        # Load tseries
        self.tseries.phi_hc      = np.zeros(len(self.data.T))
        self.tseries.w_hp        = np.zeros(len(self.data.T))
        self.tseries.phi_hp      = np.zeros(len(self.data.T))
        self.tseries.theta_i     = np.zeros(len(self.data.T))
        self.tseries.theta_hs    = np.zeros(len(self.data.T))
        self.tseries.u_hp        = np.zeros(len(self.data.T))
        self.tseries.x_su        = np.zeros(len(self.data.T))
        
    
    def gains2nodes(self, phi_int, I_opa, I_gla, T_e, T_sky):
        A_m      = self.data.Am
        H_tr_w   = self.data.H_trw
        k0       = self.data.k0_s
        k1       = self.data.k1_s
        k2       = self.data.k2_s
        A_t      = 4.5*self.data.A_floor
        S_aw     = self.data.A_aw
        phi_sol = np.zeros(len(self.data.T))
        for t in self.data.T:          
            phi_sol[t] = k0*I_gla[t] + k1/S_aw*I_opa[t] + k2*(T_e[t] - T_sky[t]) # simplified formula
            # distribute solar and internal heat gains to temperature nodes
            self.vardata.phi[t,0] = 0.5*phi_int[t]
            self.vardata.phi[t,1] = (1 - A_m/A_t - H_tr_w/(9.1*A_t))*(0.5*phi_int[t] + phi_sol[t])
            self.vardata.phi[t,2] = A_m/A_t*(0.5*phi_int[t] + phi_sol[t])
            
    def heatPump(self, t1, t2):
        # Maximum heat flow rate
        c0 = self.data.coeff_qmax_h[0]
        c  = self.data.coeff_qmax_h[1]
        if len(c) == 9:
            x = [t1, t2, t1**2, t2**2, t1*t2, t1**3, t2**3, (t1**2)*t2]
        elif len(c) == 5:
            x = [t1, t2, t1**2, t2**2, t1*t2]
        elif len(c) == 2:
            x = [t1, t2]
        else:
            print('Error: number of coefficient is not coherent with length of polynomials')
        # Maximum heat flow rate
        qmax  = c0 + np.sum(np.multiply(x,c))
        Q_hp_max = qmax[0]*1000  # converted kW (logs) --> W (optimization)
        
        # f_cop (to be updated with correlation)
        k0 = self.data.coeff_fcop_h[0]
        k  = self.data.coeff_fcop_h[1]
        fcop  = k0 + np.sum(np.multiply([t1, t2],k))
        f_cop = fcop[0]        
        return Q_hp_max, f_cop

    def update(self, fdata, cstate, *argv):
        
        # Read input data
        self.vardata.theta_e   = fdata['Te']
        self.vardata.theta_su  = fdata['Te']    # to be updated with formula of heat recovery
        self.vardata.theta_sky = fdata['Te']    # to be updated with T_sky
        self.vardata.I_opa     = fdata['I_tot_opa']
        self.vardata.I_gla     = fdata['I_tot_gla']
        self.vardata.phi_int   = 0*fdata['I_tot_gla']  # to be updated according to calibrated parameters
        self.vardata.theta_max = fdata['theta_max']
        self.vardata.theta_min = fdata['theta_min']
        self.vardata.w_pv      = fdata['W_pv']
        
        self.vardata.theta_m0   = cstate['Tm_0'];  theta_m0 = self.vardata.theta_m0
        self.vardata.theta_hs0  = cstate['Ths_0']; theta_hs0 = self.vardata.theta_hs0
        self.vardata.theta_hp_out = cstate['Thp_out_avg']
        self.vardata.u_hp0      = cstate['u_hp0']; u_hp0 = self.vardata.u_hp0
        
        season = argv[0]
        if season == 'h':
            for t in self.data.T:
                # calculate heat pump performance (max heat flow rate and COP)
                self.vardata.Q_hp_max[t], self.vardata.f_cop[t] = self.heatPump(self.vardata.theta_e[t], self.vardata.theta_hp_out) 
                self.vardata.Q_hc_max[t] = self.vardata.Q_hp_max[t]
                # update coefficients to linear constraints               
                self.model.chgCoeff(self.constraints.heatpump_5a[t], self.variables.u_hp[t], -self.data.r_min*self.vardata.Q_hp_max[t])
                self.model.chgCoeff(self.constraints.heatpump_5c[t], self.variables.phi_hp[t], -self.vardata.f_cop[t])                  
        elif season == 'c':
            for t in self.data.T:
                # calculate heat pump performance (max heat flow rate and COP)
                self.vardata.Q_hp_max[t], self.vardata.f_cop[t] = self.heatPump(self.vardata.theta_e[t], self.vardata.theta_hp_out[t]) 
                self.vardata.Q_hc_max[t] = -self.vardata.Q_hp_max[t]
                # update coefficients to linear constraints
                self.model.chgCoeff(self.constraints.heatpump_5a[t], self.variables.u_hp[t], self.data.r_min*self.vardata.Q_hp_max[t])
                self.model.chgCoeff(self.constraints.heatpump_5a[t], self.variables.Q_hp_mod[t], 1)
                self.model.chgCoeff(self.constraints.heatpump_5c[t], self.variables.phi_hp[t], self.vardata.f_cop[t])
   
        # Objective
        self.model.setObjective(gb.quicksum(self.data.pbuy*self.variables.w_buy[t] - self.data.psell*self.variables.w_sell[t] + self.data.pgamma*(self.variables.delta_up[t] + self.variables.delta_dw[t]) for t in self.data.T), gb.GRB.MINIMIZE)
                
        # Distribute heat gains to nodes
        self.gains2nodes(self.vardata.phi_int, self.vardata.I_opa, self.vardata.I_gla, self.vardata.theta_e, self.vardata.theta_sky)
        
        # Right hand side (rhs) terms of the constraints (equations)       
        for t in self.data.T:
            # Building energy balance (rhs)
            self.constraints.building_3a[t].rhs = -self.data.H_ve*self.vardata.theta_e[t]-self.vardata.phi[t,0]
            self.constraints.building_3b[t].rhs = -self.vardata.phi[t,1]-self.data.H_trw*self.vardata.theta_e[t]
            if (t==min(self.data.T)):
                self.constraints.building_3c[t].rhs = -self.vardata.phi[t,2]-self.data.H_trem*self.vardata.theta_e[t]-self.data.Cm*theta_m0
            else:
                self.constraints.building_3c[t].rhs = -self.vardata.phi[t,2]-self.data.H_trem*self.vardata.theta_e[t]
            # Thermal storage energy balance (rhs)           
            if (t==min(self.data.T)):
                self.constraints.storage_4[t].rhs = self.data.C_hs*theta_hs0                
            else:
                self.constraints.storage_4[t].rhs = 0                
            # Heat pump  (rhs)
            self.model.chgCoeff(self.constraints.heatpump_5b[t], self.variables.u_hp[t], -(1-self.data.r_min)*self.vardata.Q_hp_max[t])
            if (t==min(self.data.T)):
                self.constraints.heatpump_5d[t].rhs = u_hp0
            else:
                self.constraints.heatpump_5d[t].rhs = 0
            self.model.chgCoeff(self.constraints.heatpump_5f[t], self.variables.x_su[t], self.data.r_min_start*self.vardata.Q_hp_max[t])
            # Electrical energy balance (rhs)
            self.constraints.elbalance_6[t].rhs  = self.vardata.w_pv[t]
            # Thermal comfort (rhs)
            self.constraints.comfort_7a[t].rhs = self.vardata.theta_max[t]
            self.constraints.comfort_7b[t].rhs = self.vardata.theta_min[t]
            # Fancoils (rhs)
            self.model.chgCoeff(self.constraints.fancoil_8a[t], self.variables.u_hc[t], -self.vardata.Q_hc_max[t])

        self.model.optimize()
        
#        # Load updated variables in tseries object (for controller check)
        for t in self.data.T:
            self.tseries.phi_hc[t]   = self.variables.phi_hc[t].x
            self.tseries.phi_hp[t]   = self.variables.phi_hp[t].x
            self.tseries.w_hp[t]     = self.variables.w_hp[t].x
            self.tseries.theta_i[t]  = self.variables.theta_i[t].x
            self.tseries.theta_hs[t] = self.variables.theta_hs[t].x
            self.tseries.u_hp[t]     = self.variables.u_hp[t].x
            self.tseries.x_su[t]     = self.variables.x_su[t].x



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