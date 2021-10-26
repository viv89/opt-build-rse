# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:05:56 2020

@author: vivijac14771
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#from solarprocess import solarProcessor
#import pyswarms
#from pyswarms.single.global_best import GlobalBestPSO as gbpso
#from pyswarms.utils.plotters import plot_cost_history
#import sklearn
#from sklearn import linear_model
#from sklearn.metrics import mean_squared_error, r2_score
#from functions.io import estimateCurrentState


class obj(object):
    '''
        A small class which can have attributes set
    '''
    pass

# Calibration of RC parameters using PSO
    
class calipso:
    
    def __init__(self, building_props, df_ext, df_air, df_Qhc, df_irrad, cali_set):
        
        
#% INITIALIZATION -----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
     
        self.fixdata = obj()    # fixed parameters
        self.auxdata = obj()    # auxiliary parameters
        self.nompars = obj()    # nominal (initial) RC parameters
        self.arrays  = obj()
        self.history = obj()    # database of temperatures and heat flow rates
        self.training = obj()
        self.testing = obj()
        self.calset  = obj()    # calibration settings
        self.calpars = obj()    # calibrated RC parameters  
        self.info    = obj()       
        
        # Read building properties
        self.fixdata.A_heated  = building_props['A_heated']    # Heated/Cooled floor area
        self.fixdata.perimeter = building_props['perimeter']   # Perimeter of external walls
        self.fixdata.n_floors  = building_props['n_floors']    # Number of heated/Cooled floors
        self.fixdata.U_walls   = building_props['U_walls']     # thermal transmittance of the walls
        self.fixdata.U_windows = building_props['U_windows']   # thermal transmittance of the windows
        self.fixdata.U_roof    = building_props['U_roof']      # thermal transmittance of the roof
        self.fixdata.U_ground  = building_props['U_ground']    # thermal transmittance of the ground
        self.fixdata.A_windows = building_props['A_windows']   # windows area per each orientation
#        self.fixdata.n_zones   = building_props['n_zones']     # number of thermal zones
        self.fixdata.ACR       = building_props['ACR']         # air change rate
        self.fixdata.structure = building_props['weight']
        self.fixdata.HVAC      = building_props['HVAC']
        self.auxdata.phi_0     = building_props['avg_heat_gain']    # [W] average heat gain due to human activities
        
        # Auxiliary parameters
        self.auxdata.R_at = 4.0
        self.auxdata_h_tr_is = 3.45
        self.auxdata_h_tr_ms = 9.1
        self.auxdata.height = 3
        self.auxdata.cp_air = 1000
        self.auxdata.rho_air = 1.2
        
        self.auxdata.opaque_area = self.auxdata.height*self.fixdata.n_floors*np.array(self.fixdata.perimeter) - np.array(self.fixdata.A_windows)
        self.auxdata.glazed_area = np.array(self.fixdata.A_windows)
        
        k_conv_min, k_conv_max = 0.0, 1.0
        if self.fixdata.HVAC == 'radiant_surfaces':
            k_conv_min, k_conv_max = 0.1, 0.9
        elif self.fixdata.HVAC == 'radiators':
            k_conv_min, k_conv_max = 0.3, 0.7  
        elif self.fixdata.HVAC == 'fancoils':
            k_conv_min, k_conv_max = 0.9, 1.0
        
        self.auxdata.k_conv_nom = np.mean([k_conv_min,k_conv_max])
        self.auxdata.k_conv_min = k_conv_min
        self.auxdata.k_conv_max = k_conv_max
        
#% ESTIMATE NOMINAL PARAMETERS FROM BUILDING DESCRIPTION --------------------------------------------
#---------------------------------------------------------------------------------------------------
    
        self.auxdata.A_t = self.auxdata.R_at*self.fixdata.A_heated
        self.auxdata.V_heated = self.fixdata.A_heated*self.auxdata.height
        
        # (1) Coupling conductance H_tr_is [W/K]
        self.nompars.H_tr_is =  self.auxdata_h_tr_is*self.auxdata.A_t    
        
        # (2) Transmittance of glazed elements H_tr_w [W/K]
        self.nompars.H_tr_w = self.fixdata.U_windows*np.sum(self.fixdata.A_windows)
        
        # Transmittance of opaque elements H_tr_op [W/K]
        self.auxdata.A_walls = self.auxdata.height*self.fixdata.n_floors*np.sum(self.fixdata.perimeter) - np.sum(self.fixdata.A_windows)      
        self.auxdata.A_floor = self.fixdata.A_heated/self.fixdata.n_floors
        self.auxdata.H_tr_op = self.fixdata.U_walls*self.auxdata.A_walls + (self.fixdata.U_roof + self.fixdata.U_ground)*self.auxdata.A_floor
        
        self.auxdata.fgc   = 1  # free gains coefficient
        self.auxdata.k0_s  = 0.6
        self.auxdata.k1_s  = 0.024 
        self.auxdata.k2_s  = 0.190
               
        # Dynamic parameters 
        # (3) Internal heat capacity of the building zone Cm [J/K]
        # Effective mass area [m2]
        if self.fixdata.structure == 'very light':
            self.nompars.C_m = 80000*self.auxdata.A_t
            self.auxdata.A_m = 2.5*self.auxdata.A_t
        elif self.fixdata.structure == 'light':
            self.nompars.C_m = 110000*self.auxdata.A_t
            self.auxdata.A_m = 2.5*self.auxdata.A_t
        elif self.fixdata.structure == 'medium':
            self.nompars.C_m = 165000*self.auxdata.A_t
            self.auxdata.A_m = 2.5*self.auxdata.A_t       
        elif self.fixdata.structure == 'heavy':
            self.nompars.C_m = 260000*self.auxdata.A_t
            self.auxdata.A_m = 3*self.auxdata.A_t 
        elif self.fixdata.structure == 'very heavy':
            self.nompars.C_m = 370000*self.auxdata.A_t
            self.auxdata.A_m = 3.5*self.auxdata.A_t 
        
        # (4) Coupling conductance H_tr_ms [W/K]
        self.nompars.H_tr_ms = self.auxdata_h_tr_ms*self.auxdata.A_m
        
        # (5) Coupling conductance H_tr_ms [W/K]
        self.nompars.H_tr_em = 1/(1/self.auxdata.H_tr_op - 1/self.nompars.H_tr_ms)

        # (6) Ventilation conductance H_ve [W/K]
        self.nompars.H_ve = self.auxdata.rho_air*self.auxdata.cp_air*(self.fixdata.ACR*self.auxdata.V_heated)/3600
             
        # Calibration parameters
        self.calset.start       = 0
        self.calset.stop        = len(df_air)
        self.calset.steps       = int(self.calset.stop - self.calset.start)
        self.calset.trainSteps  = int(cali_set['training_part']*self.calset.steps)
        self.calset.trainEnd    = self.calset.start + self.calset.trainSteps
        self.calset.timeStep    = cali_set['tau']
        self.calset.duration    = (self.calset.stop - self.calset.start)*self.calset.timeStep/3600       
        self.calset.population  = cali_set['population']
        self.calset.c1          = cali_set['c1']
        self.calset.c2          = cali_set['c2']       
        self.calset.DW          = np.array(cali_set['DW']) 
        self.calset.maxiter     = cali_set['maxiter']
        self.calset.w           = cali_set['w']
        self.calset.maxloops    = cali_set['maxloops']
        self.calset.zoom        = cali_set['zoomFactor']
        self.calset.adj         = cali_set['adj']
        
        self.calset.x_nom = [self.nompars.C_m, self.nompars.H_tr_em, self.nompars.H_tr_is, 
                             self.nompars.H_tr_ms, self.nompars.H_tr_w, self.nompars.H_ve, 
                             self.auxdata.k_conv_nom, self.auxdata.fgc, self.auxdata.phi_0]
        
        # Define calibration domain
        self.calset.x0 = self.calset.x_nom
        self.calset.dims  = len(cali_set['DW'])
        self.calset.lb    = np.ones([self.calset.dims,]) - self.calset.DW      
        self.calset.ub    = np.ones([self.calset.dims,]) + self.calset.DW      
 
        self.info.parshist = np.zeros([self.calset.maxloops,self.calset.dims])
#% SET BOUNDARY CONDITIONS FOR BUILDING MODEL CALIBRATION -------------------------------------------
#---------------------------------------------------------------------------------------------------       
        
        # Initialise history 
        self.history = pd.DataFrame()
#        self.history.index = df_ext.index
        self.history['date']      = df_ext.index.values
        self.history['Text']      = df_ext.Te.values
        self.history['Tsup']      = df_ext.Te.values      # modify in case there is heat recovery on mechanical ventilation system      
        self.history['Ih']        = df_ext.ghi.values
        self.history['Qhc']       = df_Qhc.values # df_Qhc.sum(axis = 1, skipna = True).values
        self.history['Ti_meas_avg'] = df_air.mean(axis = 1, skipna = True).values          
        self.history['I_tot_gla'] = np.sum(np.multiply(df_irrad, self.auxdata.glazed_area),axis=1).values       # W
        self.history['I_tot_opa'] = np.sum(np.multiply(df_irrad, self.auxdata.opaque_area),axis=1).values       # W
        # initialize internal heat gains
        self.history['phi_int']   = 0*self.history['Text'].values
        self.history['phi_sol']   = 0*self.history['Text'].values
        # initialize calculated indoor temperature
        self.history['Ti_calc_nom']   = 0*self.history['Text'].values
        self.history['Ti_calc_opt']   = 0*self.history['Text'].values
        
        # set date as index after conversion to datetime format
        self.history['date'] = pd.to_datetime(self.history['date'])
        self.history.set_index('date', inplace=True)    
        self.history = self.history.tz_localize('Europe/Rome', ambiguous=False, nonexistent='shift_backward') 
        self.history.index = df_ext.index
        
        # Estimate initial state
        self.calset.t_m0        = self.history['Ti_meas_avg'][0]                
  

#% CALIBRATION PROCESS ----------------------------------------------------------------------------   
           
        
    def update(self):
        
        # create training and testing datasets for calibration
        train_rng = self.history.index[0:self.calset.trainSteps]
        test_rng  = self.history.index[self.calset.trainSteps+1:] 
        
        # Call dataframes with indoor air temperature           
        Ti_meas_avg = self.history['Ti_meas_avg']
        
        # Calibration test over new data with given (old) parameters
#        if any(arg):
#            self.calset.x_old = arg[0]
#            self.calset.t_m0  = arg[1] 
#            Ti_calc_old_train, _ = self.runsim(self.calset.x_old, self.history.loc[train_rng], 
#                                               self.calset.t_m0,  self.calset.timeStep)
#            Ti_calc_old_test, _  = self.runsim(self.calset.x_old, self.history.loc[test_rng], 
#                                               self.calset.t_m0,  self.calset.timeStep)
            

#        self.calset.x_nom = arg[0]
        self.calset.x0    = self.calset.x_nom
        self.calset.lb    = np.ones([self.calset.dims,]) - self.calset.DW      
        self.calset.ub    = np.ones([self.calset.dims,]) + self.calset.DW 
        
        # STEP 1 ------ evaluate RC model on training dataset with nominal parameters
        # ---------------------------------------------------------------------------
        Ti_calc_nom_train, _ = self.runsim(self.calset.x_nom, self.history.loc[train_rng], 
                                           self.calset.t_m0,  self.calset.timeStep)
        Ti_calc_nom_test, _  = self.runsim(self.calset.x_nom, self.history.loc[test_rng], 
                                           self.calset.t_m0,  self.calset.timeStep)

        self.info.rmse_nom_train = self.rmse(Ti_calc_nom_train[:,0], Ti_meas_avg.loc[train_rng].values)
        self.info.rmse_nom_test  = self.rmse(Ti_calc_nom_test[:,0], Ti_meas_avg.loc[test_rng].values)
        print('RMSE over training period with nominal values equal to ', '{:.3f}'.format(self.info.rmse_nom_train), 'K')
        
        # STEP 2 ------ calibrate model using PSO over training dataset and iteratively 
        # ---------------------------------------------------------------------------
              
        # Define calibration options
        err, tol = 1.0, 0.3
        maxloops = self.calset.maxloops   
        # Initialise positions
#        d  = self.calset.ub - self.calset.lb
        x0 = np.ones((self.calset.population,self.calset.dims))
#        x0 = self.calset.lb + np.multiply(d,np.random.rand(self.calset.population,self.calset.dims)) 
        # Initialise variables
        self.info.err_hist = np.zeros(maxloops)
        self.info.err_hist[0] =  self.info.rmse_nom_train
        loop = 1
        
        # Run calibration loops
        while err>tol and loop<=maxloops:                   
            optpars, Ti_calc_test = self.calibrate(x0, Ti_meas_avg, train_rng, test_rng)           
            Ti_calc_train, Tm_calc_train = self.runsim(optpars, self.history.loc[train_rng], self.calset.t_m0, self.calset.timeStep)
            err              = self.rmse(Ti_calc_train[:,0], Ti_meas_avg[train_rng].values, adj = self.calset.adj)
#            Ti_weighted  = self.weightzonetemps(Ti_meas_avg.values, Ti_calc, Ti_zones.values, Ti_meas_avg.index) 
#            self.updatedomain(optpars, loop)
            self.info.err_hist[loop-1] = err
            self.info.parshist[loop-1,:] = optpars  
            x_opt = np.divide(optpars, self.calset.x_nom) 
            x0 = np.tile(x_opt,(self.calset.population,1)) #+ 0.1*np.multiply(d,np.random.rand(self.calset.population,1))
            print('RMSE over training period at the end of calibration loop ', loop, ' equal to ', '{:.3f}'.format(err), 'K')
            loop = loop + 1
        
        Ti_calc_test, _      = self.runsim(optpars, self.history.loc[test_rng], Tm_calc_train[-1], self.calset.timeStep)
        self.info.best_loop  = np.argmin(self.info.err_hist[0:loop-1])
        self.info.optpars    = self.info.parshist[self.info.best_loop,:]
        self.info.rmse_train = max(0,min(self.info.err_hist))
    
        self.info.rmse_test  = self.rmse(Ti_calc_test[:,0], Ti_meas_avg[test_rng].values, adj = self.calset.adj)
        
        print('RMSE over testing period with nominal values equal to ', '{:.3f}'.format(self.info.rmse_nom_test), 'K')
        print('------------------------------------------------------------' , 
              'Building model calibration completed with RMSE = ', 
              '{:.3f}'.format(self.info.rmse_train), ' K over training period and RMSE = ',
              '{:.3f}'.format(self.info.rmse_test), ' K over testing period', 
              '------------------------------------------------------------' )    
#   
        return    


#%% AUXILIARY FUNCTIONS ---------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
    
    def calibrate(self, x0, Ti_meas, train_rng, test_rng): 
        #------------------------------------------------------
        # x_opt has relative values (adimensional)
        # optpars has absolute values
        #------------------------------------------------------
        # Set boundary conditions fr calibration     
        hist     = self.history.loc[train_rng] 
        T_m0     = self.calset.t_m0
        tau      = self.calset.timeStep
        Ti_train = Ti_meas.loc[train_rng].values         
        # Calibrate adimenional parameters over training period
        rmse, x_opt = self.mypso(x0, hist, T_m0, tau, Ti_train)          
        # Calculate indoor temperature profile over testing period with optimal parameters
        optpars = np.multiply(x_opt, self.calset.x_nom) # valori assoluti
        Ti_calc_test, _ = self.runsim(optpars, self.history.loc[test_rng], self.calset.t_m0, self.calset.timeStep)        
        return optpars, Ti_calc_test
     
                
    def mypso(self, x0, hist, T_m0, tau, Ti_train):        
        # Calibration parameters       
        itmax = self.calset.maxiter
        N     = self.calset.population
        D     = self.calset.dims
        w     = self.calset.w*np.ones((itmax,1))
        c1    = self.calset.c1
        c2    = self.calset.c2
        lb    = self.calset.lb
        ub    = self.calset.ub
        x     = self.calset.x0     
        ## Problem and velocity bounds
        q=(ub-lb)/4 #initial velocities are 1/4 of parameter space size                  
        # Random initialization of positions and velocities
        x = x0
        v = np.multiply(q, np.random.rand(N,D))    # v0    
        # Evaluate objective for all particles
        f = self.costfun(x, hist, T_m0, tau, Ti_train) 
        # Find gbest and pbest (in this case coincides with x)
        fgbest, igbest = np.min(f), np.argmin(f)
        gbest = x[igbest,:] # best position in the swarm (1x2 array)
        pbest = x         # best position reached by each individual in the swarm (Nx2 array)
        fpbest = f
#        mem = np.zeros((itmax,D+2))
        for it in range(itmax):
            # Update velocities and positions
            v = w[it]*v + c1*np.random.rand()*(pbest-x) + c2*np.random.rand()*(gbest*np.ones((N,D))-x)
            x = x + v
            # Mirror particles out of bounds
            self.mirror(x, lb, ub)
            # Evaluate o.f.
            f = self.costfun(x, hist, T_m0, tau, Ti_train)
            # Find global best
            minf, iminf = np.min(f), np.argmin(f)
            if minf<=fgbest:
                fgbest = minf
                gbest = x[iminf,:]
            # Find personal bests
            inewpb=np.where(f<=fpbest)[0]
            np.put(pbest, inewpb, x[inewpb,:])
            np.put(fpbest, inewpb, f[inewpb])
            # Save PSO convergence data
#            mem[it,:] = [gbest[0], gbest[1], fgbest]      
        rmse, x_opt = fgbest, gbest    
        return rmse, x_opt
     
        
    def costfun(self, pos, hist, T_m0, tau, Ti_meas):
        npop = np.size(pos,0)
        f    = np.zeros(npop)       
        # calculate RMSE between measured and calculated indoor temp
        for p in range(0,npop):
            x = np.multiply(pos[p,:], self.calset.x_nom)
            Ti_calc, _ = self.runsim(x, hist, T_m0, tau)
            f[p] = self.rmse(Ti_calc[:,0], Ti_meas, adj = self.calset.adj)           
        return f
    
    
    def mirror(self, x, lb, ub):   
        # Find particles out of the domain
        delta_lb = x - lb
        delta_ub = ub - x   
        below_lb = np.where(delta_lb>0)
        above_ub = np.where(delta_ub>0)
        # Mirror them back to domain
        np.put(x, below_lb, lb+delta_lb)
        np.put(x, above_ub, ub-delta_ub) 
    
        
    def updatedomain(self, optpars, it):
        self.calset.x0 = optpars
        self.calset.DW = self.calset.DW*(self.calset.zoom**it)
        self.calset.lb = np.ones([self.calset.dims,]) - self.calset.DW
        self.calset.ub = np.ones([self.calset.dims,]) + self.calset.DW
        return
          
        
    def runsim(self, x, hist, T_m0, tau):                   
        # x = parameters
        # hist = boundary conditions
        # T_m0 = initial temperature
        # tau = timestep       
        # create array A
        A = self.createArrayLeft(x, tau)            
        k0 = x[7]*self.auxdata.k0_s 
        k1 = x[7]*self.auxdata.k1_s
        k2 = x[7]*self.auxdata.k2_s   # check k2     
        deT_er = -3 # calculate T_SKY = f(Text)
        # internal heat gains
        L = len(hist.index)
        phi_0   = x[8]
        phi_int = phi_0*np.ones([L,]) # + amp*np.cos((2*np.pi/(period*3600))*time + sfas)
        # simplified formula for solar heat gains  
        phi_sol_gla = hist['I_tot_gla'].values
        phi_sol_opa = hist['I_tot_opa'].values
        phi_sol = k0*phi_sol_gla + k1/self.auxdata.A_walls*phi_sol_opa + k2*deT_er 
        # distribute solar and internal heat gains to temperature nodes
        phi_ia = 0.5*phi_int
        phi_st = (1 - self.auxdata.A_m/self.auxdata.A_t - self.nompars.H_tr_w/(9.1*self.auxdata.A_t))*(0.5*phi_int + phi_sol)
        phi_m  = self.auxdata.A_m/self.auxdata.A_t*(0.5*phi_int + phi_sol)   
        # boundary conditions belonging to input dataset        
        phi_hc = hist['Qhc'].values           
        T_e    = hist['Text'].values
        T_sup  = hist['Tsup'].values       
        # initialize variables
        b = np.array([3,1])
#        L = len(T_e)
        Ti_calc, Tm_calc = np.zeros([L,1]), np.zeros([L,1])        
        for t in np.arange(0,L):
            phi   = [phi_ia[t], phi_st[t], phi_m[t], phi_hc[t]]
            theta = [T_e[t], T_sup[t]]            
            b = self.createArrayRight(x, phi, theta, T_m0, tau)
            t_nodes = np.linalg.solve(A, b)
            Ti_calc[t] = t_nodes[0]
            T_m0       = t_nodes[2] 
            Tm_calc[t] = t_nodes[2]
        return Ti_calc, Tm_calc


    def createArrayLeft(self, x, tau):        
        C_m     = x[0]
        H_tr_em = x[1]
        H_tr_is = x[2]
        H_tr_ms = x[3]
        H_tr_w  = x[4]   
        H_ve    = x[5]
#        k_conv     = x[6]              
        # Create arrays 
        A = np.zeros([3,3])
        A[0,0] = -(H_tr_is + H_ve)
        A[0,1] = H_tr_is
        A[1,0] = H_tr_is
        A[1,1] = -(H_tr_is + H_tr_w + H_tr_ms) #-C_s/tau
        A[1,2] = H_tr_ms
        A[2,1] = H_tr_ms
        A[2,2] = -C_m/tau - H_tr_em - H_tr_ms        
        return A


    def createArrayRight(self, x, phi, theta, T_m0, tau):     
        C_m     = x[0]
        H_tr_em = x[1]
#        H_tr_is = x[2]
#        H_tr_ms = x[3]
        H_tr_w  = x[4]   
        H_ve    = x[5]        
        k_conv  = x[6]      
        phi_ia  = phi[0]
        phi_st  = phi[1]
        phi_m   = phi[2]
        phi_hc  = phi[3]       
        T_e     = theta[0]
        T_sup   = theta[1]               
        # Create arrays 
        b = np.zeros([3,1])
        b[0] = -H_ve*T_sup - phi_ia - k_conv*phi_hc         
        b[1] = -H_tr_w*T_e - phi_st - (1-k_conv)*phi_hc  #- C_s*T_s0/tau   
        b[2] = -H_tr_em*T_e - phi_m - C_m*T_m0/tau         
        return b
        
    
    def rmse(self, x, y, **kwargv):
        # no correction by default if no correction factor is provided
        adj = 1 
        # read correction factor if present
        for key, value in kwargv.items():
            adj = value
        # vector with square of residuals
        errors = (x - y) ** 2
        N = len(x)
        # correction coefficients
        correction_coefficients = np.linspace(1,adj,N)
        # corrected vector with square of residuals
        corrected_errors = np.multiply(errors, correction_coefficients)
        # corrected RMSE
        rmse_corr = np.sqrt(sum(corrected_errors)/N)     
        
        # objective function   
        alpha = 0.0
        r2 = r2_score(x, y)
        fo = rmse_corr - alpha*r2
        
        self.info.rmse = rmse_corr
        self.info.r2 = r2
        self.info.fo = fo
        
        return fo
    


#%% CALIBRATE HEAT PUMP PARAMETERS
        

class calicop:

    def __init__(self, logs, hvac_data):    
        # Initialize objects
        self.history = obj()    # logged datasets
        self.nompars = obj()    # nominal parameters
        self.calpars = obj()    # calibrated parameters
        self.info    = obj()
        self.array   = obj()       
        # Nominal power (Te = 7°C, Thp_out = 45°C)
        if hvac_data['season'] == 'h':
            self.nompars.Qhp_nom = hvac_data['Q_hp_nom_heat']/1000
        else:
            self.nompars.Qhp_nom = hvac_data['Q_hp_nom_cool']/1000
        
        self.history = logs[['Te','Thp_out','q_spv','rps','q_hp','Wel_hp']]
        
        # Subsets of logs        
        self.history2   = self.history[abs(self.history['q_hp'])>2.0]
        self.history3   = self.history2[self.history2['q_spv'].notna()]      
        self.history100 = self.history3[self.history3['rps']>59.5] # working points at full load (frequency > 59.5 Hz)
        # Calculate amount of useful data for regression
        self.info.len_hist100 = len(self.history100.index)
        self.info.len_hist3   = len(self.history3.index)
        # If no data is available, use estimation
        if self.info.len_hist100 < 5:
            self.history100 = self.history3[self.history3['rps']>30.0]
            self.history100.q_hp = np.divide(self.history100.q_hp, self.history100.rps)*60
            print('No full-load data available in the selected period: data has been estimated')    
            
            
    def update_qmax(self):
        # Define input and output variables for regression models
        self.array.Xmax = self.history100[['Te','Thp_out']].values   
        self.array.ymax = self.history100[['q_hp']].values
        # Calculate coefficients according to polynomial 
        self.calpars.coeff_qmax, self.info.score_qmax = self.heatPumpCoefficients(self.array.Xmax, 
                                                                                  self.array.ymax, 
                                                                                  2)
        return
    
    def update_qmod(self):
        # Calculate maximum heat flow rate (qmax) for all points
        self.array.X_new = self.history3[['Te','Thp_out']].values        
        # Add calculated qmax to the log dataframe as a new column
        self.history3['q_hp_max'] = self.heatPumpCapacity(self.array.X_new, self.calpars.coeff_qmax)
        ## Linear regression to calculate partial load heat flow rate
        self.history3['q_hp_mod'] = self.history3['q_hp_max']*self.history3['rps']/60
        # Input and output data for regression 
        self.array.X_q = self.history3[['q_hp_mod']].values   
        self.array.y_q = self.history3[['q_hp']].values
        # Linear regression
        reg_q = LinearRegression().fit(self.array.X_q, self.array.y_q)  # fit_intercept=False
        self.calpars.coeff_q = [reg_q.intercept_,reg_q.coef_[0]]    
        # Calculate regression performance
        self.info.score_q = reg_q.score(self.array.X_q, self.array.y_q)   
        return
    
    def update_cop(self):        
    # ottenere esattamente i coefficienti che servono nell'ottimizzatore!!   
        # Linear regression to calculate COP
        self.history3['cop']   = abs(self.history3['q_hp'])/self.history3['Wel_hp'] # cop>0
        self.history3['f_cop'] = self.history3['Wel_hp']/abs(self.history3['q_hp'])
        # Input and output data for regression
        self.array.X_w = self.history3[['Te','Thp_out']].values   #'rps'
        self.array.y_w = self.history3[['f_cop']].values
        # Linear regression
        reg_w = LinearRegression().fit(self.array.X_w, self.array.y_w)
        self.calpars.coeff_w = [reg_w.intercept_,reg_w.coef_[0]]    
        # Calculate regression performance
        self.info.score_w = reg_w.score(self.array.X_w, self.array.y_w)        
        return
               
        
    def buildDataset(self, xa, order):
        if order == 1:
            xb = xa
        elif order == 2:
            a20 = np.zeros((xa.shape[0],1))
            a02 = np.zeros((xa.shape[0],1))
            a11 = np.zeros((xa.shape[0],1))
            a20[:,0] = xa[:,0]**2
            a02[:,0] = xa[:,1]**2
            a11[:,0] = np.multiply(xa[:,0],xa[:,1])
            xb =  np.hstack((xa, a20, a02, a11))
        elif order == 3:
            a20 = np.zeros((xa.shape[0],1))
            a02 = np.zeros((xa.shape[0],1))
            a11 = np.zeros((xa.shape[0],1))
            a30 = np.zeros((xa.shape[0],1))
            a03 = np.zeros((xa.shape[0],1))
            a21 = np.zeros((xa.shape[0],1))
            a12 = np.zeros((xa.shape[0],1))
            a20[:,0] = xa[:,0]**2
            a02[:,0] = xa[:,1]**2
            a11[:,0] = np.multiply(xa[:,0],xa[:,1])
            a30[:,0] = xa[:,0]**3
            a03[:,0] = xa[:,1]**3
            a21[:,0] = np.multiply(a20[:,0],xa[:,1])
            a12[:,0] = np.multiply(xa[:,0],a02[:,0])
            xb = np.hstack((xa, a20, a02, a11, a30, a03, a21, a12))
        else:
            print('Error: maximum order is 3')        
        return xb 
    
    
    def heatPumpCoefficients(self, xa, y, order): 
        xb = self.buildDataset(xa, order)
        # Fit dataset xb with linear regression    
        reg = LinearRegression().fit(xb, y)
        # Get coefficients and score
        c0  = reg.intercept_ 
        coeff = reg.coef_[0]
        c = [c0,coeff]
        score = reg.score(xb, y)    
        return c, score
    
    
    def heatPumpCapacity(self, Xa, coefficients):
        c0 = coefficients[0]
        c  = coefficients[1]
        if len(c) == 9:
            order = 3
        elif len(c) == 5:
            order = 2
        elif len(c) == 2:
            order = 1
        else:
            print('Error: number of coefficient is not coherent with length of polynomials')
        Xb = self.buildDataset(Xa, order)
        y  = c0 + np.sum(np.multiply(Xb,c), axis = 1)
        return y
        
    

    
#%% CALIBRATE HVAC SYSTEM PARAMETERS        

#    Calibrazione di altri parametri dell'impianto come dispersioni termiche serbatoio e 
class calihvac:

    def __init__(self, logs, hvac_data):
         
        self.history = obj()    # logged datasets
        self.nompars = obj()    # nominal parameters
        self.calpars = obj()    # calibrated parameters
        self.info    = obj()
        self.array   = obj()

        self.history = logs[abs(logs['q_fc'])>1.0][['m_fc','Tst_high','Tst_low','q_fc','Ti_A', 'Ti_B','Ti_C','Ti_D']]
        self.history['Tst_avg'] = (self.history['Tst_high']+self.history['Tst_low'])/2
        self.history['Ti_avg'] = (self.history['Ti_A']+self.history['Ti_B']+self.history['Ti_C']+self.history['Ti_D'])/4
        self.history['deT'] = self.history['q_fc']/(4.183*(self.history['m_fc'])/3600)
        
        return
        
    def update_fc(self, hvac_data, fdata):

        self.info.Tmin = min(fdata.theta_min) - 0.5
        self.info.Tmax = max(fdata.theta_max) + 0.5
        self.history2 = self.history[self.history['Ti_avg']>self.info.Tmin]
        self.history2 = self.history[self.history['Ti_avg']<self.info.Tmax]
        self.array.X_fc = self.history2[['Tst_avg','Ti_avg']].values   
        self.array.y_fc = 1000*abs(self.history2[['q_fc']].values)
        if self.array.X_fc.any():
            reg_fc = LinearRegression().fit(self.array.X_fc, self.array.y_fc) 
        else:
            if hvac_data['season'] == 'h':
                self.info.Tmin, self.info.Tmax = 17, 23
                self.history = self.history[self.history['Ti_avg']>self.info.Tmin]
            elif hvac_data['season'] == 'c':    
                self.info.Tmin, self.info.Tmax = 23, 29
                self.history = self.history[self.history['Ti_avg']<self.info.Tmax]           
            self.array.X_fc = self.history[['Tst_avg','Ti_avg']].values   
            self.array.y_fc = 1000*abs(self.history[['q_fc']].values)
            reg_fc = LinearRegression().fit(self.array.X_fc, self.array.y_fc) 
        self.calpars.coeff_fc = [reg_fc.intercept_, reg_fc.coef_[0]] 
        self.info.score_fc = reg_fc.score(self.array.X_fc, self.array.y_fc)
        
        return
    
    
        

        
        
        



    
    


