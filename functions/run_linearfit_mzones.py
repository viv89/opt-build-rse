# -*- coding: utf-8 -*-
"""
Created on Jul 9 2020

@author: mazzi
"""
#%%
import numpy  as np
import pandas as pd

from sklearn.linear_model import LinearRegression

#%%

def run_fit_mzones(bdf,n,w,w0,t0,t00):

    tair1  = np.array(bdf['Tair1'])
    tair2  = np.array(bdf['Tair2'])
    text   = np.array(bdf['Text'])
    qh1    = np.array(bdf['Qh1'])
    qh2    = np.array(bdf['Qh2'])
    ih     = np.array(bdf['Ih'])
    ih1    = np.array(bdf['Ih1'])
    ih2    = np.array(bdf['Ih2'])
    tm_m   = np.array(bdf['Tm_meas'])
    tm_c   = np.array(bdf['Tm_calc'])

    qh  = qh1 + qh2
    dt1 = tair1-tm_c
    dt2 = tair2-tm_c

    # fit zone 1
    X1f = np.zeros((168*w,4*n+1))
    for i in range(n): X1f[:,i+0*n] = tm_c[t0-i:t0+168*w-i]
    for i in range(n): X1f[:,i+1*n] = text[t0-i:t0+168*w-i]-tm_c[t0-i:t0+168*w-i]
    for i in range(n): X1f[:,i+2*n] = qh[t0-i:t0+168*w-i]
    for i in range(n): X1f[:,i+3*n] = qh1[t0-i:t0+168*w-i]
    X1f[:,4*n] = ih1[t0:t0+168*w]
    y1f = dt1[t0:t0+168*w]
    G1f = LinearRegression().fit(X1f,y1f)
    # predict zone 1
    X1p=np.zeros((168*w0,4*n+1))
    for i in range(n): X1p[:,i+0*n] = tm_c[t00-i:t00+168*w0-i]
    for i in range(n): X1p[:,i+1*n] = text[t00-i:t00+168*w0-i]-tm_c[t00-i:t00+168*w0-i]
    for i in range(n): X1p[:,i+2*n] = qh[t00-i:t00+168*w0-i]
    for i in range(n): X1p[:,i+3*n] = qh1[t00-i:t00+168*w0-i]
    X1p[:,4*n] = ih1[t00:t00+168*w0]
    y1r = dt1[t00:t00+168*w0] + tm_m[t00:t00+168*w0]
    y1p = G1f.predict(X1p) + tm_c[t00:t00+168*w0]

    # fit zone 2
    X2f = np.zeros((168*w,4*n+1))
    for i in range(n): X2f[:,i+0*n] = tm_c[t0-i:t0+168*w-i]
    for i in range(n): X2f[:,i+1*n] = text[t0-i:t0+168*w-i]-tm_c[t0-i:t0+168*w-i]
    for i in range(n): X2f[:,i+2*n] = qh[t0-i:t0+168*w-i]
    for i in range(n): X2f[:,i+3*n] = qh2[t0-i:t0+168*w-i]
    X2f[:,4*n] = ih2[t0:t0+168*w]
    y2f = dt2[t0:t0+168*w]
    G2f = LinearRegression().fit(X2f,y2f)
    # predict zone 2
    X2p=np.zeros((168*w0,4*n+1))
    for i in range(n): X2p[:,i+0*n] = tm_c[t00-i:t00+168*w0-i]
    for i in range(n): X2p[:,i+1*n] = text[t00-i:t00+168*w0-i]-tm_c[t00-i:t00+168*w0-i]
    for i in range(n): X2p[:,i+2*n] = qh[t00-i:t00+168*w0-i]
    for i in range(n): X2p[:,i+3*n] = qh2[t00-i:t00+168*w0-i]
    X2p[:,4*n] = ih2[t00:t00+168*w0]
    y2r = dt2[t00:t00+168*w0] + tm_m[t00:t00+168*w0]
    y2p = G2f.predict(X2p) + tm_c[t00:t00+168*w0]

    return tm_m,tm_c,y1r,y1p,y2r,y2p