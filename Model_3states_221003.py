
import pickle, os    # packages to open pickle files
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from pylab import *
import numpy as np
import scipy.io as sio
import mpmath as mp
from numpy import diff
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import interpolate
import math, tqdm
from tifffile import imread
from skimage import img_as_bool
from scipy.optimize import fsolve
from scipy.special import lambertw
import scipy.special as sc
from scipy import integrate
from scipy.integrate import odeint
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize, minimize_scalar, rosen, rosen_der, least_squares
#import time as t
import random

sns.set_palette("PuBuGn_d")

# linear fit
def linear_fit(phi20,a):
    return a*phi20
       
# Code to solve the nonlinear 3-state system for Bra dynamics

def step2(t,t0,t2,t3):
    return 1 * (t < t0 or t > t3) + 0 * (t > t2 and t < t3)

def step(t,t0,t2):
    return 1 * (t < t0 or t > t2) 

# define phi0=1-phi20, where phi0=phi1(0) and phi20=phi2(0)
def phi0(phi20):
    return 1-phi20

# phi1(t) for mu=+
def fun(phi0,K,T1):
    return K*lambertw((1/K)*phi0*np.exp(-T1+phi0/K)).real
    #return K*lambertw((1/K)*(1-phi20)*np.exp(-T1+(1-phi20)/K)).real
    
# phi1(t) for mu=-
def fun2(phi0,K,T1):
    return (K/lambertw((K/phi0)*np.exp(T1+K/phi0))).real
    #return K*lambertw((1/K)*(1-phi20)*np.exp(-T1+(1-phi20)/K)).real    


# phi2(t) for mu=0 and nu=0 
def phi2_fit_nosignal(phi20,alpha, T1):
    phi0=1-phi20
    return (1-phi0)*np.exp(-alpha*T1)+(phi0/(alpha-1))*(np.exp(-T1)-np.exp(-alpha*T1))
   

# phi2(t) for mu=+ and nu=+ 
def phi2_fit0(phi20,alpha, K, T1):
    phi0=1-phi20
    if alpha == 1:
        return (fun(phi0,K,T1)/phi0)*(1-phi0+phi0*np.log(phi0/K)-phi0*np.log(fun(phi0,K,T1)/K))
    else: 
        return (1/(alpha-1))*(fun(phi0,K,T1)+(-1+alpha-alpha*phi0)*pow(fun(phi0,K,T1)/phi0,alpha))
    
# Multi function to fit phi2 at 3 different time points with 4 parameters
def combo_phi2_fit0(comboData,alpha, K, T24, T48):
    
    extract1 = comboData[:5] # first data
    extract2 = comboData[5:10] # second data
    extract3 = comboData[10:15] # second data
    
    result24 = phi2_fit0(extract1,alpha, K, T24)
    result48 = phi2_fit0(extract2,alpha, K, T48)
    result72 = phi2_fit0(extract3,alpha, K, T48+(T48-T24))
    
    return np.concatenate([result24,result48,result72])



# phi2(t) for mu=0 and nu=+ 
def phi2_fit1(phi20,alpha, K, T1):
    mp.dps = 80; mp.pretty = True
    phi0=1-phi20
    hypfunc = np.frompyfunc(mp.hyp2f1,4,1)
    #return float(hypfunc(2,1+alpha,2+alpha,(K*np.exp(T1)+phi0)/phi0).real)
    #return (1/phi0)*(K*(K*np.exp(T1)+phi0)*hypfunc(2,1+alpha,2+alpha,(K*np.exp(T1)+phi0)/phi0)/(1+alpha)+pow((K+phi0)/(K*np.exp(T1)+phi0),alpha)*((1+K)*phi0+K*alpha*(K+phi0)*hypfunc(1,1+alpha,2+alpha,K/phi0+1)/(1+alpha))).real
    
    size_phi20=np.size(phi20,0) # size array
    f2=[] # f2 function
    f1=[] # f1(0) function
    result = []
    for i in range(0,size_phi20):
        #f2.append(float(hypfunc(2,1+alpha,2+alpha,(K*np.exp(T1)+phi0[i])/phi0[i]).real))
        #f1.append(float(hypfunc(1,1+alpha,2+alpha,(K+phi0[i])/phi0[i]).real))
        f2.append(hypfunc(2,1+alpha,2+alpha,(K*np.exp(T1)+phi0[i])/phi0[i]))
        f1.append(hypfunc(1,1+alpha,2+alpha,(K+phi0[i])/phi0[i]))
        result.append(float((1/phi0[i])*(K*(K*np.exp(T1)+phi0[i])*f2[i]/(1+alpha)+pow((K+phi0[i])/(K*np.exp(T1)+phi0[i]),alpha)*((1+K)*phi0[i]+(K*alpha*(K+phi0[i])*f1[i])/(1+alpha))).real))
        
    return result
    

# phi2(t) for mu=+ and nu=0 
def phi2_fit2(phi20,alpha, K, T1):
    phi0=1-phi20
    Nint=100 # number of points of the time integral
    time_array=np.linspace(0,T1,Nint)
    
    if T1 == 0:
        return 1-phi0
    else:
        size_phi20=np.size(phi20,0) # size array
        a=[] # f2 function
        for i in range(0,size_phi20): # loop for phi
            a.append(np.exp(-alpha*T1)*(1-phi0[i])+integrate.simpson((fun(phi0[i],K,time_array)*np.exp(alpha*(time_array-T1)))/(1+fun(phi0[i],K,time_array)/K),time_array))  
        return np.asarray(a)
    
# phi2(t) for mu=- and nu=- 
def phi2_fit3(phi20,alpha, K, T1):
    phi0=1-phi20
    if alpha == 1:
        return (fun2(phi0,K,T1)/phi0)*(1-phi0-phi0*np.log(K/phi0)+phi0*np.log(K/fun2(phi0,K,T1)))
    else: 
        return (1/(alpha-1))*(fun2(phi0,K,T1)-(1-alpha+alpha*phi0)*pow(fun2(phi0,K,T1)/phi0,alpha))
    
    
# phi2(t) for mu=- and nu=0 
def phi2_fit4(phi20,alpha, K, T1):
    phi0=1-phi20
    Nint=100 # number of points of the time integral
    time_array=np.linspace(0,T1,Nint)
    
    if T1 == 0:
        return 1-phi0
    else:
        size_phi20=np.size(phi20,0) # size array
        a=[] # f2 function
        for i in range(0,size_phi20): # loop for phi
            a.append(np.exp(-alpha*T1)*(1-phi0[i])+integrate.simpson((pow(fun2(phi0[i],K,time_array),2)*np.exp(alpha*(time_array-T1)))/(K+fun2(phi0[i],K,time_array)),time_array))  
        return np.asarray(a)  
    
# phi2(t) for mu=0 and nu=- 
def phi2_fit5(phi20,alpha, K, T1):
    phi0=1-phi20
    gfun = K*np.exp(T1)+phi0
    return (np.exp(-alpha*T1)/(alpha-1))*(np.exp(T1*(alpha-1))*gfun-(1+K+alpha*(phi0-1))*pow(gfun/(K+phi0),alpha))
        
# phi2(t) for mu=+ and nu=- 
def phi2_fit6(phi20,alpha, K, T1):
    phi0=1-phi20
    return (1/alpha)*(K-np.exp((alpha/K)*(fun(phi0,K,T1)-fun(phi0,K,0)))*(K+alpha*(phi0-1)))

# phi2(t) for mu=- and nu=+ 
def phi2_fit7(phi20,alpha, K, T1):
    phi0=1-phi20
    return -fun2(phi0,K,T1)+np.exp(-alpha*K/fun2(phi0,K,T1))*(np.exp(alpha*K/phi0)+K*alpha*(sc.expi(K*alpha/fun2(phi0,K,T1))-sc.expi(K*alpha/phi0)))
                

# We define three different states phi1: pluripotent, phi2: T+, phi3:T-

dir_path = os.path.abspath(os.getcwd()) # get path

# get paths for each csv file

# Experiment 211130

output_unmixing_analysis_24h_GFP_211130 =os.path.join(dir_path,'output_24_GFP_211130.csv')
output_unmixing_analysis_48h_GFP_211130 =os.path.join(dir_path,'output_48_GFP_211201.csv')
output_unmixing_analysis_72h_GFP_211130 =os.path.join(dir_path,'output_72_GFP_211202.csv')

# Experiment 220118

output_unmixing_analysis_24h_GFP_220118 =os.path.join(dir_path,'output_24_GFP_220118.csv')
output_unmixing_analysis_48h_GFP_220118 =os.path.join(dir_path,'output_48_GFP_220119.csv')
output_unmixing_analysis_72h_GFP_220118 =os.path.join(dir_path,'output_72_GFP_220120.csv')

# Experiment 220125

output_unmixing_analysis_24h_GFP_220125 =os.path.join(dir_path,'output_24_GFP_220125.csv')
output_unmixing_analysis_48h_GFP_220125 =os.path.join(dir_path,'output_48_GFP_220126.csv')
output_unmixing_analysis_72h_GFP_220125 =os.path.join(dir_path,'output_72_GFP_220127.csv')

# Experiment 220503

output_unmixing_analysis_24h_GFP_220503 =os.path.join(dir_path,'output_24_GFP_220503.csv')
output_unmixing_analysis_48h_GFP_220503 =os.path.join(dir_path,'output_48_GFP_220504.csv')
output_unmixing_analysis_72h_GFP_220503 =os.path.join(dir_path,'output_72_GFP_220505.csv')


# Experiment 220517

#output_unmixing_analysis_24h_GFP_220517 =os.path.join(dir_path,'output_24_GFP_220517.csv')
#output_unmixing_analysis_48h_GFP_220517 =os.path.join(dir_path,'output_48_GFP_220518.csv')
#output_unmixing_analysis_72h_GFP_220517 =os.path.join(dir_path,'output_72_GFP_220519.csv')

# Experiment 220705

output_unmixing_analysis_24h_GFP_220705 =os.path.join(dir_path,'output_24_plate_2_GFP_220705.csv')
output_unmixing_analysis_48h_GFP_220705 =os.path.join(dir_path,'output_48_plate_2_GFP_220706.csv')
output_unmixing_analysis_72h_GFP_220705 =os.path.join(dir_path,'output_72_plate_2_GFP_220707.csv')

# Experiment 220712

output_unmixing_analysis_24h_GFP_220712=os.path.join(dir_path,'output_24_plate_2_GFP_220712.csv')
output_unmixing_analysis_48h_GFP_220712=os.path.join(dir_path,'output_48_plate_2_GFP_220713.csv')
output_unmixing_analysis_72h_GFP_220712 =os.path.join(dir_path,'output_72_plate_2_GFP_220714.csv')

# Experiment 220809

output_unmixing_analysis_24h_GFP_220809 =os.path.join(dir_path,'output_24_plate_2_GFP_220809.csv')
output_unmixing_analysis_48h_GFP_220809 =os.path.join(dir_path,'output_48_plate_2_GFP_220810.csv')
output_unmixing_analysis_72h_GFP_220809 =os.path.join(dir_path,'output_72_plate_2_GFP_220811.csv')



# get data

data_GFP_24h_211130=pd.read_csv(output_unmixing_analysis_24h_GFP_211130)
data_GFP_48h_211130=pd.read_csv(output_unmixing_analysis_48h_GFP_211130)
data_GFP_72h_211130=pd.read_csv(output_unmixing_analysis_72h_GFP_211130)

data_GFP_24h_220118=pd.read_csv(output_unmixing_analysis_24h_GFP_220118)
data_GFP_48h_220118=pd.read_csv(output_unmixing_analysis_48h_GFP_220118)
data_GFP_72h_220118=pd.read_csv(output_unmixing_analysis_72h_GFP_220118)

data_GFP_24h_220125=pd.read_csv(output_unmixing_analysis_24h_GFP_220125)
data_GFP_48h_220125=pd.read_csv(output_unmixing_analysis_48h_GFP_220125)
data_GFP_72h_220125=pd.read_csv(output_unmixing_analysis_72h_GFP_220125)

data_GFP_24h_220503=pd.read_csv(output_unmixing_analysis_24h_GFP_220503)
data_GFP_48h_220503=pd.read_csv(output_unmixing_analysis_48h_GFP_220503)
data_GFP_72h_220503=pd.read_csv(output_unmixing_analysis_72h_GFP_220503)

#data_GFP_24h_220517=pd.read_csv(output_unmixing_analysis_24h_GFP_220517)
#data_GFP_48h_220517=pd.read_csv(output_unmixing_analysis_48h_GFP_220517)
#data_GFP_72h_220517=pd.read_csv(output_unmixing_analysis_72h_GFP_220517)

data_GFP_24h_220705=pd.read_csv(output_unmixing_analysis_24h_GFP_220705)
data_GFP_48h_220705=pd.read_csv(output_unmixing_analysis_48h_GFP_220705)
data_GFP_72h_220705=pd.read_csv(output_unmixing_analysis_72h_GFP_220705)

data_GFP_24h_220712=pd.read_csv(output_unmixing_analysis_24h_GFP_220712)
data_GFP_48h_220712=pd.read_csv(output_unmixing_analysis_48h_GFP_220712)
data_GFP_72h_220712=pd.read_csv(output_unmixing_analysis_72h_GFP_220712)

data_GFP_24h_220809=pd.read_csv(output_unmixing_analysis_24h_GFP_220809)
data_GFP_48h_220809=pd.read_csv(output_unmixing_analysis_48h_GFP_220809)
data_GFP_72h_220809=pd.read_csv(output_unmixing_analysis_72h_GFP_220809)

# Change the name of some columns for practical purposes

data_GFP_24h_220705['Condition'] = data_GFP_24h_220705['Condition'].str.replace('24h Ctrl','24h control')
data_GFP_24h_220712['Condition'] = data_GFP_24h_220712['Condition'].str.replace('24h Ctrl','24h control')
data_GFP_24h_220809['Condition'] = data_GFP_24h_220809['Condition'].str.replace('24h Fgf8','24h control') # It's not Fgf8 but actually the Ctrl (change in plates from 1 to 2 DF)

data_GFP_48h_220705['Condition'] = data_GFP_48h_220705['Condition'].str.replace('48h Ctrl','48h control')
data_GFP_48h_220712['Condition'] = data_GFP_48h_220712['Condition'].str.replace('48h Ctrl','48h control')
data_GFP_48h_220809['Condition'] = data_GFP_48h_220809['Condition'].str.replace('48h Fgf8','48h control') # It's not Fgf8 but actually the Ctrl (change in plates from 1 to 2 DF)

data_GFP_72h_220705['Condition'] = data_GFP_72h_220705['Condition'].str.replace('72h Ctrl','72h control')
data_GFP_72h_220712['Condition'] = data_GFP_72h_220712['Condition'].str.replace('72h Ctrl','72h control')
data_GFP_72h_220809['Condition'] = data_GFP_72h_220809['Condition'].str.replace('72h Fgf8','72h control') # It's not Fgf8 but actually the Ctrl (change in plates from 1 to 2 DF)


# Extrapolate maximum GFP value for normalization 

popt_211130,pcov_211130 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_211130[data_GFP_24h_211130['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220118,pcov_220118 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220118[data_GFP_24h_220118['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220125,pcov_220125 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220125[data_GFP_24h_220125['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220503,pcov_220503 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220503[data_GFP_24h_220503['Condition']=='24h control']['GFP fluo'])[0:4])
#popt_220517,pcov_220517 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220517[data_GFP_24h_220517['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220705,pcov_220705 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220705[data_GFP_24h_220705['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220712,pcov_220712 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220712[data_GFP_24h_220712['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220809,pcov_220809 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220809[data_GFP_24h_220809['Condition']=='24h control']['GFP fluo'])[0:4]) 
      
    
# I normalise by the intensity

data_GFP_24h_211130['GFP fluo norm']=data_GFP_24h_211130['GFP fluo']/popt_211130
data_GFP_48h_211130['GFP fluo norm']=data_GFP_48h_211130['GFP fluo']/popt_211130
data_GFP_72h_211130['GFP fluo norm']=data_GFP_72h_211130['GFP fluo']/popt_211130

data_GFP_24h_220118['GFP fluo norm']=data_GFP_24h_220118['GFP fluo']/popt_220118
data_GFP_48h_220118['GFP fluo norm']=data_GFP_48h_220118['GFP fluo']/popt_220118
data_GFP_72h_220118['GFP fluo norm']=data_GFP_72h_220118['GFP fluo']/popt_220118

data_GFP_24h_220125['GFP fluo norm']=data_GFP_24h_220125['GFP fluo']/popt_220125
data_GFP_48h_220125['GFP fluo norm']=data_GFP_48h_220125['GFP fluo']/popt_220125
data_GFP_72h_220125['GFP fluo norm']=data_GFP_72h_220125['GFP fluo']/popt_220125

data_GFP_24h_220503['GFP fluo norm']=data_GFP_24h_220503['GFP fluo']/popt_220503
data_GFP_48h_220503['GFP fluo norm']=data_GFP_48h_220503['GFP fluo']/popt_220503
data_GFP_72h_220503['GFP fluo norm']=data_GFP_72h_220503['GFP fluo']/popt_220503

#data_GFP_24h_220517['GFP fluo norm']=data_GFP_24h_220517['GFP fluo']/popt_220517
#data_GFP_48h_220517['GFP fluo norm']=data_GFP_48h_220517['GFP fluo']/popt_220517
#data_GFP_72h_220517['GFP fluo norm']=data_GFP_72h_220517['GFP fluo']/popt_220517

data_GFP_24h_220705['GFP fluo norm']=data_GFP_24h_220705['GFP fluo']/popt_220705
data_GFP_48h_220705['GFP fluo norm']=data_GFP_48h_220705['GFP fluo']/popt_220705
data_GFP_72h_220705['GFP fluo norm']=data_GFP_72h_220705['GFP fluo']/popt_220705

data_GFP_24h_220712['GFP fluo norm']=data_GFP_24h_220712['GFP fluo']/popt_220712
data_GFP_48h_220712['GFP fluo norm']=data_GFP_48h_220712['GFP fluo']/popt_220712
data_GFP_72h_220712['GFP fluo norm']=data_GFP_72h_220712['GFP fluo']/popt_220712

data_GFP_24h_220809['GFP fluo norm']=data_GFP_24h_220809['GFP fluo']/popt_220809
data_GFP_48h_220809['GFP fluo norm']=data_GFP_48h_220809['GFP fluo']/popt_220809
data_GFP_72h_220809['GFP fluo norm']=data_GFP_72h_220809['GFP fluo']/popt_220809


data_GFP_24h=pd.concat([data_GFP_24h_211130, data_GFP_24h_220118, data_GFP_24h_220125, data_GFP_24h_220503, data_GFP_24h_220705, data_GFP_24h_220712])
data_GFP_48h=pd.concat([data_GFP_48h_211130, data_GFP_48h_220118, data_GFP_48h_220125, data_GFP_48h_220503, data_GFP_48h_220705, data_GFP_48h_220712])
data_GFP_72h=pd.concat([data_GFP_72h_211130, data_GFP_72h_220118, data_GFP_72h_220125, data_GFP_72h_220503, data_GFP_72h_220705, data_GFP_72h_220712])

num_experiments =6

# Remove 'Unnamed: 0' if necessary
if 'Unnamed: 0' in data_GFP_24h.columns:
    del data_GFP_24h['Unnamed: 0']
    
#if 'Unnamed: 0' in data_sirDNA_24h.columns:    
#    del data_sirDNA_24h['Unnamed: 0']

if 'Unnamed: 0' in data_GFP_48h.columns:
    del data_GFP_48h['Unnamed: 0']
    
#if 'Unnamed: 0' in data_sirDNA_48h.columns:    
#    del data_sirDNA_48h['Unnamed: 0']
    
if 'Unnamed: 0' in data_GFP_72h.columns:
    del data_GFP_72h['Unnamed: 0']
        
# Take only control conditions

data_GFP_24h_control = data_GFP_24h[data_GFP_24h['Condition']=='24h control']  
data_GFP_24h_PDO3 = data_GFP_24h[data_GFP_24h['Condition']=='24h PDO3'] 

data_GFP_48h_control = data_GFP_48h[data_GFP_48h['Condition']=='48h control']  
data_GFP_48h_PDO3 = data_GFP_48h[data_GFP_48h['Condition']=='48h PDO3']  

data_GFP_72h_control = data_GFP_72h[data_GFP_72h['Condition']=='72h control'] 
data_GFP_72h_PDO3 = data_GFP_72h[data_GFP_72h['Condition']=='72h PDO3'] 


# Get the mean and SD #24h

data_GFP_24h_mean_control = data_GFP_24h_control.groupby('GFP fraction', as_index=False).mean()    
data_GFP_24h_std_control = data_GFP_24h_control.groupby('GFP fraction', as_index=False).std()  

data_GFP_24h_mean_PDO3 = data_GFP_24h_PDO3.groupby('GFP fraction', as_index=False).mean()    
data_GFP_24h_std_PDO3 = data_GFP_24h_PDO3.groupby('GFP fraction', as_index=False).std()    
  
data_GFP_24h_mean_control['Num samples']=data_GFP_24h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_24h_std_control['Num samples']=data_GFP_24h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  

data_GFP_24h_mean_PDO3['Num samples']=data_GFP_24h_PDO3.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_24h_std_PDO3['Num samples']=data_GFP_24h_PDO3.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  


# Get the mean and SD #48h

data_GFP_48h_mean_control = data_GFP_48h_control.groupby('GFP fraction', as_index=False).mean()    
data_GFP_48h_std_control = data_GFP_48h_control.groupby('GFP fraction', as_index=False).std()   

data_GFP_48h_mean_PDO3 = data_GFP_48h_PDO3.groupby('GFP fraction', as_index=False).mean()    
data_GFP_48h_std_PDO3 = data_GFP_48h_PDO3.groupby('GFP fraction', as_index=False).std()    
 
data_GFP_48h_mean_control['Num samples']=data_GFP_48h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_48h_std_control['Num samples']=data_GFP_48h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  

data_GFP_48h_mean_PDO3['Num samples']=data_GFP_48h_PDO3.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_48h_std_PDO3['Num samples']=data_GFP_48h_PDO3.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  

# Get the mean and SD #72h

data_GFP_72h_mean_control = data_GFP_72h_control.groupby('GFP fraction', as_index=False).mean()    
data_GFP_72h_std_control = data_GFP_72h_control.groupby('GFP fraction', as_index=False).std()    

data_GFP_72h_mean_PDO3 = data_GFP_72h_PDO3.groupby('GFP fraction', as_index=False).mean()    
data_GFP_72h_std_PDO3= data_GFP_72h_PDO3.groupby('GFP fraction', as_index=False).std()    

data_GFP_72h_mean_control['Num samples']=data_GFP_72h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_72h_std_control['Num samples']=data_GFP_72h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  

data_GFP_72h_mean_PDO3['Num samples']=data_GFP_72h_PDO3.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_72h_std_PDO3['Num samples']=data_GFP_72h_PDO3.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  


# create mean data sets to fit #####################################################################
data_24h_mean=np.zeros((5,2))
data_48h_mean=np.zeros((5,2))
data_72h_mean=np.zeros((5,2))

data_24h_mean[:,0]=[0.01,0.2,0.5,0.8,0.99]
data_48h_mean[:,0]=[0.01,0.2,0.5,0.8,0.99]
data_72h_mean[:,0]=[0.01,0.2,0.5,0.8,0.99]

# Combo arrays to fit
comboX = np.concatenate([data_24h_mean[:,0],data_48h_mean[:,0],data_72h_mean[:,0]])        
comboY = np.concatenate([data_GFP_24h_mean_control['GFP fluo norm'],data_GFP_48h_mean_control['GFP fluo norm'],data_GFP_72h_mean_control['GFP fluo norm']])         

# # fit for phi2(t) for mu=+ and nu=+ 
#popt48h,pcov48h = curve_fit(phi2_fit0,data_48h_mean[:,0],data_GFP_48h_mean_control['GFP fluo norm'],bounds=((0.001,0.001,0.001),(inf,inf,inf)))
#popt24h,pcov24h = curve_fit(phi2_fit0,data_24h_mean[:,0],data_GFP_24h_mean_control['GFP fluo norm'],bounds=((popt48h[0]-0.001,popt48h[1]-0.001,0.001),(popt48h[0]+0.001,popt48h[1]+0.001,inf)))
#popt72h,pcov72h = curve_fit(phi2_fit0,data_72h_mean[:,0],data_GFP_72h_mean_control['GFP fluo norm'],bounds=((popt48h[0]-0.001,popt48h[1]-0.001,popt48h[2]+(popt48h[2]-popt24h[2])),(popt48h[0]+0.001,popt48h[1]+0.001,popt48h[2]+(popt48h[2]-popt24h[2]+3))))

fittedParameters,pcov = curve_fit(combo_phi2_fit0,comboX,comboY,bounds=((0.001,0.001,0.001,0.001),(inf,inf,inf,inf)))

error = np.sqrt(np.diag(pcov)) 
# plot states vs initial condition
    
#k=4 # time point

# plot states vs time

timescale = 24/(fittedParameters[3]-fittedParameters[2]) # (time in hours equivalent to 1 u.a. of time)
#timescale = 24/(popt48h[1]-popt24h[1]) # no signal (time in hours equivalent to 1 u.a. of time)
    
############ BEGIN SOLVE MODEL WITH FITTED PARAMETERS ##########################################################################################

phi0 = 0.1 # initial condition 

# parameters Model  (dimensionless)

alpha=fittedParameters[0] # (q/p)
beta=2000  # (r/k)
K=fittedParameters[1]
#t0=3 # PD03 starts
#t2=35 # PD03 is washed out
#t3 = t2+(t2-t0)

def three_state_model(y, t, params):
    phi1, phi2, phi3, fgf = y
    alpha, beta, K = params #unpack params
    dydt = [-fgf*phi1, fgf*(phi1-alpha*phi2), alpha*fgf*phi2, beta*(1/(1+(phi1/K))-fgf)] #mu=+ and nu=+ (Model 0) Good match analytical vs numerical
    #dydt = [-phi1, (phi1-fgf*alpha*phi2), alpha*fgf*phi2, beta*(1/(1+(phi1/K))-fgf)] #mu=0 and nu=+ (Model 1) Good match analytical vs numerical
    #dydt = [-fgf*phi1, fgf*phi1-alpha*phi2, alpha*phi2, beta*(1/(1+(phi1/K))-fgf)]  #mu=+ and nu=0 (Model 2) Good match analytical vs numerical
    #dydt = [-(1-fgf)*phi1, (1-fgf)*(phi1-alpha*phi2), alpha*(1-fgf)*phi2, beta*(1/(1+(phi1/K))-fgf)] #mu=- and nu=- (Model 3) Good match analytical vs numerical
    #dydt = [-(1-fgf)*phi1, (1-fgf)*phi1-alpha*phi2, alpha*phi2, beta*(1/(1+(phi1/K))-fgf)] #mu=- and nu=0 (Model 4) Good match analytical vs numerical
    #dydt = [-phi1, phi1-(1-fgf)*alpha*phi2, (1-fgf)*alpha*phi2, beta*(1/(1+(phi1/K))-fgf)] #mu=0 and nu=- (Model 5) Good match analytical vs numerical
    #dydt = [-fgf*phi1, fgf*phi1-(1-fgf)*alpha*phi2, (1-fgf)*alpha*phi2, beta*(1/(1+(phi1/K))-fgf)] #mu=0 and nu=- (Model 6) Good match analytical vs numerical
    #dydt = [-(1-fgf)*phi1, (1-fgf)*phi1-(fgf)*alpha*phi2, (fgf)*alpha*phi2, beta*(1/(1+(phi1/K))-fgf)] #mu=0 and nu=- (Model 7) Good match analytical vs numerical
    #dydt = [-phi1, phi1-alpha*phi2, alpha*phi2, beta*(1/(1+(phi1/K))-fgf)] #mu=0 and nu=0 no signal
    #dydt = [-phi1, phi1-alpha*phi2, alpha*fgf, beta*(1/(1+(phi1/K))-fgf)]
    #dydt = [-step(t,t0,t2)*fgf*phi1, step(t,t0,t2)*fgf*phi1-fgf*alpha*phi2, alpha*fgf*phi2, beta*(1/(1+step(t,t0,t2)*(phi1/K))-fgf)]
    return dydt
    

fgf0 = 0.30
phi0=0.3

# initial conditions
y0 = [phi0, 1-phi0,0,fgf0]

Tmax = 200 # Tmax
dt = 0.01 # time step
dphi = 0.01 # step initial condition
t = np.linspace(0, Tmax,int(Tmax/dt)) # time array
phi0_array = np.linspace(0.01, 0.99, int(1/dphi)) # % state 2 cells

# solve

sol=[]

for i in range(0,np.size(phi0_array,0)):
    y0 = [1-phi0_array[i],phi0_array[i],0,fgf0] # set initial condition
    params = [alpha, beta, K] # Bundle parameter for ODE Solver
    sol.append(odeint(three_state_model, y0, t, args=(params,))) # solve

time_point_y = np.linspace(0,1)
time_point_x = np.ones(np.size(time_point_y))


sol_phi=np.zeros((np.size(phi0_array,0),3,np.size(t,0)))

for j in range(3): #states
    for i in range(0,np.size(phi0_array,0)):
        sol_phi[i][j][:]=sol[i][:,j]
        
############ END SOLVE MODEL WITH FITTED PARAMETERS ##########################################################################################

j=20 # initial condition

modelfig  = plt.figure(figsize=(4,4))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([0,60,0,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.plot(t*timescale, sol[j][:,0], 'magenta', label=r'$\phi_1$')
plt.plot(t*timescale, sol[j][:,1], 'green', label=r'$\phi_2$')
plt.plot(t*timescale, sol[j][:,2], 'orange', label=r'$\phi_3$')
#plt.plot(t*timescale, sol[j][:,3], 'gray', label=r'$Fgf$')
plt.plot(fittedParameters[2]*time_point_x*timescale ,time_point_y,'--',color='gray')
plt.plot(fittedParameters[3]*time_point_x*timescale ,time_point_y,'--',color='cadetblue')
plt.plot((fittedParameters[3]+(fittedParameters[3]-fittedParameters[2]))*time_point_x*timescale ,time_point_y,'--',color='black')
#plt.plot(popt24h[1]*time_point_x*timescale ,time_point_y,'--',color='gray')    #no signal
#plt.plot(popt48h[1]*time_point_x*timescale ,time_point_y,'--',color='cadetblue')   #no signal
#plt.plot(popt72h[1]*time_point_x*timescale ,time_point_y,'--',color='black')  #no signal
#plt.legend(loc='best')
plt.xlabel('signalling time (h)')
#plt.grid()
plt.show()
modelfig.savefig('modelfig_221003.pdf',bbox_inches = "tight")   # save as .eps


nSEM=2 #number od SEM

#sns.set_palette("ch:s=-.2,r=.9") 
#s2fig = plt.figure()
s2fig  = plt.figure(figsize=(4,4))
#s2fig.set_aspect('equal')
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([-0.05,1.05,0,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
#plt.axis('scaled')
plt.ylabel(r'$\phi_2$',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'$\phi_2(0)$',fontsize=15, color = 'black')  # x-label fontsize + color # fit plot
#plt.plot(phi0_array, sol_phi[:,1,T24], 'g', label='t='+str("{:.2f}".format(t[T24]))) # 24h
#plt.plot(phi0_array, sol_phi[:,1,T48], 'b', label='t='+str("{:.2f}".format(t[T48]))) # 48h
#plt.plot(phi0_array, sol_phi[:,1,T72], 'k', label='t='+str("{:.2f}".format(t[T72]))) # 72h
#plt.plot(phi0_array, sol_phi[:,1,0], label='t='+str("{:.2f}".format(t[T24]))) # 24h
#plt.plot(phi0_array, sol_phi[:,1,40], label='t='+str("{:.2f}".format(t[T48]))) # 48h
#plt.plot(phi0_array, sol_phi[:,1,80], label='t='+str("{:.2f}".format(t[T72]))) # 72h
#plt.plot(phi0_array, sol_phi[:,1,120], label='t='+str("{:.2f}".format(t[T72]))) # 72h
#plt.plot(phi0_array, sol_phi[:,1,160], label='t='+str("{:.2f}".format(t[T72]))) # 72h
#plt.plot(phi0_array, sol_phi[:,1,200], label='t='+str("{:.2f}".format(t[T72]))) # 72h
#plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,popt24h[0],popt24h[1]),'--',color='gray')
#plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,popt48h[0],popt48h[1]),'--',color='cadetblue')
#plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,popt72h[0],popt72h[1]),'--',color='black')
plt.plot(phi0_array,phi2_fit0(phi0_array,fittedParameters[0],fittedParameters[1],fittedParameters[2]),'--',color='gray')
plt.plot(phi0_array,phi2_fit0(phi0_array,fittedParameters[0],fittedParameters[1],fittedParameters[3]),'--',color='cadetblue')
plt.plot(phi0_array,phi2_fit0(phi0_array,fittedParameters[0],fittedParameters[1],fittedParameters[3]+(fittedParameters[3]-fittedParameters[2])),'--',color='black')
#plt.plot(phi0_array,phi2_fit7(phi0_array,alpha,K,t[0]),'--',color='gray')
#plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,alpha,K,t[40]),'--',color='cadetblue')
#plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,alpha,K,t[80]),'--',color='black')
#plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,alpha,K,t[120]),'--',color='black')
#plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,alpha,K,t[160]),'--',color='black')
#plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,alpha,K,t[200]),'--',color='black')
#plt.errorbar(data_GFP_24h_mean_control['GFP fraction'],data_GFP_24h_mean_control['GFP fluo norm'],yerr=data_GFP_24h_std_control['GFP fluo norm']/np.sqrt(data_GFP_24h_mean_control['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='24h',color='gray')
#plt.errorbar(data_GFP_48h_mean_control['GFP fraction'],data_GFP_48h_mean_control['GFP fluo norm'],yerr=data_GFP_48h_std_control['GFP fluo norm']/np.sqrt(data_GFP_48h_mean_control['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='48h',color='cadetblue')
#plt.errorbar(data_GFP_72h_mean_control['GFP fraction'],data_GFP_72h_mean_control['GFP fluo norm'],yerr=data_GFP_72h_std_control['GFP fluo norm']/np.sqrt(data_GFP_72h_mean_control['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='72h',color='black')
plt.errorbar(data_GFP_24h_mean_control['GFP fraction'],data_GFP_24h_mean_control['GFP fluo norm'],yerr=nSEM*data_GFP_24h_std_control['GFP fluo norm']/np.sqrt(data_GFP_24h_mean_control['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='24h',color='gray')
plt.errorbar(data_GFP_48h_mean_control['GFP fraction'],data_GFP_48h_mean_control['GFP fluo norm'],yerr=nSEM*data_GFP_48h_std_control['GFP fluo norm']/np.sqrt(data_GFP_48h_mean_control['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='48h',color='cadetblue')
plt.errorbar(data_GFP_72h_mean_control['GFP fraction'],data_GFP_72h_mean_control['GFP fluo norm'],yerr=nSEM*data_GFP_72h_std_control['GFP fluo norm']/np.sqrt(data_GFP_72h_mean_control['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='72h',color='black')
#plt.errorbar(data_GFP_24h_mean_PDO3['GFP fraction'],data_GFP_24h_mean_PDO3['GFP fluo norm'],yerr=data_GFP_24h_std_PDO3['GFP fluo norm']/np.sqrt(3),marker='o',fmt='-',capthick=2,capsize=5, label='24h',color='gray')
#plt.errorbar(data_GFP_48h_mean_PDO3['GFP fraction'],data_GFP_48h_mean_PDO3['GFP fluo norm'],yerr=data_GFP_48h_std_PDO3['GFP fluo norm']/np.sqrt(3),marker='o',fmt='-',capthick=2,capsize=5, label='48h',color='cadetblue')
#plt.errorbar(data_GFP_72h_mean_PDO3['GFP fraction'],data_GFP_72h_mean_PDO3['GFP fluo norm'],yerr=data_GFP_72h_std_PDO3['GFP fluo norm']/np.sqrt(3),marker='o',fmt='-',capthick=2,capsize=5, label='72h',color='black')
#plt.plot(data_48h_mean[:,0],data_48h_mean[:,1],'.',markersize=15,color='cadetblue')
#plt.plot(data_72h_mean[:,0],data_72h_mean[:,1],'.',markersize=15,color='black')
plt.legend(loc='upper center',ncol=3, fontsize =11, frameon= False)
#plt.grid()
plt.show()
s2fig.savefig('T_dynamics_221003.pdf',bbox_inches = "tight")   # save as .eps

# Ouput data

output_24h_GFP_mean = pd.DataFrame({'phi2(0)':data_GFP_24h_mean_control['GFP fraction'],'phi2':data_GFP_24h_mean_control['GFP fluo norm'],'2*SE':nSEM*data_GFP_24h_std_control['GFP fluo norm']/np.sqrt(data_GFP_24h_mean_control['Num samples'])})
output_48h_GFP_mean = pd.DataFrame({'phi2(0)':data_GFP_48h_mean_control['GFP fraction'],'phi2':data_GFP_48h_mean_control['GFP fluo norm'],'2*SE':nSEM*data_GFP_48h_std_control['GFP fluo norm']/np.sqrt(data_GFP_48h_mean_control['Num samples'])})
output_72h_GFP_mean = pd.DataFrame({'phi2(0)':data_GFP_72h_mean_control['GFP fraction'],'phi2':data_GFP_72h_mean_control['GFP fluo norm'],'2*SE':nSEM*data_GFP_72h_std_control['GFP fluo norm']/np.sqrt(data_GFP_72h_mean_control['Num samples'])})
output_24h_GFP_mean.to_csv('output_24h_GFP_mean.csv')
output_48h_GFP_mean.to_csv('output_48h_GFP_mean.csv')
output_72h_GFP_mean.to_csv('output_72h_GFP_mean.csv')

# Ouput fitting params

output_fit = pd.DataFrame({'alpha':[fittedParameters[0],error[0]],'K':[fittedParameters[1],error[1]],'T24(h)':[timescale*fittedParameters[2],timescale*error[2]],'T48(h)':[timescale*fittedParameters[3],timescale*error[3]]})
output_fit.to_csv('output_fit_params.csv')
