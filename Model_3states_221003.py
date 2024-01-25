
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

def fun(phi0,K,T1):
    return K*lambertw((1/K)*phi0*np.exp(-T1+phi0/K)).real
    #return K*lambertw((1/K)*(1-phi20)*np.exp(-T1+(1-phi20)/K)).real

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

# Experiment 220705

output_unmixing_analysis_24h_GFP_220705 =os.path.join(dir_path,'output_24_plate_2_GFP_220705.csv')
output_unmixing_analysis_48h_GFP_220705 =os.path.join(dir_path,'output_48_plate_2_GFP_220706.csv')
output_unmixing_analysis_72h_GFP_220705 =os.path.join(dir_path,'output_72_plate_2_GFP_220707.csv')

# Experiment 220712

output_unmixing_analysis_24h_GFP_220712=os.path.join(dir_path,'output_24_plate_2_GFP_220712.csv')
output_unmixing_analysis_48h_GFP_220712=os.path.join(dir_path,'output_48_plate_2_GFP_220713.csv')
output_unmixing_analysis_72h_GFP_220712 =os.path.join(dir_path,'output_72_plate_2_GFP_220714.csv')

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

data_GFP_24h_220705=pd.read_csv(output_unmixing_analysis_24h_GFP_220705)
data_GFP_48h_220705=pd.read_csv(output_unmixing_analysis_48h_GFP_220705)
data_GFP_72h_220705=pd.read_csv(output_unmixing_analysis_72h_GFP_220705)

data_GFP_24h_220712=pd.read_csv(output_unmixing_analysis_24h_GFP_220712)
data_GFP_48h_220712=pd.read_csv(output_unmixing_analysis_48h_GFP_220712)
data_GFP_72h_220712=pd.read_csv(output_unmixing_analysis_72h_GFP_220712)

# Change the name of some columns for practical purposes

data_GFP_24h_220705['Condition'] = data_GFP_24h_220705['Condition'].str.replace('24h Ctrl','24h control')
data_GFP_24h_220712['Condition'] = data_GFP_24h_220712['Condition'].str.replace('24h Ctrl','24h control')

data_GFP_48h_220705['Condition'] = data_GFP_48h_220705['Condition'].str.replace('48h Ctrl','48h control')
data_GFP_48h_220712['Condition'] = data_GFP_48h_220712['Condition'].str.replace('48h Ctrl','48h control')

data_GFP_72h_220705['Condition'] = data_GFP_72h_220705['Condition'].str.replace('72h Ctrl','72h control')
data_GFP_72h_220712['Condition'] = data_GFP_72h_220712['Condition'].str.replace('72h Ctrl','72h control')

# Extrapolate maximum GFP value for normalization 

popt_211130,pcov_211130 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_211130[data_GFP_24h_211130['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220118,pcov_220118 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220118[data_GFP_24h_220118['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220125,pcov_220125 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220125[data_GFP_24h_220125['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220503,pcov_220503 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220503[data_GFP_24h_220503['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220705,pcov_220705 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220705[data_GFP_24h_220705['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220712,pcov_220712 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220712[data_GFP_24h_220712['Condition']=='24h control']['GFP fluo'])[0:4])
  
# Normalise by the intensity

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

data_GFP_24h_220705['GFP fluo norm']=data_GFP_24h_220705['GFP fluo']/popt_220705
data_GFP_48h_220705['GFP fluo norm']=data_GFP_48h_220705['GFP fluo']/popt_220705
data_GFP_72h_220705['GFP fluo norm']=data_GFP_72h_220705['GFP fluo']/popt_220705

data_GFP_24h_220712['GFP fluo norm']=data_GFP_24h_220712['GFP fluo']/popt_220712
data_GFP_48h_220712['GFP fluo norm']=data_GFP_48h_220712['GFP fluo']/popt_220712
data_GFP_72h_220712['GFP fluo norm']=data_GFP_72h_220712['GFP fluo']/popt_220712

# Final dataset to use

data_GFP_24h=pd.concat([data_GFP_24h_211130, data_GFP_24h_220118, data_GFP_24h_220125, data_GFP_24h_220503, data_GFP_24h_220705, data_GFP_24h_220712])
data_GFP_48h=pd.concat([data_GFP_48h_211130, data_GFP_48h_220118, data_GFP_48h_220125, data_GFP_48h_220503, data_GFP_48h_220705, data_GFP_48h_220712])
data_GFP_72h=pd.concat([data_GFP_72h_211130, data_GFP_72h_220118, data_GFP_72h_220125, data_GFP_72h_220503, data_GFP_72h_220705, data_GFP_72h_220712])

num_experiments =6

# Remove 'Unnamed: 0' if necessary
if 'Unnamed: 0' in data_GFP_24h.columns:
    del data_GFP_24h['Unnamed: 0']
    
if 'Unnamed: 0' in data_GFP_48h.columns:
    del data_GFP_48h['Unnamed: 0']
    
if 'Unnamed: 0' in data_GFP_72h.columns:
    del data_GFP_72h['Unnamed: 0']
        
# Take only control conditions

data_GFP_24h_control = data_GFP_24h[data_GFP_24h['Condition']=='24h control']  
data_GFP_48h_control = data_GFP_48h[data_GFP_48h['Condition']=='48h control']  
data_GFP_72h_control = data_GFP_72h[data_GFP_72h['Condition']=='72h control'] 


# Get the mean and SD #24h

data_GFP_24h_mean_control = data_GFP_24h_control.groupby('GFP fraction', as_index=False).mean()    
data_GFP_24h_std_control = data_GFP_24h_control.groupby('GFP fraction', as_index=False).std()  

data_GFP_24h_mean_control['Num samples']=data_GFP_24h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_24h_std_control['Num samples']=data_GFP_24h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  

# Get the mean and SD #48h

data_GFP_48h_mean_control = data_GFP_48h_control.groupby('GFP fraction', as_index=False).mean()    
data_GFP_48h_std_control = data_GFP_48h_control.groupby('GFP fraction', as_index=False).std()   

data_GFP_48h_mean_control['Num samples']=data_GFP_48h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_48h_std_control['Num samples']=data_GFP_48h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  

# Get the mean and SD #72h

data_GFP_72h_mean_control = data_GFP_72h_control.groupby('GFP fraction', as_index=False).mean()    
data_GFP_72h_std_control = data_GFP_72h_control.groupby('GFP fraction', as_index=False).std()    

data_GFP_72h_mean_control['Num samples']=data_GFP_72h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_72h_std_control['Num samples']=data_GFP_72h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  

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

fittedParameters,pcov = curve_fit(combo_phi2_fit0,comboX,comboY,bounds=((0.001,0.001,0.001,0.001),(inf,inf,inf,inf)))

error = np.sqrt(np.diag(pcov)) 

# plot states vs time

timescale = 24/(fittedParameters[3]-fittedParameters[2]) # (time in hours equivalent to 1 u.a. of time)

############ BEGIN SOLVE MODEL WITH FITTED PARAMETERS ##########################################################################################

phi0 = 0.1 # initial condition 

# parameters Model  (dimensionless)

alpha=fittedParameters[0] # (q/p)
beta=2000  # (r/k)
K=fittedParameters[1]

def three_state_model(y, t, params):
    phi1, phi2, phi3, fgf = y
    alpha, beta, K = params #unpack params
    dydt = [-fgf*phi1, fgf*(phi1-alpha*phi2), alpha*fgf*phi2, beta*(1/(1+(phi1/K))-fgf)] 
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
plt.xlabel('signalling time (h)')
plt.show()
modelfig.savefig('modelfig_221003.pdf',bbox_inches = "tight")   # save as .eps


nSEM=2 #number of SEM

s2fig  = plt.figure(figsize=(4,4))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([-0.05,1.05,0,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.ylabel(r'$\phi_2$',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'$\phi_2(0)$',fontsize=15, color = 'black')  # x-label fontsize + color # fit plot
plt.plot(phi0_array,phi2_fit0(phi0_array,fittedParameters[0],fittedParameters[1],fittedParameters[2]),'--',color='gray')
plt.plot(phi0_array,phi2_fit0(phi0_array,fittedParameters[0],fittedParameters[1],fittedParameters[3]),'--',color='cadetblue')
plt.plot(phi0_array,phi2_fit0(phi0_array,fittedParameters[0],fittedParameters[1],fittedParameters[3]+(fittedParameters[3]-fittedParameters[2])),'--',color='black')
plt.errorbar(data_GFP_24h_mean_control['GFP fraction'],data_GFP_24h_mean_control['GFP fluo norm'],yerr=nSEM*data_GFP_24h_std_control['GFP fluo norm']/np.sqrt(data_GFP_24h_mean_control['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='24h',color='gray')
plt.errorbar(data_GFP_48h_mean_control['GFP fraction'],data_GFP_48h_mean_control['GFP fluo norm'],yerr=nSEM*data_GFP_48h_std_control['GFP fluo norm']/np.sqrt(data_GFP_48h_mean_control['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='48h',color='cadetblue')
plt.errorbar(data_GFP_72h_mean_control['GFP fraction'],data_GFP_72h_mean_control['GFP fluo norm'],yerr=nSEM*data_GFP_72h_std_control['GFP fluo norm']/np.sqrt(data_GFP_72h_mean_control['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='72h',color='black')
plt.legend(loc='upper center',ncol=3, fontsize =11, frameon= False)
plt.show()
s2fig.savefig('T_dynamics_221003.pdf',bbox_inches = "tight")   # save as .pdf

# Output data

output_24h_GFP_mean = pd.DataFrame({'phi2(0)':data_GFP_24h_mean_control['GFP fraction'],'phi2':data_GFP_24h_mean_control['GFP fluo norm'],'2*SE':nSEM*data_GFP_24h_std_control['GFP fluo norm']/np.sqrt(data_GFP_24h_mean_control['Num samples'])})
output_48h_GFP_mean = pd.DataFrame({'phi2(0)':data_GFP_48h_mean_control['GFP fraction'],'phi2':data_GFP_48h_mean_control['GFP fluo norm'],'2*SE':nSEM*data_GFP_48h_std_control['GFP fluo norm']/np.sqrt(data_GFP_48h_mean_control['Num samples'])})
output_72h_GFP_mean = pd.DataFrame({'phi2(0)':data_GFP_72h_mean_control['GFP fraction'],'phi2':data_GFP_72h_mean_control['GFP fluo norm'],'2*SE':nSEM*data_GFP_72h_std_control['GFP fluo norm']/np.sqrt(data_GFP_72h_mean_control['Num samples'])})
output_24h_GFP_mean.to_csv('output_24h_GFP_mean.csv')
output_48h_GFP_mean.to_csv('output_48h_GFP_mean.csv')
output_72h_GFP_mean.to_csv('output_72h_GFP_mean.csv')

# Ouput fitting params

output_fit = pd.DataFrame({'alpha':[fittedParameters[0],error[0]],'K':[fittedParameters[1],error[1]],'T24(h)':[timescale*fittedParameters[2],timescale*error[2]],'T48(h)':[timescale*fittedParameters[3],timescale*error[3]]})
output_fit.to_csv('output_fit_params.csv')
