
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
from scipy.optimize import curve_fit
from scipy import interpolate
import math, tqdm
from scipy.optimize import fsolve
from scipy.special import lambertw
import scipy.special as sc
from scipy import integrate
from scipy.integrate import odeint
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize, minimize_scalar, rosen, rosen_der, least_squares
import random

sns.set_palette("PuBuGn_d")

# linear fit
def linear_fit(phi20,a):
    return a*phi20
       
# Define Lambert function
def fun(phi0,K,T1):
    return K*lambertw((1/K)*phi0*np.exp(-T1+phi0/K)).real

# Analytical solution for phi2(t) with feedback
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

# We define three different states phi1: A, phi2: B, phi3: C

dir_path = os.path.abspath(os.getcwd()) # get path

# get paths for each csv file

# Experiment 211130

output_unmixing_analysis_24h_GFP_211130 =os.path.join(dir_path,'data_24_GFP_211130.csv')
output_unmixing_analysis_48h_GFP_211130 =os.path.join(dir_path,'data_48_GFP_211201.csv')
output_unmixing_analysis_72h_GFP_211130 =os.path.join(dir_path,'data_72_GFP_211202.csv')

# Experiment 220118

output_unmixing_analysis_24h_GFP_220118 =os.path.join(dir_path,'data_24_GFP_220118.csv')
output_unmixing_analysis_48h_GFP_220118 =os.path.join(dir_path,'data_48_GFP_220119.csv')
output_unmixing_analysis_72h_GFP_220118 =os.path.join(dir_path,'data_72_GFP_220120.csv')

# Experiment 220125

output_unmixing_analysis_24h_GFP_220125 =os.path.join(dir_path,'data_24_GFP_220125.csv')
output_unmixing_analysis_48h_GFP_220125 =os.path.join(dir_path,'data_48_GFP_220126.csv')
output_unmixing_analysis_72h_GFP_220125 =os.path.join(dir_path,'data_72_GFP_220127.csv')

# Experiment 220503

output_unmixing_analysis_24h_GFP_220503 =os.path.join(dir_path,'data_24_GFP_220503.csv')
output_unmixing_analysis_48h_GFP_220503 =os.path.join(dir_path,'data_48_GFP_220504.csv')
output_unmixing_analysis_72h_GFP_220503 =os.path.join(dir_path,'data_72_GFP_220505.csv')

# Experiment 220705

output_unmixing_analysis_24h_GFP_220705 =os.path.join(dir_path,'data_24_GFP_220705.csv')
output_unmixing_analysis_48h_GFP_220705 =os.path.join(dir_path,'data_48_GFP_220706.csv')
output_unmixing_analysis_72h_GFP_220705 =os.path.join(dir_path,'data_72_GFP_220707.csv')

# Experiment 220712

output_unmixing_analysis_24h_GFP_220712=os.path.join(dir_path,'data_24_GFP_220712.csv')
output_unmixing_analysis_48h_GFP_220712=os.path.join(dir_path,'data_48_GFP_220713.csv')
output_unmixing_analysis_72h_GFP_220712 =os.path.join(dir_path,'data_72_GFP_220714.csv')


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

# Extrapolate maximum GFP value for normalization 

popt_211130,pcov_211130 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_211130[data_GFP_24h_211130['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220118,pcov_220118 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220118[data_GFP_24h_220118['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220125,pcov_220125 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220125[data_GFP_24h_220125['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220503,pcov_220503 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220503[data_GFP_24h_220503['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220705,pcov_220705 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220705[data_GFP_24h_220705['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220712,pcov_220712 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220712[data_GFP_24h_220712['Condition']=='24h control']['GFP fluo'])[0:4])
      
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

data_GFP_24h_220705['GFP fluo norm']=data_GFP_24h_220705['GFP fluo']/popt_220705
data_GFP_48h_220705['GFP fluo norm']=data_GFP_48h_220705['GFP fluo']/popt_220705
data_GFP_72h_220705['GFP fluo norm']=data_GFP_72h_220705['GFP fluo']/popt_220705

data_GFP_24h_220712['GFP fluo norm']=data_GFP_24h_220712['GFP fluo']/popt_220712
data_GFP_48h_220712['GFP fluo norm']=data_GFP_48h_220712['GFP fluo']/popt_220712
data_GFP_72h_220712['GFP fluo norm']=data_GFP_72h_220712['GFP fluo']/popt_220712

# Compile data

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
        
# Take the control conditions

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

# Perform fit

fittedParameters,pcov = curve_fit(combo_phi2_fit0,comboX,comboY,bounds=((0.001,0.001,0.001,0.001),(inf,inf,inf,inf)))

# Compute error of the fit

error = np.sqrt(np.diag(pcov)) 

# Compute timescale of the system
timescale = 24/(fittedParameters[3]-fittedParameters[2]) # (time in hours equivalent to 1 u.a. of time)
time_array = [fittedParameters[2]*timescale,fittedParameters[3]*timescale,(fittedParameters[3]+(fittedParameters[3]-fittedParameters[2]))*timescale] # time array in signalling time

############ BEGIN SOLVE MODEL WITH FITTED PARAMETERS ##########################################################################################

# parameters Model  (dimensionless)

alpha=fittedParameters[0] # (q/p)
K=fittedParameters[1]

def three_state_model(y, t, params):
    phi1, phi2, phi3 = y
    alpha, beta, K = params #unpack params
    dydt = [-phi1/(1+phi1/K), (phi1-alpha*phi2)/(1+phi1/K), (alpha*phi2)/(1+phi1/K)] 
    return dydt

# initial conditions and time limit
Tmax = 200 # Tmax
dt = 0.01 # time step
dphi = 0.01 # step initial condition
t = np.linspace(0, Tmax,int(Tmax/dt)) # time array
phi0_array = np.linspace(0.01, 0.99, int(1/dphi)) # % state 2 cells

# solve

sol=[] # initialization

for i in range(0,np.size(phi0_array,0)):
    y0 = [1-phi0_array[i],phi0_array[i],0] # set initial condition
    params = [alpha, beta, K] # Bundle parameter for ODE Solver
    sol.append(odeint(three_state_model, y0, t, args=(params,))) # solve

time_point_y = np.linspace(0,1)
time_point_x = np.ones(np.size(time_point_y))

sol_phi=np.zeros((np.size(phi0_array,0),3,np.size(t,0)))

for j in range(3): #states
    for i in range(0,np.size(phi0_array,0)):
        sol_phi[i][j][:]=sol[i][:,j]
        
############ END SOLVE MODEL WITH FITTED PARAMETERS ##########################################################################################

# Plot experimental data and fit to the model for the control case

nSEM=2 #number of SEM

# Create timelapse data
data_phi0_mean = []
data_phi0_std = []
phi0_value_array = [0.0,0.2,0.5,0.8,1.0]

for i in range(0,5): # loop over phi
    phi0_value=phi0_value_array[i]
    data_phi0_mean.append([data_GFP_24h_mean_control[data_GFP_24h_mean_control['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0],data_GFP_48h_mean_control[data_GFP_48h_mean_control['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0],data_GFP_72h_mean_control[data_GFP_72h_mean_control['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0]])
    data_phi0_std.append([nSEM*data_GFP_24h_std_control[data_GFP_24h_std_control['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0]/np.sqrt(data_GFP_24h_std_control[data_GFP_24h_std_control['GFP fraction']==phi0_value]['Num samples'].iloc[0]),nSEM*data_GFP_48h_std_control[data_GFP_48h_std_control['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0]/np.sqrt(data_GFP_48h_std_control[data_GFP_48h_std_control['GFP fraction']==phi0_value]['Num samples'].iloc[0]),nSEM*data_GFP_72h_std_control[data_GFP_72h_std_control['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0]/np.sqrt(data_GFP_72h_std_control[data_GFP_72h_std_control['GFP fraction']==phi0_value]['Num samples'].iloc[0])])

# T Dyanmics vs initial condition for the control case

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
s2fig.savefig('T_dynamics_240131.pdf',bbox_inches = "tight")   # save as .eps

# T Dynamics vs time for the control case

s2fig_time  = plt.figure(figsize=(4,4))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([0,80,0,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.ylabel(r'T+ fraction $\phi_B$',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel('signalling time (h)',fontsize=15, color = 'black')  # x-label fontsize + color # fit plot
plt.plot(t*timescale, sol[0][:,1],'--',color='black',label=r'$\phi_B(0)=0$')
plt.plot(t*timescale, sol[19][:,1],'--',color='gray',label=r'$\phi_B(0)=0.2$')
plt.plot(t*timescale, sol[49][:,1],'--',color='cadetblue',label=r'$\phi_B(0)=0.5$')
plt.plot(t*timescale, sol[79][:,1],'--',color='green',label=r'$\phi_B(0)=0.8$')
plt.plot(t*timescale, sol[99][:,1],'--',color='orange',label=r'$\phi_B(0)=1.0$')
plt.errorbar(time_array,data_phi0_mean[0],yerr=data_phi0_std[0],marker='o',fmt=' ',capthick=2,capsize=5,color='black')
plt.errorbar(time_array,data_phi0_mean[1],yerr=data_phi0_std[1],marker='o',fmt=' ',capthick=2,capsize=5,color='gray')
plt.errorbar(time_array,data_phi0_mean[2],yerr=data_phi0_std[2],marker='o',fmt=' ',capthick=2,capsize=5,color='cadetblue')
plt.errorbar(time_array,data_phi0_mean[3],yerr=data_phi0_std[3],marker='o',fmt=' ',capthick=2,capsize=5,color='green')
plt.errorbar(time_array,data_phi0_mean[4],yerr=data_phi0_std[4],marker='o',fmt=' ',capthick=2,capsize=5,color='orange')
plt.legend(loc='upper right', fontsize =8, frameon= False)
plt.show()
s2fig_time.savefig('modelfig_240131_time.pdf',bbox_inches = "tight")   # save as .eps

# Model time evolution of the different states with the fitted parameters for the control case

j=20 # initial condition for state B (phi2(0)=0.2)

modelfig  = plt.figure(figsize=(4,4))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([0,60,0,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.plot(t*timescale, sol[j][:,0], 'magenta', label=r'$\phi_1$')
plt.plot(t*timescale, sol[j][:,1], 'green', label=r'$\phi_2$')
plt.plot(t*timescale, sol[j][:,2], 'orange', label=r'$\phi_3$')
plt.plot(fittedParameters[2]*time_point_x*timescale ,time_point_y,'--',color='gray')
plt.plot(fittedParameters[3]*time_point_x*timescale ,time_point_y,'--',color='cadetblue')
plt.plot((fittedParameters[3]+(fittedParameters[3]-fittedParameters[2]))*time_point_x*timescale ,time_point_y,'--',color='black')
plt.xlabel('signalling time (h)')
plt.show()
modelfig.savefig('modelfig_240131.pdf',bbox_inches = "tight")   # save as .eps

# Ouput data

output_24h_GFP_mean = pd.DataFrame({'phi2(0)':data_GFP_24h_mean_control['GFP fraction'],'phi2':data_GFP_24h_mean_control['GFP fluo norm'],'2*SE':nSEM*data_GFP_24h_std_control['GFP fluo norm']/np.sqrt(data_GFP_24h_mean_control['Num samples'])})
output_48h_GFP_mean = pd.DataFrame({'phi2(0)':data_GFP_48h_mean_control['GFP fraction'],'phi2':data_GFP_48h_mean_control['GFP fluo norm'],'2*SE':nSEM*data_GFP_48h_std_control['GFP fluo norm']/np.sqrt(data_GFP_48h_mean_control['Num samples'])})
output_72h_GFP_mean = pd.DataFrame({'phi2(0)':data_GFP_72h_mean_control['GFP fraction'],'phi2':data_GFP_72h_mean_control['GFP fluo norm'],'2*SE':nSEM*data_GFP_72h_std_control['GFP fluo norm']/np.sqrt(data_GFP_72h_mean_control['Num samples'])})
output_24h_GFP_mean.to_csv('output_24h_GFP_mean.csv')
output_48h_GFP_mean.to_csv('output_48h_GFP_mean.csv')
output_72h_GFP_mean.to_csv('output_72h_GFP_mean.csv')

# Ouput fitting params

output_fit_control = pd.DataFrame({'alpha':[fittedParameters[0],error[0]],'K':[fittedParameters[1],error[1]],'T24(h)':[timescale*fittedParameters[2],timescale*error[2]],'T48(h)':[timescale*fittedParameters[3],timescale*error[3]]})
output_fit_control.to_csv('output_fit_params.csv')
