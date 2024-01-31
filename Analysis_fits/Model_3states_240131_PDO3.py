
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

# Analytical solution for phi2(t) with no feedback
def phi2_fit_nosignal(phi20,alpha, T1):
    phi0=1-phi20
    return (1-phi0)*np.exp(-alpha*T1)+(phi0/(alpha-1))*(np.exp(-T1)-np.exp(-alpha*T1))
    
# Multi function to fit phi2 at 3 different time points with 4 parameters
def combo_phi2_fit0_no_signal(comboData,alpha, T24, T48):
    
    extract1 = comboData[:5] # first data
    extract2 = comboData[5:10] # second data
    extract3 = comboData[10:15] # second data
    
    result24 = phi2_fit_nosignal(extract1,alpha, T24)
    result48 = phi2_fit_nosignal(extract2,alpha, T48)
    result72 = phi2_fit_nosignal(extract3,alpha, T48+(T48-T24))
    
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

# Extrapolate maximum GFP value for normalization 

popt_211130,pcov_211130 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_211130[data_GFP_24h_211130['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220118,pcov_220118 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220118[data_GFP_24h_220118['Condition']=='24h control']['GFP fluo'])[0:4])
popt_220125,pcov_220125 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),np.flipud(data_GFP_24h_220125[data_GFP_24h_220125['Condition']=='24h control']['GFP fluo'])[0:4])

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


# Compile data

data_GFP_24h=pd.concat([data_GFP_24h_211130, data_GFP_24h_220118, data_GFP_24h_220125]) 
data_GFP_48h=pd.concat([data_GFP_48h_211130, data_GFP_48h_220118, data_GFP_48h_220125]) 
data_GFP_72h=pd.concat([data_GFP_72h_211130, data_GFP_72h_220118, data_GFP_72h_220125]) 

num_experiments =3

# Remove 'Unnamed: 0' if necessary
if 'Unnamed: 0' in data_GFP_24h.columns:
    del data_GFP_24h['Unnamed: 0']

if 'Unnamed: 0' in data_GFP_48h.columns:
    del data_GFP_48h['Unnamed: 0']
    
if 'Unnamed: 0' in data_GFP_72h.columns:
    del data_GFP_72h['Unnamed: 0']
        
# Take the PDO3 conditions

data_GFP_24h_PDO3 = data_GFP_24h[data_GFP_24h['Condition']=='24h PDO3'] 
data_GFP_48h_PDO3 = data_GFP_48h[data_GFP_48h['Condition']=='48h PDO3']  
data_GFP_72h_PDO3 = data_GFP_72h[data_GFP_72h['Condition']=='72h PDO3'] 


# Get the mean and SD #24h

data_GFP_24h_mean_PDO3 = data_GFP_24h_PDO3.groupby('GFP fraction', as_index=False).mean()    
data_GFP_24h_std_PDO3 = data_GFP_24h_PDO3.groupby('GFP fraction', as_index=False).std()    
  
data_GFP_24h_mean_PDO3['Num samples']=data_GFP_24h_PDO3.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_24h_std_PDO3['Num samples']=data_GFP_24h_PDO3.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  


# Get the mean and SD #48h

data_GFP_48h_mean_PDO3 = data_GFP_48h_PDO3.groupby('GFP fraction', as_index=False).mean()    
data_GFP_48h_std_PDO3 = data_GFP_48h_PDO3.groupby('GFP fraction', as_index=False).std()    
 
data_GFP_48h_mean_PDO3['Num samples']=data_GFP_48h_PDO3.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_48h_std_PDO3['Num samples']=data_GFP_48h_PDO3.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  

# Get the mean and SD #72h

data_GFP_72h_mean_PDO3 = data_GFP_72h_PDO3.groupby('GFP fraction', as_index=False).mean()    
data_GFP_72h_std_PDO3= data_GFP_72h_PDO3.groupby('GFP fraction', as_index=False).std()    

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
comboY = np.concatenate([data_GFP_24h_mean_PDO3['GFP fluo norm'],data_GFP_48h_mean_PDO3['GFP fluo norm'],data_GFP_72h_mean_PDO3['GFP fluo norm']])         

# Perform fit

fittedParameters,pcov = curve_fit(combo_phi2_fit0_no_signal,comboX,comboY,bounds=((0.001,0.001,0.001),(inf,inf,inf)))

# Compute error of the fit

error = np.sqrt(np.diag(pcov)) 

# Compute timescale of the system
timescale = 24/(fittedParameters[2]-fittedParameters[1])  # (time in hours equivalent to 1 u.a. of time)

############ BEGIN SOLVE MODEL WITH FITTED PARAMETERS ##########################################################################################

# parameters Model  (dimensionless)

alpha=fittedParameters[0] # (q/p)
K=fittedParameters[1]

def three_state_model(y, t, params):
    phi1, phi2, phi3 = y
    alpha, beta, K = params #unpack params
    dydt = [-phi1, phi1-alpha*phi2, alpha*phi2]
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

# Plot experimental data and fit to the model for the PDO3 case

j=20 # initial condition (phi2(0)=0.2)

nSEM=2 #number of SEM

s2fig  = plt.figure(figsize=(4,4))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([-0.05,1.05,0,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.ylabel(r'$\phi_2$',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'$\phi_2(0)$',fontsize=15, color = 'black')  # x-label fontsize + color # fit plot
plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,fittedParameters[0],fittedParameters[1]),'--',color='gray')
plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,fittedParameters[0],fittedParameters[2]),'--',color='cadetblue')
plt.plot(phi0_array,phi2_fit_nosignal(phi0_array,fittedParameters[0],fittedParameters[2]+(fittedParameters[2]-fittedParameters[1])),'--',color='black')
plt.errorbar(data_GFP_24h_mean_PDO3['GFP fraction'],data_GFP_24h_mean_PDO3['GFP fluo norm'],yerr=nSEM*data_GFP_24h_std_PDO3['GFP fluo norm']/np.sqrt(data_GFP_24h_mean_PDO3['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='24h',color='gray')
plt.errorbar(data_GFP_48h_mean_PDO3['GFP fraction'],data_GFP_48h_mean_PDO3['GFP fluo norm'],yerr=nSEM*data_GFP_48h_std_PDO3['GFP fluo norm']/np.sqrt(data_GFP_48h_mean_PDO3['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='48h',color='cadetblue')
plt.errorbar(data_GFP_72h_mean_PDO3['GFP fraction'],data_GFP_72h_mean_PDO3['GFP fluo norm'],yerr=nSEM*data_GFP_72h_std_PDO3['GFP fluo norm']/np.sqrt(data_GFP_72h_mean_PDO3['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='72h',color='black')
plt.legend(loc='upper center',ncol=3, fontsize =11, frameon= False)
plt.show()
s2fig.savefig('T_dynamics_240131_PDO3.pdf',bbox_inches = "tight")   # save as .eps

modelfig  = plt.figure(figsize=(4,4))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([0,60,0,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.plot(t*timescale, sol[j][:,0], 'magenta', label=r'$\phi_1$')
plt.plot(t*timescale, sol[j][:,1], 'green', label=r'$\phi_2$')
plt.plot(t*timescale, sol[j][:,2], 'orange', label=r'$\phi_3$')
plt.plot(fittedParameters[1]*time_point_x*timescale ,time_point_y,'--',color='gray')    #no signal
plt.plot(fittedParameters[2]*time_point_x*timescale ,time_point_y,'--',color='cadetblue')   #no signal
plt.plot((fittedParameters[2]+(fittedParameters[2]-fittedParameters[1]))*time_point_x*timescale,time_point_y,'--',color='black')  #no signal
plt.xlabel('signalling time (h)')
plt.show()
modelfig.savefig('modelfig_240131_PDO3.pdf',bbox_inches = "tight")   # save as .eps


