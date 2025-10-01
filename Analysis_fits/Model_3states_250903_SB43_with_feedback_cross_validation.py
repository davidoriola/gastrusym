
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import random
from scipy.stats import chisquare
from sklearn.metrics import r2_score

sns.set_palette("PuBuGn_d")

# linear fit
def linear_fit(phi20,a):
    return a*phi20
       
# Define Lambert function
# IMPORTANT: In this model the time is reescaled by q!
def fun(phi0,alpha,K,T1):
    return K*lambertw((1/K)*phi0*np.exp(-alpha*T1+phi0/K)).real

# Analytical solution for phi2(t) with feedback
# IMPORTANT: In this model the time is reescaled by q!
def phi2_fit0(phi20,alpha, K, T1):
    phi0=1-phi20
    if alpha == 1:
        return (fun(phi0,alpha,K,T1)/phi0)*(1-phi0+phi0*np.log(phi0/K)-phi0*np.log(fun(phi0,alpha,K,T1)/K))
    if alpha == 0:
        return (1-phi0)*np.exp(-t/(1+phi0/K))
    else: 
        return (1/(1-alpha))*(alpha*fun(phi0,alpha,K,T1)+(1-phi0-alpha)*pow(fun(phi0,alpha,K,T1)/phi0,1/alpha))
    
# Multi function to fit phi2 at 3 different time points with 4 parameters
def combo_phi2_fit0(comboData,alpha, K, T24, T48):
    
    extract1 = comboData[:5] # first data
    extract2 = comboData[5:10] # second data
    extract3 = comboData[10:15] # second data
    
    result24 = phi2_fit0(extract1,alpha, K, T24)
    result48 = phi2_fit0(extract2,alpha, K, T48)
    result72 = phi2_fit0(extract3,alpha, K, T48+(T48-T24))
    
    return np.concatenate([result24,result48,result72])   

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


# We define three different states phi1: A, phi2: B, phi3: C

# get paths for each csv file

dir_path = os.path.abspath(os.getcwd()) # get path


# Experiment 220705

output_unmixing_analysis_24h_GFP_220705 =os.path.join(dir_path,'data_24_GFP_Ctrl_XAV_SB43_220705.csv')
output_unmixing_analysis_48h_GFP_220705 =os.path.join(dir_path,'data_48_GFP_Ctrl_XAV_SB43_220706.csv')
output_unmixing_analysis_72h_GFP_220705 =os.path.join(dir_path,'data_72_GFP_Ctrl_XAV_SB43_220707.csv')

# Experiment 220712

output_unmixing_analysis_24h_GFP_220712 =os.path.join(dir_path,'data_24_GFP_Ctrl_XAV_SB43_220712.csv')
output_unmixing_analysis_48h_GFP_220712 =os.path.join(dir_path,'data_48_GFP_Ctrl_XAV_SB43_220713.csv')
output_unmixing_analysis_72h_GFP_220712 =os.path.join(dir_path,'data_72_GFP_Ctrl_XAV_SB43_220714.csv')


# get data

data_GFP_24h_220705=pd.read_csv(output_unmixing_analysis_24h_GFP_220705)
data_GFP_48h_220705=pd.read_csv(output_unmixing_analysis_48h_GFP_220705)
data_GFP_72h_220705=pd.read_csv(output_unmixing_analysis_72h_GFP_220705)

data_GFP_24h_220712=pd.read_csv(output_unmixing_analysis_24h_GFP_220712)
data_GFP_48h_220712=pd.read_csv(output_unmixing_analysis_48h_GFP_220712)
data_GFP_72h_220712=pd.read_csv(output_unmixing_analysis_72h_GFP_220712)

# Separate Conditions
    
data_GFP_24_control_220705 = data_GFP_24h_220705[data_GFP_24h_220705['Condition']=='24h Ctrl']  
data_GFP_24_SB43_220705 = data_GFP_24h_220705[data_GFP_24h_220705['Condition']=='24h SB43']  
data_GFP_24_XAV_220705 = data_GFP_24h_220705[data_GFP_24h_220705['Condition']=='24h XAV']  
data_GFP_24_control_220712 = data_GFP_24h_220712[data_GFP_24h_220712['Condition']=='24h Ctrl']  
data_GFP_24_SB43_220712 = data_GFP_24h_220712[data_GFP_24h_220712['Condition']=='24h SB43']  
data_GFP_24_XAV_220712 = data_GFP_24h_220712[data_GFP_24h_220712['Condition']=='24h XAV']  

data_GFP_48_control_220706 = data_GFP_48h_220705[data_GFP_48h_220705['Condition']=='48h Ctrl']  
data_GFP_48_SB43_220706 = data_GFP_48h_220705[data_GFP_48h_220705['Condition']=='48h SB43']  
data_GFP_48_XAV_220706 = data_GFP_48h_220705[data_GFP_48h_220705['Condition']=='48h XAV']  
data_GFP_48_control_220713 = data_GFP_48h_220712[data_GFP_48h_220712['Condition']=='48h Ctrl']  
data_GFP_48_SB43_220713 = data_GFP_48h_220712[data_GFP_48h_220712['Condition']=='48h SB43']  
data_GFP_48_XAV_220713 = data_GFP_48h_220712[data_GFP_48h_220712['Condition']=='48h XAV']  

data_GFP_72_control_220707 = data_GFP_72h_220705[data_GFP_72h_220705['Condition']=='72h Ctrl']  
data_GFP_72_SB43_220707 = data_GFP_72h_220705[data_GFP_72h_220705['Condition']=='72h SB43']  
data_GFP_72_XAV_220707 = data_GFP_72h_220705[data_GFP_72h_220705['Condition']=='72h XAV']  
data_GFP_72_control_220714 = data_GFP_72h_220712[data_GFP_72h_220712['Condition']=='72h Ctrl']  
data_GFP_72_SB43_220714 = data_GFP_72h_220712[data_GFP_72h_220712['Condition']=='72h SB43']  
data_GFP_72_XAV_220714 = data_GFP_72h_220712[data_GFP_72h_220712['Condition']=='72h XAV']  

#sort according to GFP fraction

data_GFP_24_SB43_220705=data_GFP_24_SB43_220705.sort_values(by=['GFP fraction'])
data_GFP_24_SB43_220712=data_GFP_24_SB43_220712.sort_values(by=['GFP fraction'])

data_GFP_48_SB43_220706=data_GFP_48_SB43_220706.sort_values(by=['GFP fraction'])
data_GFP_48_SB43_220713=data_GFP_48_SB43_220713.sort_values(by=['GFP fraction'])

data_GFP_72_SB43_220707=data_GFP_72_SB43_220707.sort_values(by=['GFP fraction'])
data_GFP_72_SB43_220714=data_GFP_72_SB43_220714.sort_values(by=['GFP fraction'])

# Extrapolate maximum GFP value for normalization 

popt_220705,pcov_220705 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),data_GFP_24_SB43_220705[data_GFP_24_SB43_220705['Condition']=='24h SB43']['GFP fluo'][0:4])
popt_220712,pcov_220712 = curve_fit(linear_fit,np.array([0.0,0.2,0.5,0.8]),data_GFP_24_SB43_220712[data_GFP_24_SB43_220712['Condition']=='24h SB43']['GFP fluo'][0:4])

# I normalise by the intensity

data_GFP_24_SB43_220705['GFP fluo norm']=data_GFP_24_SB43_220705['GFP fluo']/popt_220705
data_GFP_48_SB43_220706['GFP fluo norm']=data_GFP_48_SB43_220706['GFP fluo']/popt_220705
data_GFP_72_SB43_220707['GFP fluo norm']=data_GFP_72_SB43_220707['GFP fluo']/popt_220705

data_GFP_24_SB43_220712['GFP fluo norm']=data_GFP_24_SB43_220712['GFP fluo']/popt_220712
data_GFP_48_SB43_220713['GFP fluo norm']=data_GFP_48_SB43_220713['GFP fluo']/popt_220712
data_GFP_72_SB43_220714['GFP fluo norm']=data_GFP_72_SB43_220714['GFP fluo']/popt_220712

# Compile data

data_GFP_24h=pd.concat([data_GFP_24_SB43_220705, data_GFP_24_SB43_220712]) 
data_GFP_48h=pd.concat([data_GFP_48_SB43_220706, data_GFP_48_SB43_220713]) 
data_GFP_72h=pd.concat([data_GFP_72_SB43_220707, data_GFP_72_SB43_220714]) 

num_experiments =2

# Remove 'Unnamed: 0' if necessary
if 'Unnamed: 0' in data_GFP_24h.columns:
    del data_GFP_24h['Unnamed: 0']

if 'Unnamed: 0' in data_GFP_48h.columns:
    del data_GFP_48h['Unnamed: 0']
    
if 'Unnamed: 0' in data_GFP_72h.columns:
    del data_GFP_72h['Unnamed: 0']
        
# Take the PDO3 conditions

data_GFP_24h_PDO3 = data_GFP_24h[data_GFP_24h['Condition']=='24h SB43'] 
data_GFP_48h_PDO3 = data_GFP_48h[data_GFP_48h['Condition']=='48h SB43']  
data_GFP_72h_PDO3 = data_GFP_72h[data_GFP_72h['Condition']=='72h SB43'] 


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

# --------------------------------------------------------------------------------
# K-fold cross-validation setup---------------------------------------------------
# --------------------------------------------------------------------------------
kf = KFold(n_splits=14, shuffle=True, random_state=42)
# To store results
param_list = []
mse_scores = []

for train_index, test_index in kf.split(comboX):
    x_train, x_test = comboX[train_index], comboX[test_index]
    y_train, y_test = comboY[train_index], comboY[test_index]

    # Fit model to training data
    try:
        fittedParameters0,pcov = curve_fit(combo_phi2_fit0,x_train, y_train,bounds=((0.001,0.001,0.001,0.001),(inf,inf,inf,inf)))
        param_list.append(fittedParameters0)
        y_pred = combo_phi2_fit0(x_test, *fittedParameters0)
        mse = mean_squared_error(y_test, y_pred)
        #popt, _ = curve_fit(model_func, x_train, y_train, p0=[1, 1, 0, 0])
       # y_pred = model_func(x_test, *popt)
       # mse = mean_squared_error(y_test, y_pred)
    except RuntimeError:
        # If fitting fails, assign a high error
        popt = [np.nan] * 4
        mse = np.inf
        param_list.append(popt)

    mse_scores.append(mse)
    
# Convert to numpy array for analysis
params_array = np.array(param_list)
    
# Mask out failed fits (nan)
valid_params = params_array[~np.isnan(params_array).any(axis=1)]

# Compute Z-scores
z_scores = np.abs(zscore(valid_params, axis=0))

# Keep rows where all parameter Z-scores are < threshold (e.g., 2.5)
keep_mask = (z_scores < 1).all(axis=1)
filtered_params = valid_params[keep_mask]

# Compute mean and std of parameters
fittedParameters = np.median(filtered_params, axis=0) # median
error = np.std(filtered_params, axis=0)

# --------------------------------------------------------------------------------

alpha=fittedParameters[0] # 
K=fittedParameters[1] # 

# Compute timescale of the system
timescale = 24/(fittedParameters[3]-fittedParameters[2]) # (time in hours equivalent to 1 u.a. of time)
q=1/timescale #(1/h)
error_q = (np.sqrt(pow(error[2],2)+pow(error[3],2)))/24 #(1/h)
p=alpha*q #(1/h)
error_p = p*np.sqrt(pow((error[0]/alpha),2)+pow((error_q /q),2)) #(1/h)
time_array = [fittedParameters[2]*timescale,fittedParameters[3]*timescale,(fittedParameters[3]+(fittedParameters[3]-fittedParameters[2]))*timescale] # time array in signalling time

time_array_realtime = [24,48,72] # time array in hpa

############ BEGIN SOLVE MODEL WITH FITTED PARAMETERS ##########################################################################################

# parameters Model  (dimensionless)

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

# Plot experimental data and fit to the model for the control case

nSEM=2 #number of SEM

# Create timelapse data
data_phi0_mean = []
data_phi0_std = []
phi0_value_array = [0.0,0.2,0.5,0.8,1.0]

for i in range(0,5): # loop over phi
    phi0_value=phi0_value_array[i]
    data_phi0_mean.append([data_GFP_24h_mean_PDO3[data_GFP_24h_mean_PDO3['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0],data_GFP_48h_mean_PDO3[data_GFP_48h_mean_PDO3['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0],data_GFP_72h_mean_PDO3[data_GFP_72h_mean_PDO3['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0]])
    data_phi0_std.append([nSEM*data_GFP_24h_std_PDO3[data_GFP_24h_std_PDO3['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0]/np.sqrt(data_GFP_24h_std_PDO3[data_GFP_24h_std_PDO3['GFP fraction']==phi0_value]['Num samples'].iloc[0]),nSEM*data_GFP_48h_std_PDO3[data_GFP_48h_std_PDO3['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0]/np.sqrt(data_GFP_48h_std_PDO3[data_GFP_48h_std_PDO3['GFP fraction']==phi0_value]['Num samples'].iloc[0]),nSEM*data_GFP_72h_std_PDO3[data_GFP_72h_std_PDO3['GFP fraction']==phi0_value]['GFP fluo norm'].iloc[0]/np.sqrt(data_GFP_72h_std_PDO3[data_GFP_72h_std_PDO3['GFP fraction']==phi0_value]['Num samples'].iloc[0])])

# Plot experimental data and fit to the model for the PDO3 case

j=20 # initial condition (phi2(0)=0.2)

nSEM=2 #number of SEM

s2fig  = plt.figure(figsize=(4,4))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([-0.05,1.05,-0.05,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.ylabel(r'$\phi_2$',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'$\phi_2(0)$',fontsize=15, color = 'black')  # x-label fontsize + color # fit plot
plt.plot(phi0_array,phi2_fit0(phi0_array,fittedParameters[0],fittedParameters[1],fittedParameters[2]),'-',color='gray')
plt.plot(phi0_array,phi2_fit0(phi0_array,fittedParameters[0],fittedParameters[1],fittedParameters[3]),'-',color='cadetblue')
plt.plot(phi0_array,phi2_fit0(phi0_array,fittedParameters[0],fittedParameters[1],fittedParameters[3]+(fittedParameters[3]-fittedParameters[2])),'-',color='black')
plt.errorbar(data_GFP_24h_mean_PDO3['GFP fraction'],data_GFP_24h_mean_PDO3['GFP fluo norm'],yerr=nSEM*data_GFP_24h_std_PDO3['GFP fluo norm']/np.sqrt(data_GFP_24h_mean_PDO3['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='24h',color='gray')
plt.errorbar(data_GFP_48h_mean_PDO3['GFP fraction'],data_GFP_48h_mean_PDO3['GFP fluo norm'],yerr=nSEM*data_GFP_48h_std_PDO3['GFP fluo norm']/np.sqrt(data_GFP_48h_mean_PDO3['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='48h',color='cadetblue')
plt.errorbar(data_GFP_72h_mean_PDO3['GFP fraction'],data_GFP_72h_mean_PDO3['GFP fluo norm'],yerr=nSEM*data_GFP_72h_std_PDO3['GFP fluo norm']/np.sqrt(data_GFP_72h_mean_PDO3['Num samples']),marker='o',fmt=' ',capthick=2,capsize=5, label='72h',color='black')
plt.legend(loc='upper center',ncol=3, fontsize =11, frameon= False)
plt.show()
s2fig.savefig('T_dynamics_250304_SB43.pdf',bbox_inches = "tight")   # save as .eps

# modelfig  = plt.figure(figsize=(4,4))
# rc('axes',linewidth=2)  # box thickness
# rc('font',size = 15)   # font size ticks
# plt.axis([0,60,0,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
# plt.plot(t*timescale, sol[j][:,0], 'magenta', label=r'$\phi_1$')
# plt.plot(t*timescale, sol[j][:,1], 'green', label=r'$\phi_2$')
# plt.plot(t*timescale, sol[j][:,2], 'brown', label=r'$\phi_3$')
# plt.errorbar(time_array,data_phi0_mean[1],yerr=data_phi0_std[1],marker='o',fmt=' ',capthick=2,capsize=5,color='green')
# plt.plot(fittedParameters[1]*time_point_x*timescale ,time_point_y,'--',color='gray')    #no signal
# plt.plot(fittedParameters[2]*time_point_x*timescale ,time_point_y,'--',color='cadetblue')   #no signal
# plt.plot((fittedParameters[2]+(fittedParameters[2]-fittedParameters[1]))*time_point_x*timescale,time_point_y,'--',color='black')  #no signal
# plt.xlabel('signalling time (h)')
# plt.show()
# modelfig.savefig('modelfig_240131_PDO3.pdf',bbox_inches = "tight")   # save as .eps

output_fit_SB43 = pd.DataFrame({'p':[p,error_p],'q':[q,error_q],'K':[fittedParameters[1],error[1]],'T24(h)':[timescale*fittedParameters[2],timescale*error[2]],'T48(h)':[timescale*fittedParameters[3],timescale*error[3]]})
#output_fit_XAV.to_csv('output_fit_params_XAV_feedback_250306.csv')

