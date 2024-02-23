#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:31:53 2020

@author: oriola 


"""
# First load the utils folder in Nicola's app such that it has acces

import pickle, os    # packages to open pickle files
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pylab import *
import numpy as np
import scipy.io as sio
from numpy import diff
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import interpolate
import math, tqdm
from tifffile import imread
from skimage import img_as_bool
from scipy.ndimage import map_coordinates
from scipy.optimize import fsolve
from scipy.integrate import odeint
import time as t


# date of the experiment

# WARNING #### the number of timepoints should be the same in all gastruloids

dir_path = os.path.abspath(os.getcwd())

folder_data24 = os.path.join(dir_path,'Data24h')
folder_data48 = os.path.join(dir_path,'Data48h')

# gastruloids to analyze at 24 and 48h

gastrs24 = ['g02A','g02D','g02E','g02G','g04F','g04G','g04H','g06A','g06B','g06C','g06F','g08D','g10C','g10D','g10E']
gastrs48 = ['g04A','g04B','g04F','g04H','g05A','g05B','g05E','g05H','g06A','g06C','g06H']

num_gast24 = len(gastrs24) # number of gastruloids 
num_gast48 = len(gastrs48) # number of gastruloids 

# Get the filenames of all the .csv files with the morphometry
filenames24 = [os.path.join(folder_data24,g+'_morpho_straight_params.csv') for g in gastrs24]
filenames48 = [os.path.join(folder_data48,g+'_morpho_straight_params.csv') for g in gastrs48]

# load area and major axis

morpho_params24=[]
morpho_params24 = [pd.read_csv(f,usecols=['area','major_axis_length']) for f in filenames24]

morpho_params48=[]
morpho_params48 = [pd.read_csv(f,usecols=['area','major_axis_length']) for f in filenames48]

# define R(theta)
def R(theta):
    return pow(2,2/3)*pow(1+np.cos(theta),-2/3)*pow(2-np.cos(theta),-1/3)

# define FuncRoot_length(length_tot,theta)
def FuncRoot_length(theta,length_tot,length_init):
    return 4*length_tot/length_init - 2*R(theta)*(1+np.cos(theta))

# define Lambda(theta)
def Lambda(theta,epsilonY):
    return (2/(np.cos(theta)*(1+np.cos(theta))))*(2*(1+epsilonY)/(R(theta)*(1+np.cos(theta)))-1)

# Full equation
def ThetaDotTime(y,t,params):
    theta = y # unpack current values of y
    tau,betaN,epsilonY = params # unpack params
    return (2/(tau*np.tan(theta)))*pow(1/R(theta),3)*(4/pow(1+np.cos(theta),2)-betaN*Lambda(theta,epsilonY))


# create time array

timepoints = morpho_params24[0]['major_axis_length'].size  # get number of timepoints
dt = 10 # timestep in minutes
dx = 0.5979 # um per pixel
time = np.linspace(0,timepoints*dt/60,timepoints) # time in hours
T_max = timepoints*dt/60 # max time in hours

# Calculate effective sin2 theta from major axis length at 24h

sin2theta24 = np.zeros((num_gast24,timepoints))
radius24 = np.zeros((num_gast24,timepoints))

sin2theta48 = np.zeros((num_gast48,timepoints))
radius48 = np.zeros((num_gast48,timepoints))

for i in range(0,num_gast24):

    for j in range(0,timepoints):
        sin2theta24[i][j] = pow(np.sin(fsolve(FuncRoot_length,x0=0.5,args=(morpho_params24[i]['major_axis_length'][j],morpho_params24[i]['major_axis_length'][0]))),2)
    radius24[i] = dx*morpho_params24[i]['major_axis_length'][0]/4 # L(0)/4 in um
    
for i in range(0,num_gast48):

    for j in range(0,timepoints):
        sin2theta48[i][j] = pow(np.sin(fsolve(FuncRoot_length,x0=0.5,args=(morpho_params48[i]['major_axis_length'][j],morpho_params48[i]['major_axis_length'][0]))),2)
    radius48[i] = dx*morpho_params48[i]['major_axis_length'][0]/4 # L(0)/4 in um    

### 24h analysis ####
    
sin2_mean24 = np.nanmean(sin2theta24,axis=0)   
sin2_std24 = np.nanstd(sin2theta24,axis=0)   

radius_mean24 = np.nanmean(radius24,axis=0) 
radius_std24 = np.nanstd(radius24,axis=0) 

#### 48h analysis ####

sin2_mean48 = np.nanmean(sin2theta48,axis=0)   
sin2_std48 = np.nanstd(sin2theta48,axis=0)   

radius_mean48 = np.nanmean(radius48,axis=0) 
radius_std48 = np.nanstd(radius48,axis=0) 

####################### Numerically solve thetadot #########################3
    
y0 = 0.01 # initial angle to start the solver

# Create numerical function to fit to sin^2 theta (ONLY THE CASE OF NO YIELD STRAIN!)
def solNumSin2(time,tau, betaN):
    params = [tau, betaN,0] # Bundle parameter for ODE Solver
    return pow(np.sin(odeint(ThetaDotTime,y0,time,args=(params,))),2).flatten() # Call the ODE solver


####################### Fit to data #########################3
    
popt24,pcov24 = curve_fit(solNumSin2,time,sin2_mean24,bounds=((0,0), (np.inf,np.inf)))
popt48,pcov48 = curve_fit(solNumSin2,time,sin2_mean48,bounds=((0,0), (np.inf,np.inf)))
error_24 = np.sqrt(np.diag(pcov24)) # 1 standard deviation
error_48 = np.sqrt(np.diag(pcov48)) # 1 standard deviation

#################### Calculations and error propagation  ###########

# Elastocapillary length  lEc = gamma/E

lEC24 = radius_mean24/popt24[1] # mean value
lEC48 = radius_mean48/popt48[1] # mean value
    
error_lEC24 = lEC24*np.sqrt(pow(error_24[1]/popt24[1],2)+pow((radius_std24/np.sqrt(num_gast24))/radius_mean24,2))
error_lEC48 = lEC48*np.sqrt(pow(error_48[1]/popt48[1],2)+pow((radius_std48/np.sqrt(num_gast48))/radius_mean48,2))

# Viscocapillary velocity  vc = gamma/eta

vc24 = radius_mean24/(60*popt24[0])  #(um/min)
vc48 = radius_mean48/(60*popt48[0])  #(um/min)

error_vc24 = vc24*np.sqrt(pow(error_24[0]/popt24[0],2)+pow((radius_std24/np.sqrt(num_gast24))/radius_mean24,2))
error_vc48 = vc48*np.sqrt(pow(error_48[0]/popt48[0],2)+pow((radius_std48/np.sqrt(num_gast48))/radius_mean48,2))


## Plot sin2 
#
fsin2 = plt.figure(figsize=(4,4))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([0,T_max,0,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.ylabel(r'$\sin^2 \theta$',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel('time (h)',fontsize=15, color = 'black')  # x-label fontsize + color
plt.plot(time,sin2_mean24,linewidth=2,label="24h (n="+str(num_gast24)+")", color = 'gray')  # plot 
plt.plot(time,solNumSin2(time,popt24[0],popt24[1])) # fit plot
plt.fill_between(time,sin2_mean24-sin2_std24, sin2_mean24+sin2_std24,alpha=0.5, color = 'gray')
plt.plot(time,sin2_mean48,linewidth=2,label="48h (n="+str(num_gast48)+")", color = 'green')  # plot 
plt.plot(time,solNumSin2(time,popt48[0],popt48[1])) # fit plot
plt.fill_between(time,sin2_mean48-sin2_std48, sin2_mean48+sin2_std48,alpha=0.5, color = 'green')
plt.legend(frameon=False,loc="lower right")
plt.show()   
fsin2.savefig('Fusion_curve_mean_24h_48h.pdf',bbox_inches = "tight")   # save as .eps

# Ouput parameters

output_sorting = pd.DataFrame({'Condition':['24h','48h'],'lEC (um)':[lEC24[0],lEC48[0]],'err_lEC (um)':[error_lEC24[0],error_lEC48[0]],'vc (um/min)':[vc24[0],vc48[0]],'err_vc (um/min)':[error_vc24[0],error_vc48[0]]})

output_sorting.to_csv('output_24h_48h.csv')

