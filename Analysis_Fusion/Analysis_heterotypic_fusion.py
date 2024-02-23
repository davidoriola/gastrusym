#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:31:53 2020

@author: oriola 


"""
# First load the utils folder in Nicola's app such that it has acces

import sys, time, tqdm, copy, os, glob
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
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.integrate import odeint
from morgana.DatasetTools import arrangemorphodata
from morgana.DatasetTools import arrangefluodata
import time as t

dir_path = os.path.abspath(os.getcwd())

folder_data= os.path.join(dir_path,'DataTpTm')

# load data from T- and T+ 

output_sin2_Tminus_data = os.path.join(dir_path,'output_sin2_Tminus.csv')
output_sin2_Tplus_data = os.path.join(dir_path,'output_sin2_Tplus.csv')
output_info_Tplus_Tminus_data = os.path.join(dir_path,'output_fits_Tplus_Tminus.csv')

sin2_Tminus_data = pd.read_csv(output_sin2_Tminus_data) 
sin2_Tplus_data = pd.read_csv(output_sin2_Tplus_data) 
info_Tplus_Tminus_data = pd.read_csv(output_info_Tplus_Tminus_data) 

#####################################################

# gastruloids to analyze for TpTm 

gastrs = ['A07','A08','A09','B07','B08','B09','B10','C10','D07','D08','D09','D10','E07','E09','E10','F09','F10','G07','G08','H07','H09','H10']

# ******************************************************************
# From now on in the code 24 -> TmTp 
# ******************************************************************


# Get the filenames of all the .csv files with the morphometry
image_folders = [os.path.join(folder_data,g) for g in gastrs]

# number of fusion events
num_gast24 = np.size(image_folders)

###########################################################################

### Select morphological parameters to be quantified

maskType = 'Straightened' # Use 'Unprocessed' or 'Straightened' binary mask
Timelapse = True # Do images in the folder belong to a timelapse?

all_morpho_params = False # select true if all parameters are to be used.
# otherwise, select which paramters you would like to compute.
area = True
eccentricity = False
major_axis_length = True
minor_axis_length = False
equivalent_diameter = False
perimeter = False
euler_number = False
extent = False
form_factor = False
orientation = False
locoefa_coeff = False

# Define number of groups/number of conditions and the image subfolders that belong to each group
#groups = [image_folders[0], image_folders[1]]
group1 = image_folders
groups = [group1]

###########################################################################

###########################################################################

morphoKeys = ['area',
              'eccentricity',
              'major_axis_length',
              'minor_axis_length',
              'equivalent_diameter',
              'perimeter',
              'euler_number',
              'extent',
              'form_factor',
              'orientation',
              'locoefa_coeff']

if all_morpho_params:
    computeMorpho = [True for key in morphoKeys]
else:
    computeMorpho = [area, eccentricity, major_axis_length, minor_axis_length, equivalent_diameter, 
                     perimeter, euler_number, extent, form_factor, orientation, locoefa_coeff]
    
# extract data from all the folders
data_all, keys = arrangemorphodata.collect_morpho_data( groups, 
                                                        morphoKeys, 
                                                        computeMorpho, 
                                                        maskType, 
                                                        Timelapse
                                                        )

###########################################################################

# define R(theta)
def R(theta):
    return pow(2,2/3)*pow(1+np.cos(theta),-2/3)*pow(2-np.cos(theta),-1/3)

# define FuncRoot_length(length_tot,theta)
def FuncRoot_length(theta,length_tot,length_init):
    return 4*length_tot/length_init - 2*R(theta)*(1+np.cos(theta))


# number of fusion events
num_gast24 = np.size(image_folders)

# create time array

timepoints = np.size(data_all[0].major_axis_length[0],0)  # get number of timepoints
dt = 10 # timestep in minutes
dx = 0.5979 # um per pixel
time = np.linspace(0,timepoints*dt/60,timepoints) # time in hours
T_max = timepoints*dt/60 # max time in hours

sin2theta24 = np.zeros((num_gast24,timepoints))

# Calculate effective sin2 theta from major axis length at 24h

for i in range(0,num_gast24):
    for j in range(0,timepoints):
        sin2theta24[i][j] = pow(np.sin(fsolve(FuncRoot_length,x0=0.5,args=(data_all[0].major_axis_length[i][j],data_all[0].major_axis_length[i][0]))),2)


### 24h analysis ####
      
sin2_mean24 = np.nanmean(sin2theta24,axis=0)   
sin2_std24 = np.nanstd(sin2theta24,axis=0)   

# define Lambda(theta)
def Lambda(theta,epsilonY):
    return (2/(np.cos(theta)*(1+np.cos(theta))))*(2*(1+epsilonY)/(R(theta)*(1+np.cos(theta)))-1)

# Full equation
def ThetaDotTime(y,t,params):
    theta = y # unpack current values of y
    tau,betaN,epsilonY = params # unpack params
    return (2/(tau*np.tan(theta)))*pow(1/R(theta),3)*(4/pow(1+np.cos(theta),2)-betaN*Lambda(theta,epsilonY))

####################### Numerically solve thetadot #########################3
    
y0 = 0.1 # initial angle to start the solver

# Create numerical function to fit to sin^2 theta (ONLY THE CASE OF NO YIELD STRAIN!)
def solNumSin2(time,tau, betaN):
    params = [tau, betaN,0] # Bundle parameter for ODE Solver
    return pow(np.sin(odeint(ThetaDotTime,y0,time,args=(params,))),2).flatten() # Call the ODE solver

####################### Fit to data #########################3

## Plot sin2 
#
fsin2 = plt.figure(figsize=(4,4))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([0,10,0,1])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.ylabel(r'$\sin^2 \theta$',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel('time (h)',fontsize=15, color = 'black')  # x-label fontsize + color
plt.plot(time,solNumSin2(time,info_Tplus_Tminus_data['tau (h)'][0],info_Tplus_Tminus_data['beta'][0]),color='gray') # fit plot
plt.fill_between(time,sin2_mean24-sin2_std24, sin2_mean24+sin2_std24,alpha=0.5, color = 'sandybrown')
plt.plot(sin2_Tminus_data['Time (h)'],sin2_Tminus_data['Sin2theta_mean'],linewidth=2,label="-/- (n="+str(info_Tplus_Tminus_data ['nsamples'][0])+")", color = 'gray')  # plot 
plt.plot(time,sin2_mean24,linewidth=2,label="+/- (n="+str(num_gast24)+")", color = 'sandybrown')  # plot 
plt.fill_between(sin2_Tminus_data['Time (h)'],sin2_Tminus_data['Sin2theta_mean']-sin2_Tminus_data['Sin2theta_std'], sin2_Tminus_data['Sin2theta_mean']+sin2_Tminus_data['Sin2theta_std'],alpha=0.5, color = 'gray')
plt.plot(sin2_Tplus_data['Time (h)'],sin2_Tplus_data['Sin2theta_mean'],linewidth=2,label="+/+ (n="+str(info_Tplus_Tminus_data ['nsamples'][1])+")", color = 'green')  # plot 
plt.fill_between(sin2_Tplus_data['Time (h)'],sin2_Tplus_data['Sin2theta_mean']-sin2_Tplus_data['Sin2theta_std'],sin2_Tplus_data['Sin2theta_mean']+sin2_Tplus_data['Sin2theta_std'],alpha=0.5, color = 'green')
plt.plot(time,solNumSin2(time,info_Tplus_Tminus_data['tau (h)'][1],info_Tplus_Tminus_data['beta'][1]),color='green') # fit plot
plt.legend(frameon=False,loc="lower right")
plt.show()  

fsin2.savefig('sin2theta_heterotypic.pdf',bbox_inches = "tight")   # save as .eps
 
# Output sin2 curves

df_sin2_TminusTplus = pd.DataFrame({'Time (h)':time,'Sin2theta_mean':sin2_mean24,'Sin2theta_std':sin2_std24})

df_sin2_TminusTplus.to_csv('output_sin2_TplusTminus.csv')
