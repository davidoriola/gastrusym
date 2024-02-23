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
from scipy.integrate import odeint
import time as t
from scipy.optimize import fsolve
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

dir_path = os.path.abspath(os.getcwd()) # get path

allfiles = os.listdir(dir_path) # get all files in that path
files = [ fname for fname in allfiles if fname.endswith('.txt')] # find for all .txt files
files = sorted(files) # sort according to numbers

# group files depending on the conditions
files24h = [ fname for fname in allfiles if fname.startswith('24')] # 24h data
files48h = [ fname for fname in allfiles if fname.startswith('48')] # 24h data
filesTplus = [ fname for fname in allfiles if fname.startswith('Tplus')] # T+ data
filesTminus = [ fname for fname in allfiles if fname.startswith('Tminus')] # T- data

elasticity24=[]  # create empty dataset for the elasticity values at 24h
elasticity48=[]  # create empty dataset for the elasticity values at 48h
elasticityTplus=[] # create empty dataset for the elasticity values of T+ aggregates
elasticityTminus=[] # create empty dataset for the elasticity values of T- aggregates

# Read elasticity .txt files with 'Gastruloid','Eeff Tplus (Pa)','Eeff Tminus (Pa)','Major axis (a.u.)','Minor axis (a.u.)'
elasticity24 = [pd.read_csv(f,usecols=['Gastruloid','Eeff Tplus (Pa)','Eeff Tminus (Pa)','Major axis (a.u.)','Minor axis (a.u.)'], delimiter = "\t") for f in files24h ]
elasticity48 = [pd.read_csv(f,usecols=['Gastruloid','Eeff Tplus (Pa)','Eeff Tminus (Pa)','Major axis (a.u.)','Minor axis (a.u.)'], delimiter = "\t") for f in files48h ]
elasticityTplus = [pd.read_csv(f,usecols=['Gastruloid','Eeff Tplus (Pa)'], delimiter = "\t") for f in filesTplus ]
elasticityTminus = [pd.read_csv(f,usecols=['Gastruloid','Eeff Tminus (Pa)'], delimiter = "\t") for f in filesTminus ]
    

# dataset at 24h from different days
df_24_list = []
for i in range (0,len(elasticity24)):
    df_24_list.append(pd.DataFrame({'Condition':'24h','Eeff Tminus (Pa)':elasticity24[i]['Eeff Tminus (Pa)']}))

df_24 = pd.concat(df_24_list)

# dataset at 48h from different days
df_48_list = []
for i in range (0,len(elasticity48)):
    df_48_list.append(pd.DataFrame({'Condition':'48h','Eeff Tminus (Pa)':elasticity48[i]['Eeff Tminus (Pa)']}))

df_48 = pd.concat(df_48_list)

# dataset T+  from different days

df_Tplus_list = []
for i in range (0,len(elasticityTplus)):
    df_Tplus_list.append(pd.DataFrame({'Condition':'24h','Eeff Tplus (Pa)':elasticityTplus[i]['Eeff Tplus (Pa)']}))

df_Tplus = pd.concat(df_Tplus_list)

# dataset T-  from different days

df_Tminus_list = []
for i in range (0,len(elasticityTminus)):
    df_Tminus_list.append(pd.DataFrame({'Condition':'24h','Eeff Tminus (Pa)':elasticityTminus[i]['Eeff Tminus (Pa)']}))

df_Tminus = pd.concat(df_Tminus_list)

#number of samples in each condition
ngast24=np.size(df_24,0)
ngast48=np.size(df_48,0)
ngastTplus=np.size(df_Tplus,0)
ngastTminus=np.size(df_Tminus,0)

# Mean values

E24_mean = np.mean(df_24['Eeff Tminus (Pa)'])
E24_std = np.std(df_24['Eeff Tminus (Pa)'])

E48_mean = np.mean(df_48['Eeff Tminus (Pa)'])
E48_std = np.std(df_48['Eeff Tminus (Pa)'])

ETminus_mean = np.mean(df_Tminus['Eeff Tminus (Pa)'])
ETminus_std = np.std(df_Tminus['Eeff Tminus (Pa)'])

ETplus_mean = np.mean(df_Tplus['Eeff Tplus (Pa)'])
ETplus_std = np.std(df_Tplus['Eeff Tplus (Pa)'])

# Create output folder


#*********************************************                    
#Plot of E for 24h, T+ and T-
#*********************************************

# the shear modulus is Eeff/4 for an incompressible material

plt.figure(figsize=(3,5)) # set size
E_TplusTminus = plt.figure(figsize=(3,5))
#theta_vs_gfp, ax = plt.subplots()
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([0.5,4.5,0,60])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.ylabel(r'Shear modulus $\mu \ (Pa)$',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel('condition',fontsize=15, color = 'black')  # x-label fontsize + color
plt.scatter(np.random.normal(1, 0.05, ngast24),df_24['Eeff Tminus (Pa)']/4,facecolors='w', edgecolors='steelblue')
plt.scatter(np.random.normal(2, 0.05, ngast48),df_48['Eeff Tminus (Pa)']/4,facecolors='w', edgecolors='steelblue')
plt.scatter(np.random.normal(3, 0.05, ngastTminus),df_Tminus['Eeff Tminus (Pa)']/4,facecolors='w', edgecolors='gray')
plt.scatter(np.random.normal(4, 0.05, ngastTplus),df_Tplus['Eeff Tplus (Pa)']/4,facecolors='w', edgecolors='green')
#plt.errorbar([1,2,3,3],[np.mean(df_24['Eeff Tminus (Pa)']),np.mean(df_48['Eeff Tminus (Pa)']),np.mean(df_96_tminus['Eeff Tminus (Pa)']),np.mean(df_96_tplus['Eeff Tplus (Pa)'])],yerr=[np.std(df_24['Eeff Tminus (Pa)'])/np.sqrt(ngast24),np.std(df_48['Eeff Tminus (Pa)'])/np.sqrt(ngast48),np.std(df_96_tminus['Eeff Tminus (Pa)'])/np.sqrt(ngast96_tminus),np.std(df_96_tplus['Eeff Tplus (Pa)'])]/np.sqrt(ngast96_tplus),fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar([1,2],[np.mean(df_24['Eeff Tminus (Pa)'])/4,np.mean(df_48['Eeff Tminus (Pa)'])/4],yerr=[np.std(df_24['Eeff Tminus (Pa)']/4),np.std(df_48['Eeff Tminus (Pa)'])/4],marker='o', markersize=8.,ls ='None', color = "steelblue",ecolor='steelblue', capthick=2,elinewidth=2,capsize=5)
plt.errorbar([3],[np.mean(df_Tminus['Eeff Tminus (Pa)'])/4],yerr=[np.std(df_Tminus['Eeff Tminus (Pa)'])/4],marker='o', markersize=8.,ls ='None', color = "gray",ecolor='gray', capthick=2,elinewidth=2,capsize=5)
plt.errorbar([4],[np.mean(df_Tplus['Eeff Tplus (Pa)'])/4],yerr=[np.std(df_Tplus['Eeff Tplus (Pa)'])/4],marker='o', markersize=8.,ls ='None', color = "green",ecolor='green', capthick=2,elinewidth=2,capsize=5)


#plt.legend(frameon=False,loc="lower right")
plt.show() 
E_TplusTminus.savefig('output_E_24_48_TplusTminus.pdf',bbox_inches = "tight")   # save as .eps

#Output file with mean values
out_elasticity = pd.DataFrame({'Condition':['24h','48h','Tminus','Tplus'],'Eeff (Pa)':[E24_mean,E48_mean,ETminus_mean,ETplus_mean],'Eeff SD (Pa)':[E24_std,E48_std,ETminus_std,ETplus_std]}) # dataset from 2nd day

out_elasticity.to_csv('output_elasticity.csv', index=False)

