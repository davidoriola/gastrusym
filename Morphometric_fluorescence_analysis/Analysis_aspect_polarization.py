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
import math, tqdm
from tifffile import imread
from skimage import img_as_bool
from scipy.ndimage import map_coordinates
import time as t
import pathlib


sns.set_palette("ch:s=-.2,r=.9")

#mac
dir_path = os.path.abspath(os.getcwd()) # get path

#windows
#dir_path = pathlib.Path().absolute()

# get paths for each csv file

# Experiment 220118

data_unmixing_analysis_24h_GFP_220118 =os.path.join(dir_path,'data_24_AP_analysis_220118.csv')
data_unmixing_analysis_48h_GFP_220118 =os.path.join(dir_path,'data_48_AP_analysis_220119.csv')
data_unmixing_analysis_72h_GFP_220118 =os.path.join(dir_path,'data_72_AP_analysis_220120.csv')
data_unmixing_analysis_96h_GFP_220118 =os.path.join(dir_path,'data_96_AP_analysis_220121.csv')


# Experiment 220125

data_unmixing_analysis_24h_GFP_220125 =os.path.join(dir_path,'data_24_AP_analysis_220125.csv')
data_unmixing_analysis_48h_GFP_220125 =os.path.join(dir_path,'data_48_AP_analysis_220126.csv')
data_unmixing_analysis_72h_GFP_220125 =os.path.join(dir_path,'data_72_AP_analysis_220127.csv')
data_unmixing_analysis_96h_GFP_220125 =os.path.join(dir_path,'data_96_AP_analysis_220128.csv')

# Experiment 220503

data_unmixing_analysis_24h_GFP_220503 =os.path.join(dir_path,'data_24_AP_analysis_220503.csv')
data_unmixing_analysis_48h_GFP_220503 =os.path.join(dir_path,'data_48_AP_analysis_220504.csv')
data_unmixing_analysis_72h_GFP_220503 =os.path.join(dir_path,'data_72_AP_analysis_220505.csv')
data_unmixing_analysis_96h_GFP_220503 =os.path.join(dir_path,'data_96_AP_analysis_220506.csv')

# Experiment 220705

data_unmixing_analysis_24h_GFP_220705 =os.path.join(dir_path,'data_24_AP_analysis_220705.csv')
data_unmixing_analysis_48h_GFP_220705 =os.path.join(dir_path,'data_48_AP_analysis_220706.csv')
data_unmixing_analysis_72h_GFP_220705 =os.path.join(dir_path,'data_72_AP_analysis_220707.csv')
data_unmixing_analysis_96h_GFP_220705 =os.path.join(dir_path,'data_96_AP_analysis_220708.csv')


# Experiment 220712

data_unmixing_analysis_24h_GFP_220712 =os.path.join(dir_path,'data_24_AP_analysis_220712.csv')
data_unmixing_analysis_48h_GFP_220712 =os.path.join(dir_path,'data_48_AP_analysis_220713.csv')
data_unmixing_analysis_72h_GFP_220712 =os.path.join(dir_path,'data_72_AP_analysis_220714.csv')
data_unmixing_analysis_96h_GFP_220712 =os.path.join(dir_path,'data_96_AP_analysis_220715.csv')


# get data

data_GFP_24h_220118=pd.read_csv(data_unmixing_analysis_24h_GFP_220118)
data_GFP_48h_220118=pd.read_csv(data_unmixing_analysis_48h_GFP_220118)
data_GFP_72h_220118=pd.read_csv(data_unmixing_analysis_72h_GFP_220118)
data_GFP_96h_220118=pd.read_csv(data_unmixing_analysis_96h_GFP_220118)

data_GFP_24h_220125=pd.read_csv(data_unmixing_analysis_24h_GFP_220125)
data_GFP_48h_220125=pd.read_csv(data_unmixing_analysis_48h_GFP_220125)
data_GFP_72h_220125=pd.read_csv(data_unmixing_analysis_72h_GFP_220125)
data_GFP_96h_220125=pd.read_csv(data_unmixing_analysis_96h_GFP_220125)

data_GFP_24h_220503=pd.read_csv(data_unmixing_analysis_24h_GFP_220503)
data_GFP_48h_220503=pd.read_csv(data_unmixing_analysis_48h_GFP_220503)
data_GFP_72h_220503=pd.read_csv(data_unmixing_analysis_72h_GFP_220503)
data_GFP_96h_220503=pd.read_csv(data_unmixing_analysis_96h_GFP_220503)

data_GFP_24h_220705=pd.read_csv(data_unmixing_analysis_24h_GFP_220705)
data_GFP_48h_220705=pd.read_csv(data_unmixing_analysis_48h_GFP_220705)
data_GFP_72h_220705=pd.read_csv(data_unmixing_analysis_72h_GFP_220705)
data_GFP_96h_220705=pd.read_csv(data_unmixing_analysis_96h_GFP_220705)

data_GFP_24h_220712=pd.read_csv(data_unmixing_analysis_24h_GFP_220712)
data_GFP_48h_220712=pd.read_csv(data_unmixing_analysis_48h_GFP_220712)
data_GFP_72h_220712=pd.read_csv(data_unmixing_analysis_72h_GFP_220712)
data_GFP_96h_220712=pd.read_csv(data_unmixing_analysis_96h_GFP_220712)

# Collect

data_GFP_24h=pd.concat([data_GFP_24h_220118, data_GFP_24h_220125, data_GFP_24h_220503, data_GFP_24h_220705, data_GFP_24h_220712]) #, data_GFP_24h_220809])
data_GFP_48h=pd.concat([data_GFP_48h_220118, data_GFP_48h_220125, data_GFP_48h_220503, data_GFP_48h_220705, data_GFP_48h_220712]) #, data_GFP_48h_220809])
data_GFP_72h=pd.concat([data_GFP_72h_220118, data_GFP_72h_220125, data_GFP_72h_220503, data_GFP_72h_220705, data_GFP_72h_220712]) #, data_GFP_72h_220809])
data_GFP_96h=pd.concat([data_GFP_96h_220118, data_GFP_96h_220125, data_GFP_96h_220503, data_GFP_96h_220705, data_GFP_96h_220712]) #, data_GFP_96h_220809])

# Separate TposRpos and TnegRneg
    
data_GFP_24h_control = data_GFP_24h[data_GFP_24h['Condition']=='24h control']  
data_GFP_48h_control = data_GFP_48h[data_GFP_48h['Condition']=='48h control']  
data_GFP_72h_control = data_GFP_72h[data_GFP_72h['Condition']=='72h control']  
data_GFP_96h_control = data_GFP_96h[data_GFP_96h['Condition']=='96h control']  

# absolute value polarization

data_GFP_24h_control['Polarization']=abs(data_GFP_24h_control['Polarization'])
data_GFP_48h_control['Polarization']=abs(data_GFP_48h_control['Polarization'])
data_GFP_72h_control['Polarization']=abs(data_GFP_72h_control['Polarization'])
data_GFP_96h_control['Polarization']=abs(data_GFP_96h_control['Polarization'])

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

# Get the mean and SD #96h

data_GFP_96h_mean_control = data_GFP_96h_control.groupby('GFP fraction', as_index=False).mean()    
data_GFP_96h_std_control = data_GFP_96h_control.groupby('GFP fraction', as_index=False).std()    

data_GFP_96h_mean_control['Num samples']=data_GFP_96h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  
data_GFP_96h_std_control['Num samples']=data_GFP_96h_control.groupby('GFP fraction', as_index=False)['Num samples'].sum()['Num samples']  


# Polarization 
# Polarization SD is from DIFFERENT EXPERIMENTS. 
# 
# The error showed is SD 

plt.figure(figsize=(5,5)) # set size
Pol_analysis = plt.figure(figsize=(5,5))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.ylabel('Polarization ($P$)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'initial fraction of T+ cells $\phi$',fontsize=15, color = 'black')  # x-label fontsize + color
plt.axis([-0.1,1.1,0,1.2])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.errorbar(data_GFP_24h_mean_control['GFP fraction'],data_GFP_24h_mean_control['Polarization'],marker='o',yerr=data_GFP_24h_std_control['Polarization'],fmt='-',capthick=2,capsize=5, label='24h', color='gray')
plt.errorbar(data_GFP_48h_mean_control['GFP fraction'],data_GFP_48h_mean_control['Polarization'],marker='o',yerr=data_GFP_48h_std_control['Polarization'],fmt='-',capthick=2,capsize=5, label='48h', color='cadetblue')
plt.errorbar(data_GFP_72h_mean_control['GFP fraction'],data_GFP_72h_mean_control['Polarization'],marker='o',yerr=data_GFP_72h_std_control['Polarization'],fmt='-',capthick=2,capsize=5, label='72h', color='black')
plt.errorbar(data_GFP_96h_mean_control['GFP fraction'],data_GFP_96h_mean_control['Polarization'],marker='o',yerr=data_GFP_96h_std_control['Polarization'],fmt='-',capthick=2,capsize=5, label='96h', color='purple')
plt.legend(frameon=False,loc="upper right")
plt.show() 
Pol_analysis.savefig("Pol_analysis.pdf", bbox_inches='tight')


# Aspect ratio 
# Aspect ratio SD is from DIFFERENT EXPERIMENTS. 
# 
# The error showed is SD

plt.figure(figsize=(5,5)) # set size
AP_analysis = plt.figure(figsize=(5,5))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.ylabel('Aspect ratio ($L/d$)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'initial fraction of T+ cells $\phi$',fontsize=15, color = 'black')  # x-label fontsize + color
plt.axis([-0.1,1.1,1,2])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.errorbar(data_GFP_24h_mean_control['GFP fraction'],data_GFP_24h_mean_control['Aspect ratio'],marker='o',yerr=data_GFP_24h_std_control['Aspect ratio'],fmt='-',capthick=2,capsize=5, label='24h', color='gray')
plt.errorbar(data_GFP_48h_mean_control['GFP fraction'],data_GFP_48h_mean_control['Aspect ratio'],marker='o',yerr=data_GFP_48h_std_control['Aspect ratio SD'],fmt='-',capthick=2,capsize=5, label='48h', color='cadetblue')
plt.errorbar(data_GFP_72h_mean_control['GFP fraction'],data_GFP_72h_mean_control['Aspect ratio'],marker='o',yerr=data_GFP_72h_std_control['Aspect ratio'],fmt='-',capthick=2,capsize=5, label='72h', color='black')
plt.errorbar(data_GFP_96h_mean_control['GFP fraction'],data_GFP_96h_mean_control['Aspect ratio'],marker='o',yerr=data_GFP_96h_std_control['Aspect ratio SD'],fmt='-',capthick=2,capsize=5, label='96h', color='purple')
plt.legend(frameon=False,loc="upper right")
plt.show() 
AP_analysis.savefig("AP_analysis.pdf", bbox_inches='tight')






