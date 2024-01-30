#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:53:49 2024

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

# get current path
dir_path = os.path.abspath(os.getcwd()) 

# get paths for each csv file

data_24_220705 =os.path.join(dir_path,'data_24_GFP_Ctrl_XAV_SB43_220705.csv')
data_24_220712 =os.path.join(dir_path,'data_24_GFP_Ctrl_XAV_SB43_220712.csv')
data_48_220706 =os.path.join(dir_path,'data_48_GFP_Ctrl_XAV_SB43_220706.csv')
data_48_220713 =os.path.join(dir_path,'data_48_GFP_Ctrl_XAV_SB43_220713.csv')
data_72_220707 =os.path.join(dir_path,'data_72_GFP_Ctrl_XAV_SB43_220707.csv')
data_72_220714 =os.path.join(dir_path,'data_72_GFP_Ctrl_XAV_SB43_220714.csv')

# get data

data_GFP_24_220705=pd.read_csv(data_24_220705)
data_GFP_24_220712=pd.read_csv(data_24_220712)
data_GFP_48_220706=pd.read_csv(data_48_220706)
data_GFP_48_220713=pd.read_csv(data_48_220713)
data_GFP_72_220707=pd.read_csv(data_72_220707)
data_GFP_72_220714=pd.read_csv(data_72_220714)


# Separate Conditions
    
data_GFP_24_control_220705 = data_GFP_24_220705[data_GFP_24_220705['Condition']=='24h Ctrl']  
data_GFP_24_SB43_220705 = data_GFP_24_220705[data_GFP_24_220705['Condition']=='24h SB43']  
data_GFP_24_XAV_220705 = data_GFP_24_220705[data_GFP_24_220705['Condition']=='24h XAV']  
data_GFP_24_control_220712 = data_GFP_24_220712[data_GFP_24_220712['Condition']=='24h Ctrl']  
data_GFP_24_SB43_220712 = data_GFP_24_220712[data_GFP_24_220712['Condition']=='24h SB43']  
data_GFP_24_XAV_220712 = data_GFP_24_220712[data_GFP_24_220712['Condition']=='24h XAV']  

data_GFP_48_control_220706 = data_GFP_48_220706[data_GFP_48_220706['Condition']=='48h Ctrl']  
data_GFP_48_SB43_220706 = data_GFP_48_220706[data_GFP_48_220706['Condition']=='48h SB43']  
data_GFP_48_XAV_220706 = data_GFP_48_220706[data_GFP_48_220706['Condition']=='48h XAV']  
data_GFP_48_control_220713 = data_GFP_48_220713[data_GFP_48_220713['Condition']=='48h Ctrl']  
data_GFP_48_SB43_220713 = data_GFP_48_220713[data_GFP_48_220713['Condition']=='48h SB43']  
data_GFP_48_XAV_220713 = data_GFP_48_220713[data_GFP_48_220713['Condition']=='48h XAV']  

data_GFP_72_control_220707 = data_GFP_72_220707[data_GFP_72_220707['Condition']=='72h Ctrl']  
data_GFP_72_SB43_220707 = data_GFP_72_220707[data_GFP_72_220707['Condition']=='72h SB43']  
data_GFP_72_XAV_220707 = data_GFP_72_220707[data_GFP_72_220707['Condition']=='72h XAV']  
data_GFP_72_control_220714 = data_GFP_72_220714[data_GFP_72_220714['Condition']=='72h Ctrl']  
data_GFP_72_SB43_220714 = data_GFP_72_220714[data_GFP_72_220714['Condition']=='72h SB43']  
data_GFP_72_XAV_220714 = data_GFP_72_220714[data_GFP_72_220714['Condition']=='72h XAV']  


# GFP Mean \pm SE 
# The background GFP intensity was already substracted using Morgana

# DMSO Control Replicate #1

plt.figure(figsize=(5,5)) # set size
GFP_vs_fraction = plt.figure(figsize=(5,5))
#theta_vs_gfp, ax = plt.subplots()
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.ylabel('Average GFP intensity $I(\phi)$ (a.u.)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'initial fraction of T+ labelled cells $\phi$',fontsize=15, color = 'black')  # x-label fontsize + color
plt.axis([-0.05,1.05,-10,300])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.errorbar(data_GFP_24_control_220705['GFP fraction'],data_GFP_24_control_220705['GFP fluo'],yerr=data_GFP_24_control_220705['GFP fluo SD']/np.sqrt(data_GFP_24_control_220705['Num samples']),label = 'Ctrl 24h',color='darkgreen',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_48_control_220706['GFP fraction'],data_GFP_48_control_220706['GFP fluo'],yerr=data_GFP_48_control_220706['GFP fluo SD']/np.sqrt(data_GFP_48_control_220706['Num samples']),label = 'Ctrl 48h',color='blue',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_72_control_220707['GFP fraction'],data_GFP_72_control_220707['GFP fluo'],yerr=data_GFP_72_control_220707['GFP fluo SD']/np.sqrt(data_GFP_72_control_220707['Num samples']),label = 'Ctrl 72h',color='black',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.legend(frameon=False,loc="upper left")
plt.show() 
GFP_vs_fraction.savefig("GFP_vs_fraction_Ctrl_Replicate1.pdf", bbox_inches='tight')

# DMSO Control Replicate #2

plt.figure(figsize=(5,5)) # set size
GFP_vs_fraction = plt.figure(figsize=(5,5))
#theta_vs_gfp, ax = plt.subplots()
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.ylabel('Average GFP intensity $I(\phi)$ (a.u.)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'initial fraction of T+ labelled cells $\phi$',fontsize=15, color = 'black')  # x-label fontsize + color
plt.axis([-0.05,1.05,-10,300])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.errorbar(data_GFP_24_control_220712['GFP fraction'],data_GFP_24_control_220712['GFP fluo'],yerr=data_GFP_24_control_220712['GFP fluo SD']/np.sqrt(data_GFP_24_control_220712['Num samples']),label = 'Ctrl 24h',color='darkgreen',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_48_control_220713['GFP fraction'],data_GFP_48_control_220713['GFP fluo'],yerr=data_GFP_48_control_220713['GFP fluo SD']/np.sqrt(data_GFP_48_control_220713['Num samples']),label = 'Ctrl 48h',color='blue',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_72_control_220714['GFP fraction'],data_GFP_72_control_220714['GFP fluo'],yerr=data_GFP_72_control_220714['GFP fluo SD']/np.sqrt(data_GFP_72_control_220714['Num samples']),label = 'Ctrl 72h',color='black',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.legend(frameon=False,loc="upper left")
plt.show() 
GFP_vs_fraction.savefig("GFP_vs_fraction_Ctrl_Replicate2.pdf", bbox_inches='tight')

# SB43 Replicate #1

plt.figure(figsize=(5,5)) # set size
GFP_vs_fraction = plt.figure(figsize=(5,5))
#theta_vs_gfp, ax = plt.subplots()
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.ylabel('Average GFP intensity $I(\phi)$ (a.u.)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'initial fraction of T+ labelled cells $\phi$',fontsize=15, color = 'black')  # x-label fontsize + color
plt.axis([-0.05,1.05,-10,300])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.errorbar(data_GFP_24_SB43_220705['GFP fraction'],data_GFP_24_SB43_220705['GFP fluo'],yerr=data_GFP_24_SB43_220705['GFP fluo SD']/np.sqrt(data_GFP_24_SB43_220705['Num samples']),label = 'SB43 24h',color='darkgreen',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_48_SB43_220706['GFP fraction'],data_GFP_48_SB43_220706['GFP fluo'],yerr=data_GFP_48_SB43_220706['GFP fluo SD']/np.sqrt(data_GFP_48_SB43_220706['Num samples']),label = 'SB43 48h',color='blue',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_72_SB43_220707['GFP fraction'],data_GFP_72_SB43_220707['GFP fluo'],yerr=data_GFP_72_SB43_220707['GFP fluo SD']/np.sqrt(data_GFP_72_SB43_220707['Num samples']),label = 'SB43 72h',color='black',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.legend(frameon=False,loc="upper left")
plt.show() 
GFP_vs_fraction.savefig("GFP_vs_fraction_SB43_Replicate1.pdf", bbox_inches='tight')

# SB43 Replicate #2

plt.figure(figsize=(5,5)) # set size
GFP_vs_fraction = plt.figure(figsize=(5,5))
#theta_vs_gfp, ax = plt.subplots()
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.ylabel('Average GFP intensity $I(\phi)$ (a.u.)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'initial fraction of T+ labelled cells $\phi$',fontsize=15, color = 'black')  # x-label fontsize + color
plt.axis([-0.05,1.05,-10,300])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.errorbar(data_GFP_24_SB43_220712['GFP fraction'],data_GFP_24_SB43_220712['GFP fluo'],yerr=data_GFP_24_SB43_220712['GFP fluo SD']/np.sqrt(data_GFP_24_SB43_220712['Num samples']),label = 'SB43 24h',color='darkgreen',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_48_SB43_220713['GFP fraction'],data_GFP_48_SB43_220713['GFP fluo'],yerr=data_GFP_48_SB43_220713['GFP fluo SD']/np.sqrt(data_GFP_48_SB43_220713['Num samples']),label = 'SB43 48h',color='blue',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_72_SB43_220714['GFP fraction'],data_GFP_72_SB43_220714['GFP fluo'],yerr=data_GFP_72_SB43_220714['GFP fluo SD']/np.sqrt(data_GFP_72_SB43_220714['Num samples']),label = 'SB43 72h',color='black',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.legend(frameon=False,loc="upper left")
plt.show() 
GFP_vs_fraction.savefig("GFP_vs_fraction_SB43_Replicate2.pdf", bbox_inches='tight')

# XAV Replicate #1

plt.figure(figsize=(5,5)) # set size
GFP_vs_fraction = plt.figure(figsize=(5,5))
#theta_vs_gfp, ax = plt.subplots()
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.ylabel('Average GFP intensity $I(\phi)$ (a.u.)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'initial fraction of T+ labelled cells $\phi$',fontsize=15, color = 'black')  # x-label fontsize + color
plt.axis([-0.05,1.05,-10,300])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.errorbar(data_GFP_24_XAV_220705['GFP fraction'],data_GFP_24_XAV_220705['GFP fluo'],yerr=data_GFP_24_XAV_220705['GFP fluo SD']/np.sqrt(data_GFP_24_XAV_220705['Num samples']),label = 'XAV 24h',color='darkgreen',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_48_XAV_220706['GFP fraction'],data_GFP_48_XAV_220706['GFP fluo'],yerr=data_GFP_48_XAV_220706['GFP fluo SD']/np.sqrt(data_GFP_48_XAV_220706['Num samples']),label = 'XAV 48h',color='blue',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_72_XAV_220707['GFP fraction'],data_GFP_72_XAV_220707['GFP fluo'],yerr=data_GFP_72_XAV_220707['GFP fluo SD']/np.sqrt(data_GFP_72_XAV_220707['Num samples']),label = 'XAV 72h',color='black',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.legend(frameon=False,loc="upper left")
plt.show() 
GFP_vs_fraction.savefig("GFP_vs_fraction_XAV_Replicate1.pdf", bbox_inches='tight')

# XAV Replicate #2

plt.figure(figsize=(5,5)) # set size
GFP_vs_fraction = plt.figure(figsize=(5,5))
#theta_vs_gfp, ax = plt.subplots()
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.ylabel('Average GFP intensity $I(\phi)$ (a.u.)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'initial fraction of T+ labelled cells $\phi$',fontsize=15, color = 'black')  # x-label fontsize + color
plt.axis([-0.05,1.05,-10,300])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.errorbar(data_GFP_24_XAV_220712['GFP fraction'],data_GFP_24_XAV_220712['GFP fluo'],yerr=data_GFP_24_XAV_220712['GFP fluo SD']/np.sqrt(data_GFP_24_XAV_220712['Num samples']),label = 'XAV 24h',color='darkgreen',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_48_XAV_220713['GFP fraction'],data_GFP_48_XAV_220713['GFP fluo'],yerr=data_GFP_48_XAV_220713['GFP fluo SD']/np.sqrt(data_GFP_48_XAV_220713['Num samples']),label = 'XAV 48h',color='blue',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.errorbar(data_GFP_72_XAV_220714['GFP fraction'],data_GFP_72_XAV_220714['GFP fluo'],yerr=data_GFP_72_XAV_220714['GFP fluo SD']/np.sqrt(data_GFP_72_XAV_220714['Num samples']),label = 'XAV 72h',color='black',fmt='o',ecolor='k',capthick=2,capsize=5)
plt.legend(frameon=False,loc="upper left")
plt.show() 
GFP_vs_fraction.savefig("GFP_vs_fraction_XAV_Replicate2.pdf", bbox_inches='tight')
