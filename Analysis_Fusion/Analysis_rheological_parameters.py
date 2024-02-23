##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:15:20 2021

@author: oriola 
"""

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

dir_path = os.path.abspath(os.getcwd()) # get path

output_fusion_analysis_24h_48h =os.path.join(dir_path,'output_24h_48h.csv')
output_fusion_analysis_sorting =os.path.join(dir_path,'output_fits_Tplus_Tminus.csv')
output_elasticity_analysis =os.path.join(dir_path,'output_elasticity.csv')

fusion_analysis_24h_48h=pd.read_csv(output_fusion_analysis_24h_48h)
fusion_analysis_sorting=pd.read_csv(output_fusion_analysis_sorting)
elasticity_analysis=pd.read_csv(output_elasticity_analysis)

# Remove 'Unnamed: 0' if necessary
if 'Unnamed: 0' in fusion_analysis_24h_48h.columns:
    del fusion_analysis_24h_48h['Unnamed: 0']
    
if 'Unnamed: 0' in fusion_analysis_sorting.columns:    
    del fusion_analysis_sorting['Unnamed: 0']

# change index [0,1,2...] to condition [24h,48h, etc]
elasticity_analysis.set_index("Condition", inplace = True)
fusion_analysis_24h_48h.set_index("Condition", inplace = True)
fusion_analysis_sorting.set_index("Condition", inplace = True)

# surface tension (mN/m) um*Pa = 1e-3 mN/m 
# the shear modulus is Eeff/4 for an incompressible material


surf_tension_Tminus = (1e-3)*(elasticity_analysis.loc['Tminus']['Eeff (Pa)']/4)*fusion_analysis_sorting.loc['Tminus']['lEC (um)']
surf_tension_Tplus = (1e-3)*(elasticity_analysis.loc['Tplus']['Eeff (Pa)']/4)*fusion_analysis_sorting.loc['Tplus']['lEC (um)']
surf_tension_24h = (1e-3)*(elasticity_analysis.loc['24h']['Eeff (Pa)']/4)*fusion_analysis_24h_48h.loc['24h']['lEC (um)']
surf_tension_48h = (1e-3)*(elasticity_analysis.loc['48h']['Eeff (Pa)']/4)*fusion_analysis_24h_48h.loc['48h']['lEC (um)']


surf_tension_Tminus_error = surf_tension_Tminus*np.sqrt(pow(elasticity_analysis.loc['Tminus']['Eeff SD (Pa)']/elasticity_analysis.loc['Tminus']['Eeff (Pa)'],2)+pow(fusion_analysis_sorting.loc['Tminus']['err_lEC (um)']/fusion_analysis_sorting.loc['Tminus']['lEC (um)'],2))
surf_tension_Tplus_error = surf_tension_Tplus*np.sqrt(pow(elasticity_analysis.loc['Tplus']['Eeff SD (Pa)']/elasticity_analysis.loc['Tplus']['Eeff (Pa)'],2)+pow(fusion_analysis_sorting.loc['Tplus']['err_lEC (um)']/fusion_analysis_sorting.loc['Tplus']['lEC (um)'],2))
surf_tension_24h_error = surf_tension_24h*np.sqrt(pow(elasticity_analysis.loc['24h']['Eeff SD (Pa)']/elasticity_analysis.loc['24h']['Eeff (Pa)'],2)+pow(fusion_analysis_24h_48h.loc['24h']['err_lEC (um)']/fusion_analysis_24h_48h.loc['24h']['lEC (um)'],2))
surf_tension_48h_error = surf_tension_48h*np.sqrt(pow(elasticity_analysis.loc['48h']['Eeff SD (Pa)']/elasticity_analysis.loc['48h']['Eeff (Pa)'],2)+pow(fusion_analysis_24h_48h.loc['48h']['err_lEC (um)']/fusion_analysis_24h_48h.loc['48h']['lEC (um)'],2))

# Compute Viscosity (mN/m)/(um/min)=1e3*60 (Pa*s)

viscosity_Tminus = (1e3*60)*surf_tension_Tminus/fusion_analysis_sorting.loc['Tminus']['vc (um/min)']
viscosity_Tplus = (1e3*60)*surf_tension_Tplus/fusion_analysis_sorting.loc['Tplus']['vc (um/min)']
viscosity_24h = (1e3*60)*surf_tension_24h/fusion_analysis_24h_48h.loc['24h']['vc (um/min)']
viscosity_48h = (1e3*60)*surf_tension_48h/fusion_analysis_24h_48h.loc['48h']['vc (um/min)']

viscosity_Tminus_error = viscosity_Tminus*np.sqrt(pow(surf_tension_Tminus_error/surf_tension_Tminus,2)+pow(fusion_analysis_sorting.loc['Tminus']['err_vc (um/min)']/fusion_analysis_sorting.loc['Tminus']['vc (um/min)'],2))
viscosity_Tplus_error = viscosity_Tplus*np.sqrt(pow(surf_tension_Tplus_error/surf_tension_Tplus,2)+pow(fusion_analysis_sorting.loc['Tplus']['err_vc (um/min)']/fusion_analysis_sorting.loc['Tplus']['vc (um/min)'],2))
viscosity_24h_error = viscosity_24h*np.sqrt(pow(surf_tension_24h_error/surf_tension_24h,2)+pow(fusion_analysis_24h_48h.loc['24h']['err_vc (um/min)']/fusion_analysis_24h_48h.loc['24h']['vc (um/min)'],2))
viscosity_48h_error = viscosity_48h*np.sqrt(pow(surf_tension_48h_error/surf_tension_48h,2)+pow(fusion_analysis_24h_48h.loc['48h']['err_vc (um/min)']/fusion_analysis_24h_48h.loc['48h']['vc (um/min)'],2))


viscosity=plt.figure(figsize=(3,5)) # set size
#theta_vs_gfp = plt.figure()
#theta_vs_gfp, ax = plt.subplots()
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([-1,4,1,22])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.ylabel('Shear viscosity  $\eta$ ($10^5$ Pa$\cdot$s)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel('condition',fontsize=15, color = 'black')  # x-label fontsize + color
plt.errorbar([0,1],[viscosity_24h/1e5,viscosity_48h/1e5],yerr=[viscosity_24h_error/1e5,viscosity_48h_error/1e5],marker='o', markersize=8.,ls ='None', color = "steelblue",ecolor='steelblue', capthick=2,elinewidth=2,capsize=5)
plt.errorbar([2],[viscosity_Tminus/1e5],yerr=[viscosity_Tminus_error/1e5],marker='o', markersize=8.,ls ='None', color = "gray",ecolor='gray', capthick=2,elinewidth=2,capsize=5)
plt.errorbar([3],[viscosity_Tplus/1e5],yerr=[viscosity_Tminus_error/1e5],marker='o', markersize=8.,ls ='None', color = "green",ecolor='green', capthick=2,elinewidth=2,capsize=5)
#plt.legend(frameon=False,loc="lower right")
plt.show() 
viscosity.savefig('viscosity.pdf',bbox_inches = "tight")   # save as .eps

surf_tension=plt.figure(figsize=(3,5)) # set size
#theta_vs_gfp = plt.figure()
#theta_vs_gfp, ax = plt.subplots()
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([-1,4,0,5])    # range of the y and x axis ([xmin,xmax,ymin,ymax])
plt.ylabel('Surface tension  $\gamma$ (mN/m)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel('condition',fontsize=15, color = 'black')  # x-label fontsize + color
plt.errorbar([0,1],[surf_tension_24h,surf_tension_48h],yerr=[surf_tension_24h_error,surf_tension_48h_error],marker='o', markersize=8.,ls ='None', color = "steelblue",ecolor='steelblue', capthick=2,elinewidth=2,capsize=5)
plt.errorbar([2],[surf_tension_Tminus],yerr=[surf_tension_Tminus_error],marker='o', markersize=8.,ls ='None', color = "gray",ecolor='gray', capthick=2,elinewidth=2,capsize=5)
plt.errorbar([3],[surf_tension_Tplus],yerr=[surf_tension_Tplus_error],marker='o', markersize=8.,ls ='None', color = "green",ecolor='green', capthick=2,elinewidth=2,capsize=5)
#plt.legend(frameon=False,loc="lower right")
plt.show() 
surf_tension.savefig('surf_tension.pdf',bbox_inches = "tight")   # save as .eps
 
