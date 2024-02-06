#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:53:21 2021

@author: oriola 
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 13:46:14 2021

@author: oriola 
"""
# First load everything


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
from numpy import diff
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
import math, tqdm
from tifffile import imread
from skimage import data
from skimage import img_as_bool
from skimage import io
import skimage.color
import skimage.filters
import skimage.io
from scipy.optimize import fsolve
from scipy.integrate import odeint
from scipy.ndimage import map_coordinates
import time as t
from tqdm import tqdm
import glob, os
from tifffile import imsave
from tifffile import TiffWriter
import random
import pylab
import imageio

#to get the current working directory
directory = os.getcwd()

#folder where the data is at 24h WINDOWS

folder_GFP = os.path.join(directory,"Analysis_data_24h_unmixing_210610","210610_20x_BF_GFP_Unmixing_24h_focal_plane")
folder_sirDNA = os.path.join(directory,"Analysis_data_24h_unmixing_210610","210610_20x_BF_sirDNA_Unmixing_24h_focal_plane")
folder_GFP_masks = os.path.join(directory,"Analysis_data_24h_unmixing_210610","210610_20x_BF_GFP_Unmixing_24h_focal_plane","result_segmentation")
folder_sirDNA_masks = os.path.join(directory,"Analysis_data_24h_unmixing_210610","210610_20x_BF_sirDNA_Unmixing_24h_focal_plane","result_segmentation")

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int64)
    
    keep = ~np.isnan(data)
    tbin = np.bincount(r[keep].ravel(), weights=data[keep].ravel())
    nr = np.bincount(r[keep].ravel())
    radialprofile = tbin / nr
    return radialprofile 

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

# Get names of tif files

filenames_GFP_tif = []
for file in os.listdir(folder_GFP):
    if file.endswith(".tif"):
        filenames_GFP_tif.append(os.path.join(folder_GFP, file))
        
# Sort properly according to the well labelling  ********************* 
        
a = [element[-7:-4] for element in filenames_GFP_tif] # get indexing e.g. 12A
sortedIndex = [i[0] for i in sorted(enumerate(a), key=lambda x:x[1])] # get how the indices will be rearranged       
filenames_GFP_tif = [filenames_GFP_tif[i] for i in sortedIndex]

# ************************************************************************************    

filenames_sirDNA_tif = []
for file in os.listdir(folder_sirDNA):
    if file.endswith(".tif"):
        filenames_sirDNA_tif.append(os.path.join(folder_sirDNA, file))
        
# Sort properly according to the well labelling  ********************* 
        
a = [element[-7:-4] for element in filenames_sirDNA_tif] # get indexing e.g. 12A
sortedIndex = [i[0] for i in sorted(enumerate(a), key=lambda x:x[1])] # get how the indices will be rearranged       
filenames_sirDNA_tif = [filenames_sirDNA_tif[i] for i in sortedIndex]

# ************************************************************************************            
        
# Get GFP final masks 
        
filenames_GFP_masks = []
for file in os.listdir(folder_GFP_masks):
    if file.endswith("finalMask.tif"):
        filenames_GFP_masks.append(os.path.join(folder_GFP_masks, file))
        
# Sort properly according to the well labelling  ********************* 
        
a = [element[-17:-14] for element in filenames_GFP_masks] # get indexing e.g. 12A
sortedIndex = [i[0] for i in sorted(enumerate(a), key=lambda x:x[1])] # get how the indices will be rearranged       
filenames_GFP_masks = [filenames_GFP_masks[i] for i in sortedIndex]

# ************************************************************************************            

# We remove the high intensity peaks in the sirDNA signal and then substract the normalised images
# The substraction is IsirDNA/<IsirDNA> - IGFP/<IGFP>

num_wells = 96
radial_intensity = []
radial_intensity_raw = []
rsize = []
radial_intensity_int = []
numsteps = 60
dx = 1/numsteps
xnew = np.linspace(0,1,numsteps) # normalized radial axis
sub_image_list =[]

for i in tqdm(range(num_wells)):
    
    # Read tiff files
    
    img_mask=io.imread(filenames_GFP_masks[i])   
    img_GFP=io.imread(filenames_GFP_tif[i])
    img_sirDNA=io.imread(filenames_sirDNA_tif[i])

    # Remove high intensity peaks from sirDNA image

    cut_off=np.percentile(img_sirDNA[1],(0,99.7))[1] # remove intensity pixels that are higher than the 99.7%
    a=(img_sirDNA[1] < cut_off).astype(int) # 0's if it's above
    img_sirDNA_nopeaks=img_sirDNA[1]*(a.astype(float))# new image with no peaks

    # apply masks to GFP and sirDNA images

    GFP_masked = (img_GFP[1]*img_mask).astype('float') # apply mask to GFP signal
    GFP_masked[GFP_masked == 0] = 'nan' # change 0's to NaN
    sirDNA_masked = (img_sirDNA_nopeaks*img_mask).astype('float') # apply mask to sirDNA signal
    sirDNA_masked[sirDNA_masked == 0] = 'nan' # change 0's to NaN

    #crop image ************************************************************************************
    
    where = np.array(np.where(img_mask))

    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)
    image = sirDNA_masked/np.nanmean(sirDNA_masked)-GFP_masked/np.nanmean(GFP_masked) # IsirDNA/<IsirDNA> - IGFP/<IGFP>
    sub_image = image[x1:x2, y1:y2]
    sub_image_list.append(image[x1:x2, y1:y2]) # list of cropped images  IsirDNA/<IsirDNA> - IGFP/<IGFP>
    sub_mask = img_mask[x1:x2, y1:y2]
    
    # **********************************************************************************************
    
    # Get radial profile
    cm_mask = ndimage.measurements.center_of_mass(sub_mask) # center of mass mask
    radial_intensity_raw.append(radial_profile(sub_image,cm_mask))
    
    # Interpolate and normalize x-axis to 1 in order to do the average of the curves 
    
    rsize.append(np.size(radial_intensity_raw[i])) # GFP data
    x0 = np.linspace(0,1,rsize[i])
    
    y =radial_intensity_raw[i]
    nans, x= nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])  # remove NaNs
    radial_intensity_int.append(interpolate.interp1d(x0, y)) #interpolate
    radial_intensity.append(radial_intensity_int[i](xnew))

# Split according to the different conditions:

# 1) 50/50 (G-)+(G-,R+) *******************************************************
condition1_set = [0,1,2,3,4,5,6,7]
condition1_radial= np.zeros((np.size(condition1_set),numsteps))

for i, num in enumerate(condition1_set):
    condition1_radial[i] = radial_intensity[num]
    
condition1_mean_radial = np.nanmean(condition1_radial,axis=0) # mean
condition1_SD_radial = np.nanstd(condition1_radial,axis=0) # SD

# *****************************************************************************

# 2) 50/50 (G+)+(G+,R+)
condition2_set =[8,9,10,11,12,13,14,15]
condition2_radial= np.zeros((np.size(condition2_set),numsteps))

for i, num in enumerate(condition2_set):
    condition2_radial[i] = radial_intensity[num]
    
condition2_mean_radial = np.nanmean(condition2_radial,axis=0) # mean
condition2_SD_radial = np.nanstd(condition2_radial,axis=0) # SD


# *****************************************************************************

# 3) 50/50 (G+)+(G-,R+)
condition3_set =[16,17,18,19,20,23,24,25,26,28,30,31]
condition3_radial= np.zeros((np.size(condition3_set),numsteps))

for i, num in enumerate(condition3_set):
    condition3_radial[i] = radial_intensity[num]
    
condition3_mean_radial = np.nanmean(condition3_radial,axis=0) # mean
condition3_SD_radial = np.nanstd(condition3_radial,axis=0) # SD


# *****************************************************************************

# 4) 50/50 (G+,R+)+(G-)
condition4_set =[32,33,34,35,36,38,39,40,41,42,43,44,45,46,47]
condition4_radial= np.zeros((np.size(condition4_set),numsteps))

for i, num in enumerate(condition4_set):
    condition4_radial[i] = radial_intensity[num]
    
condition4_mean_radial = np.nanmean(condition4_radial,axis=0) # mean
condition4_SD_radial = np.nanstd(condition4_radial,axis=0) # SD


# 5) 20/80 (G+)+(G-,R+)
condition5_set =[48,51,52,53,55,56,57,58,59,60,61,62,63]
condition5_radial= np.zeros((np.size(condition5_set),numsteps))

for i, num in enumerate(condition5_set):
    condition5_radial[i] = radial_intensity[num]
    
condition5_mean_radial = np.nanmean(condition5_radial,axis=0) # mean
condition5_SD_radial = np.nanstd(condition5_radial,axis=0) # SD


# 6) 20/80 (G+,R+)+(G-)
condition6_set =[64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79]
condition6_radial= np.zeros((np.size(condition6_set),numsteps))

for i, num in enumerate(condition6_set):
    condition6_radial[i] = radial_intensity[num]
    
condition6_mean_radial = np.nanmean(condition6_radial,axis=0) # mean
condition6_SD_radial = np.nanstd(condition6_radial,axis=0) # SD


# 7) 80/20 (G+)+(G-,R+)
condition7_set =[80,81,82,83,84,85,86,87]
condition7_radial= np.zeros((np.size(condition7_set),numsteps))

for i, num in enumerate(condition7_set):
    condition7_radial[i] = radial_intensity[num]
    
condition7_mean_radial = np.nanmean(condition7_radial,axis=0) # mean
condition7_SD_radial = np.nanstd(condition7_radial,axis=0) # SD


# 8) 80/20 (G+,R+)+(G-)
condition8_set =[88,89,90,91,92,93,94,95]
condition8_radial= np.zeros((np.size(condition8_set),numsteps))

for i, num in enumerate(condition8_set):
    condition8_radial[i] = radial_intensity[num]
    
condition8_mean_radial = np.nanmean(condition8_radial,axis=0) # mean
condition8_SD_radial = np.nanstd(condition8_radial,axis=0) # SD

# create 

all_mean_intensity_radial = np.column_stack((condition1_mean_radial,condition6_mean_radial,condition4_mean_radial,condition8_mean_radial,condition5_mean_radial,condition3_mean_radial,condition7_mean_radial,condition2_mean_radial))
all_SD_intensity_radial = np.column_stack((condition1_SD_radial,condition6_SD_radial,condition4_SD_radial,condition8_SD_radial,condition5_SD_radial,condition3_SD_radial,condition7_SD_radial,condition2_SD_radial))

plt.figure(figsize=(6,3)) # set size
color_plot_radial_profile = plt.figure(figsize=(6,3))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
pcolor(all_mean_intensity_radial,cmap='PiYG_r')
plt.ylabel(r'normalized radius $(r/r_{\rm max}$)',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'initial fraction of T+ labelled cells $\phi$',fontsize=15, color = 'black')  # x-label fontsize + color
plt.colorbar()
color_plot_radial_profile.savefig('color_plot_radial_profile_Replicate1.pdf',bbox_inches = "tight")   # save as .eps

j=0 # condition [0,1,2,3,4,5,6,7] j=0 --->  50/50 (G-)+(G-,R+)
m=2 # m=2 ---> 4) 50/50 (G+,R+)+(G-)   # m=5 ---> 3) 50/50 (G+)+(G-,R+)

plt.figure(figsize=(5,5)) # set size
delta_intensity_vs_radius = plt.figure(figsize=(5,5))
rc('axes',linewidth=2)  # box thickness
rc('font',size = 15)   # font size ticks
plt.axis([-0.05,1.05,-1,1]) 
plt.ylabel(r'$I_{\rm sirDNA}/\langle I_{\rm sirDNA} \rangle-I_{\rm GFP}/\langle I_{\rm GFP} \rangle$',fontsize=15, color = 'black') # y-label fontsize + color
plt.xlabel(r'normalized radius $(r/r_{\rm max}$)',fontsize=15, color = 'black')  # x-label fontsize + color
plt.plot(xnew,all_mean_intensity_radial[:,j],linewidth=2,label="(n="+str(np.size(condition1_set))+")", color = 'cadetblue')  # plot 
plt.fill_between(xnew,(all_mean_intensity_radial[:,j]-all_SD_intensity_radial[:,j]), (all_mean_intensity_radial[:,j]+all_SD_intensity_radial[:,j]),alpha=0.5, color = 'cadetblue')
plt.plot(xnew,all_mean_intensity_radial[:,m],linewidth=2,label="(n="+str(np.size(condition3_set))+")", color = 'sandybrown')  # plot 
plt.fill_between(xnew,(all_mean_intensity_radial[:,m]-all_SD_intensity_radial[:,m]), (all_mean_intensity_radial[:,m]+all_SD_intensity_radial[:,m]),alpha=0.5, color = 'sandybrown')
plt.legend(frameon=False,loc="lower right")
delta_intensity_vs_radius.savefig('delta_intensity_vs_radius_Replicate1.pdf',bbox_inches = "tight")   # save as .eps
plt.show() 
