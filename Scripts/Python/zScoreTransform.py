# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:17:17 2022

@author: allen
"""
#%% Functions

# Import modules
import os
import nibabel as nib
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import ndimage

# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


# Extract nifti files and parse
def extractNifti(input):
    output = []
    for root, dirs, files in os.walk(input):
        if len(files) > 0:
            output.append(os.path.join(
                root, [x for x in files if 'GM' in x][0]))
    return output

def extractNames(input):
    output = []
    for root, dirs, files in os.walk(input):
        if len(files) > 0:
            output.append([x for x in files if 'GM' in x])
            example=os.path.join(
                root, [x for x in files if 'GM' in x][0])
    return [output,example]

def NormalizeData(data):
    data[np.isnan(data)]=0
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def PreprocImg(input):
    imgs = [nib.load(file).get_fdata() for file in input]
    imgs = NormalizeData(np.stack(imgs,axis=3))
    return imgs

def vectorization(input):
    input=np.transpose(input,(3,0,1,2)).reshape(input.shape[3],1749353)
    avg=np.mean(input,axis=0)
    std=np.std(input,axis=0)
    
    return [avg,std]
    
def compHealthy(compInput, HealthyImgs_Mean, HealthyImgs_Std):
    compInput=np.transpose(compInput,(3,0,1,2)).reshape(compInput.shape[3],1749353)
    
    for row in range(compInput.shape[0]):
        tempVec=np.squeeze(compInput[row,:])
        
        for voxel in range(len(tempVec)):
            if HealthyImgs_Std[voxel]!=0:
                tempVec[voxel]=(tempVec[voxel]-HealthyImgs_Mean[voxel])/HealthyImgs_Std[voxel]
        
        compInput[row]=tempVec
    
    return compInput
            
def saveNii(zScore,save_path,dir_path):
    [names,example] = extractNames(dir_path)
    img = nib.load(example)
    
    for row in range(input.shape[0]):
        tempimg = nib.Nifti1Image(np.squeeze(np.reshape(input[row,:],(113,137,113))),img.affine)
        nib.save(tempimg,os.path.join(save_path,'ZS-'+names[row][0]))
        
#%%

# Data path
datadir=r'C:\Users\allen\Desktop\datadir'
Healthydir = os.path.join(datadir, 'Control','Control')
LTLEdir = os.path.join(datadir, 'TLE','TLE','EP_LTLE_nifti')
RTLEdir = os.path.join(datadir, 'TLE','TLE','EP_RTLE_nifti')

LTLEzscoreDir=os.path.join(datadir,'zscoreLTLE')
os.mkdir(LTLEzscoreDir)

RTLEzscoreDir=os.path.join(datadir,'zscoreRTLE')
os.mkdir(RTLEzscoreDir)



# Define parameters
matter = 'GM'

# Extract Disease imgs
LTLEImgs = PreprocImg(extractNifti(LTLEdir))  # 0
RTLEImgs = PreprocImg(extractNifti(RTLEdir))  # 1
HealthyImgs = PreprocImg(extractNifti(Healthydir))  # 2

# Healthy img means
[HealthyImgs_Mean,HealthyImgs_Std] = vectorization(HealthyImgs)

# Comp each img
zScoreLTLE=compHealthy(LTLEImgs,HealthyImgs_Mean,HealthyImgs_Std)
zScoreRTLE=compHealthy(RTLEImgs,HealthyImgs_Mean,HealthyImgs_Std)

# Save nifti
saveNii(zScoreLTLE,LTLEzscoreDir,LTLEdir)
saveNii(zScoreRTLE,RTLEzscoreDir,RTLEdir)


plt.imshow(np.squeeze(np.reshape(zScoreLTLE[1,:],(113,137,113))[:, :, 40]), cmap="gray")




