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
import matplotlib.pyplot as plt

# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


# Extract nifti files and parse
def extractNifti(input, type):
    output = []
    for root, dirs, files in os.walk(input):
        if len(files) > 0:
            if type == 'healthy':
                output.append(os.path.join(
                    root, [x for x in files if 'GM' in x][0]))
            else:
                output.append(os.path.join(
                    root, [x for x in files if 'GM' in x and 'SmoothThreshold' in x][0]))
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
    
    names = [os.path.basename(file) for file in input]
    return imgs, names

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
            
def saveNii(zScore,save_path,names,example_dir):
    for root, dirs, files in os.walk(example_dir):
        if len(files) > 0:
            example = nib.load(os.path.join(root,files[0]))
            break
            
    for count, image in enumerate(zScore):
        tempimg = nib.Nifti1Image(np.squeeze(np.reshape(image,(113,137,113))),example.affine)
        nib.save(tempimg,os.path.join(save_path,'ZS-'+names[count].split('_SmoothThreshold')[0]))
        
#%%

# Data path
Healthydir = r'F:\PatientData\thres\Control'
LTLEdir = r'F:\PatientData\LargeSet_4_7\Threshold_Smoothed\LTLE'
RTLEdir = r'F:\PatientData\LargeSet_4_7\Threshold_Smoothed\RTLE'


outputdir = r'F:\PatientData\LargeSet_4_7\zScore'
LTLEzscoreDir=os.path.join(outputdir,'zscoreLTLE')
if not os.path.exists(LTLEzscoreDir):
    os.mkdir(LTLEzscoreDir)

RTLEzscoreDir=os.path.join(outputdir,'zscoreRTLE')
if not os.path.exists(RTLEzscoreDir):
    os.mkdir(RTLEzscoreDir)



# Define parameters
matter = 'GM'

# Extract Disease imgs
LTLEImgs, LTLEnames = PreprocImg(extractNifti(LTLEdir,'LTLE'))  # 0
RTLEImgs, RTLEnames = PreprocImg(extractNifti(RTLEdir,'RTLE'))  # 1
HealthyImgs, Healthynames= PreprocImg(extractNifti(Healthydir,'healthy'))  # 2

# Healthy img means
[HealthyImgs_Mean,HealthyImgs_Std] = vectorization(HealthyImgs)

# Comp each img
zScoreLTLE=compHealthy(LTLEImgs,HealthyImgs_Mean,HealthyImgs_Std)
zScoreRTLE=compHealthy(RTLEImgs,HealthyImgs_Mean,HealthyImgs_Std)

# Save nifti
saveNii(zScoreLTLE,LTLEzscoreDir,LTLEnames,LTLEdir)
saveNii(zScoreRTLE,RTLEzscoreDir,RTLEnames,LTLEdir)


ax = plt.imshow(np.squeeze(np.reshape(zScoreLTLE[1,:],(113,137,113))[:, :, 40]), cmap="gray")
plt.colorbar(ax)

ax = plt.imshow(np.squeeze(np.reshape(HealthyImgs_Mean,(113,137,113))[:, :, 40]), cmap="gray")
plt.colorbar(ax)

ax = plt.imshow(np.squeeze(np.reshape(HealthyImgs_Std,(113,137,113))[:, :, 40]), cmap="gray")
plt.colorbar(ax)




