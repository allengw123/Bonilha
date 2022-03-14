# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import nibabel as nib
import random
import math
import itertools
import scipy

# Extract nifti files and parse
def extractNifti(input):
    output = []
    for root, dirs, files in os.walk(input):
        if len(files) > 0:
            output.append(os.path.join(
                root, [x for x in files if 'GM' in x][0]))
    return output


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def PreprocImg(input):
    imgs = [nib.load(file).get_fdata().astype(np.float32) for file in input]
    imgs = NormalizeData(np.stack(imgs,axis=3))
    return imgs


# Define CNN input class
class CNNinput:
    def __init__(self, niftifiles):

        # length of files
        nFiles = len(niftifiles)

        # shuffle files
        rNums = random.sample(range(nFiles), nFiles)
        niftifiles = [niftifiles[x] for x in rNums]

        # obtain index ratio
        trainingN = math.ceil(nFiles*ratio[0]/100)
        valN = math.ceil(nFiles*ratio[1]/100)
        testingN = math.ceil(nFiles*ratio[2]/100)

        # Parse files based on ratio
        self.trainingFiles = [niftifiles.pop(0) for x in range(trainingN)]
        self.validationFiles = [niftifiles.pop(0) for x in range(valN)]
        self.testingFiles = niftifiles

        # Load images of niftifiles
        self.trainingImg = PreprocImg(self.trainingFiles)
        self.validationImg = PreprocImg(self.validationFiles)
        self.testingImg = PreprocImg(self.testingFiles)

#%%
# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Data path
datadir=r'C:\Users\allen\Desktop\datadir'
# datadir=r'F:\PatientData\thres'
R_TLEdir = r'C:\Users\allen\Desktop\datadir\TLE\EP_RTLE_nifti'
L_TLEdir = r'C:\Users\allen\Desktop\datadir\TLE\EP_LTLE_nifti'
Healthydir = os.path.join(datadir, 'Control')

    
# Define parameters
matter = 'GM'
ratio = [60, 15, 35]
iterations = 5
SIG_THRES = 0.05

R_TLEinput = CNNinput(extractNifti(R_TLEdir))
L_TLEinput = CNNinput(extractNifti(L_TLEdir))
Healthyinput = CNNinput(extractNifti(Healthydir))

comparison_list = {'R_TLE','L_TLE','Healthy'}
for comparisons in list(itertools.combinations(comparison_list,2)):
    
    comp_1 = eval(comparisons[0]+'input')
    comp_2 = eval(comparisons[1]+'input')
    
    comp_1_images = np.concatenate([comp_1.trainingImg,comp_1.validationImg,comp_1.testingImg],axis=3)
    comp_2_images = np.concatenate([comp_2.trainingImg,comp_2.validationImg,comp_2.testingImg],axis=3)
    
    t_vol = np.empty((113,137,113))
    p_vol = np.empty((113,137,113))
    for x in range(113):
        for y in range(137):
            for z in range(113):
                comp_1_vox = [comp_1_images[x,y,z,s] for s in range(comp_1_images.shape[3])]
                comp_2_vox = [comp_2_images[x,y,z,s] for s in range(comp_2_images.shape[3])]
                
                t_vol[x,y,z],p_vol[x,y,z] = scipy.stats.ttest_ind(comp_1_vox, comp_2_vox)
    
    t_vol_nifti = nib.Nifti1Image(t_vol, np.eye(4))
    nib.save(t_vol_nifti, os.path.join(datadir, comparisons[0] + ' vs '+comparisons[1] + '_t_vol.nii.gz'))  
    
    p_vol_nifti = nib.Nifti1Image(p_vol, np.eye(4))
    nib.save(p_vol_nifti, os.path.join(datadir, comparisons[0] + ' vs '+comparisons[1] + '_p_vol.nii.gz'))  
    
    adjusted_pval = SIG_THRES/np.count_nonzero(~np.isnan(t_vol))
    
    p_vol[p_vol>adjusted_pval] = 0
    t_vol[p_vol>adjusted_pval] = 0
                                           
    t_vol_nifti = nib.Nifti1Image(t_vol, np.eye(4))
    nib.save(t_vol_nifti, os.path.join(datadir, comparisons[0] + ' vs '+comparisons[1] + '_ADJUSTED_t_vol.nii.gz'))  
    
    p_vol_nifti = nib.Nifti1Image(p_vol, np.eye(4))
    nib.save(p_vol_nifti, os.path.join(datadir, comparisons[0] + ' vs '+comparisons[1] + '_ADJUSTED_p_vol.nii.gz'))  
    
                                               
