# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 09:58:17 2022

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
    imgs = [nib.load(file).get_fdata() for file in input]
    imgs = NormalizeData(np.stack(imgs,axis=3))
    imgs = np.expand_dims(imgs,axis=4)
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

        # LoLTLE images of niftifiles
        self.trainingImg = PreprocImg(self.trainingFiles)
        self.validationImg = PreprocImg(self.validationFiles)
        self.testingImg = PreprocImg(self.testingFiles)

# Build a 3D convolutional neural network model
def get_model(width, height, depth):

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=3, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

#%%
# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Data path
datadir=r'C:\Users\allen\Desktop\datadir'
Healthydir = os.path.join(datadir, 'Control')
LTLEdir = os.path.join(datadir, 'TLE','TLE','EP_LTLE_nifti')
RTLEdir = os.path.join(datadir, 'TLE','TLE','EP_RTLE_nifti')


# Define parameters
matter = 'GM'
ratio = [60, 15, 35]
iterations = 10

accuracy=[]
for i in range(iterations):
    # Prepare disease specific CNN input
    LTLEinput = CNNinput(extractNifti(LTLEdir))  # 0
    RTLEinput = CNNinput(extractNifti(RTLEdir))  # 1
    Healthyinput = CNNinput(extractNifti(Healthydir))  # 2
    
    # Pepare CNN input
    train_images = np.concatenate(
        [LTLEinput.testingImg, RTLEinput.testingImg, Healthyinput.testingImg],axis=3).transpose((3,0,1,2,4))
    train_labels = np.concatenate([np.zeros(LTLEinput.testingImg.shape[3]),  np.ones(RTLEinput.testingImg.shape[3]), np.ones(
        Healthyinput.testingImg.shape[3])*2])
    
    validation_images = np.concatenate(
        [LTLEinput.validationImg, RTLEinput.validationImg, Healthyinput.validationImg],axis=3).transpose((3,0,1,2,4))
    validation_labels = np.concatenate([np.zeros(LTLEinput.validationImg.shape[3]), np.ones(RTLEinput.validationImg.shape[3]), np.ones(
        Healthyinput.validationImg.shape[3])*2])
    
    test_images = np.concatenate(
        [LTLEinput.testingImg, RTLEinput.testingImg, Healthyinput.testingImg],axis=3).transpose((3,0,1,2,4))
    test_labels = np.concatenate([np.zeros(LTLEinput.testingImg.shape[3]), np.ones(RTLEinput.testingImg.shape[3]), np.ones(
        Healthyinput.testingImg.shape[3])*2])
    
    model=get_model(113,137,113)
    
    loss = keras.losses.SparseCategoricalCrossentropy()
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.09)
    metrics = ["accuracy"]
    
    model.compile(optimizer=optim, loss=loss, metrics=metrics)
    
    batch_size = 10
    epoch = 30
    
    #%% Run CNN model
    
    # Train Model
    model.fit(train_images, train_labels, epochs=epoch, batch_size=batch_size, verbose=2,validation_data=(validation_images,validation_labels),shuffle=True)
    
    # Test Model
    accuracy.append(model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=2))
    

plt.plot([x[1] for x in accuracy])