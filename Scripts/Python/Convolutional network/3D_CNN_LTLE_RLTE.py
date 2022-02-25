# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:17:00 2022

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

        # LoLTLE images of niftifiles
        self.trainingImg = PreprocImg(self.trainingFiles)
        self.validationImg = PreprocImg(self.validationFiles)
        self.testingImg = PreprocImg(self.testingFiles)

# Build a 3D convolutional neural network model
def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

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
    
    # Compile model.
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],)
    
    return model



def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        
        # rotate volume
        volume = ndimage.rotate(volume,angle=random.choice(angles),axes=(0,1),reshape=False)
        volume = ndimage.rotate(volume,angle=random.choice(angles),axes=(1,2),reshape=False)
        volume = ndimage.rotate(volume,angle=random.choice(angles),axes=(0,2),reshape=False)
        
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)

    return volume, label

 

class confusionMat:
    def __init__(self, prediction_labels,test_labels):
        self.predictions = prediction_labels
        self.labels = test_labels        
        self.metrics = {}
       
        for diseases in disease_labels:
            
            disease_val = disease_labels[diseases]
            TP = sum(np.logical_and(prediction_labels==disease_val,test_labels==disease_val))
            TN = sum(np.logical_and(prediction_labels!=disease_val,test_labels!=disease_val))
            FN = sum(np.logical_and(prediction_labels!=disease_val,test_labels==disease_val)) 
            FP = sum(np.logical_and(prediction_labels==disease_val,test_labels!=disease_val))
            Acc = (TP+TN)/(TP+TN+FP+FN)
            Sensitivity = TP/(TP+FN)
            Precision = TP/(TP+FP)
            
            self.metrics.update({diseases:{"TP":TP,"TN":TN,"FP":FP,"Acc":Acc,'Sensitivity':Sensitivity,"Precision":Precision}})
            
def shuffle_array(input):
    np.random.shuffle(input)
    return input


                    
        


#%%
# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Data path
datadir=r'C:\Users\allen\Desktop\datadir'
LTLEdir = os.path.join(datadir, 'TLE','TLE','EP_LTLE_nifti')
RTLEdir = os.path.join(datadir, 'TLE','TLE','EP_RTLE_nifti')


# Define parameters
matter = 'GM'
ratio = [60, 15, 35]
iterations = 5
disease_labels = {"LTLE":0,"RTLE":1}


conMat=[]
ShuffconMat=[]
TrueModels=[]
ShuffModels=[]
for i in range(iterations):
    
    print('Running Iteration....'+str(i))
    
    # Prepare disease specific CNN input
    LTLEinput = CNNinput(extractNifti(LTLEdir))  # 0
    RTLEinput = CNNinput(extractNifti(RTLEdir))  # 1
    
    # Pepare x y data
    train_images = np.concatenate(
        [LTLEinput.trainingImg, RTLEinput.trainingImg],axis=3).transpose((3,0,1,2))
    train_labels = np.concatenate([np.zeros(LTLEinput.trainingImg.shape[3]),  np.ones(RTLEinput.trainingImg.shape[3])])
    
    validation_images = np.concatenate(
        [LTLEinput.validationImg, RTLEinput.validationImg],axis=3).transpose((3,0,1,2))
    validation_labels = np.concatenate([np.zeros(LTLEinput.validationImg.shape[3]), np.ones(RTLEinput.validationImg.shape[3])])
    
    test_images = np.concatenate(
        [LTLEinput.testingImg, RTLEinput.testingImg],axis=3).transpose((3,0,1,2))
    test_labels = np.concatenate([np.zeros(LTLEinput.testingImg.shape[3]), np.ones(RTLEinput.testingImg.shape[3])])
    
    # Define data loaders
    train_loader = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    validation_loader = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    
    Shuff_train_loader = tf.data.Dataset.from_tensor_slices((train_images,shuffle_array(train_labels)))
    Shuff_validation_loader = tf.data.Dataset.from_tensor_slices((validation_images,shuffle_array(validation_labels)))
    

    # Preproc dataset
    batch_size = 2

    train_dataset = train_loader.shuffle(len(train_labels),reshuffle_each_iteration=True).map(train_preprocessing).batch(batch_size).prefetch(2)
    validation_dataset = validation_loader.shuffle(len(validation_labels),reshuffle_each_iteration=True).map(validation_preprocessing).batch(batch_size).prefetch(batch_size)
    
    Shuff_train_dataset = Shuff_train_loader.shuffle(len(train_labels),reshuffle_each_iteration=True).map(train_preprocessing).batch(batch_size).prefetch(2)
    Shuff_validation_dataset = Shuff_validation_loader.shuffle(len(validation_labels),reshuffle_each_iteration=True).map(validation_preprocessing).batch(batch_size).prefetch(batch_size)
    
    
    # data = train_dataset.take(1)
    # images, labels = list(data)[0]
    # images = images.numpy()
    # image = images[0]
    # print("Dimension of the CT scan is:", image.shape)
    # plt.imshow(np.squeeze(image[:, :, 40]), cmap="gray")
    # plt.imshow(np.squeeze(image[:, 60, :]), cmap="gray")
    # plt.imshow(np.squeeze(image[60, :, :]), cmap="gray")

    
    # Prepare Model
    model=get_model(113,137,113)
    epoch = 30
    
    ############ Run CNN model
    # Train Model
    model.fit(train_dataset, epochs=epoch, verbose=2,validation_data=validation_dataset,shuffle=True)
    
    # Test Model
    prediction_weights = model.predict(test_images)
    prediction_labels= np.argmax(prediction_weights,axis=1)
    
    conMat.append(confusionMat(prediction_labels,test_labels))
    
    # Train Shuffle Model
    model.fit(Shuff_train_dataset, epochs=epoch, verbose=2,validation_data=Shuff_validation_dataset,shuffle=True)
     
    # Test Shuffle Model
    prediction_weights = model.predict(test_images)
    prediction_labels= np.argmax(prediction_weights,axis=1)
     
    ShuffconMat.append(confusionMat(prediction_labels,test_labels))
    

