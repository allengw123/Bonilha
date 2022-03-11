# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:32:00 2022

@author: allen
"""

#%% Functions

# Import modules
import os
import numpy as np
import random
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.experimental.AUTOTUNE

def fileParse(sbj_list,file_dir):
    diseases = {'TLE','AD','Control'}    
    sbj_list=list(sbj_list)
    file_list = os.listdir(file_dir)
    
    
    trainingFiles = []
    validationFiles = []
    testingFiles= []  
    for d in diseases:
    
        # length of files
        nSbj = len(sbj_list)
    
        # shuffle files
        rNums = random.sample(range(nSbj), nSbj)
        subjects = [sbj_list[x] for x in rNums]
    
        # obtain index ratio
        trainingN = math.ceil(nSbj*ratio[0]/100)
        valN = math.ceil(nSbj*ratio[1]/100)
        testingN = math.ceil(nSbj*ratio[2]/100)
        
        # Parse subjects based on ratio
        traingingSub = [subjects.pop(0) for x in range(trainingN)]
        valSub = [subjects.pop(0) for x in range(valN)]
        testingSub = subjects

    
        # Obtain files
        [trainingFiles.append(os.path.join(file_dir,file)) for file in file_list if file.split('_ANGLES_')[0] in traingingSub]
        [validationFiles.append(os.path.join(file_dir,file)) for file in file_list if file.split('_ANGLES_')[0] in valSub] 
        [testingFiles.append(os.path.join(file_dir,file)) for file in file_list if file.split('_ANGLES_')[0] in testingSub]
        
    
    return trainingFiles, validationFiles, testingFiles

# Build a 3D convolutional neural network model
def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    
    inputs = keras.Input((width, height, depth, 1))
    
    x = layers.Conv3D(filters=64, kernel_size=5,strides=2,padding='valid')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=3,strides=3)(x)
    
    x = layers.Conv3D(filters=128, kernel_size=3,strides=1,padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=3,strides=3)(x)
    
    x = layers.Conv3D(filters=192, kernel_size=3,strides=1,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv3D(filters=192, kernel_size=3,strides=1,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv3D(filters=128, kernel_size=3,strides=1,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2,strides=3)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(3)(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    
    # Compile model.
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08, amsgrad=False),
        metrics=["accuracy"])
    
    
    return model
 

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
            

def save_model(path, model):
    if not os.path.exists(path):
        print('save directories...', flush=True)
        os.makedirs(path)
    model.save(path)
                    

def decode(serialized_example):
    # Decode examples stored in TFRecord
    # NOTE: make sure to specify the correct dimensions for the images
    features = tf.io.parse_single_example(
        serialized_example,
        features={'image': tf.io.FixedLenFeature([113, 137, 113, 1], tf.float32),
                  'label': tf.io.FixedLenFeature([], tf.int64),
                  'fileName': tf.io.FixedLenFeature([], tf.string, default_value='')})

    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    return features['image'], features['label']

def loadTFrecord(files,set_type,batchNum):
    if set_type == 'train':
        dataset = tf.data.TFRecordDataset(files).map(decode)
        dataset = dataset.repeat()    
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
        dataset = dataset.batch(batchNum)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        
        # dataset = tf.data.TFRecordDataset(files)
    else:
        dataset = tf.data.TFRecordDataset([x for x in files if 'ANGLES_0_0_0' in x]).map(decode)
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
        dataset = dataset.batch(batchNum)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset



def storeSourceFile(Input):
    
    global gCount
    # Increment gcount
    gCount=+1
    print('Gcount')
    global SourceFile
    
    if gCount < 1:
        
        SourceFile=[]
        SourceFile.append(Input)
    else:
        print('1')
        # SourceFile.append(Input)
        
    

#%%
# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Data path
# record_dir=r'F:\test\TFRecords'
record_dir=r'F:\test\TFRecords_TEST'

# Define parameters
matter = 'GM'
ratio = [60, 15, 35]
iterations = 5
disease_labels = {"AD":0,"TLE":1,"Healthy":2}


for i in range(iterations):
    
    
    print('Running Iteration....'+str(i))
    
    # Find sbj files
    sbj_names = []
    for root, folders, file in os.walk(record_dir):
        for name in file:
            sbj_names.append(name.split('_ANGLES_')[0])
    sbj_names = set(sbj_names)
    
    # Obtain training/val/testing files
    trainingFiles, validationFiles ,testingFiles = fileParse(sbj_names,record_dir)
    
    # Load TFRecordDataset 
    batchSize = 20
    NUM_SAMPLES = 10000
    trainingDataset = loadTFrecord(trainingFiles,'train',batchSize)
    validationDataset = loadTFrecord(validationFiles,'validation',batchSize)
    testingDataset = loadTFrecord(testingFiles,'test',batchSize)

    # iterator=iter(trainingDataset)
    # nx = iterator.get_next()

    # Prepare Model
    gCount = 0
    model=get_model(113,137,113)
    epoch = 30
    
    ############ Run CNN model
    # Train Model
    # with tf.device('/CPU:0'):
    model.fit(trainingDataset,epochs=epoch, verbose=2,validation_data=validationDataset,shuffle=True,steps_per_epoch=NUM_SAMPLES/batchSize)

    # Test Model
    prediction_weights = model.predict(testingDataset)
    prediction_labels= np.argmax(prediction_weights,axis=1)
    
    conMat.append(confusionMat(prediction_labels,test_labels))
    
    model.save('Downloads')
    
    # Train Shuffle Model
    model.fit(Shuff_train_dataset, epochs=epoch, verbose=2,validation_data=Shuff_validation_dataset,shuffle=True)
    
    
    # Test Shuffle Model
    with tf.device('/CPU:0'):
        prediction_weights = model.predict(test_images)
    prediction_labels= np.argmax(prediction_weights,axis=1)
     
    ShuffconMat.append(confusionMat(prediction_labels,test_labels))
    
