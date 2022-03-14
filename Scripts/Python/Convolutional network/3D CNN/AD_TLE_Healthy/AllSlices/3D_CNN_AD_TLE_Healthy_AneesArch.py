# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:19:16 2022

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
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model


AUTOTUNE = tf.data.experimental.AUTOTUNE

def fileParse(sbj_list,file_dir):
    diseases = {'TLE','AD','Control'}    
    sbj_list=list(sbj_list)
    file_list = os.listdir(file_dir)
    
    
    trainingFiles = []
    validationFiles = []
    testingFiles= []  
    for d in diseases:
    
        # Disease subject
        dSbj_list = [x for x in sbj_list if d in x]
        
        # length of files
        nSbj = len(dSbj_list)
        
        # shuffle files
        rNums = random.sample(range(nSbj), nSbj)
        subjects = [dSbj_list[x] for x in rNums]
    
        # obtain index ratio
        trainingN = math.ceil(nSbj*ratio[0]/100)
        valN = math.ceil(nSbj*ratio[1]/100)
        testingN = math.ceil(nSbj*ratio[2]/100)
        
        # Parse subjects based on ratio
        traingingSub = [subjects.pop(0) for x in range(trainingN)]
        print(d," Training....",str(len(traingingSub)))
        valSub = [subjects.pop(0) for x in range(valN)]
        print(d," Validation....",str(len(valSub)))
        testingSub = subjects
        print(d," Testing....",str(len(testingSub)))


    
        # Obtain files
        [trainingFiles.append(os.path.join(file_dir,file)) for file in file_list if file.split('_ANGLES_')[0] in traingingSub]
        [validationFiles.append(os.path.join(file_dir,file)) for file in file_list if file.split('_ANGLES_')[0] in valSub] 
        [testingFiles.append(os.path.join(file_dir,file)) for file in file_list if file.split('_ANGLES_')[0] in testingSub]
        
    
    return trainingFiles, validationFiles, testingFiles
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
            
            self.metrics.update({diseases:{"TP":TP,"TN":TN,"FP":FP,"FN":FN,"Acc":Acc,'Sensitivity':Sensitivity,"Precision":Precision}})
            
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

def loadTFrecord(files,batchNum):
    files = [x for x in files if 'ANGLES_0_0_0' in x]
    print(str(len(files)) + ' ....Imported')
    dataset = tf.data.TFRecordDataset(files).map(decode)
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.batch(batchNum)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def save_model(path, model):
    if not os.path.exists(path):
        print('save directories...', flush=True)
        os.makedirs(path)
    model.save(path)
                    
        


#%%
# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Data path
# record_dir=r'F:\test\TFRecords'
record_dir=r'F:\test\TFRecords_TEST'
callback_outputfile = r'F:\test\my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'

# Define parameters
matter = 'GM'
ratio = [60, 15, 35]
iterations = 5
disease_labels = {"AD":0,"TLE":1,"Healthy":2}


conMat_final=[]
conMat_best=[]
ShuffconMat=[]
TrueModels=[]
ShuffModels=[]
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
    trainingDataset = loadTFrecord(trainingFiles,batchSize)
    validationDataset = loadTFrecord(validationFiles,batchSize)
    testingDataset = loadTFrecord(testingFiles,batchSize)

    
    # Prepare Model
    model=get_model(113,137,113)
    epoch = 1000
    checkpoint = ModelCheckpoint(filepath=callback_outputfile, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
    
    ############ Run CNN model
    # Train Model
    history = model.fit(trainingDataset,
              epochs=epoch,
              verbose=2,
              validation_data=validationDataset,
              shuffle=True,
              callbacks=checkpoint)
    
    # Plot the training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.savefig('model_training_history')
    plt.show()
    
    #Load and evaluate the best model version
    best_model = load_model( r'F:\test\models\Model 2\my_best_model.epoch87-loss0.26.hdf5')
    
    # Test Model
    image_batch = iter(testingDataset)
    
    final_test_predictions = []
    best_test_predictions = []
    test_labels = []
    for x in range(500):
        print(x)
        batch_image, batch_label = image_batch.next()
        for input in range(batch_image.shape[0]):
            img = batch_image[input].numpy()
            label = batch_label[input].numpy()
            test_labels.append(label)
            final_test_predictions.append(np.argmax(model.predict(tf.expand_dims(img,axis=0)),axis=1)[0])
            best_test_predictions.append(np.argmax(model.predict(tf.expand_dims(img,axis=0)),axis=1)[0])

    
    conMat_final.append(confusionMat(np.array(final_test_predictions),np.array(test_labels)))
    conMat_best.append(confusionMat(np.array(best_test_predictions),np.array(test_labels)))
    
    save_model(r'F:\test\models\Model 1\Model')
    
    # Train Shuffle Model
    model.fit(Shuff_train_dataset, epochs=epoch, verbose=2,validation_data=Shuff_validation_dataset,shuffle=True)
    
    
    # Test Shuffle Model
    with tf.device('/CPU:0'):
        prediction_weights = model.predict(test_images)
    prediction_labels= np.argmax(prediction_weights,axis=1)
     
    ShuffconMat.append(confusionMat(prediction_labels,test_labels))
    
