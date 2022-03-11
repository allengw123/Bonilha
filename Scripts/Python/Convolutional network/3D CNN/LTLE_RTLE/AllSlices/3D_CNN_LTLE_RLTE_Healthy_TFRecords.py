# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:32:26 2022

@author: bonilha
"""

#%% Functions

# Import modules
import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import datetime


AUTOTUNE = tf.data.experimental.AUTOTUNE

def fileParse(sbj_list,file_dir):
    diseases = {'LTLE','RTLE','Control'}    
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

def get_model(dimensions, arch):
    
    width, height, depth = dimensions
    print('Loading Model...', arch)
    if arch == 'Eleni':
    
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
        model = keras.Model(inputs, outputs, name="3dcnn_Eleni")
        
        loss = keras.losses.SparseCategoricalCrossentropy()
        optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.09)
        metrics = ["accuracy"]
        
        model.compile(optimizer=optim, loss=loss, metrics=metrics)
        
        batchSize = 2
        
    if arch == 'Anees':
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
        model = keras.Model(inputs, outputs, name="3dcnn_Anees")
        
        # Compile model.
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08, amsgrad=False),
            metrics=["accuracy"])
        
        batchSize = 20
        
    if arch != 'Eleni' and arch != 'Anees':
        raise ValueError('2nd Argument must be either "Eleni" or "Anees"')
        
    ct = datetime.datetime.now()
    if not (os.path.exists(os.path.join(MODEL_DIR, arch + '_arch'))):
        os.mkdir((os.path.join(MODEL_DIR, arch + '_arch')))
        
    model_save_folder = os.path.join(MODEL_DIR,arch + '_arch','Model_' + str(ct)).replace(':', '_').replace('_',':',1)
    
    if not (os.path.exists(model_save_folder)):
        os.mkdir(model_save_folder)
    
    if not os.path.exists(os.path.join(model_save_folder,'callbacks')):
        os.mkdir(os.path.join(model_save_folder,'callbacks'))
    callback_dir = os.path.join(os.path.join(model_save_folder,'callbacks'))
    callback_outputfile = os.path.join(callback_dir,'my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5')
    checkpoint = ModelCheckpoint(filepath=callback_outputfile, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
    
    return model, model_save_folder, checkpoint, batchSize

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


def save_model(savepath, model, history):
    model.save(savepath)
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.ylim((0,10))
    plt.show()
    plt.savefig(os.path.join(savepath,'model_training_history'))
                    
        

def confusionMat(predictions,labels,savepath):
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    for key, value in disease_labels.items():
        print('Writing Performance File for ...', key,)
        TP = sum(np.logical_and(predictions==value,labels==value))
        TN = sum(np.logical_and(predictions!=value,labels!=value))
        FN = sum(np.logical_and(predictions!=value,labels==value)) 
        FP = sum(np.logical_and(predictions==value,labels!=value))
        Acc = (TP+TN)/(TP+TN+FP+FN)
        Sensitivity = TP/(TP+FN)
        Precision = TP/(TP+FP)
        F1 = 2*((Precision*Sensitivity)/(Precision+Sensitivity))
        
        with open(os.path.join(savepath,key+' performance.txt'),'w') as f:
            metrics = ['TP = ' + str(TP) + '\n',
                      'TN = ' + str(TN) + '\n', 
                      'FN = ' + str(FN) + '\n',
                      'Acc = ' + str(Acc) + '\n',
                      'Sensitivity = ' + str(Sensitivity) + '\n',
                      'Precision = ' + str(Precision) + '\n',
                      'F1 Score = ' + str(F1)]
            f.writelines(metrics)
            
def saveParameters(matter,ratio,disease_labels,EPOCH,batchSize,savepath):
    with open(os.path.join(savepath,'parameters.txt'),'w') as f:
        parameters = ['Matter = ' + matter + '\n',
                  'Ratio = ' + str(ratio) + '\n', 
                  'Disease_labels = ' + str(disease_labels) + '\n',
                  'Epoch = ' + str(EPOCH) + '\n',
                  'Batch Size = ' + str(batchSize) + '\n',
                  'Script Name = ' + '3D_CNN_LTLE_RTLE_Healthy_TFRecord' + '\n',
                  ]
        f.writelines(parameters)
        
#%%
# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Data path
# RECORD_DIR=r'F:\test\TFRecords'
RECORD_DIR = r'F:\PatientData\TRRecords\LTLE_RTLE_Healthy_TFRecord'
MODEL_DIR = r'F:\CNN output\3D_CNN\RTLE_LTLE_Healthy\AllSlices\models'

# Define parameters
matter = 'GM'
ratio = [60, 15, 35]
iterations = 10
disease_labels = {"LTLE":0,"RTLE":1,"Healthy":2}
EPOCH = 1000
RUNNING_ARCH = ['Eleni', 'Anees']

for i in range(iterations):
    
    print('Running Iteration....'+str(i))
    
    for current_arch in RUNNING_ARCH:
        # Prepare Model
        model, savepath, checkpoint, batchSize = get_model((113,137,113),current_arch)
        
        # Find sbj files
        sbj_names = []
        for root, folders, file in os.walk(RECORD_DIR):
            for name in file:
                sbj_names.append(name.split('_ANGLES_')[0])
        sbj_names = set(sbj_names)
        
        # Obtain training/val/testing files
        trainingFiles, validationFiles ,testingFiles = fileParse(sbj_names,RECORD_DIR)
        
        # Load TFRecordDataset 
        trainingDataset = loadTFrecord(trainingFiles,batchSize)
        validationDataset = loadTFrecord(validationFiles,batchSize)
        testingDataset = loadTFrecord(testingFiles,batchSize)
    
        
    # =============================================================================
    #     ############ Run CNN model
    # =============================================================================
        
        # Train Model
        history = model.fit(trainingDataset,
                  epochs=EPOCH,
                  verbose=2,
                  validation_data=validationDataset,
                  shuffle=True,
                  callbacks=checkpoint)
        
        # Save Model and training progress
        save_model(savepath, model, history)
        
        #Load and evaluate the best model version
        best_model = os.listdir(os.path.join(savepath,'callbacks'))[-1]
        best_model = load_model(os.path.join(savepath,'callbacks',best_model))
        
        # Test Model
        image_batch = iter(testingDataset)
        
        final_test_predictions = []
        best_test_predictions = []
        test_labels = []
        for x in range(500):
            try:
                batch_image, batch_label = image_batch.next()
                print('Testing Batch Number...',str(x))
                for input in range(batch_image.shape[0]):
                    img = batch_image[input].numpy()
                    label = batch_label[input].numpy()
                    test_labels.append(label)
                    final_test_predictions.append(np.argmax(model.predict(tf.expand_dims(img,axis=0)),axis=1)[0])
                    best_test_predictions.append(np.argmax(model.predict(tf.expand_dims(img,axis=0)),axis=1)[0])
            except:
                print('End of Batch')
                break
        
        # Save Model Performance
        confusionMat(final_test_predictions,test_labels,savepath)
        confusionMat(best_test_predictions,test_labels,savepath)
        
        # Save Model Parematers
        saveParameters(matter,ratio,disease_labels,EPOCH,batchSize,savepath)
    
    # # Train Shuffle Model
    # model.fit(Shuff_train_dataset, epochs=epoch, verbose=2,validation_data=Shuff_validation_dataset,shuffle=True)
    
    
    # # Test Shuffle Model
    # with tf.device('/CPU:0'):
    #     prediction_weights = model.predict(test_images)
    # prediction_labels= np.argmax(prediction_weights,axis=1)
     
    # ShuffconMat.append(confusionMat(prediction_labels,test_labels))
    
