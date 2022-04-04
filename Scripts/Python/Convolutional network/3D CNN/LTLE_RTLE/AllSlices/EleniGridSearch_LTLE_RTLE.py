# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:07:26 2022

@author: bonilha
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
import itertools
from tensorboard.plugins.hparams import api as hp
from scipy.stats import linregress
from keras.callbacks import TensorBoard
from keras import backend as K
import datetime


# your code here    
AUTOTUNE = tf.data.experimental.AUTOTUNE

def fileParse(sbj_list,file_dir):
    sbj_list=list(sbj_list)
    file_list = os.listdir(file_dir)
    
    
    trainingFiles = []
    validationFiles = []
    testingFiles= []  
    for d in disease_labels:
    
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

def loadTFrecord(input_files, batchNum, set_type, augmentExpand=False, expandNum=0):
    
   
    if set_type == 'training':
        print('Augment Expand...',str(augmentExpand))
        if augmentExpand:
            
            # Angles
            angles = [-20, -10, 0 ,10, 20]
            perm = list(itertools.product(angles,repeat=3))
            random.shuffle(perm)
            
            out_files = [x for x in input_files if 'ANGLES_0_0_0' in x]
            for i in range(expandNum):
                selected_angle = perm.pop()
                print('Selecting rotation...',str(selected_angle))
                out_files = out_files + [x for x in input_files if 'ANGLES_'+str(selected_angle[0])+'_'+str(selected_angle[1])+'_'+str(selected_angle[2]) in x]
        else:
            out_files = [x for x in input_files if 'ANGLES_0_0_0' in x]
        
    else:
        out_files = [x for x in input_files if 'ANGLES_0_0_0' in x]
            
    print('Imported ',set_type,' file number....',str(len(out_files)))
    dataset = tf.data.TFRecordDataset(out_files).map(decode)
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.batch(batchNum)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

def confusionMat(weights,predictions,labels):
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    accuracy = []
    F1_score = []
    for key, value in disease_labels.items():
        print('Writing Performance File for ...', key,)
        TP = sum(np.logical_and(predictions==value,labels==value))
        TN = sum(np.logical_and(predictions!=value,labels!=value))
        FN = sum(np.logical_and(predictions!=value,labels==value)) 
        FP = sum(np.logical_and(predictions==value,labels!=value))
        Acc = (TP+TN)/(TP+TN+FP+FN)
        
        if TP == 0 and FN == 0:
            Sensitivity = 0
        else:
            Sensitivity = TP/(TP+FN)

        if TP == 0 and FP == 0:
            Precision = 0
        else:
            Precision = TP/(TP+FP)

        if Precision == 0 and Sensitivity == 0:
            F1 = 0
        else:
            F1 = 2*((Precision*Sensitivity)/(Precision+Sensitivity))
           
        
        accuracy.append(Acc)
        F1_score.append(F1)
    
    return accuracy, F1_score
        
         
def train_test_model(dimensions, hparams, run_dir):
    
    batchSize = 15
    
    # Load TFRecordDataset 
    trainingDataset = loadTFrecord(trainingFiles,batchSize,'training')
    validationDataset = loadTFrecord(validationFiles,batchSize,'validation')
    testingDataset = loadTFrecord(testingFiles,batchSize,'testing')

    width, height, depth = dimensions
    
    inputs = keras.Input((width, height, depth, 1))
    
    x = layers.Conv3D(filters=8, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=(2,2,2),strides=2)(x)

    x = layers.Conv3D(filters=16, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=(2,2,2),strides=2)(x)
    
    x = layers.Conv3D(filters=32, kernel_size=3 ,padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=(2,2,2),strides=2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(units=hparams['HP_NUM'])(x)
    x = layers.Dropout(hparams['HP_DO'])(x)
    
    outputs = layers.Dense(units=3)(x)
    
    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn_Eleni")
    
    # Compile model.
    optimizer=keras.optimizers.SGD(learning_rate=hparams['HP_LR'], momentum=0.09)
    
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=["accuracy"])
    
    # Callback for TensorBoard/Learning Rate
    tensorboard_save_folder = os.path.join(run_dir,'TensorBoard')
    if not os.path.exists(tensorboard_save_folder):
        os.mkdir(tensorboard_save_folder)
    class LRTensorBoard(TensorBoard):
        # add other arguments to __init__ if you need
        def __init__(self, log_dir, **kwargs):
            super().__init__(log_dir=log_dir, **kwargs)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs.update({'lr': K.eval(self.model.optimizer.lr)})
            super().on_epoch_end(epoch, logs)
    TB_callback = LRTensorBoard(log_dir=tensorboard_save_folder)
    
    # Reduce On Plateu CallBack
    ROP_callback =  tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,
        patience=5,
        min_delta=0.0001, 
        min_lr=0.000000001
    )
        
    # Early Stop
    ES_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=15,
        verbose=2,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    
    # Compile Callbacks
    Callbacks = [
        TB_callback,
        ROP_callback, 
        ES_callback
        ]
    
    history = model.fit(trainingDataset,
              epochs=EPOCH,
              verbose=2,
              validation_data=validationDataset,
              shuffle=True,
              callbacks =  Callbacks
              )
    metric_output = []
    
    loss_slope = linregress(np.arange(0,5),np.array(history.history['loss'][-5:],dtype=np.float64)).slope
    val_slope = linregress(np.arange(0,5),np.array(history.history['val_loss'][-5:],dtype=np.float64)).slope
    diff_slope = linregress(np.arange(0,5),np.array(history.history['loss'][-5:],dtype=np.float64)-np.array(history.history['val_loss'][-5:],dtype=np.float64)).slope

    image_batch = iter(testingDataset)
    
    test_labels = []
    test_weights = []
    test_predictions = []
    
    for x in range(500):
        try:
            batch_image, batch_label = image_batch.next()
            print('Testing Batch Number...',str(x))
            for input in range(batch_image.shape[0]):
                img = batch_image[input].numpy()
                label = batch_label[input].numpy()
                
                test_labels.append(label)
                
                weights = model.predict(tf.expand_dims(img,axis=0))
                test_weights.append(weights)
                test_predictions.append(np.argmax(weights,axis=1)[0])
        except:
            print('End of Batch')
            break
    
    acc,F1 = confusionMat(test_weights,test_predictions,test_labels)

    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    loss_diff = history.history['loss'][-1]-history.history['val_loss'][-1]
    
    metric_output.append(acc[0])
    metric_output.append(acc[1])
    metric_output.append(F1[0])
    metric_output.append(F1[1])
    metric_output.append(F1[0]+F1[1])
    metric_output.append(loss)
    metric_output.append(val_loss)
    metric_output.append(loss_diff)
    metric_output.append(loss_slope)
    metric_output.append(val_slope)
    metric_output.append(diff_slope)


    return metric_output


def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    
    model_metric = train_test_model((113, 137, 113), hparams, run_dir)
    for i in range(len(METRIC)):
        tf.summary.scalar(METRIC[i], model_metric[i],step=1)

        
        
#%%
# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Data path
# RECORD_DIR=r'F:\test\TFRecords'
RECORD_DIR = r'F:\PatientData\TRRecords\LTLE_RTLE_Healthy_TFRecord'
MODEL_DIR = r'F:\CNN output\3D_CNN\RTLE_LTLE_Healthy\AllSlices\models'
HPARAM_DIR = r'F:\CNN output\3D_CNN\RTLE_LTLE\AllSlices\hparam\V1\hparam_tuning'

# Define parameters
matter = 'GM'
ratio = [70, 20, 10]
iterations = 10
disease_labels = {"LTLE":0,
                  "RTLE":1}
EPOCH = 200
RUNNING_ARCH = ['Eleni']
Augment_arg = False
Aug_Expand_Num = 1
L2 = 0.001

HP_LEARNINGRATE = hp.HParam('LearningRate', hp.Discrete([0.1, 0.01, 0.001]))
HP_DROPOUT = hp.HParam('DropOut', hp.Discrete([0.2, 0.4, 0.8]))
HP_NUM = hp.HParam('NeuronNum', hp.Discrete([32, 64, 128]) )

METRIC = ['LTLE Accuracy',
          'RTLE Accuracy',
          'LTLE F1',
          'RTLE F1',
          'Total TLE F1',
          'Final Training Loss',
          'Final Validation Loss',
          'Final Loss Diff',
          'Training loss Slope',
          'Validation loss slope',
          'Loss Diff slope'
          ]


# Find sbj files
sbj_names = []
for root, folders, file in os.walk(RECORD_DIR):
    for name in file:
        sbj_names.append(name.split('_ANGLES_')[0])
sbj_names = set(sbj_names)


for i in range(iterations):
    # Obtain training/val/testing files
    trainingFiles, validationFiles ,testingFiles = fileParse(sbj_names,RECORD_DIR)
    
    for LR in HP_LEARNINGRATE.domain.values:
        for DO in HP_DROPOUT.domain.values:
            for NUM in HP_NUM.domain.values:
    
                
                
                hparams = {
                    'HP_LR': LR,
                    'HP_DO': DO,
                    'HP_NUM': NUM
                }
              
                run_name = str(datetime.datetime.now()).replace(':', '_')
    
              
                run(HPARAM_DIR + run_name, hparams)    
    