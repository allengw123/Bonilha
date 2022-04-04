# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:18:20 2022

@author: bonilha
"""

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import os
import random
import math
import itertools
import matplotlib
import matplotlib.pyplot as plt
import xlwt
from sklearn import model_selection
import numpy as np
import pickle

AUTOTUNE = tf.data.experimental.AUTOTUNE

def fileParse_KF(sbj_list,file_dir):
    
    sbj_list=list(sbj_list)
    file_list = os.listdir(file_dir)

    dSbj_list = []
    dSbj_label = []
    for d, label in disease_labels.items():
    
        # Disease subject
        dSbjs = [x for x in sbj_list if d in x]
        dSbj_list = dSbjs + dSbj_list
        dSbj_label = dSbj_label+([label for x in dSbjs])
        
        
    # KFold Parse  
    cv = model_selection.StratifiedKFold(KF_NUM,shuffle=True)
        
    train_sbj_files = []
    validation_sbj_files = []
    test_sbj_files = []
    count = 0
    for train_indices, test_indices in cv.split(range(len(dSbj_list)),np.array(dSbj_label)):
        print('Parsing K-Fold ',count)
        count+=1
        
        random.shuffle(train_indices)
        validation_split = 0.1
        
        train_indices = list(train_indices)
        test_indices = list(test_indices)
        validation_indices = [train_indices.pop() for x in range(math.ceil(len(train_indices)*validation_split))]
        
        train_sbjs=[dSbj_list[i] for i in train_indices]
        validation_sbj=[dSbj_list[i] for i in validation_indices]
        test_sbjs=[dSbj_list[i] for i in test_indices]
        print('   Length of training set ',str(len(train_sbjs)))
        print('   Length of validation set ',str(len(validation_sbj)))
        print('   length of testing set ',str(len(test_sbjs)))
        
        checkpoint='FAILURE'
        if len(set(train_indices+validation_indices+test_indices)) == (len(train_indices) + len(validation_indices) + len(test_indices)):
             checkpoint = 'PASS'

        print('Checkpoint...'+checkpoint)      
        train_sbj_files.append([os.path.join(file_dir,x) for x in file_list if x.split('_ANGLES_')[0] in train_sbjs])
        validation_sbj_files.append([os.path.join(file_dir,x) for x in file_list if x.split('_ANGLES_')[0] in validation_sbj])
        test_sbj_files.append([os.path.join(file_dir,x) for x in file_list if x.split('_ANGLES_')[0] in test_sbjs])

    return train_sbj_files, validation_sbj_files, test_sbj_files


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
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


matplotlib.use("Agg")
# import the necessary package
def save_plot(H, path):
	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig(path)



def build_model(hp):
        
    if arch == 'Eleni':
        inputs = keras.Input((113, 137, 113, 1))
        L2=0.001
        x = layers.Conv3D(filters=hp.Int("conv_1", min_value=8, max_value=16, step=8), kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=(2,2,2),strides=2)(x)
    
        x = layers.Conv3D(filters=hp.Int("conv_2", min_value=16, max_value=32, step=16), kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=(2,2,2),strides=2)(x)
        
        x = layers.Conv3D(filters=hp.Int("conv_2", min_value=16, max_value=32, step=16), kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=(2,2,2),strides=2)(x)
    
        x = layers.Flatten()(x)
        x = layers.Dense(hp.Choice('dense_num_3',values=[16,32,64,128]))(x)
        x = layers.Dropout(hp.Choice('dropout_3',values=[0.1,0.5,0.9]))(x)
        
        outputs = layers.Dense(units=3)(x)
        
        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn_Eleni")
        
        # Compile model.
        optimizer=keras.optimizers.SGD(learning_rate=1e-2,
                                       momentum=0.09)
        
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=["accuracy"])
    else:
        inputs = keras.Input((113, 137, 113, 1))
        
        # L2_1 = hp.Choice('L2_1',values=[0.1,0.01,0.001,0.0001])
        # L2_2 = hp.Choice('L2_2',values=[0.1,0.01,0.001,0.0001])
        # L2_3 = hp.Choice('L2_3',values=[0.1,0.01,0.001,0.0001])
        # L2_4 = hp.Choice('L2_4',values=[0.1,0.01,0.001,0.0001])
        # L2_5 = hp.Choice('L2_5',values=[0.1,0.01,0.001,0.0001])
        # L2_6 = hp.Choice('L2_6',values=[0.1,0.01,0.001,0.0001])
        
        L2_1 = 0.1
        L2_2 = 0.1
        L2_3 = 0.1
        L2_4 = 0.1
        L2_5 = 0.1
        L2_6 = 0.1

        x = layers.Conv3D(filters=64, kernel_size=5,strides=2,padding='valid',kernel_regularizer=tf.keras.regularizers.l2(L2_1))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=3,strides=3)(x)
        
        x = layers.Conv3D(filters=128, kernel_size=3,strides=1,padding='valid',kernel_regularizer=tf.keras.regularizers.l2(L2_2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=3,strides=3)(x)
        
        x = layers.Conv3D(filters=192, kernel_size=3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(L2_3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv3D(filters=192, kernel_size=3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(L2_4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv3D(filters=128, kernel_size=3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(L2_5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.MaxPool3D(pool_size=2,strides=3)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(hp.Choice('DO_1',values=[0.0, 0.1, 0.5, 0.9]))(x)
        x = layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(L2_6))(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(hp.Choice('DO_2',values=[0.0, 0.1, 0.5, 0.9]))(x)
        
        outputs = layers.Dense(3)(x)
        
        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn_Anees")
        
        # Compile model
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08, 
            amsgrad=False)
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        
    
    return model

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        inputs = keras.Input((113, 137, 113, 1))
        
        L2_1 = hp.Choice('L2_1',values=[0.1,0.01,0.001,0.0001])
        L2_2 = hp.Choice('L2_2',values=[0.1,0.01,0.001,0.0001])
        L2_3 = hp.Choice('L2_3',values=[0.1,0.01,0.001,0.0001])
        L2_4 = hp.Choice('L2_4',values=[0.1,0.01,0.001,0.0001])
        L2_5 = hp.Choice('L2_5',values=[0.1,0.01,0.001,0.0001])
        L2_6 = hp.Choice('L2_6',values=[0.1,0.01,0.001,0.0001])

        x = layers.Conv3D(filters=64, kernel_size=5,strides=2,padding='valid',kernel_regularizer=tf.keras.regularizers.l2(L2_1))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=3,strides=3)(x)
        
        x = layers.Conv3D(filters=128, kernel_size=3,strides=1,padding='valid',kernel_regularizer=tf.keras.regularizers.l2(L2_2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=3,strides=3)(x)
        
        x = layers.Conv3D(filters=192, kernel_size=3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(L2_3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv3D(filters=192, kernel_size=3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(L2_4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv3D(filters=128, kernel_size=3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(L2_5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.MaxPool3D(pool_size=2,strides=3)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(hp.Choice('DO_1',values=[0.0, 0.1, 0.5, 0.9]))(x)
        x = layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(L2_6))(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(hp.Choice('DO_2',values=[0.0, 0.1, 0.5, 0.9]))(x)
        
        outputs = layers.Dense(3)(x)
        
        return keras.Model(inputs=inputs, outputs=outputs)

    def fit(self, hp, model, train_files, validation_files,test_files, epochs, callbacks=None, **kwargs):
        # Convert the datasets to tf.data.Dataset.
        batch_size = 1
        
        # Define the optimizer.
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08, 
            amsgrad=False)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # The metric to track validation loss.
        epoch_loss_metric = keras.metrics.Mean()

        # Function to run the train step.
        @tf.function
        def run_train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Function to run the validation step.
        @tf.function
        def run_val_step(images, labels):
            logits = model(images)
            loss = loss_fn(labels, logits)
            # Update the metric.
            epoch_loss_metric.update_state(loss)

        # Assign the model to the callbacks.
        for callback in callbacks:
            callback.model = model

        
        mean_epoch_loss = []    
        # The custom training loop.
        for trainingFiles, validationFiles, testingFiles in zip(train_files,validation_files, test_files):
            # Load TFRecordDataset 
            trainingDataset = loadTFrecord(trainingFiles,batch_size,'training')
            validationDataset = loadTFrecord(validationFiles,batch_size,'validation')
            testingDataset = loadTFrecord(testingFiles,batch_size,'testing')
            
            # Record the best validation loss value
            best_epoch_loss = float("inf")
            
            for epoch in range(epochs):
                print(f"Epoch: {epoch}")
                
                for images, labels in trainingDataset:
                    run_train_step(images, labels)

                # Iterate the validation data to run the validation step.
                for images, labels in validationDataset:
                    run_val_step(images, labels)

                # Calling the callbacks after epoch.
                epoch_loss = float(epoch_loss_metric.result().numpy())
                for callback in callbacks:
                    # The "my_metric" is the objective passed to the tuner.
                    callback.on_epoch_end(epoch, logs={"my_metric": epoch_loss})
                epoch_loss_metric.reset_states()

                print(f"Epoch loss: {epoch_loss}")
                best_epoch_loss = min(best_epoch_loss, epoch_loss)
                
            mean_epoch_loss.append(best_epoch_loss)
            
        # Return the evaluation metric value.
        return np.mean(mean_epoch_loss)
#%%

# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Data path
RECORD_DIR = r'F:\PatientData\TRRecords\LTLE_RTLE_Healthy_TFRecord'
OUT_PATH = r'C:\Users\bonilha\Downloads\test10'
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)

# Define parameters
Tuners = [
    'hyperband'
    ]
arch = 'Anees'

# initialize the input shape and number of classes
INPUT_SHAPE = (113, 137, 113)
NUM_CLASSES = 2
ratio = [80,20]
disease_labels = {"LTLE":0,
                  "RTLE":1}
iters = 2

# early stopping patience
EPOCHS = 100
BS = 1
KF_NUM = 2

# Find sbj files
sbj_names = []
for root, folders, file in os.walk(RECORD_DIR):
    for name in file:
        sbj_names.append(name.split('_ANGLES_')[0])
sbj_names = set(sbj_names)

train_files, validation_files, test_files = fileParse_KF(sbj_names,RECORD_DIR)


tuner = kt.RandomSearch(
    objective=kt.Objective("my_metric", "min"),
    max_trials=100,
    hypermodel=MyHyperModel(),
    directory="results",
    project_name="custom_training",
    overwrite=True,
)

tuner.search(train_files, validation_files,test_files,EPOCHS)