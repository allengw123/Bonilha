# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 13:52:26 2022

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
import datetime
import pandas as pd
import time

from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from scipy import ndimage
from keras.models import load_model
from sklearn import model_selection


# Extract nifti files and parse
def extractNifti(input,type,kf_num):
    dSbj_list = []
    if type == 'TLE':
        for root, dirs, files in os.walk(input):
            if len(files) > 0:
                dSbj_list.append(os.path.join(
                    root, [x for x in files if 'GM' in x and 'SmoothThreshold' in x][0]))
    else:
        for root, dirs, files in os.walk(input):
            if len(files) > 0:
                dSbj_list.append(os.path.join(
                    root, [x for x in files if 'GM' in x][0]))
                

        
    # KFold Parse  
    cv = model_selection.KFold(kf_num,shuffle=True)
        
    train_sbj_files = []
    validation_sbj_files = []
    test_sbj_files = []
    
    testindices_CP = set()
    
    count = 0
    for train_indices, test_indices in cv.split(range(len(dSbj_list))):
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
        train_sbj_files.append([os.path.join(file_dir,x) for x in file_list if x.split('_slices_')[0] in train_sbjs])
        validation_sbj_files.append([os.path.join(file_dir,x) for x in file_list if x.split('_slices_')[0] in validation_sbj])
        test_sbj_files.append([os.path.join(file_dir,x) for x in file_list if x.split('_slices_')[0] in test_sbjs])
        
        testindices_CP.update(test_indices)
        
    if len(testindices_CP) == len(dSbj_list):
        print('FINAL checkpoint PASS')
    else:
        print('FINAL checkpoint FAILURE')
        
    return output


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def PreprocImg(input):
    imgs = [nib.load(file).get_fdata().astype(np.float32)[:,:,27:84] for file in input]
    imgs = NormalizeData(np.concatenate(imgs,axis=2))
    # imgs = np.concatenate(imgs,axis=2)

    return imgs


# Define CNN input class
class CNNinput:
    def __init__(self, niftifiles):

        # length of files
        nFiles = len(niftifiles)
        print('importing ... ',str(nFiles),' Files')

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


def get_model(dimensions):
    
    width, height, depth = dimensions
    inputs = keras.Input((width, height, depth))
    
    x = layers.Conv2D(filters=8, kernel_size=3,padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    
    x = layers.Conv2D(filters=16, kernel_size=3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    
    x = layers.Conv2D(filters=32, kernel_size=3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    
    x = layers.Flatten()(x)
    
    outputs = layers.Dense(units=3)(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="2dcnn")
    
    # Compile model.
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        metrics=["accuracy"])
    
    
    # Arch Save Folder
    if not (os.path.exists(os.path.join(MODEL_DIR, arch + '_arch'))):
        os.mkdir((os.path.join(MODEL_DIR, arch + '_arch')))
        
    # Model Save Folder
    ct = datetime.datetime.now()
    model_save_folder = os.path.join(MODEL_DIR,arch + '_arch','Model_' + str(ct)).replace(':', '_').replace('_',':',1)
    if not (os.path.exists(model_save_folder)):
        os.mkdir(model_save_folder)

    return model, model_save_folder
    

def callback_create(model_save_folder):
    
    # Save model architecture
    model_Arch = model.get_config()
    with open(os.path.join(model_save_folder,'Arch.txt'), "w") as w:
        for l in model_Arch['layers']:
            w.write(str(l['name'])+'\n')
            for key,value in l['config'].items():
                w.write('---'+key+' : '+str(value)+'\n')
    
    model_Op = model.optimizer.get_config()
    with open(os.path.join(model_save_folder,'Optimizer.txt'), "w") as w:
        w.writelines(str(model_Op))
        
    # Callback Save Folder
    callback_save_folder = os.path.join(model_save_folder,'callbacks')
    if not os.path.exists(callback_save_folder):
        os.mkdir(callback_save_folder)
        
    # Callback for Validation Loss
    valLoss_save_folder = os.path.join(callback_save_folder,'Val_Loss')
    if not os.path.exists(valLoss_save_folder):
        os.mkdir(valLoss_save_folder)
    valLoss_outputfile = os.path.join(valLoss_save_folder,'my_best_valLoss_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5')
    valLoss_checkpoint = ModelCheckpoint(
        filepath=valLoss_outputfile, 
        monitor='val_loss',
        verbose=1, 
        save_best_only=True,
        mode='min'
    )
    
    # Callback for Validation Accuracy
    valAcc_save_folder = os.path.join(callback_save_folder,'Val_Acc')
    if not os.path.exists(valAcc_save_folder):
        os.mkdir(valAcc_save_folder)
    valAcc_outputfile = os.path.join(valAcc_save_folder,'my_best_valLoss_model.epoch{epoch:02d}-acc{val_accuracy:.2f}.hdf5')
    valAcc_checkpoint = ModelCheckpoint(
        filepath=valAcc_outputfile, 
        monitor='val_accuracy',
        verbose=1, 
        save_best_only=True,
        mode='max'
    )
   
    # Callback for TensorBoard/Learning Rate
    tensorboard_save_folder = os.path.join(callback_save_folder,'TensorBoard')
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
        
    # Early Stop Callback
    ES_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=50,
        verbose=2,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )
    
    # Compile Callbacks
    Callbacks = [
        valLoss_checkpoint, 
        valAcc_checkpoint, 
        TB_callback,
        ES_callback
    ]
    
    return  Callbacks, valLoss_save_folder, valAcc_save_folder

# def rotate(volume):
#     """Rotate the volume by a few degrees"""

#     def scipy_rotate(volume):
#         # define some rotation angles
#         angles = [-20, -10, -5, 5, 10, 20]
        
#         # rotate volume
#         volume = ndimage.rotate(volume,angle=random.choice(angles),reshape=False)
#         return volume

#     augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
#     return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    # volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=2)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=2)

    return volume, label

def confusionMat(weights,predictions,labels,savepath,set_type):
    
    weights = np.array(weights)
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    for key, value in disease_labels.items():
        print('Writing Performance File for ...', key,)
        TP = sum(np.logical_and(predictions==value,labels==value))
        TN = sum(np.logical_and(predictions!=value,labels!=value))
        FN = sum(np.logical_and(predictions!=value,labels==value)) 
        FP = sum(np.logical_and(predictions==value,labels!=value))
        Acc = (TP+TN)/(TP+TN+FP+FN)
        if TP == 0 and FN == 0:
            Sensitivity = 0
            Precision = 0
            F1 = 0
        elif TP == 0 and FP ==0:
            Sensitivity = 0
            Precision = 0
            F1 = 0
        else:
            Sensitivity = TP/(TP+FN)
            Precision = TP/(TP+FP)
            F1 = 2*((Precision*Sensitivity)/(Precision+Sensitivity))
        
        with open(os.path.join(savepath,set_type+'-'+ key+' performance.txt'),'w') as f:
            metrics = [
                'TP = ' + str(TP) + '\n',
                'TN = ' + str(TN) + '\n', 
                'FN = ' + str(FN) + '\n',
                'FP = ' + str(FP) + '\n',
                'Acc = ' + str(Acc) + '\n',
                'Sensitivity = ' + str(Sensitivity) + '\n',
                'Precision = ' + str(Precision) + '\n',
                'F1 Score = ' + str(F1)
            ]
            f.writelines(metrics)
    
    with open(os.path.join(savepath,set_type+'weights.txt'),'w') as f:
        f.writelines(str(weights))
        
def shuffle_array(input):
    np.random.shuffle(input)
    return input


def save_model(savepath, model, history):
    
    model.save(savepath)
    
    # convert the history.history dict to a pandas DataFrame and to csv
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = os.path.join(savepath,'history.csv')
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(savepath,'model_training_history_accuracy.jpg'))
    plt.show()
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.ylim((0,2))
    plt.savefig(os.path.join(savepath,'model_training_history_loss.jpg'))
    plt.show()

                    
def saveParameters(matter,ratio,disease_labels,EPOCH,batchSize,savepath,elapsed_time):
    with open(os.path.join(savepath,'parameters.txt'),'w') as f:
        parameters = [
            'Matter = ' + matter + '\n',
            'Ratio = ' + str(ratio) + '\n', 
            'Disease_labels = ' + str(disease_labels) + '\n',
            'Epoch = ' + str(EPOCH) + '\n',
            'Batch Size = ' + str(batchSize) + '\n',
            'Script Name = ' + '2D_CNN_AD_TLE_Healthy' + '\n',
            'Elapsed Time (h) = ' + str(elapsed_time/60/60) + '\n',
        ]
        f.writelines(parameters)


#%%
# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Data path
datadir = r'F:\PatientData\FullSet'
MODEL_DIR = r'F:\CNN output\2D_CNN\Python'

Healthydir = os.path.join(datadir, 'Control')
ADdir = os.path.join(datadir, 'Alz','ADNI_Alz_nifti')
TLEdir = os.path.join(datadir, 'TLE')


# Define parameters
matter = 'WM'
ratio = [60, 15, 35]
iterations = 1
disease_labels = {"AD":0,"TLE":1,"Healthy":2}
epoch = 100
arch = 'Eleni'
batch_size = 256
KF_NUM = 10



AD_nifti = extractNifti(ADdir,'AD',KF_NUM)
TLE_nifti = extractNifti(TLEdir,'TLE',KF_NUM)
Healthy_nifti = extractNifti(Healthydir,'Healthy',KF_NUM)


for kf in range(KF_NUM):
    print('Running K-Fold number ...'+str(kf))
    
            
    # Prepare disease specific CNN input
    ADinput = CNNinput()  # 0
    TLEinput = CNNinput()  # 1
    Healthyinput = CNNinput()  # 2
    
    # Pepare x y data
    train_images = np.concatenate(
        [ADinput.trainingImg, TLEinput.trainingImg, Healthyinput.trainingImg],axis=2).transpose((2,0,1))
    train_labels = np.concatenate([np.zeros(ADinput.trainingImg.shape[2]),  np.ones(TLEinput.trainingImg.shape[2]), np.ones(
        Healthyinput.trainingImg.shape[2])*2])
    
    validation_images = np.concatenate(
        [ADinput.validationImg, TLEinput.validationImg, Healthyinput.validationImg],axis=2).transpose((2,0,1))
    validation_labels = np.concatenate([np.zeros(ADinput.validationImg.shape[2]), np.ones(TLEinput.validationImg.shape[2]), np.ones(
        Healthyinput.validationImg.shape[2])*2])
    
    test_images = np.concatenate(
        [ADinput.testingImg, TLEinput.testingImg, Healthyinput.testingImg],axis=2).transpose((2,0,1))
    test_labels = np.concatenate([np.zeros(ADinput.testingImg.shape[2]), np.ones(TLEinput.testingImg.shape[2]), np.ones(
        Healthyinput.testingImg.shape[2])*2])
    
    # Define data loaders
    train_loader = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    validation_loader = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    
    Shuff_train_loader = tf.data.Dataset.from_tensor_slices((train_images,shuffle_array(train_labels)))
    Shuff_validation_loader = tf.data.Dataset.from_tensor_slices((validation_images,shuffle_array(validation_labels)))
    
    
    # Preproc dataset
    train_dataset = train_loader.shuffle(len(train_labels),reshuffle_each_iteration=True).map(train_preprocessing).batch(batch_size).prefetch(batch_size)
    validation_dataset = validation_loader.shuffle(len(validation_labels),reshuffle_each_iteration=True).map(validation_preprocessing).batch(batch_size).prefetch(batch_size)
    
    Shuff_train_dataset = Shuff_train_loader.shuffle(len(train_labels),reshuffle_each_iteration=True).map(train_preprocessing).batch(batch_size).prefetch(batch_size)
    Shuff_validation_dataset = Shuff_validation_loader.shuffle(len(validation_labels),reshuffle_each_iteration=True).map(validation_preprocessing).batch(batch_size).prefetch(batch_size)
    
    
    data = train_dataset.take(1)
    images, labels = list(data)[0]
    images = images.numpy()
    image = images[0]
    print("Dimension of the CT scan is:", image.shape)
    plt.imshow(np.squeeze(image), cmap="gray")
    plt.show()
    
    # Prepare Model
    model, model_savefolder = get_model((113,137,1))
    checkpoint, valLoss_path, valAcc_path = callback_create(model_savefolder)
    
    
    ############ Run CNN model
    # Train Model
    tic = time.time()
    history = model.fit(train_dataset, 
                        epochs=epoch, 
                        verbose=2,
                        validation_data=validation_dataset,
                        shuffle=True,
                        callbacks=checkpoint,
                        use_multiprocessing=True)
    toc = time.time()
    
    # Save Model and training progress
    save_model(model_savefolder, model, history)
    
    #Load and evaluate the best model version
    best_valLoss_model = os.listdir(os.path.join(valLoss_path))[-1]
    best_valLoss_model = load_model(os.path.join(valLoss_path,best_valLoss_model))
    
    #Load and evaluate the best model version
    best_valAcc_model = os.listdir(os.path.join(valAcc_path))[-1]
    best_valAcc_model = load_model(os.path.join(valAcc_path,best_valAcc_model))
    
    # Test Model
    final_test_weights = model.predict(test_images, batch_size=100)
    final_test_predictions= np.argmax(final_test_weights,axis=1)
    
    valLoss_test_weights = best_valLoss_model.predict(test_images, batch_size=100)
    valLoss_test_predictions= np.argmax(valLoss_test_weights,axis=1)
    
    valAcc_test_weights = best_valAcc_model.predict(test_images, batch_size=100)
    valAcc_test_predictions= np.argmax(valAcc_test_weights,axis=1)
    
    # Save Model Performance`
    confusionMat(final_test_weights,final_test_predictions,test_labels,model_savefolder,'final')
    confusionMat(valLoss_test_weights,valLoss_test_predictions,test_labels,model_savefolder,'valLoss')
    confusionMat(valAcc_test_weights,valAcc_test_predictions,test_labels,model_savefolder,'valAcc')
    
    # Save Model Parematers
    saveParameters(matter,ratio,disease_labels,epoch,batch_size,model_savefolder,(toc-tic))
    
    # # Train Shuffle Model
    # model.fit(Shuff_train_dataset, epochs=epoch, verbose=2,validation_data=Shuff_validation_dataset,shuffle=True)
     
    # # Test Shuffle Model
    # prediction_weights = model.predict(test_images)
    # prediction_labels= np.argmax(prediction_weights,axis=1)
     
    # ShuffconMat.append(confusionMat(prediction_labels,test_labels))
    

