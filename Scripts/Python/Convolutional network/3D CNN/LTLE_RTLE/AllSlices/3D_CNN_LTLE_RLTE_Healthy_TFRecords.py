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
from keras.callbacks import TensorBoard
from keras.models import load_model
import datetime
import itertools
import time
import pandas as pd
from keras import backend as K

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
        x = layers.Dropout(DO)(x)
        outputs = layers.Dense(units=3)(x)
        
        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn_Eleni")
        
        # Compile model.
        optimizer=keras.optimizers.SGD(learning_rate=LR, momentum=0.09)
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        batchSize = 12

    if arch == 'Zunair':
    
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
        model = keras.Model(inputs, outputs, name="3dcnn_Zunair")
        
        loss = keras.losses.SparseCategoricalCrossentropy()
        optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.09)
        metrics = ["accuracy"]
        
        model.compile(
            optimizer=optim, 
            loss=loss, 
            metrics=metrics
        )
        
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
        x = layers.Dropout(DO)(x)
        x = layers.Dense(64)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(DO)(x)
        
        outputs = layers.Dense(3)(x)
        
        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn_Anees")
        
        # Compile model
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08, 
            amsgrad=False)
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        
        batchSize = 20
        
    if arch != 'Eleni' and arch != 'Anees' and arch !='Zunair':
        raise ValueError('2nd Argument must be either "Eleni", "Zunair", or "Anees"')
    
    # Arch Save Folder
    ct = datetime.datetime.now()
    if not (os.path.exists(os.path.join(MODEL_DIR, arch + '_arch'))):
        os.mkdir((os.path.join(MODEL_DIR, arch + '_arch')))
        
    # Model Save Folder
    model_save_folder = os.path.join(MODEL_DIR,arch + '_arch','Model_' + str(ct)).replace(':', '_').replace('_',':',1)
    if not (os.path.exists(model_save_folder)):
        os.mkdir(model_save_folder)
    
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
    
    # Reduce On Plateu CallBack
    ROP_callback =  tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,
        patience=ROP_PATIENCE,
        min_delta=0.0001, 
        min_lr=0.000000001
    )
    
    # Save model architecture
    model_Arch = model.get_config()
    with open(os.path.join(model_save_folder,'Arch.txt'), "w") as w:
        w.writelines(str(model_Arch))
    
    model_Op = model.optimizer.get_config()
    with open(os.path.join(model_save_folder,'Optimizer.txt'), "w") as w:
        w.writelines(str(model_Op))
        
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
        valLoss_checkpoint, 
        valAcc_checkpoint, 
        TB_callback,
        ROP_callback, 
        ES_callback
    ]
    
    return model, model_save_folder, Callbacks, batchSize, valLoss_save_folder, valAcc_save_folder

def decode(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={'image': tf.io.FixedLenFeature([113, 137, 113, 1], tf.float32),
                  'label': tf.io.FixedLenFeature([], tf.int64),
                  'fileName': tf.io.FixedLenFeature([], tf.string, default_value='')}
    )
    return features['image'], features['label']

def loadTFrecord(input_files, batchNum, set_type, augmentExpand=False, expandNum=0):
   
    if set_type == 'training':
        print('Augment Expand...',str(augmentExpand))
        if augmentExpand:
            
            # Angles
            # angles = [-20, -10, 0 ,10, 20]
            angles = [0 ,10]
            perm = list(itertools.product(angles,repeat=3))
            random.shuffle(perm)
            
            out_files = [x for x in input_files if 'ANGLES_0_0_0' in x]
            # for i in range(expandNum):
                # selected_angle = perm.pop()
                # print('Selecting rotation...',str(selected_angle))
                # out_files = out_files + [x for x in input_files if 'ANGLES_'+str(selected_angle[0])+'_'+str(selected_angle[1])+'_'+str(selected_angle[2]) in x]
            out_files = out_files + [x for x in input_files if 'ANGLES_10_0_0' in x]
            out_files = out_files + [x for x in input_files if 'ANGLES_0_10_0' in x]
            out_files = out_files + [x for x in input_files if 'ANGLES_0_0_10' in x]

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
            
def saveParameters(matter,ratio,disease_labels,EPOCH,batchSize,savepath,elapsed_time):
    with open(os.path.join(savepath,'parameters.txt'),'w') as f:
        parameters = [
            'Matter = ' + matter + '\n',
            'Ratio = ' + str(ratio) + '\n', 
            'Disease_labels = ' + str(disease_labels) + '\n',
            'Epoch = ' + str(EPOCH) + '\n',
            'Batch Size = ' + str(batchSize) + '\n',
            'Script Name = ' + '3D_CNN_LTLE_RTLE_Healthy_TFRecord' + '\n',
            'Augment Argument = ' + str(Augment_arg) + '\n',
            'Augment Number = ' + str(Aug_Expand_Num) + '\n',
            'Elapsed Time (h) = ' + str(elapsed_time/60/60) + '\n',
            'ReduceOnPlateu Patience = ' + str(ROP_PATIENCE) + '\n'
        ]
        f.writelines(parameters)
        
def saveFileNames(trainingFiles, validationFiles ,testingFiles,savepath):
    with open(os.path.join(savepath,'trainingSubjectNames.txt'),'w') as f:
        trainingSet = set([x.split('_ANGLES_')[0] for x in trainingFiles])
        trainingList = [x+'\n' for x in list(trainingSet)]
        f.writelines(trainingList)
        num_training_sbjs = len(trainingSet)
        
    with open(os.path.join(savepath,'validationSubjectNames.txt'),'w') as f:
        validationSet = set([x.split('_ANGLES_')[0] for x in validationFiles])
        validationList = [x+'\n' for x in list(validationSet)]
        f.writelines(validationList)
        num_val_sbjs = len(validationSet)
        
    with open(os.path.join(savepath,'testingSubjectNames.txt'),'w') as f:
        testingSet = set([x.split('_ANGLES_')[0] for x in testingFiles])
        testingList = [x+'\n' for x in list(testingSet)]
        f.writelines(testingList)
        num_testing_sbjs = len(testingSet)
    
    if not len(set.union(trainingSet, validationSet, testingSet)) == num_training_sbjs + num_val_sbjs + num_testing_sbjs:
        raise ValueError('OVERLAP BETWEEN SETS')
    else:
        print('Overlap Check ... PASS')

        
        
#%%
# eliminate error notifications
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Data path
# RECORD_DIR=r'F:\test\TFRecords'
RECORD_DIR = r'F:\PatientData\TRRecords\LTLE_RTLE_Healthy_TFRecord'
MODEL_DIR = r'F:\CNN output\3D_CNN\RTLE_LTLE_Healthy\AllSlices\models\Iter'

# Define parameters
matter = 'GM'
ratio = [70, 20, 10]
iterations = 1
disease_labels = {
    "LTLE":0,
    "RTLE":1,
    "Healthy":2}
EPOCH = 200
RUNNING_ARCH = ['Eleni']
Augment_arg = False
Aug_Expand_Num = 1
LR = 0.01
DO = 0.7
L2 = 0.001
ROP_PATIENCE = 5
iterations = 10

for i in range(iterations):
    
    print('Running Iteration....'+str(i))
    
    for current_arch in RUNNING_ARCH:
        
        # Prepare Model
        model, savepath, checkpoint, batchSize, valLoss_path, valAcc_path = get_model((113,137,113),current_arch)
        
        # Find sbj files
        sbj_names = []
        for root, folders, file in os.walk(RECORD_DIR):
            for name in file:
                sbj_names.append(name.split('_ANGLES_')[0])
        sbj_names = set(sbj_names)
        
        # Obtain training/val/testing files
        trainingFiles, validationFiles ,testingFiles = fileParse(sbj_names,RECORD_DIR)
        
        # Save file names
        saveFileNames(trainingFiles, validationFiles ,testingFiles,savepath)

        # Load TFRecordDataset 
        trainingDataset = loadTFrecord(trainingFiles,batchSize,'training',Augment_arg,Aug_Expand_Num)
        validationDataset = loadTFrecord(validationFiles,batchSize,'validation')
        testingDataset = loadTFrecord(testingFiles,batchSize,'testing')
    
        # =============================================================================
        #     ############ Run CNN model
        # =============================================================================
        # Train Model
        tic = time.time()
        history = model.fit(trainingDataset,
                  epochs=EPOCH,
                  verbose=2,
                  validation_data=validationDataset,
                  shuffle=True,
                  callbacks=checkpoint)
        toc = time.time()

        # Save Model and training progress
        save_model(savepath, model, history)
        
        #Load and evaluate the best model version
        best_valLoss_model = os.listdir(os.path.join(valLoss_path))[-1]
        best_valLoss_model = load_model(os.path.join(valLoss_path,best_valLoss_model))
        
        #Load and evaluate the best model version
        best_valAcc_model = os.listdir(os.path.join(valAcc_path))[-1]
        best_valAcc_model = load_model(os.path.join(valAcc_path,best_valAcc_model))
        
        # Test Model
        image_batch = iter(testingDataset)
        
        final_test_weights = []
        valLoss_test_weights = []
        valAcc_test_weights = []
        
        final_test_predictions = []
        valLoss_test_predictions = []
        valAcc_test_predictions = []
        
        test_labels = []
        
        for x in range(500):
            try:
                x,y = image_batch.take(1)
                [print(x) for x in image_batch.as_numpy_iterator()]
                y = np.concatenate([y for x, y in image_batch], axis=0)
                batch_image, batch_label = image_batch.next()
                print('Testing Batch Number...',str(x))
                for input in range(batch_image.shape[0]):
                    img = batch_image[input].numpy()
                    label = batch_label[input].numpy()
                    
                    test_labels.append(label)
                    
                    ft_weights = model.predict(tf.expand_dims(img,axis=0))
                    vl_weights = best_valLoss_model.predict(tf.expand_dims(img,axis=0))
                    va_weights = best_valAcc_model.predict(tf.expand_dims(img,axis=0))

                    final_test_weights.append(ft_weights)
                    valLoss_test_weights.append(vl_weights)
                    valAcc_test_weights.append(va_weights)
                    
                    final_test_predictions.append(np.argmax(ft_weights,axis=1)[0])
                    valLoss_test_predictions.append(np.argmax(vl_weights,axis=1)[0])
                    valAcc_test_predictions.append(np.argmax(va_weights,axis=1)[0])
            except:
                print('End of Batch')
                break
        
        # Save Model Performance`
        confusionMat(final_test_weights,final_test_predictions,test_labels,savepath,'final')
        confusionMat(valLoss_test_weights,valLoss_test_predictions,test_labels,savepath,'valLoss')
        confusionMat(valAcc_test_weights,valAcc_test_predictions,test_labels,savepath,'valAcc')

        
        # Save Model Parematers
        saveParameters(matter,ratio,disease_labels,EPOCH,batchSize,savepath,(toc-tic))
    
        # # Train Shuffle Model
        # model.fit(Shuff_train_dataset, epochs=epoch, verbose=2,validation_data=Shuff_validation_dataset,shuffle=True)
        
        
        # # Test Shuffle Model
        # with tf.device('/CPU:0'):
        #     prediction_weights = model.predict(test_images)
        # prediction_labels= np.argmax(prediction_weights,axis=1)
         
        # ShuffconMat.append(confusionMat(prediction_labels,test_labels))

    