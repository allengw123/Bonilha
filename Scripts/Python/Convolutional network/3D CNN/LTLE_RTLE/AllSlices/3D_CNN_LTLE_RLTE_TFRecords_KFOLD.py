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
from sklearn import model_selection
import xlwt
import pandas as pd

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
    
    testindices_CP = set()
    
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
        
        testindices_CP.update(test_indices)
    if len(testindices_CP) == len(dSbj_list):
        print('FINAL checkpoint PASS')
    else:
        print('FINAL checkpoint FAILURE')
    return train_sbj_files, validation_sbj_files, test_sbj_files


def get_model(dimensions, arch, K_Fold_num):
    
    width, height, depth = dimensions
    if arch == 'Anees':
        inputs = keras.Input((width, height, depth, 1))
        
        x = layers.Conv3D(filters=64, kernel_size=5,strides=2,padding='valid', kernel_regularizer=tf.keras.regularizers.l2(L2_1))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=3,strides=3)(x)
        
        x = layers.Conv3D(filters=128, kernel_size=3,strides=1,padding='valid', kernel_regularizer=tf.keras.regularizers.l2(L2_2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=3,strides=3)(x)
        
        x = layers.Conv3D(filters=192, kernel_size=3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv3D(filters=192, kernel_size=3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv3D(filters=128, kernel_size=3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.MaxPool3D(pool_size=2,strides=3)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(DO_1)(x)
        x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(L2_6))(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(DO_2)(x)
        
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
        
    elif arch == 'Zunair':
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
        x = layers.Dropout(0.1)(x)
        
        outputs = layers.Dense(units=3, activation="sigmoid")(x)
        
        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn_Zunair")
        
        loss = keras.losses.SparseCategoricalCrossentropy()
        optim = keras.optimizers.SGD(learning_rate=0.1, momentum=0.09)
        metrics = ['accuracy']
        
        model.compile(
            optimizer=optim, 
            loss=loss, 
            metrics=metrics
        )
        
    if arch != 'Eleni' and arch != 'Anees' and arch !='Zunair':
        raise ValueError('2nd Argument must be either "Eleni", "Zunair", or "Anees"')
    
    # Arch Save Folder
    if not (os.path.exists(os.path.join(MODEL_DIR, arch + '_arch'))):
        os.mkdir((os.path.join(MODEL_DIR, arch + '_arch')))
        
    # Model Save Folder
    if K_Fold_num == 0:
        ct = datetime.datetime.now()
        model_save_folder = os.path.join(MODEL_DIR,arch + '_arch','Model_' + str(ct)).replace(':', '_').replace('_',':',1)
        if not (os.path.exists(model_save_folder)):
            os.mkdir(model_save_folder)
        return model, model_save_folder
    else:
        return model

def callback_create(model_save_folder):
    
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
        for l in model_Arch['layers']:
            w.write(str(l['name'])+'\n')
            for key,value in l['config'].items():
                w.write('---'+key+' : '+str(value)+'\n')
    
    model_Op = model.optimizer.get_config()
    with open(os.path.join(model_save_folder,'Optimizer.txt'), "w") as w:
        w.writelines(str(model_Op))
        
        
    # Early Stop
    ES_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=50,
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
    
    return  Callbacks, valLoss_save_folder, valAcc_save_folder

def decode(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={'image': tf.io.FixedLenFeature([113, 137, 113, 1], tf.float32),
                  'label': tf.io.FixedLenFeature([], tf.int64),
                  'fileName': tf.io.FixedLenFeature([], tf.string, default_value='')}
    )
    return features['image'], features['label']

def loadTFrecord(input_files, batchNum, set_type, augmentExpand=False, expandNum=0):
    wk_angles = []
    if set_type == 'training':
        print('Augment Expand...',str(augmentExpand))
        if augmentExpand:
            
            # Angles
            angles =[-2, 0, 2]
            perm = list(itertools.product(angles,repeat=3))
            random.shuffle(perm)
            
            out_files = [x for x in input_files if 'ANGLES_0_0_0' in x]
            for i in range(expandNum):
                selected_angle = perm.pop()
                print('Selecting rotation...',str(selected_angle))
                out_files = out_files + [x for x in input_files if 'ANGLES_'+str(selected_angle[0])+'_'+str(selected_angle[1])+'_'+str(selected_angle[2]) in x]
            wk_angles.append(selected_angle)
        else:
            out_files = [x for x in input_files if 'ANGLES_0_0_0' in x]
            
        
    else:
        out_files = [x for x in input_files if 'ANGLES_0_0_0' in x]
            
    print('Imported ',set_type,' file number....',str(len(out_files)))
    dataset = tf.data.TFRecordDataset(out_files)
    dataset = dataset.map(decode)
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.batch(batchNum)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset, wk_angles

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
RECORD_DIR = r'F:\PatientData\TRRecords\LTLE_RTLE_TFRecord_4'
MODEL_DIR = r'F:\CNN output\3D_CNN\RTLE_LTLE\AllSlices'

# Define parameters
matter = 'GM'
ratio = [70, 20, 10]
iterations = 1
disease_labels = {
    "LTLE":0,
    "RTLE":1
    }
EPOCH = 500
RUNNING_ARCH = 'Zunair'
Augment_arg = False
Aug_Expand_Num = 1
ROP_PATIENCE = 5

############## Testing Parameters
batchSize = 1
KF_NUM = 10

# Anees
L2_1 = 0.001
L2_2 = 0.0001
L2_3 = 0.1
L2_4 = 0.1
L2_5 = 0.1
L2_6 = 0.1
DO_1 = 0.5
DO_2 = 0.0

# 

   

# Find sbj files
sbj_names = []
for root, folders, file in os.walk(RECORD_DIR):
    for name in file:
        sbj_names.append(name.split('_ANGLES_')[0])
sbj_names = set(sbj_names)

# Obtain training/val/testing files
trainingFiles, validationFiles ,testingFiles = fileParse_KF(sbj_names,RECORD_DIR)



for kf in range(KF_NUM):
    print('Running K-Fold number ...'+str(kf))
        
    # Prepare Model
    if kf ==0:
        model, savepath = get_model((113,137,113),RUNNING_ARCH,kf)
        
        # Create K-Fold Folder
        kf_folder = os.path.join(savepath,'KFold_Models')
        if not os.path.exists(kf_folder):
            os.mkdir(kf_folder)
            print('K-Model folder created...')
    else:
        model = get_model((113,137,113),RUNNING_ARCH,kf)    
    
    
    
    
    # Create working K-Fold folder
    wk_kf_folder = os.path.join(kf_folder,'KF_'+str(kf))
    if not os.path.exists(wk_kf_folder):
        os.mkdir(wk_kf_folder)
        
    # Save file names
    saveFileNames(trainingFiles[kf], validationFiles[kf] ,testingFiles[kf], wk_kf_folder)


    # Load TFRecordDataset 
    trainingDataset, angels = loadTFrecord(trainingFiles[kf],batchSize,'training',Augment_arg,Aug_Expand_Num)
    validationDataset, angels = loadTFrecord(validationFiles[kf],batchSize,'validation')
    testingDataset, angels = loadTFrecord(testingFiles[kf],batchSize,'testing')

    checkpoint, valLoss_path, valAcc_path = callback_create(wk_kf_folder)
    
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
    save_model(wk_kf_folder, model, history)

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
    confusionMat(final_test_weights,final_test_predictions,test_labels,wk_kf_folder,'final')
    confusionMat(valLoss_test_weights,valLoss_test_predictions,test_labels,wk_kf_folder,'valLoss')
    confusionMat(valAcc_test_weights,valAcc_test_predictions,test_labels,wk_kf_folder,'valAcc')
    
    
    # Save Model Parematers
    saveParameters(matter,ratio,disease_labels,EPOCH,batchSize,wk_kf_folder,(toc-tic))

def rolling_avg(arr,window_size):
      
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
      
    # Loop through the array t o
    #consider every window of size 3
    while i < len(arr) - window_size + 1:
      
        # Calculate the average of current window
        window_average = round(np.sum(arr[
          i:i+window_size]) / window_size, 2)
          
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
          
        # Shift window to right by one position
        i += 1
    
    mask = np.empty((500,1))*np.nan
    
    mask[0:len(moving_averages)] = np.expand_dims(np.array(moving_averages),axis=1)
    
    return mask
#%% Examine K-Fold Results

# savepath = r'F:\CNN output\3D_CNN\RTLE_LTLE\AllSlices\Anees_arch\Model_2022-04-01 13_05_20.210907'
# kf_folder = r'F:\CNN output\3D_CNN\RTLE_LTLE\AllSlices\Anees_arch\Model_2022-04-01 13_05_20.210907\KFold_Models'

wb = xlwt.Workbook()

loss = []
acc = []
val_loss = []
val_acc = []

kf_count = 0
for kf in os.listdir(kf_folder):
    kf_count += 1

    wk_kf_folder = os.path.join(kf_folder,kf)
    
    for d, label in disease_labels.items():
        if kf_count == 1:
            exec(d+'_sheet = wb.add_sheet("'+str(d)+'")')
        
        with open(os.path.join(wk_kf_folder,'valLoss-'+ d +' performance.txt'),'r') as perf:
            lines = perf.readlines()
            
            row_c = 0
            for l in lines:
                parts = l.split(' = ')
                header = parts[0]
                value = parts[1].split('\n')[0]
                if kf_count == 1:
                    exec(d+'_sheet.write(row_c,0,header)')
                exec(d+'_sheet.write(row_c,kf_count,float(value))')
                row_c +=1
    
    history = pd.read_csv(os.path.join(wk_kf_folder,'history.csv'))
    
    
    temp_loss = (rolling_avg(history['loss'].to_numpy(),3))
    temp_acc = (rolling_avg(history['accuracy'].to_numpy(),3))
    temp_val_acc = (rolling_avg(history['val_accuracy'].to_numpy(),3))
    temp_val_loss = (rolling_avg(history['val_loss'].to_numpy(),3))

    if kf_count == 1:
        loss = temp_loss
        acc = temp_acc
        val_acc = temp_val_acc
        val_loss = temp_val_loss
    else:
        loss = np.hstack((loss,temp_loss))
        acc = np.hstack((acc,temp_acc))
        val_acc = np.hstack((val_acc,temp_val_acc))
        val_loss = np.hstack((val_loss,temp_val_loss))
        
wb.save(os.path.join(savepath,'K-Fold_Performance.xls'))


plt.figure()
std = np.nanstd(loss,axis=1)
mean = np.nanmean(loss,axis=1)
plt.fill_between(range(0,500), mean-std, mean+std,color = 'g', alpha=0.1)
plt.plot(mean, color = 'g')
std = np.nanstd(val_loss,axis=1)
mean = np.nanmean(val_loss,axis=1)
plt.fill_between(range(0,500), mean-std, mean+std,color = 'r', alpha=0.1)
plt.plot(mean,color = 'r')
plt.title('Summary Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim((0,2))
plt.legend(['Train','Validation'])
plt.savefig(os.path.join(savepath,'loss.jpg'))


plt.figure()
std = np.nanstd(acc,axis=1)
mean = np.nanmean(acc,axis=1)
plt.fill_between(range(0,500), mean-std, mean+std,color = 'g', alpha=0.1)
plt.plot(mean,color = 'g')
std = np.nanstd(val_acc,axis=1)
mean = np.nanmean(val_acc,axis=1)
plt.fill_between(range(0,500), mean-std, mean+std,color = 'r', alpha=0.1)
plt.plot(mean,color = 'r')
plt.title('Summary Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Validation'])
plt.savefig(os.path.join(savepath,'accuracy.jpg'))