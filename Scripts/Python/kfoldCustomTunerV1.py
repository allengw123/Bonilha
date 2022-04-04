# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:25:26 2022

@author: bonilha
"""

# import the necessary packages
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

class CVTuner(kt.Tuner):
    def run_trial(self, trial, train_files, validation_files, test_files, batch_size, epochs, callbacks):
    
        test_losses = []
        
        for trainingFiles, validationFiles, testingFiles in zip(train_files,validation_files, test_files):
              
            # Load TFRecordDataset 
            trainingDataset = loadTFrecord(trainingFiles,batch_size,'training')
            validationDataset = loadTFrecord(validationFiles,batch_size,'validation')
            testingDataset = loadTFrecord(testingFiles,batch_size,'testing')
            
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(trainingDataset, 
                      batch_size=batch_size, 
                      epochs=epochs,
                      validation_data=validationDataset,
                      shuffle=True,
                      workers=4,
                      use_multiprocessing=True,
                      callbacks=callbacks)
            test_loss,_ = model.evaluate(testingDataset)
            test_losses.append(test_loss)
            
        self.oracle.update_trial(trial.trial_id, {'test_loss': np.mean(test_losses)})
        self.save_model(trial.trial_id, model)
    
    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), "model.pickle")
        with tf.io.gfile.GFile(fname, "wb") as f:
            pickle.dump(model, f)


        
def get_tuner(args, OUTPUT_PATH):

    # check if we will be using the hyperband tuner
    if args == "hyperband":
    	# instantiate the hyperband tuner object
    	print("[INFO] instantiating a hyperband tuner object...")
    	tuner = CVTuner(
            hypermodel=build_model,
            oracle=kt.oracles.Hyperband(
                objective=kt.Objective('test_loss','min'),
                max_epochs=10,
        		factor=3,
                hyperband_iterations=1),
            directory=OUTPUT_PATH
            )
        # check if we will be using the random search tuner
    # elif args == "random":
    # 	# instantiate the random search tuner object
    # 	print("[INFO] instantiating a random search tuner object...")
    # 	tuner = kt.RandomSearch(
    # 		build_model,
    # 		objective=kt.Objective('val_loss','min'),
    # 		max_trials=10,
    # 		directory=OUTPUT_PATH,
    # 		project_name=args)
    #     # otherwise, we will be using the bayesian optimization tuner
    # elif args == "bayesian":
    # 	# instantiate the bayesian optimization tuner object
    # 	print("[INFO] instantiating a bayesian optimization tuner object...")
    # 	tuner = kt.BayesianOptimization(
    # 		build_model,
    # 		objective=kt.Objective('val_loss','min'),
    # 		max_trials=10,
    # 		directory=OUTPUT_PATH,
    # 		project_name=args)
    
    # initialize an early stopping callback
    es_callback = EarlyStopping(
                    	monitor='val_loss',
                        min_delta=0.0001,
                        patience=3,
                        verbose=2,
                        mode="auto",
                        baseline=None,
                        restore_best_weights=True,
                    )
    # tensorboard callbacks
    tb_callback = keras.callbacks.TensorBoard(
                        os.path.join(OUTPUT_PATH,'tb_logs')
                        )
    
    # Reduce On Plateu CallBack
    ROP_callback =  tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_delta=0.0001, 
        min_lr=0.000000001
    )
    

    return tuner, [es_callback, tb_callback, ROP_callback]


#%%

#https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f

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
EPOCHS = 500
BS = 1
KF_NUM = 2

# Find sbj files
sbj_names = []
for root, folders, file in os.walk(RECORD_DIR):
    for name in file:
        sbj_names.append(name.split('_ANGLES_')[0])
sbj_names = set(sbj_names)

train_files, validation_files, test_files = fileParse_KF(sbj_names,RECORD_DIR)

for wk_tuner in Tuners:
    for i in range(iters):
        # perform the hyperparameter search
        print("Performing hyperparameter search using...",wk_tuner)
        
        wk_out_path = os.path.join(OUT_PATH,wk_tuner)
        if not os.path.exists(wk_out_path):
            os.mkdir(wk_out_path)
            
        # Obtain training/val/testing files   
        model_dir =os.path.join(wk_out_path,'model'+str(i))
        tuner, callbacks = get_tuner(wk_tuner, model_dir)
    
     
        tuner.search(
        	train_files,
            validation_files,
            test_files,
        	batch_size=BS,
        	callbacks=callbacks,
        	epochs=10,
            )
       
        # Grab best models
        bestHP = tuner.oracle.get_best_trials(10)
        count = 0
        wb = xlwt.Workbook()
        for model in bestHP:
            count+=1
            if count ==1:
                sheet1 = wb.add_sheet('Sheet 1')
                sheet1.write(0, 0, 'Score')
                sheet1.write(count, 0, model.score)
                column = 0
                for key,value in model.hyperparameters.values.items():
                    column+=1
                    sheet1.write(0,column,key)
                    sheet1.write(count,column,value)
            else:
                sheet1.write(count, 0, model.score)
                column = 0
                for key,value in model.hyperparameters.values.items():
                    column+=1
                    sheet1.write(count,column,value)
        wb.save(os.path.join(wk_out_path,wk_tuner+'_results'+str(i)+'.xls'))
                    
            
# # build the best model and train it
# print("[INFO] training the best model...")
# model = tuner.hypermodel.build(bestHP)
# H = model.fi(trainingDataset,
#              validation_data=validationDataset, 
#             batch_size=BS,
#             epochs=20,
#             callbacks=[es, keras.callbacks.TensorBoard(os.path.join(OUT_PATH+'/tmp/tb_logs'))], 
#             verbose=)
# # evaluate the network
# print("[INFO] evaluating network...")
# predictions = model.predict(x=testX, batch_size=32)
# print(classification_report(testY.argmax(axis=1),
#  	predictions.argmax(axis=1), target_names=labelNames))
# # generate the training loss/accuracy plot
# utils.save_plot(H, args["plot"])