# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:12:39 2022

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
import matplotlib
import matplotlib.pyplot as plt
import xlwt

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
        features={'image': tf.io.FixedLenFeature([113, 137, 113, 2], tf.float32),
                  'label': tf.io.FixedLenFeature([], tf.int64),
                  'fileName': tf.io.FixedLenFeature([], tf.string, default_value='')})

    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    return features['image'], features['label']

def loadTFrecord(input_files, batchNum, set_type, augmentExpand=False, expandNum=0):   
    if set_type == 'training':
        print('Augment Expand...',str(augmentExpand))
        # if augmentExpand:
            
        #     # Angles
        #     angles = [-20, -10, 0 ,10, 20]
        #     perm = list(itertools.product(angles,repeat=3))
        #     random.shuffle(perm)
            
        #     out_files = [x for x in input_files if 'ANGLES_0_0_0' in x]
        #     for i in range(expandNum):
        #         selected_angle = perm.pop()
        #         print('Selecting rotation...',str(selected_angle))
        #         out_files = out_files + [x for x in input_files if 'ANGLES_'+str(selected_angle[0])+'_'+str(selected_angle[1])+'_'+str(selected_angle[2]) in x]
        # else:
        #     out_files = [x for x in input_files if 'ANGLES_0_0_0' in x]
        out_files = [x for x in input_files]

        
    else:
        # out_files = [x for x in input_files if 'ANGLES_0_0_0' in x]
        out_files = [x for x in input_files]

            
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
        inputs = keras.Input((113, 137, 113, 2))

        x = layers.Conv3D(filters=hp.Int("conv_1", min_value=8, max_value=16, step=8), kernel_size=3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=(2,2,2),strides=2)(x)
    
        x = layers.Conv3D(filters=hp.Int("conv_2", min_value=16, max_value=32, step=16), kernel_size=3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=(2,2,2),strides=2)(x)
        
        x = layers.Conv3D(filters=hp.Int("conv_2", min_value=16, max_value=32, step=16), kernel_size=3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool3D(pool_size=(2,2,2),strides=2)(x)
    
        x = layers.Flatten()(x)
        x = layers.Dense(hp.Choice('dense_num_3',values=[16,32,64,128,256]))(x)
        # x = layers.Dropout(hp.Choice('dropout_3',values=[0.1,0.5,0.9]))(x)
        
        outputs = layers.Dense(units=len(disease_labels))(x)
        
        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn_Eleni")
        
        # Compile model.
        optimizer=keras.optimizers.SGD(learning_rate=hp.Choice('LR',values=[0.1, 0.01, 0.001, 0.0001]),
                                       momentum=0.09)
        
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=["accuracy"])
    elif arch == 'Anees':
        inputs = keras.Input((113, 137, 113, 2))
        
        # L2_1 = hp.Choice('L2_1',values=[0.1,0.01,0.001,0.0001])
        # L2_2 = hp.Choice('L2_2',values=[0.1,0.01,0.001,0.0001])
        # L2_3 = hp.Choice('L2_3',values=[0.1,0.01,0.001,0.0001])
        # L2_4 = hp.Choice('L2_4',values=[0.1,0.01,0.001,0.0001])
        # L2_5 = hp.Choice('L2_5',values=[0.1,0.01,0.001,0.0001])
        # L2_6 = hp.Choice('L2_6',values=[0.1,0.01,0.001,0.0001])


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
        # x = layers.Dropout(hp.Choice('DO_1',values=[0.0, 0.1, 0.5, 0.9]))(x)
        x = layers.Dense(hp.Choice('Dense',values=[32,64,128,256]))(x)
        x = layers.ReLU()(x)
        # x = layers.Dropout(hp.Choice('DO_2',values=[0.0, 0.1, 0.5, 0.9]))(x)
        
        outputs = layers.Dense(len(disease_labels))(x)
        
        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn_Anees")
        
        # Compile model
        optimizer=keras.optimizers.Adam(
            learning_rate=(hp.Choice('LR',values=[0.1,0.01,0.001])),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08, 
            amsgrad=False)
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        
    elif arch == 'Zunair':
        inputs = keras.Input((113, 137, 113, 1))
        
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
        x = layers.Dropout(hp.Choice('DO_1',values=[0.0, 0.1, 0.5, 0.9]))(x)
        
        outputs = layers.Dense(units=len(disease_labels), activation="sigmoid")(x)
        
        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn_Zunair")
        
        loss = keras.losses.SparseCategoricalCrossentropy()
        optim = keras.optimizers.SGD(learning_rate=hp.Choice('Learning_rate',values=[0.1,0.01,0.001,0.0001]), momentum=0.09)
        metrics = ['accuracy','AUC']
        
        model.compile(
            optimizer=optim, 
            loss=loss, 
            metrics=metrics
        )
        
    return model

def get_tuner(args, OUTPUT_PATH):
        
    # check if we will be using the hyperband tuner
    if args == "hyperband":
    	# instantiate the hyperband tuner object
    	print("[INFO] instantiating a hyperband tuner object...")
    	tuner = kt.Hyperband(
    		build_model,
    		objective=kt.Objective('val_loss','min'),
    		max_epochs=500,
    		factor=3,
    		directory=OUTPUT_PATH,
    		project_name=args,
            hyperband_iterations=1)
        # check if we will be using the random search tuner
    elif args == "random":
    	# instantiate the random search tuner object
    	print("[INFO] instantiating a random search tuner object...")
    	tuner = kt.RandomSearch(
    		build_model,
    		objective=kt.Objective('val_loss','min'),
    		max_trials=10,
    		directory=OUTPUT_PATH,
    		project_name=args)
        # otherwise, we will be using the bayesian optimization tuner
    elif args == "bayesian":
    	# instantiate the bayesian optimization tuner object
    	print("[INFO] instantiating a bayesian optimization tuner object...")
    	tuner = kt.BayesianOptimization(
    		build_model,
    		objective=kt.Objective('val_loss','min'),
    		max_trials=10,
    		directory=OUTPUT_PATH,
    		project_name=args)
    
    # initialize an early stopping callback
    es_callback = EarlyStopping(
                    	monitor='val_loss',
                        min_delta=0.0001,
                        patience=50,
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
RECORD_DIR = r'F:\PatientData\LargeSet_4_7\TFRecords_LTLE_RTLE_T1_zScore'
OUT_PATH = r'F:\HP_tests\test14'
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)

# Define parameters
Tuners = [
    'bayesian',
    'random',
    'hyperband'
    ]
arch = 'Eleni'

# initialize the input shape and number of classes
INPUT_SHAPE = (113, 137, 113)
ratio = [70, 20, 10]
disease_labels = {"LTLE":0,
                  "RTLE":1}
iters = 1

# early stopping patience
EPOCHS = 500
BS = 1


# Find sbj files
sbj_names = []
for root, folders, file in os.walk(RECORD_DIR):
    for name in file:
        sbj_names.append(name.split('_ANGLES_')[0])
sbj_names = set(sbj_names)


for wk_tuner in Tuners:
    for i in range(iters):
        # perform the hyperparameter search
        print("Performing hyperparameter search using...",wk_tuner)
        
        wk_out_path = os.path.join(OUT_PATH,wk_tuner)
        if not os.path.exists(wk_out_path):
            os.mkdir(wk_out_path)
            
        # Obtain training/val/testing files
        trainingFiles, validationFiles ,testingFiles = fileParse(sbj_names,RECORD_DIR)
            
        # Load TFRecordDataset 
        trainingDataset = loadTFrecord(trainingFiles,BS,'training')
        validationDataset = loadTFrecord(validationFiles,BS,'validation')
        testingDataset = loadTFrecord(testingFiles,BS,'testing')
    
        tuner, callbacks = get_tuner(wk_tuner, os.path.join(wk_out_path,'model'+str(i)))
    
     
        tuner.search(
        	trainingDataset,
        	validation_data=validationDataset,
        	batch_size=BS,
        	callbacks=callbacks,
        	epochs=EPOCHS,
            use_multiprocessing=True
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