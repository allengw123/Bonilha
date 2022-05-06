#   This assumes your niftis are soreted into subdirs by directory, and a regex
#   can be written to match a volume-filenames and label-filenames
#
# USAGE
#  python ./genTFrecord.py <data-dir> <input-vol-regex> <label-vol-regex>
# EXAMPLE:
#  python ./genTFrecord.py ./buckner40 'norm' 'aseg' buckner40.tfrecords
#
# Based off of this: 
#   http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

# imports
import numpy as np
import tensorflow as tf
import nibabel as nib
import os

def _int64_feature(value):# LOGIC AND INTEGER
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value): # FLOATING NUMBER
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value): # STRING OR BYTE
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def listfiles(folder,matter):
    file_list = []
    disease_list = []
    label_list = []
    
    for root, folders, files in os.walk(folder):
        for f in files:
            # if matter in f and 'SmoothThreshold' in f:
            if matter in f:
                file_list.append(os.path.join(root,f))
                if 'LTLE' in root:
                    label_list.append(0)
                    disease_list.append('LTLE')
                elif 'RTLE' in root:
                    label_list.append(1)
                    disease_list.append('RTLE')
                elif 'Alz' in root:
                    label_list.append(2)
                    disease_list.append('AD')
                elif 'Control' in root:
                    label_list.append(3)
                    disease_list.append('Control')
                    
                    
    print(len(file_list),' Files Detected')
                    
    return zip(file_list,disease_list,label_list)

def TFcreate(file):
    
    v_filename_GM, disease, label = file[0]
    v_filename_WM, disease, label = file[1]

    
    sbj_name = 'GMWM2_'+os.path.basename(v_filename_GM).split("GM_")[1]
    
    print("Processing Subject:   ", sbj_name)
    print("   Filename:   ", v_filename_GM)
    print("   Label     ", label)
    print("   Disease   ", disease)
    	
    # The volume, in nifti format	
    v_nii_GM = nib.load(v_filename_GM)
    # The volume, in numpy format
    v_np_GM = v_nii_GM.get_fdata().astype(np.float32)
    # Normalize
    v_np_GM = (v_np_GM - np.min(v_np_GM)) / (np.max(v_np_GM) - np.min(v_np_GM))
    v_np_GM = v_np_GM[:,:,27:84]
     
    # The volume, in nifti format	
    v_nii_WM = nib.load(v_filename_GM)
    # The volume, in numpy format
    v_np_WM = v_nii_WM.get_fdata().astype(np.float32)
    # Normalize
    v_np_WM = (v_np_WM - np.min(v_np_WM)) / (np.max(v_np_WM) - np.min(v_np_WM))
    v_np_WM = v_np_WM[:,:,27:84]
    
    v_np = np.stack((v_np_GM,v_np_WM),3)

    
    # The label, in raw string format
    l_raw = label

    # Write TFRecords
    filename=(disease+'_'+sbj_name+'58_slices')
    
    for slice in range(v_np.shape[2]):
            
        # Features
        data_point = tf.train.Example(features=tf.train.Features(feature={
            'image': _float_feature(v_np[:,:,slice].squeeze().ravel()),
        		'label': _int64_feature(l_raw),
            'fileName': _bytes_feature(eval('b"'+filename+'"'))}))
    
        # Check if dimensions are correct
        writer = tf.io.TFRecordWriter(os.path.join(out_dir,filename+'_S'+str(slice)+'.record'))
       
        writer.write(data_point.SerializeToString())
        writer.close()
    
#%%
# parse args
preproc_dir = r'F:\PatientData\smallSet\thres\Control'

out_dir  = r'F:\PatientData\LargeSet_4_7\TFRecords\58Slices\TFRecords_LTLE_RTLE_T1_58Slices_GMWM2'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
    
preproc_files_GM = list(listfiles(preproc_dir,'GM'))
preproc_files_WM = list(listfiles(preproc_dir,'WM'))



[TFcreate(i) for i in zip(preproc_files_GM, preproc_files_WM)]
