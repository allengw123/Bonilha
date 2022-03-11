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
import os, sys, re, itertools
from scipy import ndimage

def _int64_feature(value):# LOGIC AND INTEGER
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value): # FLOATING NUMBER
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value): # STRING OR BYTE
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def listfiles(folder):
    file_list = []
    disease_list = []
    label_list = []
    
    for root, folders, files in os.walk(folder):
        
        for f in files:
            if 'GM' in f:
                file_list.append(os.path.join(root,f))
                
                if 'TLE' in root:
                    label_list.append(0)
                    disease_list.append('TLE')
                if 'Alz' in root:
                    label_list.append(1)
                    disease_list.append('AD')                    
                if 'Control' in root:
                    label_list.append(2)
                    disease_list.append('Control')
                    
    return zip(file_list,disease_list,label_list)

def ProgressIndicator(current_count,prev_perc,total_count):
    current_perc = current_count/total_count*100
    if round(current_perc,1) != round(prev_perc,1):
        output_perc = round(current_perc,1) 
    else:
        output_perc = round(prev_perc,1)
        
    return output_perc

#%%
# parse args
data_dir = r'F:\PatientData\thres'
out_dir  = r'F:\test\TFRecords_TEST'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
    
files = list(listfiles(data_dir))
count = 0
REF_DIM = (113, 137, 113)

running_perc = 0
for v_filename, disease, label in files:
    
    sbj_name = os.path.basename(v_filename).split(".")[0]
    
    print("Processing:")
    print("   Volume:   ", v_filename)
    print("   Label     ", label)
	
	# The volume, in nifti format	
    v_nii = nib.load(v_filename)
	# The volume, in numpy format
    v_np = v_nii.get_fdata().astype(np.float32)
    # Normalize
    v_np = (v_np - np.min(v_np)) / (np.max(v_np) - np.min(v_np))

    
	# The label, in raw string format
    l_raw = label
    
    # Angles
    angles =[-20, -10, 0 ,10, 20]
    perm = list(itertools.product(angles,repeat=3))
    
    # Progress indicator
    count=+1
    running_perc = ProgressIndicator(count,running_perc,len(files))
    print("   Total Progress     ", str(running_perc),'%')
    
    # Write TFRecords
    correct_dim_log = 'Pass'
    for p in perm:
        filename=(disease+'_'+sbj_name+'_ANGLES_'+str(p[0])+'_'+str(p[1])+'_'+str(p[2]))
        aug_vol = ndimage.rotate(v_np,angle=p[0],axes=(0,1),reshape=False)
        aug_vol = ndimage.rotate(aug_vol,angle=p[1],axes=(0,2),reshape=False)
        aug_vol = ndimage.rotate(aug_vol,angle=p[2],axes=(1,2),reshape=False)
        
    	# Features
        data_point = tf.train.Example(features=tf.train.Features(feature={
            'image': _float_feature(aug_vol.ravel()),
    		'label': _int64_feature(l_raw),
            'fileName': _bytes_feature(eval('b"'+filename+'"'))}))
        
        # Check if dimensions are correct
        if aug_vol.shape == REF_DIM:
            writer = tf.io.TFRecordWriter(os.path.join(out_dir,filename+'.record'))
        else:
            writer = tf.io.TFRecordWriter(os.path.join(out_dir,'DIM_ERROR_'+filename+'.record'))
            correct_dim_log = 'FAIL'
        
        writer.write(data_point.SerializeToString())
        writer.close()
    print("   Dimension Check     ", correct_dim_log)


##############################################################
#  TEST: Reconstruct images from outfile and compare to originals
#  Make sure the line `original_images.append((v_np, l_np))` is uncommented above
##############################################################
#reconstructed_images = []

#record_iterator = tf.python_io.tf_record_iterator(path=outfile)

#for string_record in record_iterator:
    
#	example = tf.train.Example()
#	example.ParseFromString(string_record)
    
#	x_dim = int(example.features.feature['x_dim'].int64_list.value[0])    
#	y_dim = int(example.features.feature['y_dim'].int64_list.value[0])
#	z_dim = int(example.features.feature['z_dim'].int64_list.value[0])    
#	image_raw = (example.features.feature['image_raw'].bytes_list.value[0])    
#	label_raw = (example.features.feature['label_raw'].bytes_list.value[0])
    
#	img_1d = np.fromstring(image_raw, dtype=np.uint16)
#	reconstructed_img = img_1d.reshape((x_dim, y_dim, z_dim))
    
#	label_1d = np.fromstring(label_raw, dtype=np.uint16)
#	reconstructed_label = label_1d.reshape((x_dim, y_dim, z_dim))
    
#	reconstructed_images.append((reconstructed_img, reconstructed_label))

#for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
    
#    img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
#                                                         reconstructed_pair)
#    print(np.allclose(*img_pair_to_compare))
#    print(np.allclose(*annotation_pair_to_compare))