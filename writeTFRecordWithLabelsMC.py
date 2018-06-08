import numpy as np
import tensorflow as tf
import pdb

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = '/home/a/TF/srez/dataset1/a1.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare
# to the reconstructed ones
original_images = []

#img = np.array(Image.open(img_path))

# The reason to store image sizes was demonstrated
# in the previous example -- we have to know sizes
# of images to later read raw serialized string,
# convert to 1d array and convert to respective
# shape that image used to have.

# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

#img = np.float32(np.random.randn(height,width,channels)*128)

import scipy.io
FullData=scipy.io.loadmat('/home/a/TF/ImgsMCD3x.mat')
Data=FullData['Data']
Labels=FullData['Labels']
#pdb.set_trace()
nSamples=Data.shape[0]
DataH = Data.shape[1]
if Data.ndim<3 :
    DataW = 1
else:
    DataW = Data.shape[2]

if Data.ndim<4 :
    channelsIn=1
else :
    channelsIn = Data.shape[3]

LabelsH = Labels.shape[1]
LabelsW = Labels.shape[2]
if Labels.ndim<4 :
    channelsOut=1
else :
    channelsOut = Labels.shape[3]

print(str(channelsIn))
print(str(channelsOut))

for x in range(0, nSamples):
    CurData=np.float32(Data[x])
    CurData_raw = CurData.tostring()

    CurLabels=np.float32(Labels[x])
    CurLabels_raw = CurLabels.tostring()

    print(str(x))
    #print(img[0,0,0])
    #print(img_raw[5000:5005])

    example = tf.train.Example(features=tf.train.Features(feature={
        'DataH': _int64_feature(DataH),
        'DataW': _int64_feature(DataW),
        'channelsIn': _int64_feature(channelsIn),
        'LabelsH': _int64_feature(LabelsH),
        'LabelsW': _int64_feature(LabelsW),
        'channelsOut': _int64_feature(channelsOut),
        'data_raw': _bytes_feature(CurData_raw),
        'labels_raw': _bytes_feature(CurLabels_raw)}))

    writer = tf.python_io.TFRecordWriter('/home/a/TF/srez/dataset1/a' + str(x) + '.tfrecords')

    writer.write(example.SerializeToString())

    
writer.close()