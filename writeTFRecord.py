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
height = 218
width = 178
channels = 3

# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

#img = np.float32(np.random.randn(height,width,channels)*128)

import scipy.io
FullData=scipy.io.loadmat('/home/a/TF/Imgs.mat')
FD=FullData['TFDataP']
nSamples=FD.shape[0]

#pdb.set_trace()

for x in range(0, nSamples):
    img=np.float32(FD[x])
    img_raw = img.tostring()

    print(str(x))
    print(img[0,0,0])
    print(img_raw[5000:5005])
    print('-------------')

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'channels': _int64_feature(channels),
        'image_raw': _bytes_feature(img_raw)}))

    writer.write(example.SerializeToString())

    writer = tf.python_io.TFRecordWriter('/home/a/TF/srez/dataset1/a' + str(x) + '.tfrecords')

writer.close()