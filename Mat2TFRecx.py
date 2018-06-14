import numpy as np
import tensorflow as tf
import pdb
import scipy.io
import sys

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#tfrecords_filename = '/home/a/TF/srez/dataset1/a1.tfrecords'

#writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare
# to the reconstructed ones
original_images = []

#print("aaa")
#print(str(sys.argv[0]))
#print("---")
#print(str(len(sys.argv)))
#print("---")
#print(str(sys.argv))
#print("***")
#exit()

MatFileName = sys.argv[1]
FolderName=sys.argv[2]

#img = np.array(Image.open(img_path))

# The reason to store image sizes was demonstrated
# in the previous example -- we have to know sizes
# of images to later read raw serialized string,
# convert to 1d array and convert to respective
# shape that image used to have.

# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

#img = np.float32(np.random.randn(height,width,channels)*128)
FullData=scipy.io.loadmat(MatFileName)

FullData.keys()
KeysL=list(FullData.keys())
KeysS=set(KeysL)-{'FNs','__header__', '__version__', '__globals__'}
KeysL=list(KeysS)

FNs=FullData['FNs']
nSamples=FNs.shape[0]

for x in range(0, nSamples):
    print(FNs[x])
    feature={}
    for K in KeysL:
        A=FullData[K][x]
        dt = A.dtype
        #print(K)
        #print(dt.name)
        if dt.name=='float32':
            feature[K]=_bytes_feature(np.float32(A).tostring())
        if dt.name=='uint8':
            feature[K]=_int64_feature(A[0])
            #print(A)
            #print(feature[K])
    #        pdb.set_trace()
    
    
    #print('----')
    #print(feature.keys())
    #print('----')

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    #pdb.set_trace()
    
    writer = tf.python_io.TFRecordWriter(FolderName + FNs[x] + '.tfrecords')

    writer.write(example.SerializeToString())

    
writer.close()
