# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#"""A deep MNIST classifier using convolutional layers.

#See extensive documentation at
#https://www.tensorflow.org/get_started/mnist/pros
#"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np 

""" ggg: Here we create a dataset, i.e. a class that contains
      data: input to the net
      labels: output

    It gives part of the data every time by calling "next_batch"
    Every time is goes over the whole data it shuffles it.
    """
class Dataset:

  def __init__(self,data,labels):
    self._index_in_epoch = 0
    self._epochs_completed = 0
    self._data = data
    self._labels = labels
    self._num_examples = data.shape[0]
    pass


  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  def next_batch(self,batch_size,shuffle = True):
    start = self._index_in_epoch
    if start == 0 and self._epochs_completed == 0:
      idx = np.arange(0, self._num_examples)  # get all possible indexes
      np.random.shuffle(idx)  # shuffle indexe
      self._data = self.data[idx]  # get list of `num` random samples
      self._labels = self.labels[idx]  # get list of `num` random samples

    # go to the next batch
    if start + batch_size > self._num_examples:
      self._epochs_completed += 1
      rest_num_examples = self._num_examples - start
      data_rest_part = self.data[start:self._num_examples]
      labels_rest_part = self.labels[start:self._num_examples]
      idx0 = np.arange(0, self._num_examples)  # get all possible indexes
      np.random.shuffle(idx0)  # shuffle indexes
      self._data = self.data[idx0]  # get list of `num` random samples
      self._labels = self.labels[idx0]  # get list of `num` random samples

      start = 0
      self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
      end =  self._index_in_epoch  
      data_new_part =  self._data[start:end]  
      labels_new_part =  self._labels[start:end]  
      return [np.concatenate((data_rest_part, data_new_part), axis=0),np.concatenate((labels_rest_part, labels_new_part), axis=0)]
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return [self._data[start:end],self._labels[start:end]]



"""ggg: Here we load the data from Matlab and put it in our dataset class
In matlab it was simply:
Data=rand(5000,28,28,2);
Labels=rand(5000,28,28,2);
save('/home/a/tensorflowtest/MyData_Img.mat','Data','Labels')
"""

import scipy.io
FullData=scipy.io.loadmat('/home/a/tensorflowtest/MyData_Img.mat')

dataset = Dataset(FullData['Data'],FullData['Labels'])

Sz=FullData['Data'].shape
OutSz=FullData['Labels'].shape

import tensorflow as tf

FLAGS = None


nPE=Sz[1];
nRO=Sz[2];

nFeaturesInput=Sz[3]; # Should be 2: real and imag
nFeaturesAfterFC=4;
nFeaturesLayer1=32;
nFeaturesLayer2=17;
nFeaturesOutput=OutSz[3]; # Should be 2: real and imag

KernelSizeLayer1PE=5;
KernelSizeLayer1RO=7;

KernelSizeLayer2PE=5;
KernelSizeLayer2RO=5;

KernelSizeLayer3PE=3;
KernelSizeLayer3RO=5;


"""ggg: Defining the neural network"""

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  #with tf.name_scope('reshape'):
  x_flatted = tf.reshape(x, [-1, nPE*nRO*nFeaturesInput])

  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([nPE*nRO*nFeaturesInput, nPE*nRO*nFeaturesAfterFC])
    b_fc1 = bias_variable([nPE*nRO*nFeaturesAfterFC])

    h_fc1 = tf.nn.relu(tf.matmul(x_flatted, W_fc1) + b_fc1)
    h_fc1_as_Img=tf.reshape(h_fc1, [-1, nPE,nRO,nFeaturesAfterFC])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([KernelSizeLayer1PE, KernelSizeLayer1RO, nFeaturesAfterFC, nFeaturesLayer1])
    b_conv1 = bias_variable([nFeaturesLayer1])
    #h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    """ggg chaned here to not do reshape"""
    #h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_conv1 = tf.nn.relu(conv2d(h_fc1_as_Img, W_conv1) + b_conv1)

  # Second convolutional layer -- maps nFeaturesLayer1 feature maps to 17.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([KernelSizeLayer2PE, KernelSizeLayer2RO, nFeaturesLayer1, nFeaturesLayer2])
    b_conv2 = bias_variable([nFeaturesLayer2])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

  # Third convolutional layer -- maps 17 feature maps to 2.
  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([KernelSizeLayer3PE, KernelSizeLayer3RO, nFeaturesLayer2, nFeaturesOutput])
    b_conv3 = bias_variable([nFeaturesOutput])
    #h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    h_conv3 = conv2d(h_conv2, W_conv3) + b_conv3
    y_conv = h_conv3;

  return y_conv

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# Create the model
x = tf.placeholder(tf.float32, np.concatenate(([None],Sz[1:10])))
# x = tf.placeholder(tf.float32, [None, 28,28,2])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, np.concatenate(([None],OutSz[1:10])))

# Build the graph for the deep net
y_conv = deepnn(x)

with tf.name_scope('loss'):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                          logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

#graph_location = tempfile.mkdtemp()
#print('Saving graph to: %s' % graph_location)
#train_writer = tf.summary.FileWriter(graph_location)
#train_writer.add_graph(tf.get_default_graph())

# saver = tf.train.Saver()

"""ggg: Running the training """
my_batch_size=40;
nIters=3;

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(nIters):
    batch=dataset.next_batch(my_batch_size)
    
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1] })
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1] })
    
  #save_path = saver.save(sess, "/tmp/aaa/model.ckpt")

  """ggg: Printing how the current net is doing on the full data:
          Don't run that on the GPU"""
  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: FullData['Data'], y_: FullData['Labels'] }))

  """ggg: Accessing data from a layer of the net
    This is just to understand stuff. Don't run that on the GPU"""
  #LayerData=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'conv1')]
  #Bias1=LayerData[-1].eval();
  #W1=LayerData[-2].eval();


print('OK finished')

""" to run a TF command: """
# qq=tf.truncated_normal([3, 4],stddev=0.1)
# ww=tf.InteractiveSession().run(qq)

""" To run this file: """
# execfile('/home/a/tensorflowtest/deep_try6_OutImg_FCstart.py')

""" To change directory #"""
# import os
# os.chdir('/home/a/tensorflowtest')
# os.getcwd()

""" to load data from matlab: """
# import scipy.io
# A=scipy.io.loadmat('/home/a/tensorflowtest/a5kx784.mat')
# A['a'].shape

""" Manual for simple matlab stuff in Python:
    http://mathesaurus.sourceforge.net/matlab-python-xref.pdf  """

import numpy as np
a = tf.constant(np.arange(1, 25, dtype=np.int32),              shape=[2, 2, 2, 3])
b=tf.reshape(a,[ 2,1,4,3 ])
aa=tf.InteractiveSession().run(a)
bb=tf.InteractiveSession().run(b)