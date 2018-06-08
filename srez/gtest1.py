import tensorflow as tf
import pdb
import os


os.chdir('/home/a/TF/srez')


filenames=['dataset1/a3.tfrecords', 'dataset1/a1.tfrecords']

reader = tf.WholeFileReader()
filename_queue = tf.train.string_input_producer(filenames)
key, value = reader.read(filename_queue)

example = tf.train.Example()
features=tf.parse_single_example(value,{'X':tf.FixedLenFeature([], tf.float32)})
#example.ParseFromString(value)
#x_1 = np.array(example.features.feature['X'].float_list.value)

#ww=tf.InteractiveSession().run(filename_queue)
#qq=tf.truncated_normal([3, 4],stddev=0.1)
#ww=tf.InteractiveSession().run(qq)

pdb.set_trace()

