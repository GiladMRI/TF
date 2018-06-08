import tensorflow as tf
import pdb
import numpy as np

import myParams
import GTools as GT

import scipy.io


FLAGS = tf.app.flags.FLAGS

def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):

    if image_size is None:
        image_size = FLAGS.sample_size
    
    #pdb.set_trace()

    
    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    

    AlsoLabel=True
    kKick= myParams.myDict['Mode'] == 'kKick'
    if kKick or myParams.myDict['Mode'] == '1DFTx' or myParams.myDict['Mode'] == '1DFTy' or myParams.myDict['Mode'] == '2DFT':
        AlsoLabel=False

    batch_size=myParams.myDict['batch_size']

    channelsIn=myParams.myDict['channelsIn']
    channelsOut=myParams.myDict['channelsOut']
    DataH=myParams.myDict['DataH']
    DataW=myParams.myDict['DataW']
    LabelsH=myParams.myDict['LabelsH']
    LabelsW=myParams.myDict['LabelsW']
    
    #image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")

    #print('1')
    if AlsoLabel:
        featuresA = tf.parse_single_example(
            value,
            features={
                'DataH': tf.FixedLenFeature([], tf.int64),
                'DataW': tf.FixedLenFeature([], tf.int64),
                'channelsIn': tf.FixedLenFeature([], tf.int64),
                'LabelsH': tf.FixedLenFeature([], tf.int64),
                'LabelsW': tf.FixedLenFeature([], tf.int64),
                'channelsOut': tf.FixedLenFeature([], tf.int64),
                'data_raw': tf.FixedLenFeature([], tf.string),
                'labels_raw': tf.FixedLenFeature([], tf.string)
            })
        labels = tf.decode_raw(featuresA['labels_raw'], tf.float32)
    else:
        featuresA = tf.parse_single_example(
            value,
            features={
                'DataH': tf.FixedLenFeature([], tf.int64),
                'DataW': tf.FixedLenFeature([], tf.int64),
                'channelsIn': tf.FixedLenFeature([], tf.int64),
                'data_raw': tf.FixedLenFeature([], tf.string)
            })
    feature = tf.decode_raw(featuresA['data_raw'], tf.float32)

    print('setup_inputs')
    print('Data   H,W,#ch: %d,%d,%d -> Labels H,W,#ch %d,%d,%d' % (DataH,DataW,channelsIn,LabelsH,LabelsW,channelsOut))
    print('------------------')
    
    if myParams.myDict['Mode'] == '1DFTy':
        feature = tf.reshape(feature, [256, 256, 1])
        feature = tf.random_crop(feature, [DataH, DataW, channelsIn])
        
        mm=tf.reduce_mean(feature)
        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        #feature = tf.Print(feature,[mm,mx],message='QQQ:')        
        #assert_op = tf.Assert(tf.greater(mx, 0), [mx])
        #with tf.control_dependencies([assert_op]):

        feature = tf.cast(feature/mx, tf.complex64)
        
        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        IQ=feature*tf.reshape(Q,[DataH,DataW,channelsIn])
        
        label=tf.concat([tf.real(IQ),tf.imag(IQ)],axis=2)
        feature=label

        HalfDataW=DataW/2

        Id=np.hstack([np.arange(HalfDataW,DataW), np.arange(0,HalfDataW)])
        Id=Id.astype(int)

        IQ2=tf.reshape(IQ,IQ.shape[0:2])
        feature=tf.fft(IQ2)
        feature = tf.gather(feature,Id,axis=1)
        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature=tf.concat([tf.real(feature),tf.imag(feature)],axis=2)

        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    if myParams.myDict['Mode'] == '1DFTx':
        feature = tf.reshape(feature, [256, 256, 1])
        feature = tf.random_crop(feature, [DataH, DataW, channelsIn])
        
        mm=tf.reduce_mean(feature)
        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        #feature = tf.Print(feature,[mm,mx],message='QQQ:')        
        #assert_op = tf.Assert(tf.greater(mx, 0), [mx])
        #with tf.control_dependencies([assert_op]):

        feature = tf.cast(feature/mx, tf.complex64)
        
        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        IQ=feature*tf.reshape(Q,[DataH,DataW,channelsIn])
        
        label=tf.concat([tf.real(IQ),tf.imag(IQ)],axis=2)
        feature=label

        HalfDataH=DataH/2

        Id=np.hstack([np.arange(HalfDataH,DataH), np.arange(0,HalfDataH)])
        Id=Id.astype(int)

        IQ2=tf.reshape(IQ,IQ.shape[0:2])
        IQ2 = tf.transpose(IQ2, perm=[1, 0])
        feature=tf.fft(IQ2)
        feature = tf.gather(feature,Id,axis=1)
        feature = tf.transpose(feature, perm=[1,0])
        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature=tf.concat([tf.real(feature),tf.imag(feature)],axis=2)

        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    if myParams.myDict['Mode'] == '2DFT':
        feature = tf.reshape(feature, [256, 256, 1])
        feature = tf.random_crop(feature, [DataH, DataW, channelsIn])
        
        mm=tf.reduce_mean(feature)
        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        #feature = tf.Print(feature,[mm,mx],message='QQQ:')        
        #assert_op = tf.Assert(tf.greater(mx, 0), [mx])
        #with tf.control_dependencies([assert_op]):

        feature = tf.cast(feature/mx, tf.complex64)
        
        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        IQ=feature*tf.reshape(Q,[DataH,DataW,channelsIn])
        
        label=tf.concat([tf.real(IQ),tf.imag(IQ)],axis=2)
        feature=label

        HalfDataH=DataH/2
        HalfDataW=DataW/2

        IdH=np.hstack([np.arange(HalfDataH,DataH), np.arange(0,HalfDataH)])
        IdH=IdH.astype(int)

        IdW=np.hstack([np.arange(HalfDataW,DataW), np.arange(0,HalfDataW)])
        IdW=IdW.astype(int)

        IQ2=tf.reshape(IQ,IQ.shape[0:2])

        IQ2=tf.fft(IQ2)
        IQ2=tf.gather(IQ2,IdW,axis=1)

        IQ2 = tf.transpose(IQ2, perm=[1, 0])
        feature=tf.fft(IQ2)
        feature = tf.gather(feature,IdH,axis=1)
        feature = tf.transpose(feature, perm=[1,0])
        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature=tf.concat([tf.real(feature),tf.imag(feature)],axis=2)

        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    if kKick:
        filename_queue2 = tf.train.string_input_producer(filenames)
        key2, value2 = reader.read(filename_queue2)
        featuresA2 = tf.parse_single_example(
            value2,
            features={
                'DataH': tf.FixedLenFeature([], tf.int64),
                'DataW': tf.FixedLenFeature([], tf.int64),
                'channelsIn': tf.FixedLenFeature([], tf.int64),
                'data_raw': tf.FixedLenFeature([], tf.string)
            })
        feature2 = tf.decode_raw(featuresA2['data_raw'], tf.float32)

        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature2 = tf.reshape(feature2, [DataH, DataW, channelsIn])


        feature.set_shape([None, None, channelsIn])
        feature2.set_shape([None, None, channelsIn])

        feature = tf.cast(feature, tf.float32)/tf.reduce_max(feature)
        feature2 = tf.cast(feature2, tf.float32)/tf.reduce_max(feature)
        
        feature= tf.concat([feature,feature*0,feature2,feature2*0], 2)
        label=feature

        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    if myParams.myDict['Mode'] == 'RegridTry1' or myParams.myDict['Mode'] == 'RegridTry1C' or myParams.myDict['Mode'] == 'RegridTry1C2' or myParams.myDict['Mode'] == 'RegridTry1C2_TS' or myParams.myDict['Mode'] == 'RegridTry1C2_TS2':
        # FullData=scipy.io.loadmat('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/NMapIndTesta.mat')
        FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
        
        NMapCR=FullData['NMapCR']
        NMapCR = tf.constant(NMapCR)

        feature=tf.gather(feature,NMapCR,validate_indices=None,name=None)

        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature = tf.cast(feature, tf.float32)
        
        labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
        label = tf.cast(labels, tf.float32)

        # Using asynchronous queues
        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    """if myParams.myDict['Mode'] == 'RegridTry1C2':
        FullData=scipy.io.loadmat('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/NMapIndC.mat')
        NMapCR=FullData['NMapCRC']
        NMapCR = tf.constant(NMapCR)

        feature=tf.gather(feature,NMapCR,validate_indices=None,name=None)

        feature = tf.reshape(feature, [DataH, DataW, channelsIn,2])
        feature = tf.cast(feature, tf.float32)
        
        labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
        label = tf.cast(labels, tf.float32)

        # Using asynchronous queues
        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels"""



    feature = tf.reshape(feature, [DataH, DataW, channelsIn])
    labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
    
    #print('44')
    #example.ParseFromString(serialized_example)
    #x_1 = np.array(example.features.feature['X'].float_list.value)

    # Convert from [depth, height, width] to [height, width, depth].
    #result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    feature.set_shape([None, None, channelsIn])
    labels.set_shape([None, None, channelsOut])

    

    # Crop and other random augmentations
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, .95, 1.05)
    #image = tf.image.random_brightness(image, .05)
    #image = tf.image.random_contrast(image, .95, 1.05)

    #print('55')
    #wiggle = 8
    #off_x, off_y = 25-wiggle, 60-wiggle
    #crop_size = 128
    #crop_size_plus = crop_size + 2*wiggle
    #print('56')
    #image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
    #print('57')
    #image = tf.image.crop_to_bounding_box(image, 1, 2, crop_size, crop_size)
    #image = tf.random_crop(image, [crop_size, crop_size, 3])

    feature = tf.reshape(feature, [DataH, DataW, channelsIn])
    feature = tf.cast(feature, tf.float32) #/255.0

    
    labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
    label = tf.cast(labels, tf.float32) #/255.0


    #if crop_size != image_size:
    #    image = tf.image.resize_area(image, [image_size, image_size])

    # The feature is simply a Kx downscaled version
    #K = 1
    #downsampled = tf.image.resize_area(image, [image_size//K, image_size//K])

    #feature = tf.reshape(downsampled, [image_size//K, image_size//K, 3])
    #feature = tf.reshape(downsampled, [image_size//K, image_size//K, 3])
    #label   = tf.reshape(image,       [image_size,   image_size,     3])

    #feature = tf.reshape(image,     [image_size,    image_size,     channelsIn])
    #feature = tf.reshape(image,     [1, image_size*image_size*2,     channelsIn])
    #label   = tf.reshape(labels,    [image_size,    image_size,     channelsOut])

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)
    
    return features, labels
