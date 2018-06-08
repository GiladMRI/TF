import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

HomeA='/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/'

sys.path.insert(0, HomeA + 'TF/')

DatasetsBase='/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TFDatasets/'


os.chdir(HomeA + 'TF/srez')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pdb

import srez_demo
import srez_input
import srez_model
import srez_train

import os.path
import random
import numpy as np
import numpy.random
import scipy.io

import tensorflow as tf

import sys

import datetime

from shutil import copyfile

import srez_modelBase

import myParams


FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('batch_size', -7,"Number of samples per batch.")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 10000,   "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', 'dataset1', "Path to the dataset directory.")

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'train',
                            "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('gene_l1_factor', 1  , "Multiplier for generator L1 loss term")
#tf.app.flags.DEFINE_float('gene_l1_factor', .90, "Multiplier for generator L1 loss term")

tf.app.flags.DEFINE_float('learning_beta1', 0.9, #0.5,
                          "Beta1 parameter used for AdamOptimizer")

# tf.app.flags.DEFINE_float('learning_rate_start', 0.0002,"Starting learning rate used for AdamOptimizer")
tf.app.flags.DEFINE_float('learning_rate_start', 0.002,"Starting learning rate used for AdamOptimizer")
#tf.app.flags.DEFINE_float('learning_rate_start', 0.00001, #0.00020,"Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_integer('summary_period', 200, "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

#tf.app.flags.DEFINE_integer('test_vectors', 16,
tf.app.flags.DEFINE_integer('test_vectors', 16,
                            """Number of features to use for testing""")
                            
tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

#tf.app.flags.DEFINE_integer('train_time', 20,"Time in minutes to train the model")
tf.app.flags.DEFINE_integer('train_time', 180,"Time in minutes to train the model")

tf.app.flags.DEFINE_integer('DataH', 64*64,"DataH")
tf.app.flags.DEFINE_integer('DataW', 64*64,"DataW")
tf.app.flags.DEFINE_integer('channelsIn', 64*64,"channelsIn")
tf.app.flags.DEFINE_integer('LabelsH', 64*64,"LabelsH")
tf.app.flags.DEFINE_integer('LabelsW', 64*64,"LabelsW")
tf.app.flags.DEFINE_integer('channelsOut', 64*64,"channelsOut")

tf.app.flags.DEFINE_string('SessionName', 'SessionName', "Which operation to run. [demo|train]")

def getParam(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


import myParams
myParams.init()
# defaults
myParams.myDict['DataH']=1
myParams.myDict['MapSize']=3
myParams.myDict['learning_rate_half_life']=5000
myParams.myDict['learning_rate_start']=0.002
myParams.myDict['dataset']='dataKnee'
myParams.myDict['batch_size']=-8
myParams.myDict['WL1_Lambda']=0.0001
myParams.myDict['WL2_Lambda']=0.0001
myParams.myDict['QuickFailureTimeM']=3
myParams.myDict['QuickFailureTthresh']=0.3
myParams.myDict['summary_period']=0.5 # in minutes
myParams.myDict['checkpoint_period']=20 # in minutes

myParams.myDict['DiscStartMinute']=5 # in minutes

myParams.myDict['Mode']='kKick'

myParams.myDict['noise_level']=0.0

myParams.myDict['ShowRealData'] = 0
myParams.myDict['CmplxBias'] = 0


ParamFN=HomeA +'TF/Params.txt'
ParamsD = {}
with open(ParamFN) as f:
    for line in f:
        if len(line)<3:
            continue
        # print(line)
        #print(line.replace("\n",""))
        (key, val) = line.split()
        valx=getParam(val)
        ParamsD[key] = valx
        myParams.myDict[key]=ParamsD[key]
        print(key + " : " + str(val) + " " + type(valx).__name__)

#pdb.set_trace()
#FLAGS.nChosen=2045
#FLAGS.random_seed=0
#tf.app.flags.random_seed=0
# FullData=scipy.io.loadmat('/home/a/TF/ImgsMC1.mat')
# FullData=scipy.io.loadmat('/home/a/TF/CurChunk.mat')
# Data=FullData['Data']
# Labels=FullData['Labels']
# #pdb.set_trace()
# nSamples=Data.shape[0]
# DataH = Data.shape[1]
# if Data.ndim<3 :
#     DataW = 1
# else:
#     DataW = Data.shape[2]

# if Data.ndim<4 :
#     channelsIn=1
# else :
#     channelsIn = Data.shape[3]

# LabelsH = Labels.shape[1]
# LabelsW = Labels.shape[2]
# if Labels.ndim<4 :
#     channelsOut=1
# else :
#     channelsOut = Labels.shape[3]

# FLAGS.DataH=DataH
# FLAGS.DataW=DataW
# FLAGS.channelsIn=channelsIn
# FLAGS.LabelsH=LabelsH
# FLAGS.LabelsW=LabelsW
# FLAGS.channelsOut=channelsOut

# del FullData
# del Data
# del Labels


# pdb.set_trace()

# MatlabParams=scipy.io.loadmat('/home/a/TF/MatlabParams.mat')

#FLAGS.dataset='dataKnee'
#FLAGS.dataset='dataFaceP4'

# SessionNameBase=MatlabParams['SessionName'][0];
SessionNameBase= myParams.myDict['SessionNameBase']

#SessionName=MatlabParams['SessionName'][0] + '__' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

SessionName= SessionNameBase + '_'+myParams.myDict['dataset'] + '__' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

myParams.myDict['Full_dataset']=DatasetsBase+ParamsD['dataset']

# FLAGS.checkpoint_dir=SessionName+'_checkpoint'
# FLAGS.train_dir=SessionName+'_train'
# FLAGS.SessionName=SessionName

myParams.myDict['SessionName']=SessionName
myParams.myDict['train_dir']=SessionName+'_train'
myParams.myDict['checkpoint_dir']=SessionName+'_checkpoint'

remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
assert(remaining_args == [sys.argv[0]])

def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(myParams.myDict['checkpoint_dir']):
        tf.gfile.MakeDirs(myParams.myDict['checkpoint_dir'])
    
    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(myParams.myDict['train_dir']):
            tf.gfile.DeleteRecursively(myParams.myDict['train_dir'])
        tf.gfile.MakeDirs(myParams.myDict['train_dir'])

    # ggg
    copyfile(ParamFN, myParams.myDict['train_dir'] + '/ParamsUsed.txt')

    # Return names of training files
    if not tf.gfile.Exists(myParams.myDict['Full_dataset']) or \
       not tf.gfile.IsDirectory(myParams.myDict['Full_dataset']):
        raise FileNotFoundError("Could not find folder `%s'" % (myParams.myDict['Full_dataset'],))

    filenames = tf.gfile.ListDirectory(myParams.myDict['Full_dataset'])
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(myParams.myDict['Full_dataset'], f) for f in filenames]

    return filenames


def setup_tensorflow():
    print("setup_tensorflow")
    # Create session
    #config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    FLAGS
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    summary_writer = tf.summary.FileWriter(myParams.myDict['train_dir'], sess.graph)
    print("setup_tensorflow end")
    return sess, summary_writer

def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(myParams.myDict['checkpoint_dir']):
        raise FileNotFoundError("Could not find folder `%s'" % (myParams.myDict['checkpoint_dir'],))

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup async input queues
    features, labels = srez_input.setup_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_modelBase.create_model(sess, features, labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(myParams.myDict['checkpoint_dir'], filename)
    saver.restore(sess, filename)

    # Execute demo
    srez_demo.demo1(sess)

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():
    LoadAndRunOnData=False
    if LoadAndRunOnData:
        # Setup global tensorflow state
        sess, summary_writer = setup_tensorflow()

        # Prepare directories
        filenames = prepare_dirs(delete_train_dir=False)

        # Setup async input queues
        features, labels = srez_input.setup_inputs(sess, filenames)

        # Create and initialize model
        [gene_minput, gene_moutput,
         gene_output, gene_var_list,
         disc_real_output, disc_fake_output, disc_var_list] = \
                srez_modelBase.create_model(sess, features, labels)

        # Restore variables from checkpoint
        saver = tf.train.Saver()
        filename = 'checkpoint_new.txt'
        filename = os.path.join('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/srez/RegridTry1C2_TS2_dataNeighborhoodRCB0__2018-06-08_16-17-56_checkpoint', filename)
        saver.restore(sess, filename)

        if myParams.myDict['Mode'] == 'RegridTry1' or myParams.myDict['Mode'] == 'RegridTry1C' or myParams.myDict['Mode'] == 'RegridTry1C2' or myParams.myDict['Mode'] == 'RegridTry1C2_TS' or myParams.myDict['Mode'] == 'RegridTry1C2_TS2':
            FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
            NMapCR=FullData['NMapCR']

        for r in range(80,81):
            ifilename='/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/srez/RealData/b_Ben14May_Sli5_r' +  f'{r:02}' + '.mat'
            RealData=scipy.io.loadmat(ifilename)
            RealData=RealData['Data']
            
            if RealData.ndim==2:
                RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],1,1))
            if RealData.ndim==3:
                RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],RealData.shape[2],1))
            
            Real_feature=RealData

            if myParams.myDict['Mode'] == 'RegridTry1' or myParams.myDict['Mode'] == 'RegridTry1C' or myParams.myDict['Mode'] == 'RegridTry1C2' or myParams.myDict['Mode'] == 'RegridTry1C2_TS' or myParams.myDict['Mode'] == 'RegridTry1C2_TS2':
                batch_size=myParams.myDict['batch_size']

                Real_feature=np.reshape(RealData[0],[RealData.shape[1]])
                Real_feature=np.take(Real_feature,NMapCR)
                Real_feature=np.tile(Real_feature, (batch_size,1,1,1))

            Real_dictOut = {gene_minput: Real_feature}


            gene_RealOutput = sess.run(gene_moutput, feed_dict=Real_dictOut)

            OnRealData={}
            OnRealDataM=gene_RealOutput
            filenamex = 'OnRealData' + f'{r:02}' + '.mat'
            filename = os.path.join('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/Out/', filenamex)
            OnRealData['x']=OnRealDataM
            scipy.io.savemat(filename,OnRealData)

            print('Finished!')
            sys.exit("Finished outputing data!")


    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    all_filenames = prepare_dirs(delete_train_dir=True)

    # Separate training and test sets
    train_filenames = all_filenames[:-FLAGS.test_vectors]
    test_filenames  = all_filenames[-FLAGS.test_vectors:]

    # TBD: Maybe download dataset here

    #pdb.set_trace()
    # Setup async input queues
    train_features, train_labels = srez_input.setup_inputs(sess, train_filenames)
    test_features,  test_labels  = srez_input.setup_inputs(sess, test_filenames)

    print('train_features %s' % (train_features))
    print('train_labels %s' % (train_labels))


    # Add some noise during training (think denoising autoencoders)
    noise_level=myParams.myDict['noise_level']
    AddNoise=noise_level>0.0
    if AddNoise:
        noisy_train_features = train_features + tf.random_normal(train_features.get_shape(), stddev=noise_level)
    else:
        noisy_train_features = train_features

    
    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_modelBase.create_model(sess, noisy_train_features, train_labels)
    
    # gene_VarNamesL=[];
    # for line in gene_var_list: gene_VarNamesL.append(line.name+'           ' + str(line.shape.as_list()))
    # gene_VarNamesL.sort()

    # for line in gene_VarNamesL: print(line)
    # # var_23 = [v for v in tf.global_variables() if v.name == "gene/GEN_L020/C2D_weight:0"][0]

    # for line in sess.graph.get_operations(): print(line)
    # Gen3_ops=[]
    # for line in sess.graph.get_operations():
    #     if 'GEN_L003' in line.name:
    #         Gen3_ops.append(line)

    #     # LL=QQQ.outputs[0]
        
    # for x in Gen3_ops: print(x.name +'           ' + str(x.outputs[0].shape))

    # GenC2D_ops= [v for v in sess.graph.get_operations()]

    # GenC2D_ops= [v for v in tf.get_operations() if "weight" in v.name]
    # GenC2D_ops= [v for v in GenC2D_ops if "C2D" in v.name]
    # for x in GenC2D_ops: print(x.name +'           ' + str(x.outputs[0].shape))

    # for x in GenC2D_ops: print(x.name)

    AEops = [v for v in sess.graph.get_operations() if "AE" in v.name and not ("_1/" in v.name)]
    AEouts = [v.outputs[0] for v in AEops]
    varsForL1=AEouts
    # varsForL1=AEouts[0:-1]
    # varsForL1=AEouts[1:]

    # for line in sess.graph.get_operations():
    #     if 'GEN_L003' in line.name:
    #         Gen3_ops.append(line)

    #     # LL=QQQ.outputs[0]
        
    # for x in Gen3_ops: print(x.name +'           ' + str(x.outputs[0].shape))

    print("Vars for l2 loss:")
    varws = [v for v in tf.global_variables() if "weight" in v.name]
    varsForL2 = [v for v in varws if "C2D" in v.name]
    for line in varsForL2: print(line.name+'           ' + str(line.shape.as_list()))

    

    # pdb.set_trace()

    gene_loss, MoreOut, MoreOut2 = srez_modelBase.create_generator_loss(disc_fake_output, gene_output, train_features,train_labels,varsForL1,varsForL2)
    disc_real_loss, disc_fake_loss = \
                     srez_modelBase.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    
    (global_step, learning_rate, gene_minimize, disc_minimize) = \
            srez_modelBase.create_optimizers(gene_loss, gene_var_list, disc_loss, disc_var_list)

    # Train model
    train_data = TrainData(locals())
    
    #pdb.set_trace()
    # ggg: to restore session
    RestoreSession=False
    if RestoreSession:
        saver = tf.train.Saver()
        filename = 'checkpoint_new.txt'
        filename = os.path.join(myParams.myDict['checkpoint_dir'], filename)
        saver.restore(sess, filename)

    srez_train.train_model(train_data)

def main(argv=None):
    print("aaa")
    _train()
    # Training or showing off?
    #_train()
    #if FLAGS.run == 'demo':
    #    _demo()
    #elif FLAGS.run == 'train':
    #    _train()

if __name__ == '__main__':
  tf.app.run()

#print("asd40")
_train()
#setup_tensorflow()
#tf.app.run()

#print("asd5")
