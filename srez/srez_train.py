import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time
import scipy.io
import pdb
import myParams

FLAGS = tf.app.flags.FLAGS

def _summarize_progress(train_data, feature, label, gene_output, batch, suffix, max_samples=8):
    td = train_data

    size = [label.shape[1], label.shape[2]]

    if myParams.myDict['Mode'] == '1DFTx' or myParams.myDict['Mode'] == '1DFTy' or myParams.myDict['Mode'] == '2DFT' or myParams.myDict['Mode'] == 'RegridTry1' or myParams.myDict['Mode'] == 'RegridTry1C' or myParams.myDict['Mode'] == 'RegridTry1C2' or myParams.myDict['Mode'] == 'RegridTry1C2_TS' or myParams.myDict['Mode'] == 'RegridTry1C2_TS2':

        LabelA=np.sqrt(np.power(label[:,:,:,0],2)+np.power(label[:,:,:,1],2))
        LabelP=np.arctan2(label[:,:,:,1],label[:,:,:,0])/(2*np.pi)+0.5;

        GeneA=np.sqrt(np.power(gene_output[:,:,:,0],2)+np.power(gene_output[:,:,:,1],2))
        GeneA=np.maximum(np.minimum(GeneA, 1.0), 0.0)
        GeneP=np.arctan2(gene_output[:,:,:,1],gene_output[:,:,:,0])/(2*np.pi)+0.5;

        gene_output=(gene_output+1)/2
        label=(label+1)/2

        clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

        image   = tf.concat([clipped[:,:,:,0], label[:,:,:,0], clipped[:,:,:,1], label[:,:,:,1],LabelA,GeneA,LabelP,GeneP], 2)

        image   = tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        image   = tf.concat([image,image,image], 3)

        image = image[0:max_samples,:,:,:]
        image = tf.concat([image[i,:,:,:] for i in range(int(max_samples))], 0)
        image = td.sess.run(image)

        filename = 'batch%06d_%s.png' % (batch, suffix)
        filename = os.path.join(myParams.myDict['train_dir'], filename)
        scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)

        print("    Saved %s" % (filename,))
        return

    SAE = myParams.myDict['Mode'] == 'SAE'
    if SAE:
        clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

        # pdb.set_trace()
        image   = tf.concat([clipped[:,:,:,0],label[:,:,:,0]], 2)

        image=tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        image   = tf.concat([image,image,image], 3)

        image = image[0:max_samples,:,:,:]
        image = tf.concat([image[i,:,:,:] for i in range(int(max_samples))], 0)
        image = td.sess.run(image)

        filename = 'batch%06d_%s.png' % (batch, suffix)
        filename = os.path.join(myParams.myDict['train_dir'], filename)
        scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)

        print("    Saved %s" % (filename,))
        return

    kKick= myParams.myDict['Mode'] == 'kKick'
    if kKick:
        clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

        image   = tf.concat([clipped[:,:,:,0], label[:,:,:,0], clipped[:,:,:,1], label[:,:,:,1]], 2)

        image=tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        image   = tf.concat([image,image,image], 3)

        image = image[0:max_samples,:,:,:]
        image = tf.concat([image[i,:,:,:] for i in range(int(max_samples))], 0)
        image = td.sess.run(image)

        filename = 'batch%06d_%s.png' % (batch, suffix)
        filename = os.path.join(myParams.myDict['train_dir'], filename)
        scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)

        print("    Saved %s" % (filename,))
        return

    #nearest = tf.image.resize_nearest_neighbor(feature, size)
    #nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    #bicubic = tf.image.resize_bicubic(feature, size)
    #bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    gene_outputx=np.sqrt(np.power(gene_output[:,:,:,0],2)+np.power(gene_output[:,:,:,1],2))
    Theta=np.arctan2(gene_output[:,:,:,1],gene_output[:,:,:,0])/(2*np.pi)+0.5;
    labelx=np.sqrt(np.power(label[:,:,:,0],2)+np.power(label[:,:,:,1],2))
    labelx[0]=Theta[0];

    clipped = tf.maximum(tf.minimum(gene_outputx, 1.0), 0.0)

    #image   = tf.concat([nearest, bicubic, clipped, label], 2)
    image   = tf.concat([clipped, labelx], 2)

    image=tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
    # pdb.set_trace()
    image   = tf.concat([image,image,image], 3)

    image = image[0:max_samples,:,:,:]
    #image = tf.concat([image[i,:,:,:] for i in range(max_samples)], 0)
    image1 = tf.concat([image[i,:,:,:] for i in range(int(max_samples/2))], 0)
    image2 = tf.concat([image[i,:,:,:] for i in range(int(max_samples/2),max_samples)], 0)
    image  = tf.concat([image1, image2], 1)
    image = td.sess.run(image)

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(myParams.myDict['train_dir'], filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)

    OnRealData={}
    OnRealDataM=gene_output[0]
    filenamex = 'OnRealData.mat'
    filename = os.path.join(myParams.myDict['train_dir'], filenamex)
    OnRealData['x']=OnRealDataM
    scipy.io.savemat(filename,OnRealData)

    print("    Saved %s" % (filename,))

def _save_checkpoint(train_data, batch,G_LossV):
    td = train_data

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(myParams.myDict['checkpoint_dir'], oldname)
    newname = os.path.join(myParams.myDict['checkpoint_dir'], newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver()
    saver.save(td.sess, newname)

    print("    Checkpoint saved")

    # save all:
    TrainSummary={}
    filenamex = 'TrainSummary_%06d.mat' % (batch)
    filename = os.path.join(myParams.myDict['train_dir'], filenamex)
    VLen=td.gene_var_list.__len__()
    var_list=[]
    for i in range(0, VLen): 
        var_list.append(td.gene_var_list[i].name);
        tmp=td.sess.run(td.gene_var_list[i])
        s1=td.gene_var_list[i].name
        print("Saving  %s" % (s1))
        s1=s1.replace(':','_')
        s1=s1.replace('/','_')
        TrainSummary[s1]=tmp
    
    TrainSummary['var_list']=var_list
    TrainSummary['G_LossV']=G_LossV

    scipy.io.savemat(filename,TrainSummary)

    print("saved to %s" % (filename))

def train_model(train_data):
    td = train_data

    summaries = tf.summary.merge_all()
    RestoreSession=False
    if not RestoreSession:
        td.sess.run(tf.global_variables_initializer())

    # lrval       = FLAGS.learning_rate_start
    learning_rate_start=myParams.myDict['learning_rate_start']
    lrval       = myParams.myDict['learning_rate_start']
    start_time  = time.time()
    last_summary_time  = time.time()
    last_checkpoint_time  = time.time()
    done  = False
    batch = 0

    print("lrval %f" % (lrval))

    # assert FLAGS.learning_rate_half_life % 10 == 0

    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])

    G_LossV=np.zeros((1000000), dtype=np.float32)
    filename = os.path.join(myParams.myDict['train_dir'], 'TrainSummary.mat')
    
    feed_dictOut = {td.gene_minput: test_feature}
    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
    _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')

    feed_dict = {td.learning_rate : lrval}
    opsx = [td.gene_minimize, td.gene_loss]
    _, gene_loss = td.sess.run(opsx, feed_dict=feed_dict)

    # opsy = [td.gene_loss]
    # gene_loss = td.sess.run(opsy, feed_dict=feed_dict)

    # ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]
    # _, _, gene_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops, feed_dict=feed_dict)

    batch += 1

    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
    _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')

    feed_dict = {td.learning_rate : lrval}
    # ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]

    opsx = [td.gene_minimize, td.gene_loss]
    _, gene_loss = td.sess.run(opsx, feed_dict=feed_dict)

    batch += 1

    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
    _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')

    # load model
    #saver.restore(sess,tf.train.latest_checkpoint('./'))
    # running model on data:test_feature
    RunOnData=False
    if RunOnData:
        filenames = tf.gfile.ListDirectory('DataAfterpMat')
        filenames = sorted(filenames)
        #filenames = [os.path.join('DataAfterpMat', f) for f in filenames]
        Ni=len(filenames)
        OutBase=myParams.myDict['SessionName']+'_OutMat'
        tf.gfile.MakeDirs(OutBase)

        #pdb.set_trace()

        for index in range(Ni):
            print(index)
            print(filenames[index])
            CurData=scipy.io.loadmat(os.path.join('DataAfterpMat', filenames[index]))
            Data=CurData['CurData']
            Data=Data.reshape((1,64,64,1))
            test_feature=np.kron(np.ones((16,1,1,1)),Data)
            #test_feature = np.array(np.random.choice([0, 1], size=(16,64,64,1)), dtype='float32')


            feed_dictOut = {td.gene_minput: test_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)

            filenameOut=os.path.join(OutBase, filenames[index][:-4] + '_out.mat') 

            SOut={}
            SOut['X']=gene_output[0]
            scipy.io.savemat(filenameOut,SOut)

    # pdb.set_trace()

    #_summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')
    # to get value of var:
    # ww=td.sess.run(td.gene_var_list[1])
    
    if myParams.myDict['ShowRealData']>0:
        # ifilename=os.path.join('RealData', 'b.mat')
        ifilename=myParams.myDict['RealDataFN']
        RealData=scipy.io.loadmat(ifilename)
        RealData=RealData['Data']
        
        if RealData.ndim==2:
            RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],1,1))
        if RealData.ndim==3:
            RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],RealData.shape[2],1))
        
        Real_feature=RealData

        if myParams.myDict['Mode'] == 'RegridTry1' or myParams.myDict['Mode'] == 'RegridTry1C' or myParams.myDict['Mode'] == 'RegridTry1C2' or myParams.myDict['Mode'] == 'RegridTry1C2_TS' or myParams.myDict['Mode'] == 'RegridTry1C2_TS2':
            # FullData=scipy.io.loadmat('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/NMapIndTesta.mat')
            FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
            NMapCR=FullData['NMapCR']

            batch_size=myParams.myDict['batch_size']

            Real_feature=np.reshape(RealData[0],[RealData.shape[1]])
            Real_feature=np.take(Real_feature,NMapCR)
            Real_feature=np.tile(Real_feature, (batch_size,1,1,1))

        Real_dictOut = {td.gene_minput: Real_feature}

    # LearningDecayFactor=np.power(2,(-1/FLAGS.learning_rate_half_life))
    learning_rate_half_life=myParams.myDict['learning_rate_half_life']
    LearningDecayFactor=np.power(2,(-1/learning_rate_half_life))

    # train_time=FLAGS.train_time
    train_time=myParams.myDict['train_time']

    QuickFailureTimeM=myParams.myDict['QuickFailureTimeM']
    QuickFailureThresh=myParams.myDict['QuickFailureThresh']

    summary_period=myParams.myDict['summary_period'] # in Minutes
    checkpoint_period=myParams.myDict['checkpoint_period'] # in Minutes

    DiscStartMinute=myParams.myDict['DiscStartMinute']
    while not done:
        batch += 1
        gene_loss = disc_real_loss = disc_fake_loss = -1.234

        # elapsed = int(time.time() - start_time)/60
        CurTime=time.time()
        elapsed = (time.time() - start_time)/60
            
        # Update learning rate
        lrval*=LearningDecayFactor
        if(learning_rate_half_life<1000): # in minutes
            lrval=learning_rate_start*np.power(0.5,elapsed/learning_rate_half_life)

        

        #print("batch %d gene_l1_factor %f' " % (batch,FLAGS.gene_l1_factor))
        # if batch==200:
        if elapsed>DiscStartMinute:
            FLAGS.gene_l1_factor=0.9
        
        RunDiscriminator= FLAGS.gene_l1_factor < 0.999

        feed_dict = {td.learning_rate : lrval}
        if RunDiscriminator:
            ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]
            _, _, gene_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops, feed_dict=feed_dict)
        else:
            ops = [td.gene_minimize, td.gene_loss, td.MoreOut, td.MoreOut2]
            _, gene_loss, MoreOut, MoreOut2 = td.sess.run(ops, feed_dict=feed_dict)
        
        
        G_LossV[batch]=gene_loss
        
        if batch % 10 == 0:

            # pdb.set_trace()

            # Show we are alive
            #print('Progress[%3d%%], ETA[%4dm], Batch [%4d], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
            #      (int(100*elapsed/train_time), train_time - int(elapsed), batch, gene_loss, disc_real_loss, disc_fake_loss))

            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f], MoreOut[%3.3f, %3.3f]' %
                  (int(100*elapsed/train_time), train_time - int(elapsed), batch, gene_loss, disc_real_loss, disc_fake_loss, MoreOut, MoreOut2))

            if np.isnan(gene_loss):
                print('NAN!!')
                done = True

            # ggg: quick failure test
            if elapsed>QuickFailureTimeM :
                if gene_loss>QuickFailureThresh :
                    print('Quick failure!!')
                    done = True
                else:
                    QuickFailureTimeM=10000000

            # Finished?            
            current_progress = elapsed / train_time
            if current_progress >= 1.0:
                done = True
            

            StopFN='/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/stop.a'
            if os.path.isfile(StopFN):
                print('Stop file used!!')
                done = True
                try:
                    tf.gfile.Remove(StopFN)
                except:
                    pass


            # Update learning rate
            # if batch % FLAGS.learning_rate_half_life == 0:
            #     lrval *= .5

        # if batch % FLAGS.summary_period == 0:
        if (CurTime-last_summary_time)/60>summary_period:
            # Show progress with test features
            # feed_dict = {td.gene_minput: test_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
            
            if myParams.myDict['ShowRealData']>0:
                gene_RealOutput = td.sess.run(td.gene_moutput, feed_dict=Real_dictOut)
                gene_output[0]=gene_RealOutput[0]
            
            Asuffix = 'out_%06.4f' % (gene_loss)
            _summarize_progress(td, test_feature, test_label, gene_output, batch, Asuffix)

            last_summary_time  = time.time()
    
            
        # if batch % FLAGS.checkpoint_period == 0:
        SaveCheckpoint_ByTime=(CurTime-last_checkpoint_time)/60>checkpoint_period
        CheckpointFN='/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/save.a'
        SaveCheckPointByFile=os.path.isfile(CheckpointFN)
        if SaveCheckPointByFile:
            tf.gfile.Remove(CheckpointFN)

        if SaveCheckpoint_ByTime or SaveCheckPointByFile:
            last_checkpoint_time  = time.time()
            # Save checkpoint
            _save_checkpoint(td, batch,G_LossV)

    _save_checkpoint(td, batch,G_LossV)
    
    print('Finished training!')