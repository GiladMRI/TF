import numpy as np
import tensorflow as tf
import scipy.io

FLAGS = tf.app.flags.FLAGS

import myParams

from srez_modelBase import Model
import srez_modelBase

def DFT_matrix(N):
    HalfN=N/2
    Id=np.hstack([np.arange(HalfN,N), np.arange(0,HalfN)])
    i, j = np.meshgrid(Id, Id)

    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W

def ConstConvKernel(K1,K2,FCOut):
    W=np.zeros([K1,K2,K1,K2,FCOut,FCOut])
    for i in range(0,K1):
        for j in range(0,K2):
            for t in range(0,FCOut):
                W[i,j,i,j,t,t]=1
    W=np.reshape(W,[K1,K2,K1*K2*FCOut,FCOut])
    return W

def _generator_model(sess, features, labels, channels):
    # Upside-down all-convolutional resnet

    mapsize = 3
    mapsize = myParams.myDict['MapSize']
    res_units  = [256, 128, 96]

    old_vars = tf.global_variables()

    # See Arxiv 1603.05027
    model = Model('GEN', features)

    # H=FLAGS.LabelsH;
    # W=FLAGS.LabelsW;
    H=myParams.myDict['LabelsH']
    W=myParams.myDict['LabelsW']
    channelsOut=myParams.myDict['channelsOut']

    print("_generator_model")
    print("%d %d %d" % (H, W,channels))

    if myParams.myDict['NetMode'] == '1DFTy':
        print("1DFTy mode")
        model.add_Mult2DMCy(W,channelsOut)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == '1DFTx':
        print("1DFTx mode")
        model.add_Mult2DMCx(H,channelsOut)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == '2DFT':
        print("2DFT mode")
        model.add_Mult2DMCy(W,channelsOut)
        model.add_Mult2DMCx(H,channelsOut)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'RegridTry1':
        print("RegridTry1 mode")
        model.add_PixelwiseMult(2, stddev_factor=1.0)
        model.add_Mult2DMCy(W,channelsOut)
        model.add_Mult2DMCx(H,channelsOut)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'RegridTry1C':
        print("RegridTry1C mode")
        addBias=myParams.myDict['CmplxBias']>0
        if addBias:
            print("with bias")
        else:
            print("without bias")
        model.add_PixelwiseMult(2, stddev_factor=1.0)
        model.add_5thDim()
        model.add_Permute45()
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'RegridTry1C2':
        print("RegridTry1C2 mode")
        addBias=myParams.myDict['CmplxBias']>0
        if addBias:
            print("with bias")
        else:
            print("without bias")
        model.add_Split4thDim(2)
        model.add_PixelwiseMultC(1, stddev_factor=1.0)
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'RegridTry1C2_TS':
        print("RegridTry1C2_TS mode")
        addBias=myParams.myDict['CmplxBias']>0
        if addBias:
            print("with bias")
        else:
            print("without bias")
        nTS=7
        model.add_Split4thDim(2)
        model.add_PixelwiseMultC(nTS, stddev_factor=1.0)
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.add_PixelwiseMultC(1, stddev_factor=1.0)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'RegridTry1C2_TS2':
        print("RegridTry1C2_TS mode")
        addBias=myParams.myDict['CmplxBias']>0
        if addBias:
            print("with bias")
        else:
            print("without bias")
        nTS=7
        model.add_Split4thDim(2)
        model.add_PixelwiseMultC(nTS, stddev_factor=1.0)
        model.add_Mult2DMCyCSharedOverFeat(W,1,add_bias=addBias)
        model.add_Mult2DMCxCSharedOverFeat(H,1,add_bias=addBias)
        model.add_PixelwiseMultC(1, stddev_factor=1.0)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASHTry1':
        print("SMASHTry1 mode")
        addBias=myParams.myDict['CmplxBias']>0
    
        model.add_PixelwiseMultC(2, stddev_factor=1.0)
        model.add_Combine34()
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASHTry1_CC':
        print("SMASHTry1_CC mode")
        addBias=myParams.myDict['CmplxBias']>0
        DataH=myParams.myDict['DataH']
        # we're [Batch, kH,kW,AllChannels*Neighbors*RI]
        model.add_Split4thDim(2) # Now [Batch, kH,kW,AllChannels*Neighbors,RI]

        model.add_Mult2DMCxCSharedOverFeat(DataH, 1) # Now [Batch, H,kW,AllChannels*Neighbors,RI]
        model.add_Split4thDim(6) # Now [Batch, H,kW,AllChannels,Neighbors,RI]

        ncc=4
        model.add_einsumC('abcde,dx->abcxe',[8, ncc])

        model.add_Combine45(squeeze=True) # Now [Batch, H,kW,CompressedChannels*Neighbors,RI]
        model.add_Mult2DMCxCSharedOverFeat(DataH, 1) # Now [Batch, kH,kW,CompressedChannels*Neighbors,RI]

        model.add_PixelwiseMultC(2, stddev_factor=1.0)
        model.add_Combine34()
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars



    if myParams.myDict['NetMode'] == 'SMASHTry1_GCC':
        print("SMASHTry1_GCC mode")
        addBias=myParams.myDict['CmplxBias']>0
        DataH=myParams.myDict['DataH']
        # we're [Batch, kH,kW,AllChannels*Neighbors*RI]
        model.add_Split4thDim(2) # Now [Batch, kH,kW,AllChannels*Neighbors,RI]

        model.add_Mult2DMCxCSharedOverFeat(DataH, 1) # Now [Batch, H,kW,AllChannels*Neighbors,RI]
        model.add_Split4thDim(6) # Now [Batch, H,kW,AllChannels,Neighbors,RI]

        ncc=4
        model.add_einsumC('abcde,bdx->abcxe',[DataH,8, ncc])

        model.add_Combine45(squeeze=True) # Now [Batch, H,kW,CompressedChannels*Neighbors,RI]
        model.add_Mult2DMCxCSharedOverFeat(DataH, 1) # Now [Batch, kH,kW,CompressedChannels*Neighbors,RI]

        model.add_PixelwiseMultC(2, stddev_factor=1.0)
        model.add_Combine34()
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASHTry1_GCCF':
        print("SMASHTry1_GCCF mode")
        addBias=myParams.myDict['CmplxBias']>0
        DataH=myParams.myDict['DataH']
        # we're [Batch, kH,kW,AllChannels*Neighbors*RI]
        model.add_Split4thDim(2) # Now [Batch, H,kW,AllChannels*Neighbors,RI]

        model.add_Split4thDim(6) # Now [Batch, H,kW,AllChannels,Neighbors,RI]

        ncc=4
        model.add_einsumC('abcde,bdx->abcxe',[DataH,8, ncc])

        model.add_Combine45(squeeze=True) # Now [Batch, H,kW,CompressedChannels*Neighbors,RI]

        DFTM=DFT_matrix(DataH)
        model.add_Mult2DMCxCSharedOverFeat(DataH, 1,add_bias=addBias,Trainable=False,InitC=DFTM) # Now [Batch, kH,kW,CompressedChannels*Neighbors,RI]

        model.add_PixelwiseMultC(2, stddev_factor=1.0)
        model.add_Combine34()
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars









    # SAE
    SAE = myParams.myDict['NetMode'] == 'SAE'
    if SAE:
        model.add_conv2d(64, mapsize=mapsize, stride=1, stddev_factor=2.)
        model.add_elu()
        model.add_conv2dWithName(128, name="AE", mapsize=mapsize, stride=1, stddev_factor=2.)

        model.add_conv2d(64, mapsize=mapsize, stride=1, stddev_factor=2.)   
        model.add_elu()
        model.add_conv2d(channels, mapsize=7, stride=1, stddev_factor=2.)  
        model.add_sigmoid()
        # model.add_tanh()

    # kKick:
    kKick= myParams.myDict['NetMode'] == 'kKick'
    if kKick:
        model.add_conv2d(64, mapsize=1, stride=1, stddev_factor=2.)   
        model.add_elu()
        b=np.array([[64,0,0,128,128,0,0,64],[128,0,0,256,256,0,0,128],[512,0,0,0,0,0,0,512]])
        model.add_UnetKsteps(b, mapsize=mapsize, stride=2, stddev_factor=1e-3)
        model.add_conv2dWithName(50, name="AE", mapsize=3, stride=1, stddev_factor=2.)
        model.add_elu()
        model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=2.)    

        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars



    # AUTOMAP
    # AUTOMAP_units  = [64, 64, channels]
    # AUTOMAP_mapsize  = [5, 5, 7]

    # ggg option 1: FC
    # model.add_flatten() # FC1
    # model.add_dense(num_units=H*W*2)
    # model.add_reshapeTo4D(H,W)

    TSRECON = myParams.myDict['NetMode'] == 'TSRECON'
    if TSRECON:
        # ggg option 2: FC per channel, and then dot multiplication per pixel, then conv
        ChannelsPerCoil=myParams.myDict['NumFeatPerChannel']
        NumTotalFeat=myParams.myDict['NumTotalFeat']
        model.add_Mult2DMC(H*W,ChannelsPerCoil)
        model.add_reshapeTo4D(H, W)
        model.add_PixelwiseMult(NumTotalFeat, stddev_factor=1.0)
        model.add_elu()


        #model.add_denseFromM('piMDR')
        #model.add_reshapeTo4D(FLAGS.LabelsH,FLAGS.LabelsW)
        # #model.add_tanh() # FC2

        #model.add_Unet1Step(128, mapsize=5, stride=2, num_layers=2, stddev_factor=1e-3)
        #model.add_conv2d(channels, mapsize=5, stride=1, stddev_factor=2.)

        b=np.array([[64,0,0,128,128,0,0,64],[128,0,0,256,256,0,0,128],[512,0,0,0,0,0,0,512]])
        #b=np.array([[64,0,0,128,128,0,0,64],[128,0,0,256,256,0,0,128]])
        #b=np.array([[64,0,0,0,0,0,0,64]])

        model.add_UnetKsteps(b, mapsize=mapsize, stride=2, stddev_factor=1e-3)
        # model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=2.)

        # ggg: Autoencode
        model.add_conv2dWithName(50, name="AE", mapsize=3, stride=1, stddev_factor=2.)
        model.add_elu()

        # ggg: Finish
        model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=2.)    

        # #model.add_flatten()
        # #model.add_dense(num_units=H*W*1)
        # model.add_reshapeTo4D(FLAGS.LabelsH,FLAGS.LabelsW)
        # #model.add_batch_norm()
        # #model.add_tanh() # TC3

        # # model.add_conv2d(AUTOMAP_units[0], mapsize=AUTOMAP_mapsize[0], stride=1, stddev_factor=2.)
        # # model.add_batch_norm()
        # #model.add_relu()

        # #model.add_conv2d(AUTOMAP_units[1], mapsize=AUTOMAP_mapsize[1], stride=1, stddev_factor=2.)
        # # model.add_batch_norm()
        # #model.add_relu()

        # #model.add_conv2d(AUTOMAP_units[2], mapsize=AUTOMAP_mapsize[2], stride=1, stddev_factor=2.)
        # # model.add_conv2d(AUTOMAP_units[2], mapsize=1, stride=1, stddev_factor=2.)
        # # model.add_relu()


        #model.add_constMatMul()
        #for ru in range(len(res_units)-1):
        #    nunits  = res_units[ru]

        #    for j in range(2):
        #        model.add_residual_block(nunits, mapsize=mapsize)

            # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
            # and transposed convolution
        #    model.add_upscale()
            
        #    model.add_batch_norm()
        #    model.add_relu()
        #    model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

        # model.add_flatten()
        # model.add_dense(num_units=H*W*4)
        # model.add_reshapeTo4D(FLAGS.LabelsH,FLAGS.LabelsW)

        # #model.add_Mult2D()
        # #model.add_Mult3DComplexRI()

    SrezOrigImagePartModel=False
    if SrezOrigImagePartModel:
        nunits  = res_units[0]
        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)
        #model.add_upscale()
        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

        nunits  = res_units[1]
        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)
        #model.add_upscale()
        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

        # Finalization a la "all convolutional net"
        nunits = res_units[-1]
        model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()

        model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()

        # Last layer is sigmoid with no batch normalization
        model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
        model.add_sigmoid()
    
    new_vars  = tf.global_variables()
    gene_vars = list(set(new_vars) - set(old_vars))

    # ggg = tf.identity(model.get_output(), name="ggg")

    return model.get_output(), gene_vars
