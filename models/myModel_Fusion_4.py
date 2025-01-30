import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Conv1D, BatchNormalization, LeakyReLU, ZeroPadding2D, \
    UpSampling2D, Flatten, Dense, Reshape, MaxPool2D, UpSampling3D, \
    Conv2DTranspose, Dropout, SpatialDropout3D, \
    Add, Concatenate, Activation, Softmax, LayerNormalization
import numpy as np
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import activations
from tensorflow.keras.applications import ResNet50, MobileNetV2, InceptionV3
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras import activations

import config_model as cfg
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import add, Activation
from tensorflow.keras.regularizers import l2

max_group = cfg.max_group
max_pillars = cfg.max_pillars
nb_channels = cfg.nb_channels
batch_size = cfg.BATCH_SIZE
image_size = cfg.image_shape
nb_anchors = cfg.nb_anchors
nb_classes = cfg.nb_classes

def PillarFeature(input_pillar,input_pillar_pos):
    # PILLAR Network 
    if tf.keras.backend.image_data_format() == "channels_first":
        reshape_shape = (nb_channels, max_pillars)
    else:
        reshape_shape = (max_pillars, nb_channels)

    c = Conv2D(64,(1,1), activation = 'linear', use_bias = False, name = 'pillar/conv1')(input_pillar)
    c = BatchNormalization(fused = True, epsilon = 1e-3,momentum = 0.99, name = 'pillar/BN1')(c)
    c = Activation('relu',name = 'pillar/RELU1')(c)
    c = MaxPool2D((1,max_group), name = 'pillar/Maxpool2D_1')(c)
    c = Reshape(reshape_shape, name = 'pillar/Reshape1')(c) # 1,10000,64
    input_pillar_pos = tf.cast(input_pillar_pos,dtype=tf.int32)

    def correct_batch_indices(tensor, batch_size):
        array = np.zeros((batch_size, max_pillars, 3), dtype=np.float32)
        for i in range(batch_size):
            array[i, :, 0] = i
        return tensor + tf.constant(array, dtype=tf.int32)

    if batch_size > 1:
            corrected_indices = tf.keras.layers.Lambda(lambda t: correct_batch_indices(t, batch_size))(input_pillar_pos)
    else:
        corrected_indices = input_pillar_pos  

    pillars = tf.keras.layers.Lambda(lambda inp: tf.scatter_nd(inp[0], inp[1],
                                                               ([batch_size,image_size[0],image_size[1],nb_channels])),
                                     name="pillars/scatter_nd")([corrected_indices, c])

    return pillars

def SmallBlock(entrylayer, channels, kernelsize, stride, pad = 'same', activation = 'relu', number='1'):
    c = Conv2D(filters = channels,
                kernel_size = kernelsize,
                strides = stride,
                padding=pad,
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'conv'+number)(entrylayer)
    c = BatchNormalization(fused = True,name = 'BN'+ number)(c)
    c = Activation(tf.keras.activations.swish)(c)
    return c

def SmallBlock_up(entrylayer, channels, kernelsize, stride, pad = 'same', activation = 'relu', number='1'):
    c = Conv2DTranspose(filters = channels,
                        kernel_size = kernelsize,
                        strides = stride,
                        padding=pad,
                        kernel_initializer = tf.keras.initializers.HeNormal(),
                        name = 'convTransp'+number)(entrylayer)
    c = BatchNormalization(fused = True,name = 'BN'+ number)(c)
    c = Activation(tf.keras.activations.swish)(c)
    return c

def ConvFeature(entry_layer):
    c = SmallBlock(entry_layer, nb_channels,(3,3),(2,2),pad='same',activation= 'relu', number='1_1')
    c1 = SmallBlock(c, nb_channels,(3,3),(1,1),pad='same',activation= 'relu', number='1_4')

    c = SmallBlock(c1, nb_channels*2,(3,3),(2,2),pad='same',activation= 'relu', number='2_1')
    c2 = SmallBlock(c, nb_channels*2,(3,3),(1,1),pad='same',activation= 'relu', number='2_6')

    c = SmallBlock(c2, nb_channels*4,(3,3),(2,2),pad='same',activation= 'relu', number='3_1')
    c3 = SmallBlock(c, nb_channels*4,(3,3),(1,1),pad='same',activation= 'relu', number='3_6')

    up1 = SmallBlock_up(c1, nb_channels*2,(3,3),(1,1),pad='same',activation= 'relu', number='4_1')

    up2 = SmallBlock_up(c2, nb_channels*2,(3,3),(2,2),pad='same',activation= 'relu', number='5_1')

    up3 = SmallBlock_up(c3, nb_channels*2,(3,3),(4,4),pad='same',activation= 'relu', number='6_1')

    conc = Concatenate(name = 'Concatenate1')([up1,up2,up3])
    return conc

def Head(entry_layer):

    last_class = Conv2D(nb_classes*nb_anchors, (1, 1))(entry_layer)
    last_class = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, nb_classes), name="F_clf")(last_class)

    last_conf = Conv2D(32, (3,3), padding = 'same', name="occ1", activation="linear")(entry_layer)
    last_conf = Conv2D(nb_anchors, (1, 1), name="occ2", activation="sigmoid")(last_conf)
    last_conf = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, 1), name="F_occ")(last_conf)

    last_pos = Conv2D(32, (3,3), padding = 'same', name="loc1", activation="linear")(entry_layer)
    last_pos = Conv2D(nb_anchors*3, (1, 1), name="loc2")(last_pos)
    last_pos =  Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="F_pos")(last_pos)

    last_dim = Conv2D(32, (3,3), padding = 'same', name="size1", activation="linear")(entry_layer)
    last_dim = Conv2D(nb_anchors*3, (1, 1), name="size2")(last_dim)
    last_dim = Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="F_dim")(last_dim)

    last_rot = Conv2D(32, (3,3), padding = 'same', name="rot1", activation="linear")(entry_layer)
    last_rot = Conv2D(nb_anchors, (1, 1), name="rot2")(last_rot)
    last_rot = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, 1), name="F_rot")(last_rot)



    output = Concatenate()([last_conf,last_pos,last_dim,last_rot,last_class])

    return output

def Img2BEV(img):
    # Input = 512x512x3

    # ----------------------------------------------------------------------------------------------------#

    c1 = Conv2D(filters = 188,
                kernel_size = (3,3),
                strides = (1,1),
                padding='same',
                # activation = 'relu',
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'IMG_decay_1')(img)
    #c1 = 504,504
    c1 = BatchNormalization()(c1)
    c1 = Activation(tf.keras.activations.swish)(c1)
    
    Reshape_c1 = Reshape((512,188,512), name = 'Out_c1')(c1)

    out_c1 = Conv2D(filters = 64,
                kernel_size = (3,3),
                strides = (1,1),
                padding='same',
                # activation = 'relu',
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'IMG_decay_1_1')(Reshape_c1)
    
    out_c1 = BatchNormalization()(out_c1)
    out_c1 = Activation(tf.keras.activations.swish)(out_c1)
    # Out_VEC_c1 = (188,512,64) ------ 50-80

    # ----------------------------------------------------------------------------------------------------#

    c2 = Conv2D(filters = 126,
                kernel_size = (3,3),
                strides = (2,2),
                padding='same',
                # activation = 'relu',
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'IMG_decay_2')(c1)
    #c1 = 256,256
    c2 = BatchNormalization()(c2)
    c2 = Activation(tf.keras.activations.swish)(c2)

    Reshape_c2 = Reshape((512,126,128), name = 'Out_c2')(c2)

    out_c2 = Conv2D(filters = 64,
                kernel_size = (3,3),
                strides = (1,1),
                padding='same',
                # activation = 'relu',
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'IMG_decay_2_1')(Reshape_c2)
    
    out_c2 = BatchNormalization()(out_c2)
    out_c2 = Activation(tf.keras.activations.swish)(out_c2)
    # Out_VEC_c2 = (126,512,64) ------ 30-50

    # ----------------------------------------------------------------------------------------------------#

    c3 = Conv2D(filters = 72,
                kernel_size = (3,3),
                strides = (2,2),
                padding='same',
                # activation = 'relu',
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'IMG_decay_3')(c2)
    #c3 = 128,128

    c3 = BatchNormalization()(c3)
    c3 = Activation(tf.keras.activations.swish)(c3)

    Reshape_c3 = Reshape((512,72,32), name = 'Out_c3')(c3)

    out_c3 = Conv2D(filters = 64,
                kernel_size = (3,3),
                strides = (1,1),
                padding='same',
                # activation = 'relu',
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'IMG_decay_3_1')(Reshape_c3)

    out_c3 = BatchNormalization()(out_c3)
    out_c3 = Activation(tf.keras.activations.swish)(out_c3)
    # Out_VEC_c3 = (76,512,64) ------ 20-30

    # ----------------------------------------------------------------------------------------------------#

    c4 = Conv2D(filters = 66,
                kernel_size = (3,3),
                strides = (2,2),
                padding='same',
                # activation = 'relu',
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'IMG_decay_4')(c3)
    #c1 = 64,64

    c4 = BatchNormalization()(c4)
    c4 = Activation(tf.keras.activations.swish)(c4)

    Reshape_c4 = Reshape((512,66,8), name = 'Out_c4')(c4)

    out_c4 = Conv2D(filters = 64,
                kernel_size = (3,3),
                strides = (1,1),
                padding='same',
                # activation = 'relu',
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'IMG_decay_4_1')(Reshape_c4)

    out_c4 = BatchNormalization()(out_c4)
    out_c4 = Activation(tf.keras.activations.swish)(out_c4)

    # Out_VEC_c4 = (64,512,64) ------ 10-30

    # ----------------------------------------------------------------------------------------------------#

    c5 = Conv2D(filters = 240,
                kernel_size = (3,3),
                strides = (2,2),
                padding='same',
                # activation = 'relu',
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'IMG_decay_5')(c4)
    #c1 = 32,32

    c5 = BatchNormalization()(c5)
    c5 = Activation(tf.keras.activations.swish)(c5)

    Reshape_c5 = Reshape((512,60,8), name = 'Out_c5')(c5)

    out_c5 = Conv2D(filters = 64,
                kernel_size = (3,3),
                strides = (1,1),
                padding='same',
                # activation = 'relu',
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = 'IMG_decay_5_1')(Reshape_c5)

    out_c5 = BatchNormalization()(out_c5)
    out_c5 = Activation(tf.keras.activations.swish)(out_c5)
    
    # Out_VEC_c5 = (60,512,64) ------ 00-10

    # ----------------------------------------------------------------------------------------------------#

    out = Concatenate(axis = 2)([out_c5,out_c4,out_c3,out_c2,out_c1])

    return out

def CrossAtt(Flidar,Fimg,number = '0'):
    # Reshape_Flidar = Reshape(((image_size[0]//4)*(image_size[0]//4),64), name ='Reshape_Flidar')(Flidar)
    # Reshape_Fimg = Reshape(((image_size[0]//4)*(image_size[0]//4),64), name ='Reshape_Fimg')(Fimg)

    # Self-Attention

    Query= Conv2D(64,(1,1), activation = 'linear', use_bias = False, name = 'SA/Query'+ number)(Flidar)
    Key= Conv2D(64,(1,1), activation = 'linear', use_bias = False, name = 'SA/Key'+ number)(Fimg)
    Value= Conv2D(64,(1,1), activation = 'linear', use_bias = False, name = 'SA/Value'+ number)(Fimg)

    out = tf.matmul(Query,tf.transpose(Key,perm=[0,2,3,1]))
    out = out/np.sqrt(image_size[0]//4)
    out = tf.keras.activations.softmax(out, axis = -1)
    out = tf.matmul(out,Value)
    out = Conv2D(64,(1,1), activation = 'linear', use_bias = False, name = 'SA/out'+ number)(out)
    # out = Reshape((128,128,64), name = 'Reshape_out_CrAtt')(out)
    return out


def My_Model(input_pillar, input_pillar_mean, input_img):
    # Enter the pillar and pillar mean
    Flidar = PillarFeature(input_pillar,input_pillar_mean) # Output (?,img_size,img_size,64)

    # Convolutional Network feature
    Fimg = Img2BEV(input_img)
    
    c_lidar1 = ConvFeature(Flidar)#256
    c_lidar2 = SmallBlock(c_lidar1, nb_channels,(3,3),(2,2),pad='same', number='_c_lidar2')#128
    c_img1 = SmallBlock(Fimg, nb_channels,(3,3),(2,2),pad='same', number='_c_img')#256
    c_img2 = SmallBlock(c_img1, nb_channels,(3,3),(2,2),pad='same', number='_c_img2')#128




    CrAtt1 = CrossAtt(c_lidar2,c_img2,number = '1')
    add1 = Add(name = 'Add_CrAtt_Flidar1')([c_lidar2,CrAtt1])#64
    CrAtt2 = CrossAtt(c_lidar2,c_img2,number = '2')
    add2 = Add(name = 'Add_CrAtt_Flidar2')([c_lidar2,CrAtt2])#64
    CrAtt3 = CrossAtt(c_lidar2,c_img2,number = '3')
    add3 = Add(name = 'Add_CrAtt_Flidar3')([c_lidar2,CrAtt3])#64
    CrAtt4 = CrossAtt(c_lidar2,c_img2,number = '4')
    add4 = Add(name = 'Add_CrAtt_Flidar4')([c_lidar2,CrAtt4])#64
    CrAtt5 = CrossAtt(c_lidar2,c_img2,number = '5')
    add5 = Add(name = 'Add_CrAtt_Flidar5')([c_lidar2,CrAtt5])#64
    CrAtt6 = CrossAtt(c_lidar2,c_img2,number = '6')
    add6 = Add(name = 'Add_CrAtt_Flidar6')([c_lidar2,CrAtt6])#64

    conc = Concatenate()([add1,add2,add3,add4,add5,add6]) #384




    Add_up = SmallBlock_up(conc, nb_channels*2,(3,3),(2,2),pad='same', number='_up_1')

    # conc = Add(name = 'Add_CrAtt_Flidar')([c_lidar1,A_up])

    # c = ConvFeature(Fimg)
    # c = SmallBlock_up(conc, nb_channels*2,(3,3),(2,2),pad='same',activation= 'relu', number='up_1')
    # c = SmallBlock(Add_up, nb_channels*2,(3,3),(1,1),pad='same',activation= 'relu', number='2')

    # Head detection
    out = Head(Add_up)
    out = tf.cast(out, tf.float32)
    return out


if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model

    input_pillar_shape = (12000, 100, 9)
    input_pillar_mean_shape = (12000, 3)
    input_img_shape = (512,512,3)


    input_pillar = Input(input_pillar_shape, batch_size=batch_size)
    input_pillar_mean = Input(input_pillar_mean_shape, batch_size=batch_size)
    input_img = Input(input_img_shape,batch_size = batch_size)

    output = My_Model(input_pillar, input_pillar_mean,input_img)
    model = Model(inputs=[input_pillar, input_pillar_mean,input_img], outputs=output)
    model.summary()
    # tf.keras.utils.plot_model(
    # model,
    # to_file="D:/SCRIPTS/Doc_lidar_v2/models/model2.png",
    # show_shapes=True,
    # dpi = 300)