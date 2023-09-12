import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Conv3D, BatchNormalization, LeakyReLU, ZeroPadding2D, \
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
image_size = cfg.img_shape
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
    c = Activation(tf.keras.activations.relu)(c)
    return c

def SmallBlock_up(entrylayer, channels, kernelsize, stride, pad = 'same', activation = 'relu', number='1'):
    c = Conv2DTranspose(filters = channels,
                        kernel_size = kernelsize,
                        strides = stride,
                        padding=pad,
                        kernel_initializer = tf.keras.initializers.HeNormal(),
                        name = 'convTransp'+number)(entrylayer)
    c = BatchNormalization(fused = True,name = 'BN'+ number)(c)
    c = Activation(tf.keras.activations.relu)(c)
    return c

def ConvFeature(entry_layer):
    c = SmallBlock(entry_layer, nb_channels,(3,3),(2,2),pad='same',activation= 'relu', number='1_1')
    c = SmallBlock(c, nb_channels,(3,3),(1,1),pad='same',activation= 'relu', number='1_2')
    c = SmallBlock(c, nb_channels,(3,3),(1,1),pad='same',activation= 'relu', number='1_3')
    c1 = SmallBlock(c, nb_channels,(3,3),(1,1),pad='same',activation= 'relu', number='1_4')

    c = SmallBlock(c1, nb_channels*2,(3,3),(2,2),pad='same',activation= 'relu', number='2_1')
    c = SmallBlock(c , nb_channels*2,(3,3),(1,1),pad='same',activation= 'relu', number='2_2')
    c = SmallBlock(c , nb_channels*2,(3,3),(1,1),pad='same',activation= 'relu', number='2_3')
    c = SmallBlock(c , nb_channels*2,(3,3),(1,1),pad='same',activation= 'relu', number='2_4')
    c = SmallBlock(c , nb_channels*2,(3,3),(1,1),pad='same',activation= 'relu', number='2_5')
    c2 = SmallBlock(c, nb_channels*2,(3,3),(1,1),pad='same',activation= 'relu', number='2_6')

    c = SmallBlock(c2, nb_channels*4,(3,3),(2,2),pad='same',activation= 'relu', number='3_1')
    c = SmallBlock(c , nb_channels*4,(3,3),(1,1),pad='same',activation= 'relu', number='3_2')
    c = SmallBlock(c , nb_channels*4,(3,3),(1,1),pad='same',activation= 'relu', number='3_3')
    c = SmallBlock(c , nb_channels*4,(3,3),(1,1),pad='same',activation= 'relu', number='3_4')
    c = SmallBlock(c , nb_channels*4,(3,3),(1,1),pad='same',activation= 'relu', number='3_5')
    c3 = SmallBlock(c, nb_channels*4,(3,3),(1,1),pad='same',activation= 'relu', number='3_6')

    up1 = SmallBlock_up(c1, nb_channels*2,(3,3),(1,1),pad='same',activation= 'relu', number='4_1')

    up2 = SmallBlock_up(c2, nb_channels*2,(3,3),(2,2),pad='same',activation= 'relu', number='5_1')

    up3 = SmallBlock_up(c3, nb_channels*2,(3,3),(4,4),pad='same',activation= 'relu', number='6_1')

    conc = Concatenate(name = 'Concatenate1')([up1,up2,up3])
    return conc

def Head(entry_layer):

    # last_class = Conv2D(nb_anchors*nb_classes, (1, 1))(entry_layer)
    # last_class = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, nb_classes), name="F_clf")(last_class)

    # last_conf = Conv2D(nb_anchors, (1, 1), name="F_occ", activation="sigmoid")(entry_layer)
    
    # last_pos = Conv2D(nb_anchors * 3, (1, 1), name="loc/conv2d", activation="sigmoid")(entry_layer)
    # last_pos =  Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="F_pos")(last_pos)

    # last_dim = Conv2D(nb_anchors * 3, (1, 1), name="size/conv2d", activation="relu")(entry_layer)
    # last_dim = Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="F_dim")(last_dim)

    # last_rot = Conv2D(nb_anchors, (1, 1), name="F_rot" , activation="sigmoid")(entry_layer)


    last_class = Conv2D(nb_classes*nb_anchors, (1, 1))(entry_layer)
    last_class = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, nb_classes), name="F_clf")(last_class)

    last_conf = Conv2D(nb_anchors, (1, 1), name="occ", activation="sigmoid")(entry_layer)
    last_conf = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, 1), name="F_occ")(last_conf)

    last_pos = Conv2D(nb_anchors*3, (1, 1), name="loc/conv2d")(entry_layer)
    last_pos =  Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="F_pos")(last_pos)

    last_dim = Conv2D(nb_anchors*3, (1, 1), name="size/conv2d")(entry_layer)
    last_dim = Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="F_dim")(last_dim)

    last_rot = Conv2D(nb_anchors, (1, 1), name="rot")(entry_layer)
    last_rot = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, 1), name="F_rot")(last_rot)



    output = Concatenate()([last_conf,last_pos,last_dim,last_rot,last_class])
    # output = ([last_conf,last_pos,last_dim,last_rot,last_class])
    return output

def My_Model(input_pillar, input_pillar_mean):
    # Enter the pillar and pillar mean
    PF = PillarFeature(input_pillar,input_pillar_mean) # Output (?,img_size,img_size,64)

    # Convolutional Network feature
    CF = ConvFeature(PF)
    
    # Head detection
    out = Head(CF)
    
    return out


if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model

    input_pillar_shape = (12000, 100, 9)
    input_pillar_mean_shape = (12000, 3)

    input_pillar = Input(input_pillar_shape, batch_size=batch_size)
    input_pillar_mean = Input(input_pillar_mean_shape, batch_size=batch_size)

    output = My_Model(input_pillar, input_pillar_mean)
    model = Model(inputs=[input_pillar, input_pillar_mean], outputs=output)
    model.summary()