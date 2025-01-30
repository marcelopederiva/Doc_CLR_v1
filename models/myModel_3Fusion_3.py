import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, LeakyReLU, \
    Flatten, Reshape, MaxPool2D, \
    Conv2DTranspose,  \
    Add, Concatenate, Activation, Softmax, LayerNormalization
import numpy as np
import tensorflow.keras.backend as K
import sys
from pathlib import Path
from groupnorm import GroupNormalization
import time
# Get the directory of the script being run
current_dir = Path(__file__).resolve().parent

# Get the parent directory
parent_dir = current_dir.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import config_model as cfg

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import add, Activation
from tensorflow.keras.regularizers import l2

max_group_l = cfg.max_group_l
max_pillars_l = cfg.max_pillars_l
max_group_r = cfg.max_group_r
max_pillars_r = cfg.max_pillars_r
nb_channels = cfg.nb_channels
batch_size = cfg.BATCH_SIZE
image_size = cfg.image_shape
nb_anchors = cfg.nb_anchors
nb_classes = cfg.nb_classes


def check_numerics(layer, message):
    return tf.keras.layers.Lambda(lambda x: tf.debugging.check_numerics(x, message=message))(layer)


def PillarFeature(input_pillar,input_pillar_pos,max_pillars,max_group,sensor = 'lidar'):
    # PILLAR Network 
    if tf.keras.backend.image_data_format() == "channels_first":
        reshape_shape = (nb_channels, max_pillars)
    else:
        reshape_shape = (max_pillars, nb_channels)

    c = Conv2D(64,(1,1), activation = 'linear', use_bias = False, name = sensor + '_pillar/conv1')(input_pillar)
    c = BatchNormalization(fused = True, epsilon = 1e-3,momentum = 0.99, name = sensor+ '_pillar/BN1')(c)
    c = Activation('relu',name = sensor + '_pillar/RELU1')(c)
    c = MaxPool2D((1,max_group), name = sensor + '_pillar/Maxpool2D_1')(c)
    c = Reshape(reshape_shape, name = sensor + '_pillar/Reshape1')(c) # 1,10000,64
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
    if sensor == 'radar':
        div=4 #128,128
    else:
        div=1
    pillars = tf.keras.layers.Lambda(lambda inp: tf.scatter_nd(inp[0], inp[1],
                                                               ([batch_size,image_size[0]//div,image_size[1]//div,nb_channels])),
                                     name= sensor + "_pillars/scatter_nd")([corrected_indices, c])

    return pillars

def SmallBlock(entrylayer, channels, kernelsize, stride, pad='same', activation='relu', number='1'):
    # Depthwise Convolution
    c = DepthwiseConv2D(kernel_size=kernelsize, strides=stride, padding=pad, use_bias=False)(entrylayer)
    c = GroupNormalization(axis=-1,groups=c.shape[-1])(c)
    c = Activation(tf.keras.activations.relu)(c)

    # Pointwise Convolution
    c = Conv2D(filters=channels, kernel_size=(1, 1), use_bias=False)(c)
    c = BatchNormalization(fused=True)(c)
    c = Activation(tf.keras.activations.relu)(c)
    return c

def SmallBlock_up(entrylayer, channels, kernelsize, stride, pad = 'same', activation = 'relu', number='1'):
    c = Conv2DTranspose(filters = channels,
                        kernel_size = kernelsize,
                        strides = stride,
                        padding=pad,
                        kernel_initializer = tf.keras.initializers.HeNormal())(entrylayer)
    c = GroupNormalization(axis=-1,groups=c.shape[-1])(c)
    c = Activation(tf.keras.activations.swish)(c)
    return c

def SimpleConv2D(entry_layer, out_features, Ksize, stride=1, group=1, pad='same'):
    c = Conv2D(out_features,
               kernel_size=(Ksize, Ksize),
               strides=(stride, stride),
               groups=group,
               kernel_initializer = tf.keras.initializers.HeNormal(),
               padding=pad)(entry_layer)
    return c

def BottleNeckBlock(entry_layer, in_features, out_features, exp, stride=1, Ksize = 5):
    expanded_features = out_features * exp

    conv_seq = DepthwiseConv2D(kernel_size=Ksize,strides = stride, use_bias = False, padding = 'same')(entry_layer)

    conv_seq = GroupNormalization(axis=-1,groups=conv_seq.shape[-1])(conv_seq)

    conv_seq = SimpleConv2D(conv_seq, expanded_features, Ksize=1, stride=1)
    conv_seq = Activation('relu')(conv_seq)

    conv_seq = SimpleConv2D(conv_seq, out_features, Ksize=1, stride=1)

    #     conv_shortcut = SimpleConv2D(entry_layer,out_features, Ksize = 1,stride = stride)

    out = Add()([conv_seq, entry_layer])
    #     out = Activation('relu')(add)

    return out

def conv_block(entrylayer, channels_out, kernelsize, stride, pad):
    c = Conv2D(channels_out,kernel_size=kernelsize, strides=stride,
               kernel_initializer = tf.keras.initializers.HeNormal(),
               padding = pad, use_bias = False)(entrylayer)
    c = GroupNormalization(axis=-1, groups=c.shape[-1])(c)
    c = Activation('relu')(c)
    return c
def conv_block_up(entrylayer, channels, kernelsize, stride, pad = 'same'):
    c = Conv2DTranspose(filters = channels,
                        kernel_size = kernelsize,
                        strides = stride,
                        padding=pad,
                        kernel_initializer = tf.keras.initializers.HeNormal())(entrylayer)
    c = GroupNormalization(axis=-1, groups=c.shape[-1])(c)
    c = Activation('relu')(c)
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

def Img2BEV(img):
    # Input = 512x512x3

    # ----------------------------------------------------------------------------------------------------#

    c1 = SimpleConv2D(img, out_features=16,
                     Ksize=3, stride=2, group=1,  pad='same')
    # c1 = BottleNeckBlock(c1, 64, 64, 4, stride=1)
    # for i in range(1):
    #     c1 = BottleNeckBlock(c1, 32, 32, 2, stride=1)
    c1 = GroupNormalization(axis=-1, groups=c1.shape[-1])(c1)
    c1 = Activation('relu')(c1) # 256 x 256 x 16

    
    # c2 = LayerNormalization()(c1)
    
    c2 = SimpleConv2D(c1, out_features=24,
                     Ksize=3, stride=2, group=1, pad='same')
    c2 = GroupNormalization(axis=-1, groups=c2.shape[-1])(c2)
    c2 = Activation('relu')(c2) # 128 x 128 x 24

    c2 = BottleNeckBlock(c2, 24, 24, exp=4, stride=1,Ksize=3)

    

    c3 = LayerNormalization()(c2)
    c3 = SimpleConv2D(c3, out_features=40,
                     Ksize=3, stride=2, group=1, pad='same')
    c3 = GroupNormalization(axis=-1, groups=c3.shape[-1])(c3)
    c3 = Activation('relu')(c3) # 64 x 64 x 40
    # c3 = BottleNeckBlock(c3, 128, 128, 4, stride=1)

    for i in range(3):
        c3 = BottleNeckBlock(c3, 40, 40, 4, stride=1)


    c4 = LayerNormalization(axis=-1)(c3)
    c4 = SimpleConv2D(c4, out_features=48,
                     Ksize=3, stride=1, group=1, pad='same')
    # c4 = BottleNeckBlock(c4, 256, 256, 4, stride=1)
    c4 = GroupNormalization(axis=-1, groups=c4.shape[-1])(c4)
    c4 = Activation('relu')(c4) # 64 x 64 x 48
    for i in range(3):
        c4 = BottleNeckBlock(c4, 48, 48, 4, stride=1)


    c5 = LayerNormalization()(c4)
    c5 = SimpleConv2D(c5, out_features=96,
                     Ksize=3, stride=2, group=1, pad='same')
    c5 = GroupNormalization(axis=-1, groups=c5.shape[-1])(c5)
    c5 = Activation('relu')(c5) # 32 x 32 x 96

    # c5 = BottleNeckBlock(c5, 256, 256, 4, stride=1)

    for i in range(3):
        c5 = BottleNeckBlock(c5, 96, 96, 4, stride=1)

    # UP step
    # print(c1.shape)  
    # print(c2.shape)  
    # print(c3.shape)  
    # print(c4.shape)  
    # print(c5.shape)  
    
    c_up1 = conv_block_up(c5,96,(3,3),(2,2)) # 64x64
    conc1 = Concatenate()([c_up1,c4,c3]) # 64x64x96+48+40 = 184
    c_up2 = conv_block_up(conc1,conc1.shape[-1],(3,3),(2,2)) # 128x128
    conc2 = Concatenate()([c_up2,c2]) # 128x128x184+24 = 208
    c_up3 = conv_block_up(conc2,conc2.shape[-1],(3,3),(2,2)) # 256x256
    out = Concatenate()([c_up3,c1]) # 128x128x184+24 = 208

    return out
def Radar_Unet(entry_layer):
    c1 = SmallBlock(entry_layer, nb_channels,(3,3),(2,2),pad='same')
    c2 = SmallBlock(c1, nb_channels,(3,3),(2,2),pad='same')
    c3 = SmallBlock(c2, nb_channels*2,(3,3),(2,2),pad='same')
    c2_up = SmallBlock_up(c3, nb_channels,(3,3),(2,2),pad='same')
    c2_up_add = Add()([c2,c2_up])
    c1_up = SmallBlock_up(c2_up_add, nb_channels,(3,3),(2,2),pad='same')
    c1_up_add = Add()([c1,c1_up])

    c0_up = SmallBlock_up(c1_up_add, nb_channels,(3,3),(2,2),pad='same')
    c0_up_add = Add()([entry_layer,c0_up])

    out = SmallBlock_up(c0_up_add, nb_channels*2,(3,3),(2,2),pad='same')
    # out = check_numerics(out, 'NaN found after RADAR Unet')
    # print(c1.shape)
    # print(c2.shape)
    # print(c3.shape)
    # print(c2_up_add.shape)
    # print(c1_up_add.shape)
    # print(c0_up_add.shape)
    # print(out.shape)
    return out

def Head(entry_layer, out_rad):
    last_class = Conv2D(nb_anchors, (1, 1))(entry_layer)
    last_class = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, nb_classes), name="F_clf")(last_class)

    last_conf = Conv2D(128, (1, 1), padding = 'valid', name="occ1", kernel_initializer = tf.keras.initializers.HeNormal())(entry_layer)
    last_conf = GroupNormalization(axis=-1,groups=last_conf.shape[-1])(last_conf)
    last_conf = Activation('relu')(last_conf)
    last_conf = Conv2D(nb_anchors, (1, 1), name="occ2", activation="sigmoid")(last_conf)
    last_conf = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, 1), name="F_occ")(last_conf)

    last_pos = Conv2D(64, (1, 1), padding = 'valid', name="loc1", kernel_initializer = tf.keras.initializers.HeNormal())(entry_layer)
    last_pos = GroupNormalization(axis=-1, groups=last_pos.shape[-1])(last_pos)
    last_pos = Activation('relu')(last_pos)
    last_pos = Conv2D(nb_anchors*3, (1, 1), name="loc2")(last_pos)
    last_pos =  Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="F_pos")(last_pos)

    last_dim = Conv2D(64, (1, 1), padding = 'valid', name="size1",  kernel_initializer = tf.keras.initializers.HeNormal())(entry_layer)
    last_dim = GroupNormalization(axis=-1, groups= last_dim.shape[-1])(last_dim)
    last_dim = Activation('relu')(last_dim)
    last_dim = Conv2D(nb_anchors*3, (1, 1), name="size2")(last_dim)
    last_dim = Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="F_dim")(last_dim)

    last_rot = Conv2D(16, (1, 1), padding = 'valid', name="rot1", kernel_initializer = tf.keras.initializers.HeNormal())(entry_layer)
    last_rot = GroupNormalization(axis=-1, groups= last_rot.shape[-1])(last_rot)
    last_rot = Activation('relu')(last_rot)
    last_rot = Conv2D(nb_anchors, (1, 1), name="rot2")(last_rot)
    last_rot = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, 1), name="F_rot")(last_rot)



    output = Concatenate()([last_conf,last_pos,last_dim,last_rot,out_rad,last_class])

    return output

def Head_radar(entry_layer):
    last_vel = Conv2D(32, (3,3), padding = 'same', name="vel1",  kernel_initializer = tf.keras.initializers.HeNormal())(entry_layer)
    last_vel = GroupNormalization(axis=-1,groups=last_vel.shape[-1])(last_vel)
    last_vel = Activation('relu')(last_vel)
    last_vel = Conv2D(nb_anchors*2, (1, 1), name="vel2", activation="sigmoid")(last_vel)
    last_vel =  Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 2), name="F_vel")(last_vel)

    return last_vel



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


def My_Model(input_pillar_l, input_pillar_pos_l,input_pillar_r, input_pillar_pos_r,  input_img):
    # LIDAR FEATURES
    Flidar = PillarFeature(input_pillar_l,input_pillar_pos_l, max_pillars_l, max_group_l, sensor = 'lidar') # Output (?,512,512,64)
    
    # CAMEAR FEATURES
    Fimg = Img2BEV(input_img)# Output (?,256,256,64)
    # RADAR FEATURES
    Fradar = PillarFeature(input_pillar_r,input_pillar_pos_r, max_pillars_r, max_group_r, sensor = 'radar') # Output (?,512,512,64)
    
    
    c_lidar = ConvFeature(Flidar)#256
    c_lidar = SmallBlock(c_lidar, nb_channels*2,(3,3),(2,2),pad='same', number='_c_LIDAR2')#128
    c_img = SmallBlock(Fimg, nb_channels,(3,3),(2,2),pad='same', number='_c_img')#128
    # c_img2 = SmallBlock(c_img1, nb_channels,(3,3),(2,2),pad='same', number='_c_img2')#128

    # print('LIDARF shape: ', c_lidar.shape)
    # print('ImageF shape: ', c_img.shape)
    # print('RADARF shape: ', Fradar.shape)
    # exit()

    # CrAtt1 = CrossAtt(c_lidar2,c_img2,number = '1_LIDAR_cam')
    # add1 = Add(name = 'Add_CrAtt_FLIDAR1')([c_lidar2,CrAtt1])#64
    # CrAtt2 = CrossAtt(c_lidar2,c_img2,number = '2_LIDAR_cam')
    # add2 = Add(name = 'Add_CrAtt_FLIDAR2')([c_lidar2,CrAtt2])#64
    # CrAtt3 = CrossAtt(c_lidar2,c_img2,number = '3_LIDAR_cam')
    # add3 = Add(name = 'Add_CrAtt_FLIDAR3')([c_lidar2,CrAtt3])#64
    # CrAtt4 = CrossAtt(c_lidar2,c_img2,number = '4_LIDAR_cam')
    # add4 = Add(name = 'Add_CrAtt_FLIDAR4')([c_lidar2,CrAtt4])#64
    # CrAtt5 = CrossAtt(c_lidar2,c_img2,number = '5_LIDAR_cam')
    # add5 = Add(name = 'Add_CrAtt_FLIDAR5')([c_lidar2,CrAtt5])#64
    # CrAtt6 = CrossAtt(c_lidar2,c_img2,number = '6_LIDAR_cam')
    # add6 = Add(name = 'Add_CrAtt_FLIDAR6')([c_lidar2,CrAtt6])#64

    conc = Concatenate()([c_img,c_lidar]) 

    Add_up = SmallBlock_up(conc, nb_channels*2,(3,3),(2,2),pad='same', number='_up_1_LIDAR_cam') #?, 256, 256, 128

    
    c_radar = Radar_Unet(Fradar)

    LC_plus_R = Concatenate()([Add_up,c_radar])
    LC_plus_R = SimpleConv2D(LC_plus_R, out_features=128, Ksize=1, stride=1, group=1, pad='same')
    LCRadar = Add()([c_radar,LC_plus_R])
    # LCRadar = check_numerics(LCRadar, 'NaN found after LCRadar')
    out_rad = Head_radar(LCRadar)
    # out_rad = check_numerics(out_rad, 'NaN found after out_rad')
    # Head detection
    out = Head(Add_up,out_rad)
    
    return out

# -----------------

if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model
    from tensorflow.python.client import timeline
    import time
    tf.compat.v1.disable_eager_execution()
    input_pillar_l_shape = (12000, 100, 9)
    input_pillar_pos_l_shape = (12000, 3)

    input_pillar_r_shape = (100, 10, 4)
    input_pillar_pos_r_shape = (100, 3)

    input_img_shape = (512,512,3)


    input_pillar_l = Input(input_pillar_l_shape, batch_size=batch_size)
    input_pillar_pos_l = Input(input_pillar_pos_l_shape, batch_size=batch_size)

    input_pillar_r = Input(input_pillar_r_shape, batch_size=batch_size)
    input_pillar_pos_r = Input(input_pillar_pos_r_shape, batch_size=batch_size)

    input_img = Input(input_img_shape,batch_size = batch_size)

    output = My_Model(input_pillar_l, input_pillar_pos_l,input_pillar_r, input_pillar_pos_r, input_img)
    model = Model(inputs=[input_pillar_l, input_pillar_pos_l,input_pillar_r, input_pillar_pos_r,input_img], outputs=output)
    
    batch_size = 1

    # Generate random input data
    # input_pillar_l = np.random.rand(batch_size, *input_pillar_l_shape).astype(np.float32)
    # input_pillar_pos_l = np.random.rand(batch_size, *input_pillar_pos_l_shape).astype(np.float32)

    # input_pillar_r = np.random.rand(batch_size, *input_pillar_r_shape).astype(np.float32)
    # input_pillar_pos_r = np.random.rand(batch_size, *input_pillar_pos_r_shape).astype(np.float32)

    # input_img = np.random.rand(batch_size, *input_img_shape).astype(np.float32)
    # input_shapes = [
    #     (batch_size, *input_pillar_l_shape), 
    #     (batch_size, *input_pillar_pos_l_shape),
    #     (batch_size, *input_pillar_r_shape),
    #     (batch_size, *input_pillar_pos_r_shape),
    #     (batch_size, *input_img_shape)
    # ]


    model.summary()