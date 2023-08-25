import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow.keras.backend as K
import config_model as cfg
import tensorflow as tf
import numpy as np
# import tensorflow_probability as tfp
# import tensorflow_addons as tfa
# from utils2 import intersection
# from utils3 import iou3d
Sx = cfg.X_div
Sy = cfg.Y_div
Sz = cfg.Z_div

# lambda_iou = cfg.lambda_iou
# lambda_rot = cfg.lambda_rot
# lambda_obj = cfg.lambda_obj
# lambda_noobj = cfg.lambda_noobj
# lambda_class = cfg.lambda_class

x_min = cfg.x_min
x_max = cfg.x_max
x_diff = cfg.x_diff

y_min = cfg.y_min
y_max = cfg.y_max
y_diff = cfg.y_diff

z_min = cfg.z_min
z_max = cfg.z_max
z_diff = cfg.z_diff

rot_norm = cfg.rot_norm

lambda_class = cfg.lambda_class
lambda_occ = cfg.lambda_occ
lambda_pos = cfg.lambda_pos
lambda_dim = cfg.lambda_dim
lambda_rot = cfg.lambda_rot
# Smooth_L1 = tf.keras.losses.Huber(reduction = 'none')

def Smooth_L1(label,pos):
    delta = 1.0
    diff = tf.abs(label - pos)
    lower_mask = K.cast(diff <= delta,  dtype = tf.float32)
    higher_mask = K.cast(diff > delta,  dtype = tf.float32)
    
    lower = lower_mask*(tf.pow(diff,2)/2)
    higher = higher_mask*(delta * (diff - delta/2))

    out = lower+higher

    return out

def focal_loss(y_true, y_pred):
    """ y_true value from occ in {-1, 0, 1}, i.e. {bad match, neg box, pos box} """
    # print('True:', y_true.shape,'\n')
    # print('Pred:', y_pred.shape)
    y_true = y_true[...,0]
    y_pred = y_pred[...,0]

    mask_ = tf.equal(y_true, 1)
    pos_mask = tf.cast(mask_, dtype = tf.float32)
    Npos = tf.clip_by_value(tf.reduce_sum(pos_mask),1,1e+15)

    neg_mask = tf.equal(y_true, 0)
    neg_mask = tf.cast(neg_mask, dtype = tf.float32)
    # Nneg = tf.clip_by_value(tf.reduce_sum(neg_mask),1,1e+15)

    gamma = 2
    alpha = 0.75
    
    # obj_loss = - alpha * K.pow( (1 - y_pred), gamma) * K.log(tf.clip_by_value(y_pred,1e-15,1.0))
    # noobj_loss = - (1 - alpha)* K.pow((y_pred),gamma)* K.log(tf.clip_by_value((1 - y_pred), 1e-15, 1.0))

    # obj_loss_pos = pos_mask* tf.clip_by_value(obj_loss,1e-15,1e+15)
    # obj_loss_neg = neg_mask * tf.clip_by_value(noobj_loss,1e-15,1e+15)

    # object_loss = ((tf.reduce_sum(obj_loss_pos)) + (tf.reduce_sum(obj_loss_neg)))/Npos

    ce = K.binary_crossentropy(y_true, y_pred, from_logits=False)

    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = tf.pow((1.0 - p_t), gamma)

    object_loss = tf.reduce_sum(alpha_factor * modulating_factor * ce)/Npos
    object_loss = tf.clip_by_value(object_loss,0,1e+15)

    # fl = tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred, alpha=alpha, gamma=gamma)
    # # focal_loss = fl(y_true, y_pred)
    # # print(focal_loss)
    # object_loss = tf.reduce_sum(fl)/Npos

    return (lambda_occ * object_loss)


def loc_loss(y_true, y_pred):
    
    
    mask = y_true[...,0]
    # print(y_true)
    # print(mask.shape)
    # exit()
    mask_loc = tf.tile(tf.expand_dims(mask, -1), [1, 1, 1, 1, 3])
    # print(mask.shape)
    # print(mask)

    Npos = tf.clip_by_value(tf.reduce_sum(mask),1,1e+15)
    y_true = y_true[...,1:4]
    y_pred = y_pred[...,1:4]

    # mask_loc = tf.cast(mask, dtype = tf.float32)
    # real_true = pos_to_real(y_true)
    # real_pred = pos_to_real(y_pred)

    # loss = tf.compat.v1.losses.huber_loss(y_true,
    #                             y_pred,
    #                             reduction="none")
    # print(loss.shape)
    loss = Smooth_L1(y_true,y_pred)
    # 
    # print(loss.shape)
    # exit()
    # print(mask_loc)
    # print(np.multiply(mask_loc,loss))
    # exit()
    loc_loss = tf.math.multiply(mask_loc, loss)
    # print(loc_loss)
    
    # print(loc_loss)
    # exit()
    lloss = tf.reduce_sum(loc_loss)/(Npos)

    return lambda_pos * lloss

def size_loss(y_true, y_pred):
    # mask_size = tf.tile(tf.expand_dims(mask, -1), [1, 1, 1, 1, 3])
    # mask_size = tf.cast(mask_size, dtype = tf.float32)
    # real_true = dim_to_real(y_true)
    # real_pred = dim_to_real(y_pred)
    mask = y_true[...,0]
    mask_size = tf.tile(tf.expand_dims(mask, -1), [1, 1, 1, 1, 3])
    Npos = tf.clip_by_value(tf.reduce_sum(mask),1,1e+15)
    y_true = y_true[...,4:7]
    y_pred = y_pred[...,4:7]

    # loss = tf.compat.v1.losses.huber_loss(y_true,
    #                             y_pred,
    #                             reduction="none")

    loss = Smooth_L1(y_true,y_pred)
    
    size_loss = tf.math.multiply(mask_size , loss)
    sloss = tf.reduce_sum(size_loss)/(Npos)

    return lambda_dim * sloss

def angle_loss(y_true, y_pred):
    # mask_angle = tf.cast(mask, dtype = tf.float32)
    mask = y_true[...,0]
    Npos = tf.clip_by_value(tf.reduce_sum(mask),1,1e+15)
    y_true = y_true[...,7]
    y_pred = y_pred[...,7]

    # real_true = rot_to_real(y_true)
    # real_pred = rot_to_real(y_pred)
    # loss = tf.compat.v1.losses.huber_loss(y_true,
    #                             y_pred,
    #                             reduction="none")

    loss = Smooth_L1(y_true,y_pred)

    angle_loss = tf.math.multiply(mask , loss)
    aloss = tf.reduce_sum(angle_loss)/(Npos)

    return lambda_rot * aloss

def class_loss(y_true, y_pred):
    mask = y_true[...,0]

    Npos = tf.clip_by_value(tf.reduce_sum(mask),1,1e+15)
    y_true = y_true[...,8]
    y_pred = y_pred[...,8]
    mask_ = tf.equal(mask, 1)
    mask_class = tf.cast(tf.math.logical_or(mask_[...,0],mask_[...,1]), dtype = tf.float32)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    # print(mask_class)
    # print(loss)
    # exit()
    class_loss = tf.math.multiply(mask_class , loss)
    cls_ = tf.reduce_sum(class_loss)/(Npos)

    return lambda_class * cls_


def PointPillarNetworkLoss(y_true,y_pred):

    focal_loss_ = focal_loss(y_true,y_pred)
    loc_loss_ = loc_loss(y_true,y_pred)
    size_loss_ = size_loss(y_true,y_pred)
    angle_loss_ = angle_loss(y_true,y_pred)
    class_loss_ = class_loss(y_true,y_pred)
    # Testing
    # print(focal_loss_)
    # print(loc_loss_)
    # print(size_loss_)
    # print(angle_loss_)
    # print(class_loss_)

    # exit()
    total_loss = focal_loss_+ loc_loss_+ size_loss_+ angle_loss_+ class_loss_

    return total_loss

if __name__=='__main__':
    import numpy as np
    
    occ_t =K.constant([[[[[0],[0]],[[0],[0]],[[0],[0]]],
                       [[[0],[0]],[[0],[0]],[[0],[1]]],
                       [[[0],[0]],[[0],[0]],[[0],[0]]]]])
    
    pos_t = K.constant([[[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]],
                       [[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]],
                       [[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]]]])
    
    size_t = K.constant([[[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]],
                       [[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]],
                       [[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]]]])
    
    angle_t = K.constant([[[[[0],[0]],[[0],[0]],[[0],[0]]],
                       [[[0],[0]],[[0],[0]],[[0],[1]]],
                       [[[0],[0]],[[0],[0]],[[0],[0]]]]])
    
    class_t = K.constant([[[[[0],[0]],[[0],[0]],[[0],[0]]],
                       [[[0],[0]],[[0],[0]],[[0],[1]]],
                       [[[0],[0]],[[0],[0]],[[0],[0]]]]])
    # ------------------------------------------------------------------------------------------
    occ_p = K.constant([[[[[0],[0]],[[0],[0]],[[0],[0]]],
                       [[[0],[0]],[[0],[0]],[[0],[1]]],
                       [[[0],[0]],[[0],[0]],[[0],[0]]]]])
    
    pos_p = K.constant([[[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]],
                       [[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[-0.5,-0.5,-0.5]]],
                       [[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]]]])
    
    size_p = K.constant([[[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]],
                       [[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]],
                       [[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]]]])
    
    angle_p = K.constant([[[[[0],[0]],[[0],[0]],[[0],[0]]],
                       [[[0],[0]],[[0],[0]],[[0],[1]]],
                       [[[0],[0]],[[0],[0]],[[0],[0]]]]])
    
    class_p = K.constant([[[[[0],[0]],[[0],[0]],[[0],[0]]],
                       [[[0],[0]],[[0],[0]],[[0],[10]]],
                       [[[0],[0]],[[0],[0]],[[0],[0]]]]])
    

    # occ_t = K.constant(occ_t)
    
    pred = K.concatenate([occ_p,pos_p,size_p,angle_p,class_p])
    true = K.concatenate([occ_t,pos_t,size_t,angle_t,class_t])
    # print(pred.shape)
    # for x in true: print(x.shape)
    # 
    # print(pred)
    # exit()
    print('\nLoss Score: ', K.eval(PointPillarNetworkLoss(true, pred)))
    # exit()
