import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow.keras.backend as K
import config_model as cfg
import tensorflow as tf
import tensorflow_probability as tfp
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

##################### TO REAL ################################

def pos_to_real(detect):
    # real_dtc = detect/3

    x_real = (detect[..., 0:1] * (x_diff)) - abs(x_min)
    y_real = (detect[..., 1:2] * (y_diff)) - abs(y_min)
    z_real = (detect[..., 2:3] * (z_diff)) - abs(z_min)

    return K.concatenate([x_real, y_real, z_real])


def dim_to_real(detect):
    w_real = detect[..., 0:1] * (x_diff)
    h_real = detect[..., 1:2] * (y_diff)
    l_real = detect[..., 2:3] * (z_diff)

    return K.concatenate([w_real, h_real, l_real])


def rot_to_real(detect):
    rot_real = detect * (rot_norm)

    return rot_real



class PointPillarNetworkLoss:

    def __init__(self):
        self.alpha = 0.25
        self.gamma = 2
        self.focal_weight = cfg.lambda_occ
        self.loc_weight = cfg.lambda_pos
        self.size_weight = cfg.lambda_dim
        self.angle_weight = cfg.lambda_rot
        self.class_weight = cfg.lambda_class
        # self.bce = tf.keras.losses.BinaryCrossentropy(reduction = 'none')
        # self.Smooth_L1 = tf.keras.losses.Huber(delta = 1/9,
        #                                        reduction = 'none')

    def losses(self):
        return [self.focal_loss, self.loc_loss, self.size_loss, self.angle_loss, self.class_loss]
    
    

    def focal_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """ y_true value from occ in {-1, 0, 1}, i.e. {bad match, neg box, pos box} """
        # print('True:', y_true.shape,'\n')
        # print('Pred:', y_pred.shape)

        self.mask_ = tf.equal(y_true, 1)
        pos_mask = tf.cast(self.mask_, dtype = tf.float32)
        self.Npos = tf.clip_by_value(tf.reduce_sum(pos_mask),1,1e+15)

        # neg_mask = tf.equal(y_true, 0)
        # neg_mask = tf.cast(neg_mask, dtype = tf.float32)
        # Nneg = tf.clip_by_value(tf.reduce_sum(neg_mask),1,1e+15)

        gamma = 2
        alpha = 0.25
        
        # obj_loss = - alpha * K.pow( (1 - y_pred), gamma) * K.log(tf.clip_by_value(y_pred,1e-15,1.0))
        # noobj_loss = - (1 - alpha)* K.pow((y_pred),gamma)* K.log(tf.clip_by_value((1 - y_pred), 1e-15, 1.0))

        # obj_loss_pos = pos_mask* tf.clip_by_value(obj_loss,1e-15,1e+15)
        # obj_loss_neg = neg_mask * tf.clip_by_value(noobj_loss,1e-15,1e+15)

        # object_loss = (tf.reduce_sum(obj_loss_pos)+tf.reduce_sum(obj_loss_neg))/self.Npos


        ce = K.binary_crossentropy(y_true, y_pred, from_logits=False)

        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

        object_loss = tf.reduce_sum(alpha_factor * modulating_factor * ce)/self.Npos



        return self.focal_weight * object_loss


    def loc_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        # print(self.mask_)
        mask_loc = tf.tile(tf.expand_dims(self.mask_, -1), [1, 1, 1, 1, 3])
        mask_loc = tf.cast(mask_loc, dtype = tf.float32)
        # real_true = pos_to_real(y_true)
        # real_pred = pos_to_real(y_pred)

        loss = tf.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    delta=1/9,
                                    reduction="none")

        # loss = self.Smooth_L1(y_pred,y_true)


        loc_loss = mask_loc * loss
        lloss = tf.reduce_sum(loc_loss)/(self.Npos)

        return self.loc_weight * lloss

    def size_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        mask_size = tf.tile(tf.expand_dims(self.mask_, -1), [1, 1, 1, 1, 3])
        mask_size = tf.cast(mask_size, dtype = tf.float32)
        # real_true = dim_to_real(y_true)
        # real_pred = dim_to_real(y_pred)

        loss = tf.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    delta=1/9,
                                    reduction="none")

        # loss = self.Smooth_L1(y_pred,y_true)
        
        size_loss = mask_size * loss
        sloss = tf.reduce_sum(size_loss)/(self.Npos)

        return self.size_weight * sloss

    def angle_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):

        mask_angle = tf.cast(self.mask_, dtype = tf.float32)

        # real_true = rot_to_real(y_true)
        # real_pred = rot_to_real(y_pred)
        loss = tf.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    delta=1/9,
                                    reduction="none")

        # loss = self.Smooth_L1(y_pred,y_true)

        angle_loss = mask_angle * loss
        aloss = tf.reduce_sum(angle_loss)/(self.Npos)

        return self.angle_weight * aloss

    def class_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        mask_class = tf.tile(tf.expand_dims(self.mask_, -1), [1, 1, 1, 1, 1])
        mask_class = tf.cast(mask_class, dtype = tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        
        class_loss = mask_class * loss
        cls_ = tf.reduce_sum(loss)/(self.Npos)

        return self.class_weight * cls_

if __name__=='__main__':
    import numpy as np
    # p = np.array([[[[1.0, 1.0, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    #               [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    #               [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [1.0, 1.0, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.0]]]])


    # t = np.array([[[[1.0, 1.0, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    #               [[[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    #               [[[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    #                [[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [1.0, 1.0, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.0]]]])

    # p = np.reshape(p, [-1, 3, 3, 3, 9])
    # t = np.reshape(t, [-1, 3, 3, 3, 9])
    # p = K.constant(p)
    # t = K.constant(t)
    # print(t.shape)
    # # input()


    class_pred = K.random_uniform((1, 252,  252, 4), minval=0.0, maxval=1.0, seed=400)
    conf_pred = K.random_uniform((1, 252,  252, 4), minval=0.0, maxval=1.0, seed=401)
    pos_pred = K.random_uniform((1, 252,  252, 4, 3), minval=0.0, maxval=1.0, seed=402)
    dim_pred = K.random_uniform((1, 252,  252, 4, 3), minval=0.0, maxval=1.0, seed=403)
    angle_pred = K.random_uniform((1, 252,  252, 4), minval=0.0, maxval=1.0, seed=404)
    
    class_true = K.random_uniform((1, 252,  252, 4), minval=0.0, maxval=1.0, seed=405)
    conf_true = K.random_uniform((1, 252,  252, 4), minval=0.0, maxval=1.0, seed=406)
    pos_true = K.random_uniform((1, 252,  252, 4, 3), minval=0.0, maxval=1.0, seed=407)
    dim_true = K.random_uniform((1, 252,  252, 4, 3), minval=0.0, maxval=1.0, seed=408)
    angle_true = K.random_uniform((1, 252,  252, 4), minval=0.0, maxval=1.0, seed=409)

    pred = ([class_pred,conf_pred,pos_pred,dim_pred,angle_pred])
    true = ([class_true,conf_true,pos_true,dim_true,angle_true])


    loss = PointPillarNetworkLoss(true,pred)
    print(loss.losses())
    # print(loss.loc_loss(true,pred
    # print(K.eval())

    # print('\nLoss Score: ', K.eval(My_loss(t, p, test=True)))