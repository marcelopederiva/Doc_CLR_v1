import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow.keras.backend as K
import config_model as cfg
import tensorflow as tf
import tensorflow_probability as tfp
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

w_max = cfg.w_max
h_max = cfg.h_max
l_max = cfg.l_max

class PointPillarNetworkLoss:

    def __init__(self):
        self.alpha = 0.25
        self.gamma = 2
        self.focal_weight = cfg.lambda_occ
        self.loc_weight = cfg.lambda_pos
        self.size_weight = cfg.lambda_dim
        self.angle_weight = cfg.lambda_rot
        self.class_weight = cfg.lambda_class

    def losses(self):
        return [self.focal_loss, self.loc_loss, self.size_loss, self.angle_loss, self.class_loss]
    
    

    def focal_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """ y_true value from occ in {-1, 0, 1}, i.e. {bad match, neg box, pos box} """
        # print('True:', y_true.shape,'\n')
        # print('Pred:', y_pred.shape)

        # exit()
        self.mask_ = tf.equal(y_true, 1)

        cross_entropy = K.binary_crossentropy(y_true, y_pred)

        p_t = y_true * y_pred + (tf.subtract(1.0, y_true) * tf.subtract(1.0, y_pred))

        gamma_factor = tf.pow(1.0 - p_t, self.gamma)

        alpha_factor = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)

        focal_loss = gamma_factor * alpha_factor * cross_entropy

        neg_mask = tf.equal(y_true, 0)
        thr = tfp.stats.percentile(tf.boolean_mask(focal_loss, neg_mask), 90.)
        hard_neg_mask = tf.greater(focal_loss, thr)
        # mask = tf.logical_or(tf.equal(y_true, 0), tf.equal(y_true, 1))
        mask = tf.logical_or(self.mask_, tf.logical_and(neg_mask, hard_neg_mask))
        masked_loss = tf.boolean_mask(focal_loss, mask)

        return self.focal_weight * tf.reduce_mean(masked_loss)


    def loc_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        # print(self.mask_)
        mask_loc = tf.tile(tf.expand_dims(self.mask_, -1), [1, 1, 1, 1, 3])
        # h = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        # loss = h(y_true,y_pred)
        # print(loss)
        loss = tf.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    reduction="none")



        mask_loc = tf.cast(mask_loc, dtype = tf.float32)
        loc_loss = mask_loc * loss
        lloss = tf.reduce_sum(loc_loss)/(tf.reduce_sum(mask_loc)+1)
        # masked_loss = tf.boolean_mask(loss, mask_loc)

        # print(self.loc_weight * tf.reduce_mean(masked_loss))
        return self.loc_weight * lloss

    def size_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        mask_size = tf.tile(tf.expand_dims(self.mask_, -1), [1, 1, 1, 1, 3])
        loss = tf.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    reduction="none")

        # masked_loss = tf.boolean_mask(loss, mask)

        mask_size = tf.cast(mask_size, dtype = tf.float32)
        size_loss = mask_size * loss
        sloss = tf.reduce_sum(size_loss)/(tf.reduce_sum(mask_size)+1)

        return self.size_weight * sloss

    def angle_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        loss = tf.compat.v1.losses.huber_loss(y_true,
                                    y_pred,
                                    reduction="none")

        # masked_loss = tf.boolean_mask(loss, self.mask_)

        mask_angle = tf.cast(self.mask_, dtype = tf.float32)
        angle_loss = mask_angle * loss
        aloss = tf.reduce_sum(angle_loss)/(tf.reduce_sum(mask_angle)+1)

        return self.angle_weight * aloss

    def class_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        # masked_loss = tf.boolean_mask(loss, self.mask_)
        mask_class = tf.cast(self.mask_, dtype = tf.float32)
        class_loss = mask_class * loss
        closs = tf.reduce_sum(class_loss)/(tf.reduce_sum(mask_class)+1)

        return self.class_weight * closs

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