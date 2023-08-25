import os
# import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras.backend as K
import config_model as cfg
import tensorflow as tf
from utils2 import intersection
trust_threshold = 0.5
def mean_IoU(y_true,y_pred):
    response_mask = y_true[0]  # ? * 6 * 6 * 6 * 1
    label_box_4iou = K.concatenate(y_true[1][:,:,0,:],y_true[2][:,:,0,:])  # ? * 6 * 6 * 6 * 6
    predict_box_4iou = K.concatenate(y_pred[1][:,:,0,:],y_pred[2][:,:,0,:])  # ? * 6 * 6 * 6 * 6


    response_mask_f = K.cast(response_mask > trust_threshold, dtype = tf.float32)
    iou_scores = intersection(label_box_4iou, predict_box_4iou)  # ? * 6 * 6 * 6 * 1
    iou_det = response_mask_f*iou_scores
    iou_mean = K.sum(iou_det)/K.sum(response_mask_f)
    return iou_mean

def mAP(y_true, y_pred, test=False):

    trust_treshold = 0.7
    iou_threshould = 0.7

    label_class = y_true[..., :1]  # ? * 6 * 6 * 6 * 1
    response_mask = y_true[..., 1:2]  # ? * 6 * 6 * 6 * 1
    label_box_pos = y_true[..., 2:5]  # ? * 6 * 6 * 6 * 3
    label_box_x = y_true[..., 2:3]  # ? * 6 * 6 * 6 * 1
    label_box_y = y_true[..., 3:4]  # ? * 6 * 6 * 6 * 1
    label_box_z = y_true[..., 4:5]  # ? * 6 * 6 * 6 * 1
    label_box_dim = y_true[..., 5:8]  # ? * 6 * 6 * 6 * 3
    label_box_rot = y_true[..., 8:9] # ? * 6 * 6 * 6 * 1

    label_box_4iou = y_true[..., 2:8]  # ? * 6 * 6 * 6 * 6

    # response_mask = K.expand_dims(response_mask)  # ? * 6 * 6 * 6 * 1

    predict_class = y_pred[..., :1]  # ? * 6 * 6 * 6 * 20
    predict_trust = y_pred[..., 1:2]  # ? * 6 * 6 * 6 * 1
    predict_box_pos = y_pred[..., 2:5]  # ? * 6 * 6 * 6 * 3
    predict_box_x = y_pred[..., 2:3]  # ? * 6 * 6 * 6 * 1
    predict_box_y = y_pred[..., 3:4]  # ? * 6 * 6 * 6 * 1
    predict_box_z = y_pred[..., 4:5]  # ? * 6 * 6 * 6 * 1
    predict_box_dim = y_pred[..., 5:8]  # ? * 6 * 6 * 6 * 3
    predict_box_rot = y_pred[..., 8:9]  # ? * 6 * 6 * 6 * 1

    predict_box_4iou = y_pred[..., 2:8]  # ? * 6 * 6 * 6 * 6

    # predict_pos_real = pos_to_real(predict_box_pos)
    # label_pos_real = pos_to_real(label_box_pos)

    # predict_dim_real = dim_to_real(predict_box_dim)
    # label_dim_real = dim_to_real(label_box_dim)

    # predict_box_4iou = K.concatenate([predict_pos_real, predict_dim_real])
    # label_box_4iou = K.concatenate([label_pos_real, label_dim_real])
    iou_scores = intersection(label_box_4iou, predict_box_4iou)  # ? * 6 * 6 * 6 * 1

    box_mask = K.cast(predict_trust >= 0.7,  dtype = tf.float32)  # ? * 6 * 6 * 6 * 1
    response_mask_f = K.cast(response_mask > trust_threshold, dtype = tf.float32)
    iou_det = box_mask*iou_scores
    total_true = K.sum(response_mask_f)
    total_pred = K.sum(box_mask)

    corr_mask = K.cast(iou_det >= iou_threshould,  dtype = tf.float32)
    corr = K.sum(response_mask_f*corr_mask)
    result = corr/total_true
    # Lambda(lambda x: x[0] / x[1])([tensor1, tensor2])
    # print(K.eval(box_mask*iou_scores))
    # result = round(result, 2)
    return result*100

def correct_grid(y_true, y_pred):
    response_mask = y_true[...,0]
    mask = K.cast(response_mask > 0.5,  dtype = tf.float32)
    box_mask = y_pred[...,0]

    total_grid = tf.clip_by_value(K.sum(mask),1,1e+15)

    box_mask_truth = mask * K.cast(box_mask >= 0.7, K.dtype(box_mask))  # ? * 6 * 6 * 6 * 1

    total_pred_grid = K.sum(box_mask_truth)

    return  (total_pred_grid/total_grid)*100

def incorrect_grid(y_true, y_pred):

    response_mask = y_true[...,0]
    mask = K.cast(response_mask > trust_threshold,  dtype = tf.float32)
    box_mask = y_pred[...,0]

    

    total_grid = tf.clip_by_value(K.sum(1-mask),1,1e+15)
    # print(total_grid)
    box_mask_truth = (1 - mask) * K.cast(box_mask >= 0.7, K.dtype(box_mask))  # ? * 6 * 6 * 6 * 1

    total_pred_grid = K.sum(box_mask_truth) 
    # print(total_pred_grid)

    return (total_pred_grid/total_grid)*100

if __name__ == '__main__':
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
                       [[[0],[0]],[[0],[1]],[[0],[1]]],
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

    print('\nCorrect Grid: ', K.eval(correct_grid(true, pred)))
