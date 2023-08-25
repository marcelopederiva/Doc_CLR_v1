import numpy as np
# from scipy.spatial import ConvexHull
# from numpy import *
#import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import tensorflow.keras.backend as K
def vertices(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    return box1_x1, box2_x1, box1_y1, box2_y1, box1_x2, box2_x2, box1_y2, box2_y2

def intersection(label,predict):
    # Label -> ? * 6 * 6 * 6 * 6 (x,y,z,w,h,l)
    # Predict -> ? * 6 * 6 * 6 * 6 (x,y,z,w,h,l)

    # Using X/Y to calculate iou2D
    label1 = K.concatenate([label[...,0:2],label[...,3:5]])
    pred1 = K.concatenate([predict[...,0:2],predict[...,3:5]])
    box1_x1, box2_x1, box1_y1, box2_y1, box1_x2, box2_x2, box1_y2, box2_y2 = vertices(pred1,label1)
    x1 = K.maximum(box1_x1, box2_x1)
    y1 = K.maximum(box1_y1, box2_y1)
    x2 = K.minimum(box1_x2, box2_x2)
    y2 = K.minimum(box1_y2, box2_y2)

    label2 = K.concatenate([label[..., 1:3], label[..., 4:6]])
    pred2 = K.concatenate([predict[..., 1:3], predict[..., 4:6]])
    box1_y1, box2_y1, box1_z1, box2_z1, box1_y2, box2_y2, box1_z2, box2_z2 = vertices(pred2, label2)
    y1 = K.maximum(box1_y1, box2_y1)
    z1 = K.maximum(box1_z1, box2_z1)
    y2 = K.minimum(box1_y2, box2_y2)
    z2 = K.minimum(box1_z2, box2_z2)

    inter = K.clip((x2 - x1),min_value = 0, max_value=None) * K.clip((y2 - y1), min_value = 0,max_value=None) * K.clip((z2 - z1), min_value = 0,max_value=None)
    box1 = label[...,3:4] * label[...,4:5] * label[...,5:6]
    box2 = predict[..., 3:4] * predict[..., 4:5] * predict[..., 5:6]

    iou = inter/(box1+box2-inter+1e-7)
    return iou

def Bc(label,predict):
    # Using X/Y
    label1 = K.concatenate([label[..., 0:2], label[..., 3:5]])
    pred1 = K.concatenate([predict[..., 0:2], predict[..., 3:5]])
    box1_x1, box2_x1, box1_y1, box2_y1, box1_x2, box2_x2, box1_y2, box2_y2 = vertices(pred1, label1)
    x1 = K.minimum(box1_x1, box2_x1)
    x2 = K.maximum(box1_x2, box2_x2)
    y1 = K.minimum(box1_y1, box2_y1)
    y2 = K.maximum(box1_y2, box2_y2)

    # Using Y/Z
    label2 = K.concatenate([label[..., 1:3], label[..., 4:6]])
    pred2 = K.concatenate([predict[..., 1:3], predict[..., 4:6]])
    box1_y1, box2_y1, box1_z1, box2_z1, box1_y2, box2_y2, box1_z2, box2_z2 = vertices(pred2, label2)
    y1 = K.minimum(box1_y1, box2_y1)
    y2 = K.maximum(box1_y2, box2_y2)
    z1 = K.minimum(box1_z1, box2_z1)
    z2 = K.maximum(box1_z2, box2_z2)

    hipo_xz = K.sqrt(K.square(K.clip((x2 - x1), min_value=0, max_value=None))  +
                     K.square(K.clip((z2 - z1), min_value=0, max_value=None))
                     )
    c = K.sqrt(K.square(hipo_xz) + K.square(K.clip((y2 - y1), min_value=0, max_value=None)))

    # bc = K.clip((x2 - x1), min_value=0, max_value=None) * K.clip((y2 - y1), min_value=0, max_value=None) * K.clip(
    #     (z2 - z1), min_value=0, max_value=None)

    return c


if __name__=='__main__':
    print('------------------')
    # get_3d_box(box_size, heading_angle, center)
    corners_3d_ground  = get_3d_box((0.6 ,0.5 ,0.4), (0.5 ,0.5 ,0.5)) 
    corners_3d_predict = get_3d_box((0.61 ,0.51 ,0.38),  (0.51 ,0.48 ,0.48))
    IOU_3d= box3d_iou(corners_3d_predict,corners_3d_ground)
    print (IOU_3d) #3d IoU/ 2d IoU of BEV(bird eye's view)
