import numpy as np
from scipy.spatial import ConvexHull
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import keras.backend as K

def plotingcubes(Z1,Z2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(Z1[:, 0], Z1[:, 1], Z1[:, 2])
    ax.scatter3D(Z2[:, 0], Z2[:, 1], Z2[:, 2])

    verts1 = [[Z1[0],Z1[1],Z1[2],Z1[3]],
     [Z1[4],Z1[5],Z1[6],Z1[7]], 
     [Z1[0],Z1[1],Z1[5],Z1[4]], 
     [Z1[2],Z1[3],Z1[7],Z1[6]], 
     [Z1[1],Z1[2],Z1[6],Z1[5]],
     [Z1[4],Z1[7],Z1[3],Z1[0]]]

    verts2 = [[Z2[0],Z2[1],Z2[2],Z2[3]],
     [Z2[4],Z2[5],Z2[6],Z2[7]], 
     [Z2[0],Z2[1],Z2[5],Z2[4]], 
     [Z2[2],Z2[3],Z2[7],Z2[6]], 
     [Z2[1],Z2[2],Z2[6],Z2[5]],
     [Z2[4],Z2[7],Z2[3],Z2[0]]]

    ax.add_collection3d(Poly3DCollection(verts1,  facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
    ax.add_collection3d(Poly3DCollection(verts2,  facecolors='red', linewidths=1, edgecolors='r', alpha=.25))
    plt.show()

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    # plotingcubes(corners1,corners2)
    rect1 = [(corners1[...,i,0], corners1[...,i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[...,i,0], corners2[...,i,2]) for i in range(3,-1,-1)] 
    # rect1 = K.permute_dimensions(rect1,(2,3,4,5,1,0))
    # rect2 = K.permute_dimensions(rect2,(2,3,4,5,1,0))
    # print(rect1.shape)
    # input()
    # area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    # area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    
    # iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[...,0,1], corners2[...,0,1])
    ymin = max(corners1[...,4,1], corners2[...,4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)

    if iou>1.0:iou=1.0
    return iou
# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    # def roty(t):
    #     c = np.cos(t)
    #     s = np.sin(t)
    #     return np.array([[c,  0,  s],
    #                      [0,  1,  0],
    #                      [-s, 0,  c]])

    # R = roty(heading_angle)
    w,h,l = box_size[...,:1],box_size[...,1:2],box_size[...,2:] # ?, 6, 6, 6, 1

    # w = K.expand_dims(w)
    # x_corners = {}
    # x_corners[1] = l/2
    # x_corners = K.tile(l,[1,1,1,1]) # ?, 6, 6, 6, 8
    # x_corners = K.stack([x_corners,[l[...,0]/2, l[...,0]/2,-l[...,0]/2,-l[...,0]/2,
    #                       l[...,0]/2,l[...,0]/2,-l[...,0]/2,-l[...,0]/2]], axis = 4)

    x_corners = [l[...,0]/2 + center[...,0], l[...,0]/2 + center[...,0],-l[...,0]/2 + center[...,0],-l[...,0]/2 + center[...,0],
                          l[...,0]/2 + center[...,0],l[...,0]/2 + center[...,0],-l[...,0]/2 + center[...,0],-l[...,0]/2 + center[...,0]]
    # x_corners = x_corners[...,0:] + center[...,0]
    x_corners = K.permute_dimensions(x_corners, (1, 2, 3, 4, 0))


    # A = K.variable(x_corners)
    # x_corners = K.permute_dimensions(x_corners,(1,2,3,4,0))
    y_corners =[h[...,0]/2+ center[...,1],h[...,0]/2+ center[...,1],h[...,0]/2+ center[...,1],h[...,0]/2+ center[...,1],-h[...,0]/2+ center[...,1],
                    -h[...,0]/2+ center[...,1],-h[...,0]/2+ center[...,1],-h[...,0]/2+ center[...,1]]
    # y_corners = y_corners[...,0:] + center[...,1]
    y_corners = K.permute_dimensions(y_corners,(1,2,3,4,0))

    z_corners = [w[...,0]/2+ center[...,2],-w[...,0]/2+ center[...,2],-w[...,0]/2+ center[...,2],w[...,0]/2+ center[...,2],w[...,0]/2+ center[...,2],
                    -w[...,0]/2+ center[...,2],-w[...,0]/2+ center[...,2],w[...,0]/2+ center[...,2]]
    # z_corners = z_corners[...,0:] + center[...,2]

    z_corners = K.permute_dimensions(z_corners,(1,2,3,4,0))

    # x1,x2,x3,x4,x5,x6,x7,x8 = l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2

    # y1,y2,y3,y4,y5,y6,y7,y8 = h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2

    # z1,z2,z3,z4,z5,z6,z7,z8 = w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2

    

    

    corners_3d = K.stack([x_corners,y_corners,z_corners])

    corners_3d = K.permute_dimensions(corners_3d,(1,2,3,4,0,5)) # ?,6,6,6,3,8

    # print(corners_3d.shape)
    # input()
    # corners_3d[0,:] = corners_3d[0,:] + center[...,0:1]
    # corners_3d[1,:] = corners_3d[1,:] + center[...,1:2]
    # corners_3d[2,:] = corners_3d[2,:] + center[...,2:]
    # corners_3d = np.transpose(corners_3d)

    # print(corners_3d.shape)
    # input()

    return corners_3d

def iou3d(label,predict):
    # label ? x 6 x 6 x 6 x 6
    # predict ? x 6 x 6 x 6 x 6
    # label[...,:3]
    corners_3d_ground  = get_3d_box(label[...,3:], label[...,:3])   # ?,6,6,6,3,8
    corners_3d_predict = get_3d_box(predict[...,3:], predict[...,:3])  # ?,6,6,6,3,8
    x = corners_3d_predict[0,0,0,0,:]
    # ---------------------------------------------------------------------------
    print(x)
    input()
    iou3d = K.tile(label,[1,1,1,1,1]) # ? , 6, 6, 6, 1
    # iou3d = np.copy(label[:,:,:,:,:1])
    # print(label.shape)
    # print(iou3d.shape)
    # input()
    a,b,c,d,e = iou3d.shape[0],iou3d.shape[1],iou3d.shape[2],iou3d.shape[3],iou3d.shape[4]

    # print(a)
    # print(b)
    # input()
    # print(corners_3d_predict.shape)
    # print(corners_3d_predict[0,0,0,0,:,:])
    # input()
    for j in range(b-1):
        for k in range(c-1):
            for l in range(d-1):
                iou3d[a,b,c,d,e-1] = box3d_iou(K.eval(corners_3d_predict[:,j,k,l,:,:]),
                                             K.eval(corners_3d_ground[:,j,k,l,:,:]))
                                                 
    # print(iou3d.shape)
    # input()                



    # IOU_3d= box3d_iou(corners_3d_predict,corners_3d_ground) # ?,6,6,6,1
    # print(IOU_3d.shape)
    # print(iou3d[0,0,0,0,:])
    # input()
    return iou3d # ?, 6, 6, 6, 1

def iou2d(boxes_preds, boxes_labels, box_format="midpoint"):
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

    # if box_format == "corners":
    #     box1_x1 = boxes_preds[..., 0:1]
    #     box1_y1 = boxes_preds[..., 1:2]
    #     box1_x2 = boxes_preds[..., 2:3]
    #     box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
    #     box2_x1 = boxes_labels[..., 0:1]
    #     box2_y1 = boxes_labels[..., 1:2]
    #     box2_x2 = boxes_labels[..., 2:3]
    #     box2_y2 = boxes_labels[..., 3:4]

    x1 = K.max(box1_x1, box2_x1)
    y1 = K.max(box1_y1, box2_y1)
    x2 = K.min(box1_x2, box2_x2)
    y2 = K.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = K.clip((x2 - x1),min_value = 0) * K.clip((y2 - y1), min_value = 0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def intersection(label,predict):
    # Label -> ? * 6 * 6 * 6 * 6 (x,y,z,w,h,l)
    # Predict -> ? * 6 * 6 * 6 * 6 (x,y,z,w,h,l)

    # Using X/Y to calculate iou2D
    label1 = K.concatenate([label[...,0:2],label[...,3:5]])
    pred1 = K.concatenate([predict[...,0:2],predict[...,3:5]])
    inter2 = iou2d(pred1,label1)
    print(inter2.shape)
    input()
if __name__=='__main__':
    print('------------------')


    corners_3d_ground  = get_3d_box((0.6 ,0.5 ,0.4), (0.5 ,0.5 ,0.5)) 
    corners_3d_predict = get_3d_box((0.61 ,0.51 ,0.38),  (0.51 ,0.48 ,0.48))
    plotingcubes(corners1, corners2)

    IOU_3d= box3d_iou(corners_3d_predict,corners_3d_ground)
    print (IOU_3d) #3d IoU/ 2d IoU of BEV(bird eye's view)