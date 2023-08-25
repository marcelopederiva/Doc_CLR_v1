from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.lines import Line2D
from inter_area import intersection_area
import numpy as np
import config_model as cfg

Sx = cfg.X_div
Sy = cfg.Y_div
Sz = cfg.Z_div
# x_min = cfg.x_min
# x_max = cfg.x_max
# x_diff = cfg.x_diff

# y_min = cfg.y_min
# y_max = cfg.y_max
# y_diff = cfg.y_diff

# z_min = cfg.z_min
# z_max = cfg.z_max
# z_diff = cfg.z_diff

rot_norm = 3.1416
def right_rot(box):
    return - box - 3.1416 / 2
    # if box<=0:
    #     return (- box - 3.1416 / 2)
    # else:
    #     return ( box + 3.1416 / 2)

def roty(t):
    t_c = t
    c = np.cos(t_c)
    s = np.sin(t_c)
    # return np.array([[c,  0,  s],    #Z
    #                  [0,  1,  0],
    #                  [-s, 0,  c]])
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

    # return np.array([[c, s, 0],
    #                  [-s, c, 0],
    #                  [0, 0, 1]])
    # return np.array([[1, 0, 0],   #X
    #                  [0, c, -s],
    #                  [0, s, c]])


def vertices(boxes_preds, boxes_labels):

    box1_x1 = boxes_preds[0] - boxes_preds[2] / 2
    box1_y1 = boxes_preds[1] - boxes_preds[3] / 2

    # ex 1,1
    box1_x2 = boxes_preds[0] + boxes_preds[2] / 2
    box1_y2 = boxes_preds[1] + boxes_preds[3] / 2
    #LABEL
    box2_x1 = boxes_labels[0] - boxes_labels[2] / 2
    box2_y1 = boxes_labels[1] - boxes_labels[3] / 2

    box2_x2 = boxes_labels[0] + boxes_labels[2] / 2
    box2_y2 = boxes_labels[1] + boxes_labels[3] / 2


    return box1_x1,box2_x1,box1_y1,box2_y1,box1_x2,box2_x2,box1_y2,box2_y2

def iou2d(label,predict):
    # Label -> ? * 6 * 6 * 6 * 6 (x,y,z,w,h,l)
    # Predict -> ? * 6 * 6 * 6 * 6 (x,y,z,w,h,l)

    # Using X/Y to calculate iou2D
    # label1 = np.concatenate([label[0:1],label[2:3], label[3:4], label[5:6], label[6:7]])
    #
    # pred1 = np.concatenate([predict[0:1],predict[2:3],predict[3:4],predict[5:6], predict[6:7]])


    # box1_x1,box2_x1,box1_z1,box2_z1,box1_x2,box2_x2,box1_z2,box2_z2 = vertices(pred1,label1,True)
    #
    #
    #
    # x1 = max(box1_x1, box2_x1)
    # z1 = max(box1_z1, box2_z1)
    # x2 = min(box1_x2, box2_x2)
    # z2 = min(box1_z2, box2_z2)
    #
    # print('Max: ', x1, z1)
    # inter = np.clip((x2 - x1),0,None) * np.clip((z2 - z1), 0,None)
    # rot_label = (label[6] * 2 * rot_norm) - rot_norm
    # rot_pred = (predict[6] * 2 * rot_norm) - rot_norm

    # rot_label = label[6] + (3.1416/2)
    # rot_pred = predict[6] + (3.1416/2)

    rot_label = right_rot(label[6])
    rot_pred = right_rot(predict[6])


    r1 = (label[0],label[2], label[3], label[5], rot_label)
    r2 = (predict[0],predict[2], predict[3], predict[5], rot_pred)

    inter = intersection_area(r1,r2)

    box1 = abs(label[3] * label[5])
    box2 = abs(predict[3] * predict[5])


    iou = inter/(box1+box2-inter)

    return iou
def iou3d(label,predict):
    # print(label)
    # print(predict)

    # input()
    # Label -> ? * 6 * 6 * 6 * 6 (x,y,z,w,h,l)
    # Predict -> ? * 6 * 6 * 6 * 6 (x,y,z,w,h,l)

    # Using X/Y to calculate iou2D

    label1 = np.concatenate([label[0:2],label[3:5]])

    pred1 = np.concatenate([predict[0:2],predict[3:5]])

    # print(label1)
    # print(pred1)
    #
    # input()

    box1_x1,box2_x1,box1_y1,box2_y1,box1_x2,box2_x2,box1_y2,box2_y2 = vertices(pred1,label1)
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    # print(box1_y1, box2_y1)
    # Using X/Z to calculate iou2D

    # label2 = np.concatenate([label[0:1],label[2:3], label[3:4], label[5:6], label[6:7]])
    # pred2 = np.concatenate([predict[0:1],predict[2:3], predict[3:4], predict[5:6], predict[6:7]])

    # rot_label = (label[6] * 2 * rot_norm) - rot_norm
    # rot_pred = (predict[6] * 2 * rot_norm) - rot_norm

    rot_label = right_rot(label[6] )
    rot_pred = right_rot(predict[6])

    # rot_label = label[6]  +3.1416/2
    # rot_pred = predict[6] +3.1416/2


    # rot_label = label[6] - (3.1416/2)
    # rot_pred = predict[6]- (3.1416/2)
    # print(label[0], label[2], label[3], label[5])
    # input()
    r1 = (label[0], label[2], label[3], label[5], rot_label)
    r2 = (predict[0], predict[2], predict[3], predict[5], rot_pred)
    # print(r1)
    # print(r2)
    inter2d = intersection_area(r1, r2)

    # print(inter2d)
    # print(np.clip((y2 - y1), 0,None))
    inter = inter2d * np.clip((y2 - y1), 0,None)
    # print(inter)
    # inter = np.clip((x2 - x1),0,None) * np.clip((y2 - y1), 0,None) * np.clip((z2 - z1), 0,None)

    box1 = abs(label[3] * label[4] * label[5])
    box2 = abs(predict[3] * predict[4] * predict[5])
    # print(box1,box2)
    iou = inter/(box1+box2-inter)
    return iou


def get_3d_box(box_size):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    # rot_final = (box_size[6] * 2 * rot_norm) - rot_norm
    # print(box_size[6])
    # rot_final = box_size[6] *(2*rot_norm) - rot_norm
    # rot_final = box_size[6]+(3.1416/2)
    # rot_final = box_size[6]

    # rot_final = box_size[6] - (3.1416/2)

    # print(box_size[6])
    rot_final =  right_rot(box_size[6])
    # if box_size[6]<=0:
    #     rot_final = - box_size[6] +3.1416 / 2
    # else:
    #     rot_final = - box_size[6] -3.1416 / 2


    # rot_final = box_size[6]
    # print(rot_final)
    # print(rot_final)
    w, h, l = box_size[3], box_size[4], box_size[5]  # ?, 6, 6, 6, 1


    z_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]

    R = roty(rot_final)
    # print(R)
    # print(R)
    # print(np.vstack([x_corners,  z_corners, y_corners]).shape)
    corners_3d = np.dot(R,np.vstack([x_corners,  z_corners, y_corners]))
    corners_3d[0, :] = corners_3d[0, :] + box_size[0];
    corners_3d[1, :] = corners_3d[1, :] + box_size[2];
    corners_3d[2, :] = corners_3d[2, :] + box_size[1];

    corners_3d = np.transpose(corners_3d)

    return corners_3d
def plotingcubes(Box_pred,Box_True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # x1= -23
    # x2 = 23
    # y1 = -5
    # y2 = 45
    # z1 = 0
    # z2 = 50
    x1 = -40
    x2 = 40
    y1 = -5
    y2 = 45
    z1 = 0
    z2 = 80

    ax.set_xlabel('X')
    ax.set_zlabel('Y')
    ax.set_ylabel('Z')
    ax.set_xticks(np.arange(x1, x2, ((x2-x1)/(15+1))))
    ax.set_zticks(np.arange(y1, y2, ((y2-y1)/(5+5))))
    ax.set_yticks(np.arange(z1, z2, ((z2-z1)/(15+1))))
    ax.view_init(elev=24 , azim=-54)

    custom_lines = [Line2D([1,0], [0,0], color=cfg.color_true[str(0)]),
                    Line2D([1,0], [0,0], color=cfg.color_true[str(1)]),
                    Line2D([1,0], [0,0], color=cfg.color_true[str(2)]),
                    Line2D([1,0], [0,0], color=cfg.color_true[str(3)])]

    ax.legend(custom_lines, ['Car', 'Pedestrian','Cyclist', 'Truck/Van'], loc= 'best')
    for Z in Box_pred:
        Z1 = Z[0]
        c = Z[1]
        verts1 = [[Z1[0], Z1[1], Z1[2], Z1[3]],
                  [Z1[4], Z1[5], Z1[6], Z1[7]],
                  [Z1[0], Z1[1], Z1[5], Z1[4]],
                  [Z1[2], Z1[3], Z1[7], Z1[6]],
                  [Z1[1], Z1[2], Z1[6], Z1[5]],
                  [Z1[4], Z1[7], Z1[3], Z1[0]]]
        ax.add_collection3d(Poly3DCollection(verts1, facecolors='red', linewidths=0.5, edgecolors='black', alpha=.25))

    for Z in Box_True:
        Z2 = Z[0]
        c = Z[1]
        verts2 = [[Z2[0], Z2[1], Z2[2], Z2[3]],
                  [Z2[4], Z2[5], Z2[6], Z2[7]],
                  [Z2[0], Z2[1], Z2[5], Z2[4]],
                  [Z2[2], Z2[3], Z2[7], Z2[6]],
                  [Z2[1], Z2[2], Z2[6], Z2[5]],
                  [Z2[4], Z2[7], Z2[3], Z2[0]]]
        ax.add_collection3d(Poly3DCollection(verts2, facecolors=cfg.color_true[str(c)], linewidths=0.5, edgecolors='black', alpha=.25))

    plt.show()
def plotingcubes1(Box):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1= -23
    x2 = 23
    y1 = 0
    y2 = 45
    z1 = 5
    z2 = 70

    plt.subplots_adjust(top = 0.985,
                        bottom = 0.015,
                        left = 0.008,
                        right = 0.992,
                        hspace = 0.2,
                        wspace = 0.2)

    #
    ax.set_xlabel('X')
    ax.set_zlabel('Y')
    ax.set_ylabel('Z')
    ax.set_xticks(np.arange(x1, x2, int((x2-x1)/Sx)))
    ax.set_zticks(np.arange(y1, y2, int((y2-y1)/(Sy+5))))
    ax.set_yticks(np.arange(z1, z2, int((z2-z1)/Sz)))
    ax.view_init(elev=26 , azim=-90)
    # ax.view_init(elev=90, azim=-90)

    for Z1 in Box:
        verts1 = [[Z1[0], Z1[1], Z1[2], Z1[3]],
                  [Z1[4], Z1[5], Z1[6], Z1[7]],
                  [Z1[0], Z1[1], Z1[5], Z1[4]],
                  [Z1[2], Z1[3], Z1[7], Z1[6]],
                  [Z1[1], Z1[2], Z1[6], Z1[5]],
                  [Z1[4], Z1[7], Z1[3], Z1[0]]]
        ax.add_collection3d(Poly3DCollection(verts1, facecolors='blue', linewidths=0.5, edgecolors='black', alpha=.25))
    plt.show()
    # plt.savefig('C:/Users/Marcelo/Desktop/SCRIPTS/MySCRIPT/Doc_code_R/Results/test_results_Path3_3D/' + name)
    # plt.close()
    # plt.clf()
    # plt.show()

if __name__ == '__main__':

    rot = np.pi/4
    rot2 = 0
    # print(rot)
    r = (rot+ rot_norm)/(2*rot_norm)
    r2 = (rot2 + rot_norm)/(2*rot_norm)

    # true = [-8.190929676805219, 1.4739772081375122, 18.45955181121826, 1.5634794107505252, 1.6248242855072021, 3.782201290130615, -1.3181436728954314,3]
    # pred = [-8.283353298902512, 1.4421186447143555, 22.063607692718506, 1.6227973358971732, 1.634529948234558, 3.794888973236084, -1.3328595864772796, 3]

    # true = [-8.190929676805219, 1.4739772081375122, 18.45955181121826, 1.5634794107505252, 1.6248242855072021,
    #         3.782201290130615, -3.1416/2, 3]
    # pred = [-8.283353298902512, 1.4421186447143555, 22.063607692718506, 1.6227973358971732, 1.634529948234558,
    #         3.794888973236084, -3.1416/2, 3]

    true = [0.5, 0.5, 0.5, 2, 1, 4, 0, 3]
    pred = [0.6, 0.6, 0.7, 2, 1, 4, 0, 3]

    # true = [0, 0, 0, 2/4, 1/4, 1, 0, 3]
    # pred = [3/4, 0, 2/4, 2/4, 1/4, 1, -3.1416 / 4, 3]

    corners = get_3d_box(true)
    corners2 = get_3d_box(pred)
    iou2d = iou2d(true, pred)
    iou3d = iou3d(true, pred)
    print('IoU2D: ',iou2d)
    print('IoU3D: ', iou3d)
    plotingcubes([corners2],[corners])





    # lbl = [[3.280000000000001, 1.7699999999999996, 17.69, 1.7300000000000002, 1.38, 3.9699999999999998, -1.56, 0],
    #  [11.549999999999997, 1.5500000000000007, 16.47, 1.58, 1.74, 3.9, -2.44, 1],
    #  [3.3799999999999955, 1.8500000000000014, 23.47, 1.62, 1.5, 4.11, -1.54, 2],
    #  [-4.710000000000001, 2.1099999999999994, 33.26, 1.68, 1.58, 3.7599999999999993, 1.6199999999999997, 0],
    #  [11.090000000000003, 1.5899999999999999, 22.82, 1.53, 1.49, 4.38, -2.48, 2],
    #  [11.560000000000002, 1.5799999999999983, 19.52, 1.8, 1.55, 3.9800000000000004, -2.46, 2],
    #  [5.049999999999997, 2.039999999999999, 52.28, 1.7699999999999998, 1.77, 5.1, -1.4899999999999998, 1],
    #  [-15.310000000000002, 2.219999999999999, 38.07, 1.87, 1.5, 4.23, -0.08000000000000007, 2]]

    # corners = []
    # for dtc in lbl:
    #     corners.append(get_3d_box(dtc))
    # corners2 = []
    # for dtc in pred:
    #     corners2.append(get_3d_box(dtc))

    # label = [
    #     [0.5298181818181819, 0.5589999999999999, 0.1769, 0.01572727272727273, 0.046, 0.0397, 0.2517188693659282] ,
    #     [0.605, 0.5516666666666666, 0.16469999999999999, 0.014363636363636365, 0.058, 0.039, 0.1116628469569646 ],
    #     [0.5307272727272727, 0.5616666666666668, 0.2347, 0.014727272727272728, 0.05, 0.041100000000000005, 0.2549019607843137],
    #     [0.4571818181818182, 0.5703333333333334, 0.3326, 0.015272727272727273, 0.05266666666666667, 0.037599999999999995, 0.7578304048892284],
    #     [0.6008181818181818, 0.553, 0.22820000000000001, 0.01390909090909091, 0.049666666666666665, 0.0438, 0.10529666412019353],
    #     [0.6050909090909091, 0.5526666666666666, 0.19519999999999998, 0.016363636363636365, 0.051666666666666666, 0.0398, 0.10847975553857907],
    #     [0.5459090909090909, 0.568, 0.5228, 0.01609090909090909, 0.059000000000000004, 0.051, 0.2628596893302776],
    #     [0.3608181818181818, 0.574, 0.3807, 0.017, 0.05, 0.042300000000000004, 0.48726763432645787]
    # ]
    # corners = get_3d_box(lbl)
    # corners2 = get_3d_box([])
    # plotingcubes(corners,corners2)