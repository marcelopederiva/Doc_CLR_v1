import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from utils3 import iou3d, iou2d, plotingcubes, get_3d_box
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
# import open3d as o3d
import tensorflow as tf
import numpy as np
# from models.myModel_27 import My_Model
from models.myModel_pillar_1 import My_Model
from vox_pillar import pillaring
import cv2
import config_model as cfg
import time

######## CFG imports #############
X_div = cfg.X_div
Y_div = cfg.Y_div
Z_div = cfg.Z_div

input_pillar_shape = cfg.input_pillar_shape
input_pillar_indices_shape = cfg.input_pillar_indices_shape

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

w_max = cfg.w_max
h_max = cfg.h_max
l_max = cfg.l_max
#################################

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

####### PATHs ##################
'''RTX'''
# label_path = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/label_norm_lidar_v1/'
# lidar_path = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/input/lidar_crop/'
'''PC-MARCELO'''
label_path = cfg.LABEL_PATH
lidar_path = cfg.LIDAR_PATH
'''PC-DESKTOP'''
# label_path = 'D:/SCRIPTS/Doc_code/data/label_2/'
# lidar_path = 'D:/SCRIPTS/Doc_code/data/input/lidar_crop_mini/'

weights_path = '20230530-Model_minimum.hdf5'
###############################

np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})

trust_treshould = 0.13
iou_treshould = 0.7

input_pillar = Input(input_pillar_shape,batch_size = 1)
input_pillar_indices = Input(input_pillar_indices_shape,batch_size = 1)

output = My_Model(input_pillar, input_pillar_indices)
model = Model(inputs=[input_pillar, input_pillar_indices], outputs=output)
model.load_weights(weights_path, by_name=True)


def to_real(detect):
    real_detect = []
    for dt in detect:
        occ = dt[0]
        x_final = (dt[1] * x_diff) - abs(x_min)
        y_final = (dt[2] * y_diff) - abs(y_min)
        z_final = (dt[3] * z_diff) - abs(z_min)

        width_final = dt[4] * (x_diff)
        height_final = dt[5] * (y_diff)
        length_final = dt[6] * (z_diff)
        rot_final = (dt[7] * rot_norm)
        cls = dt[8]
        try:
            oclu = dt[9]
        except:
            oclu = 3
        real_detect.append([occ, x_final, y_final, z_final, width_final, height_final, length_final, rot_final, cls, oclu])

    return real_detect


def reading_label_ground(label_path, dataset):
    dtc = []
    with open(label_path + dataset + '.txt', 'r') as f:
        label = f.readlines()
    for l in label:
        l = l.replace('\n', '')
        l = l.split(' ')
        l = np.array(l)
        if l[0] != 'DontCare':
            cla = int(cfg.classes[l[0]]) # class --> int
            pos_x = float(l[11])  # Center Position X in relation to max X
            pos_y = float(l[12])  # Center Position Y in relation to max Y
            pos_z = float(l[13])  # Center Position Z in relation to max Z

            dim_x = float(l[9])  # Dimension W in relation to max 2X
            dim_y = float(l[8])  # Dimension H in relation to max 2Y
            dim_z = float(l[10])  # Dimension L in relation to max Z

            rot = float(l[14])
            if rot < 0: rot = -rot
            # occ = int(l[8])
            oclu = int(l[2])
            # dtc.append([pos_x,pos_y,pos_z,dim_x,dim_y,dim_z,rot_y,occ])
            dtc.append([pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, rot, cla, oclu])
    return dtc


def building_ipt(lidar_path, data):
    # LIDAR read
    cam3d = np.load(lidar_path + data + '.npy')  # [0,1,2] -> Z,X,Y

    vox_pillar, pos = pillaring(cam3d)  # (10000,20,7)/ (10000,3)

    return vox_pillar, pos


def predict(lidar_path, data):
    # input_shape = (1,input_size[0],input_size[1],input_size[2])
    pillar_ipt, pos_ipt = building_ipt(lidar_path, data)
    pillar_ipt = np.reshape(pillar_ipt, [-1,
                                         input_pillar_shape[0],
                                         input_pillar_shape[1],
                                         input_pillar_shape[2]])
    pos_ipt = np.reshape(pos_ipt, [-1,
                               input_pillar_indices_shape[0],
                               input_pillar_indices_shape[1]])
    # print(pillar_ipt.shape)
    # print(pos_ipt.shape)
    # exit()
    ipt = (pillar_ipt, pos_ipt)
    occupancy, position, size, angle, classification = model.predict(ipt, batch_size=1) # y = [last_conf,last_pos,last_dim,last_rot,last_class]

    # print(occupancy.shape)
    # print(position.shape)
    # print(size.shape)
    # print(angle.shape)
    # print(classification.shape)


    # exit()
    f = []
    trust_boxes = np.where(occupancy[i] >= trust_treshould)
    coordinates = list(zip(trust_boxes[0], trust_boxes[1], trust_boxes[2])) # (Xdiv,Zdiv,Anchor)

    for idx, value in enumerate(coordinates):
        occ = occupancy[0, value[0], value[1], value[2]]

        x_f = (position[0, value[0], value[1], value[2], 0] + value[0]) / X_div
        y_f = position[0, value[0], value[1], value[2], 1]
        z_f = (position[0, value[0], value[1], value[2], 2] + value[1]) / Z_div

        w_f = size[0, value[0], value[1], value[2], 0] / X_div
        h_f = size[0, value[0], value[1], value[2], 1]
        l_f = size[0, value[0], value[1], value[2], 2] / Z_div

        rot_y = angle[0, value[0], value[1], value[2]]

        cls = np.argmax(classification[0, value[0], value[1], value[2], :])

        f.append([occ,x_f,y_f,z_f,w_f,h_f,l_f,rot_y,cls])

            # print('Occ: ', occ)
            # print('X: ', x_f)
            # print('Y: ', y_f)
            # print('Z: ', z_f)
            # print('W: ', w_f)
            # print('H: ', h_f)
            # print('L: ', l_f)
            # print('Rot: ', rot_y)
            # print('Cls: ', cls)
            #
            # exit()
    return f


def detect(data, label_path):
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    f = predict(lidar_path, data)
    # print(f)
    # print('\n')
    f = sorted(f, key=lambda x: x[0], reverse=True)
    # for i in range(len(f)):
        # f[i] = f[i][1:]

    bboxes = to_real(f)
    # print(bboxes)
    # exit()
    ########## Non-Max-Supression #####################
    final_detect_real = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        new_bboxes = []
        for box in bboxes:
            # print(iou3d(chosen_box[1:-1], box[1:-1]))
            if iou3d(chosen_box[1:-1], box[1:-1]) < 0.01:
                new_bboxes.append(box)
        bboxes = new_bboxes
        final_detect_real.append(chosen_box)
    ########## Non-Max-Supression #####################

    # final_detect_real = bboxes
    # print('PRed')
    # print(final_detect_real)
    # print('\n True')
    # for i in final_detect_real: print(i)
    # exit()
    corners_predict = []

    corners_real = []
    true_detect = reading_label_ground(label_path, data)
    # plt.imshow(true_detect[:, :, 1])
    # plt.show()
    # exit()
    # print('\n')
    # print(true_detect)
    # input()
    # print('\n\n----------------------')
    # true_detect_real = to_real(true_detect)
    # print(true_detect_real)
    # exit()
    # for i in true_detect_real: print(i)

    # print('\n Real:')

    for dtc in true_detect:
        # print(np.round(dtc, decimals=3))
        # print('True')
        # print(dtc)
        # input()
        corners_real.append([get_3d_box(dtc[:-1]),dtc[-2]])

    # print('Predict:')
    for dtc in final_detect_real:
        # print('\nPredict')
        dtc = dtc[1:]

        # print(np.round(dtc,decimals=3))

        corners_predict.append([get_3d_box(dtc[:-1]),dtc[-2]])
        # print(corners_predict)
        # exit()
    #

    mAP_ez, mAP_med, mAP_hard, mean_i_ez, mean_i_med, mean_i_hard, max_i_ez, max_i_med, max_i_hard, detections_ez, detections_med, detections_hard = maP(
        final_detect_real, true_detect)
    # exit() # FAZER O MAP
    plotingcubes(corners_predict, corners_real)
    # exit()
    return final_detect_real, true_detect, mAP_ez, mAP_med, mAP_hard, mean_i_ez, mean_i_med, mean_i_hard, max_i_ez, max_i_med, max_i_hard, detections_ez, detections_med, detections_hard


def maP(corners_predict, corners_real):
    correct_ez = 0
    correct_med = 0
    correct_hard = 0

    corr_dtc_ez = 0
    corr_dtc_med = 0
    corr_dtc_hard = 0

    i_ez = []
    i_med = []
    i_hard = []

    number_dtc_ez = 0
    number_dtc_med = 0
    number_dtc_hard = 0
    for d in corners_real:
        if d[-1] == 0:
            number_dtc_ez += 1
        elif d[-1] == 1:
            number_dtc_med += 1
        elif d[-1] == 2:
            number_dtc_hard += 1
        # elif d[-1]==3:
        #     if (0.0<d[-2]<0.15):
        #         number_dtc_ez += 1
        #     if (0.15<d[-2]<0.30):
        #         number_dtc_med += 1
        #     if (0.30<d[-2]<0.50):
        #         number_dtc_hard += 1

    while (len(corners_predict) > 0):

        for r in range(len(corners_real)):
            # print(corners_predict[0])
            # input()
            iou_score = iou3d(corners_predict[0][:-2], corners_real[r][:-2])
            # print(iou_score)
            if (iou_score > 0.1) and (corners_predict[0][-2] == corners_real[r][-2]):
                # print(iou_score)
                if corners_real[r][-1] == 0:
                    i_ez.append(iou_score)
                    corr_dtc_ez += 1
                elif corners_real[r][-1] == 1:
                    i_med.append(iou_score)
                    corr_dtc_med += 1
                elif corners_real[r][-1] == 2:
                    i_hard.append(iou_score)
                    corr_dtc_hard += 1

            if (iou_score > iou_treshould) and (corners_predict[0][-2] == corners_real[r][-2]) :
                if corners_real[r][-1] == 0:
                    corners_real.pop(r)
                    correct_ez += 1
                elif corners_real[r][-1] == 1:
                    corners_real.pop(r)
                    correct_med += 1
                elif corners_real[r][-1] == 2:
                    corners_real.pop(r)
                    correct_hard += 1

                break
        corners_predict.pop(0)

    non_detected = len(corners_real)

    if i_ez == []:
        i_ez = 0
    if i_med == []:
        i_med = 0
    if i_hard == []:
        i_hard = 0

    if number_dtc_ez == 0:
        mAP_ez = 'nd'
        detections_ez = 'nd'
    elif number_dtc_ez != 0:
        mAP_ez = correct_ez / number_dtc_ez
        detections_ez = corr_dtc_ez / number_dtc_ez

    if number_dtc_med == 0:
        mAP_med = 'nd'
        detections_med = 'nd'
    elif number_dtc_med != 0:
        mAP_med = correct_med / number_dtc_med
        detections_med = corr_dtc_med / number_dtc_med

    if number_dtc_hard == 0:
        mAP_hard = 'nd'
        detections_hard = 'nd'
    elif number_dtc_hard != 0:
        mAP_hard = correct_hard / number_dtc_hard
        detections_hard = corr_dtc_hard / number_dtc_hard

    mean_i_ez = np.mean(i_ez)
    mean_i_med = np.mean(i_med)
    mean_i_hard = np.mean(i_hard)

    max_i_ez = np.max(i_ez)
    max_i_med = np.max(i_med)
    max_i_hard = np.max(i_hard)

    return mAP_ez, mAP_med, mAP_hard, mean_i_ez, mean_i_med, mean_i_hard, max_i_ez, max_i_med, max_i_hard, detections_ez, detections_med, detections_hard


def clearning_nd(a):
    while 'nd' in a:
        a.pop(a.index('nd'))
    return a


if __name__ == '__main__':

    #
    tst_dataset = '/home/rtxadmin/Documents/Marcelo/Doc_code/data'
    # tst_dataset = 'C:/Users/Marcelo/Desktop/SCRIPTS/MySCRIPT/Doc_code/data'
    # datasets = ['000010','000011','000013','000036','000072','007349']
    # datasets = ['004833']
    # datasets = ['000010','005382','000036']
    # datasets = ['000012','000018','000035','000102','007381']
    # datasets = ['000190','006420','007019','003701']
    # datasets = ['000004','000007','000009','000010','000012']

    # datasets = ['002937','004834','000975','000021','001595'] # val10_norm6
    # datasets = ['005248','000585','001465','000510'] # val10_norm
    # datasets = ['004029']
    # with open(os.path.join(tst_dataset, 'test_R_tr.txt'), 'r') as f:
    #     datasets = f.readlines()
    with open(os.path.join(tst_dataset, 'val_20.txt'), 'r') as f:
        datasets = f.readlines()

    MAcc_ez = []
    MAcc_med = []
    MAcc_hard = []

    i_mean_ez = []
    i_mean_med = []
    i_mean_hard = []

    i = 0
    maxmax_iez = 0
    maxmax_imed = 0
    maxmax_ihard = 0

    yes_detect_ez = []
    yes_detect_med = []
    yes_detect_hard = []
    d = 1
    for data in datasets:
        # data = '004283.txt'
        # input('Enter to evalutate')
        print(data[:6])

        pred, real, mAP_ez, mAP_med, mAP_hard, mean_i_ez, mean_i_med, mean_i_hard, max_i_ez, max_i_med, max_i_hard, detections_ez, detections_med, detections_hard = detect(
            data[:6], label_path)

        if max_i_ez > maxmax_iez:
            maxmax_iez = max_i_ez
        if max_i_med > maxmax_imed:
            maxmax_imed = max_i_med
        if max_i_hard > maxmax_ihard:
            maxmax_ihard = max_i_hard


        # print('Ap:', ap)
        MAcc_ez.append(mAP_ez)
        MAcc_med.append(mAP_med)
        MAcc_hard.append(mAP_hard)
        # print('A_Iou:', i)
        i_mean_ez.append(mean_i_ez)
        i_mean_med.append(mean_i_med)
        i_mean_hard.append(mean_i_hard)
        # print('Corr_dtc:',corr_dtc)
        yes_detect_ez.append(detections_ez)
        yes_detect_med.append(detections_med)
        yes_detect_hard.append(detections_hard)
        # print('\n')

        print('*************************************************************************************')
        print('Data: ', d, ' of ', len(datasets))
        # ---------------------------------- EASY--------------------------------------------
        print('*************************************************************************************')
        print('---------------------------------EASY--------------------------------------------')
        print('mAp:', round(np.mean(clearning_nd(MAcc_ez)) * 100, 2), '%')
        print('Mean Iou:', round(np.mean(i_mean_ez), 4))
        print('Max Iou:', round(maxmax_iez, 4))
        print('Average detection: ', round(np.mean(clearning_nd(yes_detect_ez)) * 100, 2), '%')
        print('\n')

        print('---------------------------------MEDIUM--------------------------------------------')
        print('mAp:', round(np.mean(clearning_nd(MAcc_med)) * 100, 2), '%')
        print('Mean Iou:', round(np.mean(i_mean_med), 4))
        print('Max Iou:', round(maxmax_imed, 4))
        print('Average detection: ', round(np.mean(clearning_nd(yes_detect_med)) * 100, 2), '%')
        print('\n')

        print('---------------------------------HARD--------------------------------------------')
        print('mAp:', round(np.mean(clearning_nd(MAcc_hard)) * 100, 2), '%')
        print('Mean Iou:', round(np.mean(i_mean_hard), 4))
        print('Max Iou:', round(maxmax_ihard, 4))
        print('Average detection: ', round(np.mean(clearning_nd(yes_detect_hard)) * 100, 2), '%')
        print('*************************************************************************************')
        print('\n\n\n\n')
        d += 1


