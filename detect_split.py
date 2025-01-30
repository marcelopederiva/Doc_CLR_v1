import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from utils3 import iou3d, iou2d, plotingcubes, get_3d_box
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
#import open3d as o3d
import tensorflow as tf
import numpy as np
# from models.myModel_27 import My_Model
from models.myModel_3Fusion_3 import My_Model
import vox_pillar_l
import vox_pillar_r
import cv2
import config_model as cfg
import time
from map import mAPNuscenes
# import kitti_data_utils as K_U

######## CFG imports #############
X_div = cfg.X_div
Y_div = cfg.Y_div
Z_div = cfg.Z_div

input_pillar_l_shape = cfg.input_pillar_l_shape
input_pillar_l_indices_shape = cfg.input_pillar_l_indices_shape

input_pillar_r_shape = cfg.input_pillar_r_shape
input_pillar_r_indices_shape = cfg.input_pillar_r_indices_shape

x_min = cfg.x_min
x_max = cfg.x_max
x_diff = cfg.x_diff

y_min = cfg.y_min
y_max = cfg.y_max
y_diff = cfg.y_diff

z_min = cfg.z_min
z_max = cfg.z_max
z_diff = cfg.z_diff

velo_min = cfg.vel_min
velo_max = cfg.vel_max

rot_norm = cfg.rot_max

x_step = cfg.stepx
z_step = cfg.stepz


lidar_path = cfg.LIDAR_PATH
radar_path = cfg.RADAR_PATH
img_path = cfg.IMG_PATH



NUSC_PATH = cfg.NUSC_PATH
#################################
classes_names = [k for k, v in cfg.classes.items()]
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

####### PATHs ##################
'''RTX'''
# label_path = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/label_norm_lidar_v1/'
# lidar_path = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/input/lidar_crop/'
'''PC-MARCELO'''
# label_path = cfg.LABEL_PATH
lidar_path = cfg.LIDAR_PATH
'''PC-DESKTOP'''
# label_path = 'D:/SCRIPTS/Doc_code/data/label_2/'
# lidar_path = 'D:/SCRIPTS/Doc_code/data/input/lidar_crop_mini/'
# chkp_path = '/home/rtxadmin/Documents/Marcelo/Doc_CLR_v1/checkpoints/val_loss/Temp_loss/'
weights_path = 'model_100_Model_3Fusion_3_3_dtvelofix.hdf5'
###############################-

np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})

trust_treshould = 0.5
iou_treshould = 0.5

input_pillar_l = Input(input_pillar_l_shape,batch_size = 1)
input_pillar_indices_l = Input(input_pillar_l_indices_shape,batch_size = 1)

input_pillar_r = Input(input_pillar_r_shape,batch_size = 1)
input_pillar_indices_r = Input(input_pillar_r_indices_shape,batch_size = 1)

input_img = Input((cfg.image_shape[0],cfg.image_shape[1],3), batch_size = 1)

output = My_Model(input_pillar_l, input_pillar_indices_l,input_pillar_r, input_pillar_indices_r ,input_img)
model = Model(inputs=[input_pillar_l, input_pillar_indices_l,input_pillar_r, input_pillar_indices_r ,input_img], outputs=output)
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


def reading_label_ground(dataset):
    dtc = []
    with open('C:/Users/maped/Documents/Scripts/Nuscenes_Scripts/' + 'data_an_velo_fix/' + dataset + '.txt', 'r') as f:
        label = f.readlines()
    for l in label:
        l = l.replace('\n', '')
        l = l.split(' ')
        l = np.array(l)
        
        if l[0] in classes_names:
            # print(l)
            cla = int(cfg.classes[l[0]])  # class --> int
            pos_x = float(l[1])  # Center Position X in relation to max X
            pos_y = float(l[2])  # Center Position Y in relation to max Y
            pos_z = float(l[3])  # Center Position Z in relation to max Z

            dim_x = float(l[4])  # Dimension W in relation to max 2X
            dim_y = float(l[5])  # Dimension H in relation to max 2Y
            dim_z = float(l[6])  # Dimension L in relation to max Z
            rot = float(l[7])
            vel_x = float(l[8])
            vel_z = float(l[9])

            # if axis_z < 0: rot = -rot
            # if rot < 0: rot = rot + np.pi
            # occ = int(l[8])
            # oclu = int(l[2])
            # dtc.append([pos_x,pos_y,pos_z,dim_x,dim_y,dim_z,rot_y,occ])
            norm_x = (pos_x + abs(cfg.x_min)) / (
                    cfg.x_max - cfg.x_min)  # Center Position X in relation to max X 0-1
            norm_y = (pos_y + abs(cfg.y_min)) / (
                    cfg.y_max - cfg.y_min)  # Center Position Y in relation to max Y 0-1
            norm_z = (pos_z + abs(cfg.z_min)) / (
                    cfg.z_max - cfg.z_min)  # Center Position Z in relation to max Z 0-1

            norm_vel_x = (vel_x + abs(cfg.vel_min)) / (
                    cfg.vel_max - cfg.vel_min)  # Norm velocity 0-1
            norm_vel_z = (vel_z + abs(cfg.vel_min)) / (
                    cfg.vel_max - cfg.vel_min)  # Norm velocity 0-1


            if pos_x > pos_z or -pos_x > pos_z: # Inside the camera FOV
                fov = 2
            else:
                 fov = 0.5
            # fov = 0.5
            out_of_size = np.array([norm_x, norm_y, norm_z, norm_vel_x, norm_vel_z, fov])
            # print(out_of_size)
            if (np.any(out_of_size > 1) or np.any(out_of_size < 0)) or (np.isnan(vel_x) or np.isnan(vel_z)):
                continue
            else:
                dtc.append([pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, rot, vel_x, vel_z, cla])




            #if pos_z > 0 and fov== 0.5:
            #    dtc.append([pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, rot, vel_x, vel_z, cla])
    return dtc


def building_ipt(data):
    # LIDAR read
    print(data) 
    #exit()
    img = cv2.imread(img_path + data + '.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img)
    #plt.show()
    img = img / 255.
    img = cv2.resize(img, (cfg.image_shape))

    lidar = np.load(lidar_path + data + '.npy')  # [0,1,2] -> Z,X,Y
    vox_pillar_L, pos_L = vox_pillar_l.pillaring_l(lidar)  # (10000,20,7)/ (10000,3)

    radar = np.load(radar_path + data + '.npy')
    vox_pillar_R, pos_R, cam_3d = vox_pillar_r.pillaring_r(radar)  # (10,5,5)/ (10,3)


    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cam3d[...,:3])
    # o3d.visualization.draw_geometries([pcd], width=504, height=504 ,
    #                                 zoom=0.02,
    #                                 front=[ -0.92862581169634939, 0.1699284137233735, 0.32981576078282615 ],
    #                                 lookat=[ 6.613722928178019, -0.21300680832693597, -0.39229345398463189 ],
    #                                 up=[ 0.33417852221359673, -0.0030960494529159504, 0.9425047107409712 ])
    # print(cam3d.shape)
    # print(cam3d[:30])
    # exit()
    # print(img.shape)
    # print(vox_pillar_L.shape)
    # print(pos_L.shape)
    # print(vox_pillar_R.shape)
    # print(pos_R.shape)
    # exit()

    return img, vox_pillar_L, pos_L, vox_pillar_R, pos_R, cam_3d


def predict(data):
    # data = 'e9c6888ba9c542b8ad820b6e1cc2790a'
    # input_shape = (1,input_size[0],input_size[1],input_size[2])
    img, vox_pillar_L, pos_L, vox_pillar_R, pos_R, cam_3d = building_ipt(data)
    # pillar_ipt = np.reshape(pillar_ipt, [-1,
    #                                      input_pillar_shape[0],
    #                                      input_pillar_shape[1],
    #                                      input_pillar_shape[2]])
    # pos_ipt = np.reshape(pos_ipt, [-1,
    #                            input_pillar_indices_shape[0],
    #                            input_pillar_indices_shape[1]])
    img_ipt = np.expand_dims(img, axis=0)
    pillar_L_ipt = np.expand_dims(vox_pillar_L, axis=0)
    pos_L_ipt = np.expand_dims(pos_L, axis=0)
    pillar_R_ipt = np.expand_dims(vox_pillar_R, axis=0)
    pos_R_ipt = np.expand_dims(pos_R, axis=0)




    # img_ipt = np.reshape(img_ipt, [-1,
    #                                      img_ipt.shape[0],
    #                                      img_ipt.shape[1],
    #                                      img_ipt.shape[2]])

    # print(pillar_ipt.shape)
    # print(pos_ipt.shape)
    # print(img_ipt.shape)
    # exit()
    ipt = [pillar_L_ipt, pos_L_ipt, pillar_R_ipt, pos_R_ipt, img_ipt]
    dt = model.predict(ipt, batch_size=1) # y = [last_conf,last_pos,last_dim,last_rot,last_class]

    occupancy = np.reshape(dt[...,0], (1, X_div, Z_div, 2))
    position = dt[...,1:4]
    size = dt[...,4:7]
    angle = np.reshape(dt[...,7], (1, X_div, Z_div, 2))
    velo = dt[..., 8:10]

    classification = np.reshape(dt[...,10], (1, X_div, Z_div, 2))


    #####

    # from occ_radar_ex import tensor_see
    # v = (velo * (velo_max-velo_min)) - abs(velo_min) 
    # tensor_see(occupancy,v,trust_treshould) 
    #################################################
    # tensor_ann = np.reshape(velo, (1, 256, 256, 4))
    # tensor_ann = tf.convert_to_tensor(tensor_ann, dtype=tf.float32)
    # tensor_ann = tf.nn.max_pool(tensor_ann, ksize=2, strides=2, padding='VALID')
    # tensor_ann = tf.nn.max_pool(tensor_ann, ksize=2, strides=2, padding='VALID')
    # tensor_ann = tf.reshape(tensor_ann, (1, 64, 64, 2, 2))
    # tensor_ann = tensor_ann.numpy()

    # tensor_ann = tensor_ann[0,:,:,0,:]
    # tensor_ann = np.transpose(tensor_ann, axes=(1,0,2))
    # X_an, Y_an = np.meshgrid(np.linspace(x_min, x_max, 64), np.linspace(y_min, y_max, 64))
    # U_an = tensor_ann[:, :, 0]
    # V_an = tensor_ann[:, :, 1]

    # plt.figure(figsize=(10, 10))
    # plt.quiver(X_an, Y_an, U_an, V_an,scale =150, color='b')
    # plt.show()
    
    # ###################################################
    # plt.figure(figsize=(10, 10))
    # plt.imshow(occupancy[0,:,:,0], cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='Occupancy')
    # plt.title('Occupancy Grid')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    # exit()
    # img = np.dstack((occupancy[0,:,:,:2], np.zeros((X_div, Z_div, 1))))
    # plt.imshow(img)
    # plt.imshow(p[0,:,:,1])
    
    # print(position.shape)
    # print(size.shape)
    # print(angle.shape)
    # print(classification.shape)

    # print(i)
    # exit()
    f = []
    trust_boxes = np.where(occupancy[i] >= trust_treshould)
    # print(trust_boxes)
    # exit()
    coordinates = list(zip(trust_boxes[0], trust_boxes[1], trust_boxes[2])) # (Xdiv,Zdiv,Anchor)
    # print(coordinates)
    # exit()
    anchors = cfg.anchor
    diag = [np.sqrt(pow(a[0],2)+pow(a[2],2)) for a in anchors]
    # print(x_step,z_step)
    # exit()
    for idx, value in enumerate(coordinates):
        occ = occupancy[0, value[0], value[1], value[2]]

        real_x = value[0] * (x_step) * 2 - abs(x_min)
        real_z = value[1] * (z_step) * 2 - abs(z_min)
        # print(anchors)
        # print(position[0, value[0], value[1], value[2]])
        # print( real_x, real_z)
        # print(position[0, value[0], value[1], value[2], 2] * diag[value[2]])

        bb_x = position[0, value[0], value[1], value[2], 0] * diag[value[2]] + real_x

        bb_y = position[0, value[0], value[1], value[2], 1] * anchors[value[2]][1] + anchors[value[2]][3]
        bb_z = position[0, value[0], value[1], value[2], 2] * diag[value[2]] + real_z
        # print(bb_z)
        # print('\n')
        # print(bb_x,bb_y,bb_z)
        # exit()

        bb_w = np.exp(size[0, value[0], value[1], value[2], 0]) * anchors[value[2]][0]
        bb_h = np.exp(size[0, value[0], value[1], value[2], 1]) * anchors[value[2]][1]
        bb_l = np.exp(size[0, value[0], value[1], value[2], 2]) * anchors[value[2]][2]

        bb_rot = -np.arcsin(np.clip(angle[0, value[0], value[1], value[2]],-1,1)) + anchors[value[2]][4]
        # print(velo)
        bb_velox = ((velo[0, value[0], value[1], value[2], 0] * (velo_max-velo_min)) - abs(velo_min))
        bb_veloz = ((velo[0, value[0], value[1], value[2], 1] * (velo_max-velo_min)) - abs(velo_min))
        # bb_velox = velo[0, value[0], value[1], value[2], 0] 
        # bb_veloz = velo[0, value[0], value[1], value[2], 1]
        bb_cls = np.argmax(classification[0, value[0], value[1], value[2]])
        
        # exit()

        f.append([occ,bb_x,bb_y,bb_z,bb_w,bb_h,bb_l,bb_rot,bb_velox,bb_veloz,bb_cls])

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


def detect(data):
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    # print('oi')
    f = predict(data)
    # print(f)
    # print('\n')
    f = sorted(f, key=lambda x: x[0], reverse=True)
    # for i in range(len(f)):
        # f[i] = f[i][1:]
    # print(f)
    # exit()
    # bboxes = to_real(f)
    bboxes = f
    # print(bboxes)
    # print('-----')
    # exit()
    dtcs = np.zeros((len(bboxes),len(bboxes),11))
    # print(bboxes)
    # exit()
    ########## Non-Max-Supression #####################
    d_idx = 0
    d_g_idx = 0
    final_detect_real = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        dtcs[d_idx][d_g_idx] = chosen_box
        new_bboxes = []

        for box in bboxes:
            # print(iou3d(chosen_box[1:-1], box[1:-1]))
            if iou3d(chosen_box[1:-1], box[1:-1]) < 0.1:
            # if iou3d(chosen_box[1:-1], box[1:-1]) < 0.01:
                new_bboxes.append(box)
            else:
                d_g_idx += 1
                dtcs[d_idx,d_g_idx,:] = box
                # print(dtcs)
        d_idx+=1
        bboxes = new_bboxes
        final_detect_real.append(chosen_box)
    dt_mean = []
    dt_mean_w = []
    for dt in dtcs:
        dt_res = []
        for idx in dt:
            if idx[0]!=idx[1]!=0.0:
                dt_res.append(idx)
        if dt_res != []:
            dt_res2 = np.copy(np.array(dt_res))
            dt_size = len(dt_res)
            # print(dt_res2,'\n')
            dt_res2[:,1:] *= dt_res2[:,0:1]
            dt_res2_sum  = np.sum(dt_res2,axis=0)
            dt_res2 =dt_res2_sum/dt_res2_sum[0]

            dt_mean_w.append(dt_res2)
            dt_mean.append(np.mean(np.array(dt_res),axis=0))

            # print(dt_mean)
            # print(dt_mean_w,'\n')
            # exit()
    # final_detect_real = bboxes
    # final_detect_real = dt_mean
    final_detect_real = dt_mean_w
    ########## Non-Max-Supression #####################



    corners_predict = []

    corners_real = []
    true_detect = reading_label_ground(data)
    


    # calib_path = KITTI_PATH + 'calib/' + data + '.txt'
    # calib = K_U.Calibration(calib_path)
    # img = plt.imread(cfg.KITTI_PATH + 'image_2/' + data + '.png')
    # print('True:')
    for dtc in true_detect:
        print('\nTrue')
        print(np.round(dtc, decimals=3))
        corners_real.append([get_3d_box(dtc[:-3]),dtc[-3],dtc[-2],dtc[-1]])



    # print('Predict:')
    for dtc in final_detect_real:
        # print('\nPredict')
        dtc = dtc[1:]

        # print(np.round(dtc,decimals=3))

        corners_predict.append([get_3d_box(dtc[:-3]),dtc[-3],dtc[-2],dtc[-1]])

        # dtc_2d = projection_2d(get_3d_box(dtc[:-1]), data)
        # img = draw_projected_box3d(img, dtc_2d)

    # print(dtc)
    # print(np.array(corners_predict).shape)
    # exit()

    # plt.imshow(img)
    # plt.show()

    f = final_detect_real
    t = true_detect
    # mAP_ez, mAP_med, mAP_hard, mean_i_ez, mean_i_med, mean_i_hard, max_i_ez, max_i_med, max_i_hard, detections_ez, detections_med, detections_hard, FP,FN,mave_x, mave_y = maP(
    #     final_detect_real, true_detect)
    # print(mean_i)

    mAP, mean_i, mean_iou, max_i, max_iou, detections, F1, Recall, mave = mAPNuscenes(f,t)
    # print(mave)
    # exit()
    # exit()
    # plotingcubes(corners_predict, corners_real)

    # exit()
    return mAP, mean_i, mean_iou, max_i, max_iou, detections, F1, Recall, mave
    # return final_detect_real, true_detect, mAP_ez, mAP_med, mAP_hard, mean_i_ez, mean_i_med, mean_i_hard, max_i_ez, max_i_med, max_i_hard, detections_ez, detections_med, detections_hard, FP, FN, mave_x, mave_y


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

    mave_x = []
    mave_y = []

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
    # print(corners_predict[0][:-1])
    # print(corners_real[0][:-2])
    # exit()
    FP = 0
    # print(len(corners_predict))

    while (len(corners_predict) > 0):
        c = True
        for r in range(len(corners_real)):
            # print(corners_predict[0])
            # input()
            iou_score = iou3d(corners_predict[0][1:-1], corners_real[r][:-1])
            # print(corners_predict[0][0])
            # print(iou_score,'\n')
            #
            # print(iou_score)
            if (iou_score > 0.1) and (corners_predict[0][-1] == corners_real[r][-1]):

                # print(iou_score)
                # print(corners_predict[0][1:-1])
                # print(corners_real[r][:-2],'\n')
                if corners_real[r][-1] == 0:
                    i_ez.append(iou_score)
                    corr_dtc_ez += 1
                elif corners_real[r][-1] == 1:
                    i_med.append(iou_score)
                    corr_dtc_med += 1
                elif corners_real[r][-1] == 2:
                    i_hard.append(iou_score)
                    corr_dtc_hard += 1

            if (iou_score >= iou_treshould) and (corners_predict[0][-1] == corners_real[r][-1]):
                
                mave_x.append(abs(corners_predict[0][8] - corners_real[r][8]))
                mave_y.append(abs(corners_predict[0][9] - corners_real[r][9]))
                c = False
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
        if c:
            FP += 1

        corners_predict.pop(0)

    FN = len(corners_real)
    # exit()
    if mave_x == []:
        mave_x=0
    if mave_y == []:
        mave_y=0
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

    TP = correct_ez + correct_med + correct_hard

    F1 = TP / np.clip((TP + (FP + FN) / 2),10^-3,None)

    Recall = TP / np.clip((TP + FN),10^-3,None)

    mave_x = np.mean(mave_x)
 
    mave_y = np.mean(mave_y)

    return mAP_ez, mAP_med, mAP_hard, mean_i_ez, mean_i_med, mean_i_hard, max_i_ez, max_i_med, max_i_hard, detections_ez, detections_med, detections_hard, F1, Recall, mave_x, mave_y


def clearning_nd(a):
    while 'nd' in a:
        a.pop(a.index('nd'))
    return a


if __name__ == '__main__':

    #
    tst_dataset = 'C:/Users/maped/Documents/Scripts/Nuscenes/'
    # tst_dataset = '/home/rtxadmin/Documents/Marcelo/Nuscenes/Nuscenes/'
    # datasets = ['000010','000011','000013','000036','000072','007349']"C:/Users/maped/Documents/Scripts/Nuscenes"
    # datasets = ['006953']
    # datasets = ['000010','005382','000036']
    # datasets = ['000012','000018','000035','000102','007381']
    # datasets = ['000190','006420','007019','003701']
    # datasets = ['000004','000007','000009','000010','000012']

    # datasets = ['002937','004834','000975','000021','001595'] # val10_norm6
    # datasets = ['005248','000585','001465','000510'] # val10_norm
    # datasets = ['921b2a5eeff840be8d77a3e040b45434\n']
    # with open(os.path.join(tst_dataset, 'test_R_tr.txt'), 'r') as f:
    #     datasets = f.readlines()

    with open(os.path.join(tst_dataset, 'val_20.txt'), 'r') as f:
        datasets = f.readlines()
    # datasets = ['921b2a5eeff840be8d77a3e040b45434\n']
    MAcc_ez = []
    MAcc_med = []
    MAcc_hard = []

    mAcc = [[],[],[],[]]

    i_mean_ez = []
    i_mean_med = []
    i_mean_hard = []

    i_mean_ns = [[],[],[],[]]
    i = 0
    maxmax_iez = 0
    maxmax_imed = 0
    maxmax_ihard = 0

    recog = [[],[],[],[]]
    yes_detect_ez = []
    yes_detect_med = []
    yes_detect_hard = []
    d = 1
    mean_mave_x, mean_mave_y = [],[]
    mean_F1 = []
    mean_recall = []
    for data in datasets:
        # data = '000107.txt'
        # input('Enter to evalutate')
        # print(data)
        # print(data[:-1])
        # exit()

        # pred, real, mAP_ez, mAP_med, mAP_hard, mean_i_ez, mean_i_med, mean_i_hard, max_i_ez, max_i_med, max_i_hard, detections_ez, detections_med, detections_hard, F1,Recall,mave_x, mave_y = detect(
        #     data[:-1])

        mAP_ns, mean_i, mean_iou, max_i, max_iou, detections, F1, Recall, mave = detect(data[:-1])
        # exit()


        # dist threshould = dist_threshoulds = [0.5,1,2,4]
        try:
            mean_mave_x = mean_mave_x + mave[0]
            mean_mave_y = mean_mave_y + mave[1]
        except:
            print(mave)
        # print(mean_mave_x)
        # input()
        mean_F1.append(F1)
        mean_recall.append(Recall)
        # print('Ap:', ap)
        #print(mAP_ns)
        for j in range(4):
            mAcc[j].append(mAP_ns[j])
            i_mean_ns[j].append(mean_iou[j])
            recog[j].append(detections[j])

        #MAcc_ez.append(mAP_ez)
        #MAcc_med.append(mAP_med)
        #MAcc_hard.append(mAP_hard)
        # print('A_Iou:', i)
        #i_mean_ez.append(mean_i_ez)
        #i_mean_med.append(mean_i_med)
        #i_mean_hard.append(mean_i_hard)
        # print('Corr_dtc:',corr_dtc)
        #yes_detect_ez.append(detections_ez)
        #yes_detect_med.append(detections_med)
        #yes_detect_hard.append(detections_hard)
        # print('\n')

        print('*************************************************************************')
        print('Data: ', d, ' of ', len(datasets))
        # ---------------------------------- EASY--------------------------------------------
        print('*************************************************************************')
        print('---------------------------Disc 0.5--------------------------------------')
        print('mAp:', round(np.mean(clearning_nd(mAcc[0])) * 100, 2), '%')
        print('Mean Iou:', round(np.mean(i_mean_ns[0]), 4))
        print('Average detection: ', round(np.mean(clearning_nd(recog[0])) * 100, 2), '%')
        print('---------------------------Disc 1.0--------------------------------------')
        print('mAp:', round(np.mean(clearning_nd(mAcc[1])) * 100, 2), '%')
        print('Mean Iou:', round(np.mean(i_mean_ns[1]), 4))
        print('Average detection: ', round(np.mean(clearning_nd(recog[1])) * 100, 2), '%')
        print('---------------------------Disc 2.0--------------------------------------')
        print('mAp:', round(np.mean(clearning_nd(mAcc[2])) * 100, 2), '%')
        print('Mean Iou:', round(np.mean(i_mean_ns[2]), 4))
        print('Average detection: ', round(np.mean(clearning_nd(recog[2])) * 100, 2), '%')
        print('---------------------------Disc 4.0--------------------------------------')
        print('mAp:', round(np.mean(clearning_nd(mAcc[3])) * 100, 2), '%')
        print('Mean Iou:', round(np.mean(i_mean_ns[3]), 4))
        print('Average detection: ', round(np.mean(clearning_nd(recog[3])) * 100, 2), '%')
        print('-------------')
        print('MEAV X:', round(np.mean(mean_mave_x), 4))
        print('MEAV Y:', round(np.mean(mean_mave_y), 4))
        print('MEAV:', round((np.mean(mean_mave_x)+np.mean(mean_mave_y))/2, 4))
        print('F1-Score', round(np.mean(mean_F1), 4))
        print('Recall', round(np.mean(mean_recall), 4))
        print('\n')
        #exit()
        d += 1

''' 

        if max_i_ez > maxmax_iez:
            maxmax_iez = max_i_ez
        if max_i_med > maxmax_imed:
            maxmax_imed = max_i_med
        if max_i_hard > maxmax_ihard:
            maxmax_ihard = max_i_hard


        mean_mave_x.append(mave_x)


        mean_mave_y.append(mave_y)

        mean_F1.append(F1)
        mean_recall.append(Recall)
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
        print('MEAV X:', round(np.mean(mean_mave_x), 4))
        print('MEAV Y:', round(np.mean(mean_mave_y), 4))
        print('MEAV:', round((np.mean(mean_mave_x)+np.mean(mean_mave_y))/2, 4))
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
        print('F1-Score', round(np.mean(mean_F1), 4))
        print('Recall', round(np.mean(mean_recall), 4))
        print('\n\n\n\n')
        
        d += 1

'''

