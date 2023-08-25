# NUSCENES

from tensorflow.keras.utils import Sequence
import math
import numpy as np
import os
import config_model as cfg
from utils3 import iou3d, iou2d
import matplotlib.pyplot as plt
import cv2
from nuscenes.nuscenes import NuScenes

from vox_pillar import pillaring
nusc = NuScenes(version='v1.0-trainval', dataroot='D:/SCRIPTS/Nuscenes', verbose=True)

X_div = cfg.X_div
# Y_div = cfg.Y_div
Z_div = cfg.Z_div

x_step = cfg.stepx
z_step = cfg.stepz
class SequenceData(Sequence):

    def __init__(self, model, dir, target_size, batch_size, shuffle=True, data_aug=False):
        self.model = model
        self.data_aug = data_aug
        self.datasets = []
        self.nb_anchors = cfg.nb_anchors
        self.nb_classes = cfg.nb_classes
        self.anchor = cfg.anchor
        self.pos_iou = cfg.positive_iou_threshold
        self.neg_iou = cfg.negative_iou_threshold

        self.TRAIN_TXT = cfg.TRAIN_TXT
        self.VAL_TXT = cfg.VAL_TXT
        self.LABEL_PATH = cfg.LABEL_PATH
        self.LIDAR_PATH = cfg.LIDAR_PATH
        self.IMG_PATH = cfg.IMG_PATH
        self.image_shape = cfg.img_shape
        self.classes = cfg.classes
        self.x_max = cfg.x_max
        self.x_min = cfg.x_min

        self.y_max = cfg.y_max
        self.y_min = cfg.y_min

        self.z_max = cfg.z_max
        self.z_min = cfg.z_min

        self.rot_max = cfg.rot_norm
        self.classes_names = [k for k, v in self.classes.items()]

        # self.w_max = cfg.w_max
        # self.h_max = cfg.h_max
        # self.l_max = cfg.l_max

        if self.model == 'train':
            with open(os.path.join(dir, self.TRAIN_TXT), 'r') as f:
                self.datasets = self.datasets + f.readlines()

            # elif self.data_aug == True:

            #     with open(os.path.join(dir, 'train_R_Half_aug.txt'), 'r') as f:
            #         self.datasets = self.datasets + f.readlines()




        elif self.model == 'val':
            with open(os.path.join(dir, self.VAL_TXT), 'r') as f:
                self.datasets = self.datasets + f.readlines()

        self.image_size = target_size[0:2]
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.datasets))
        self.shuffle = shuffle

    def __len__(self):
        num_imgs = len(self.datasets)
        return math.ceil(num_imgs / float(self.batch_size))

    def __getitem__(self, idx):
        batch_indexs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.datasets[k] for k in batch_indexs]
        X, y = self.data_generation(batch)
        # print(y.shape)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read(self, dataset):
        dataset = dataset.strip()

        label_path = self.LABEL_PATH
        lidar_path = self.LIDAR_PATH
        radar_path = self.RADAR_PATH
        img_path = self.IMG_PATH
        # dataset = dataset[:-4]


        my_sample = nusc.get('sample', dataset)
        
        RADAR = nusc.get('sample_data', my_sample['data']['RADAR_FRONT'])
        LIDAR = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
        CAMERA = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])

        my_ego_token = LIDAR['ego_pose_token']
        


        radar_name = RADAR['filename']
        radar_name = radar_name.replace('samples/RADAR_FRONT/','')
        radar_name = radar_name.replace('.pcd','.npy')
        # 'samples/RADAR_FRONT/n015-2018-07-18-11-07-57+0800__RADAR_FRONT__1531883530960489.pcd'

        lidar_name = LIDAR['filename']
        lidar_name = lidar_name.replace('samples/LIDAR_TOP/','')
        lidar_name = lidar_name.replace('.pcd.bin','.npy')
        # 'samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530949817.pcd.bin'

        camera_name = CAMERA['filename']
        'samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530912460.jpg'



        # -------------------------- INPUT CAM ------------------------------------------
        img = cv2.imread(Nusc_path + camera_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.
        img = cv2.resize(img,(self.image_shape))
        # -------------------------- INPUT LIDAR ----------------------------------------
        #             z up  y front
        #             ^    ^
        #             |   /
        #             |  /
        #             | /
        #             |/
        # left ------ 0 ------> x right


        # PASSING TO DIFFERENT REFERENCE
        #               y up  z front
        #               ^    ^
        #               |   /
        #               |  /
        #               | /
        #               |/
        # left x ------ 0 ------>  right

        lidar = np.load(lidar_path + lidar_name + '.npy')  # [0,1,2] -> Z,X,Y
        vox_pillar_L, pos_L = pillaring(lidar)  # (10000,20,7)/ (10000,3)

        # -------------------------- INPUT RADAR ----------------------------------------
        # Original RADAR   CONFIRM WITH PHOTO
            #                    X front
            #               ^    ^
            #               |   /
            #               |  /
            #               | /
            #               |/
            # leftY <------ 0 ------  right

            # NUSCENES REFERENCE
            #                     Y front
            #               ^    ^
            #               |   /
            #               |  /
            #               | /
            #               |/
            # left X ------ 0 ------>  right
        radar = np.load(radar_path + radar_name + '.npy')  
        vox_pillar_R, pos_R = pillaring(radar)  # (10,5,5)/ (10,3)




        # ------------------------- ANNOTATION Tensor ---------------------------------

        class_matrix = np.zeros([X_div, Z_div, self.nb_anchors, self.nb_classes])
        conf_matrix = np.zeros([X_div, Z_div, self.nb_anchors,1])
        pos_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 3])
        dim_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 3])
        rot_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 1])
        velo_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 2])

        #             z up  y front
        #             ^    ^
        #             |   /
        #             |  /
        #             | /
        #             |/
        # left ------ 0 ------> x right


        # PASSING TO DIFFERENT REFERENCE
        #               y up  z front
        #               ^    ^
        #               |   /
        #               |  /
        #               | /
        #               |/
        # left x ------ 0 ------>  right


        # with open(label_path + dataset + '.txt', 'r') as f:
        #     label = f.readlines()
        
        my_ego = nusc.get('ego_pose',my_ego_token)

        my_pos = my_ego['translation']

        for i in range(len(my_sample['anns'])):
            my_annotation_token = my_sample['anns'][i]
            label = nusc.get('sample_annotation', my_annotation_token)
            category = label['category_name']

            pos_x = (label['translation'][0] - my_pos[0])   # original X --->  X  changed
            pos_y = (label['translation'][1] - my_pos[2])   # original Y --->  Z  changed
            pos_z = (label['translation'][2] - my_pos[1])   # original Z --->  Y  changed

            width = label['size'][0]
            lenght = label['size'][1]
            height = label['size'][2]
            
            quaternion = label['rotation']
            rot = 2*np.arccos(quaternion[0])

            velo_x = nusc.box_velocity(my_annotation_token)[0]
            velo_z = nusc.box_velocity(my_annotation_token)[1]

            # l = l.replace('\n', '')
            # l = l.split(' ')
            # l = np.array(l)
            # maxIou = 0
            #######  Normalizing the Data ########
            if category in self.classes_names:
                cla = int(self.classes[l[0]])

                norm_x = (pos_x + abs(self.x_min)) / (
                            self.x_max - self.x_min)  # Center Position X in relation to max X 0-1
                norm_y = (pos_y + abs(self.y_min)) / (
                            self.y_max - self.y_min)  # Center Position Y in relation to max Y 0-1
                norm_z = (pos_z + abs(self.z_min)) / (
                            self.z_max - self.z_min)  # Center Position Z in relation to max Z 0-1

                out_of_size = np.array([norm_x, norm_y, norm_z])

                if np.any(out_of_size > 1) or np.any([velo_x,velo_z] == np.nan):
                    continue
                else:
                    loc = [X_div * norm_x, Z_div * norm_z]

                    loc_i = int(loc[0])
                    loc_k = int(loc[1])

                    if conf_matrix[loc_i, loc_k, 0] == 0:

                        lbl = [pos_x, pos_y, pos_z, width, height, lenght, rot]
                        
                        diag = [np.sqrt(pow(a[0],2)+pow(a[2],2)) for a in self.anchor]
                        for i in range(-1,2):
                            for j in range(-1,2):
                                if (0 < loc_i + i < X_div) and (0 < loc_k + j < Z_div):
                                    # x_v = (loc_i+i)*(0.16)* 2 + self.x_min # Real --- xId * xStep * downscalingFactor + xMin;
                                    # z_v = (loc_k+j)*(0.16)* 2 + self.z_min # Real --- zId * zStep * downscalingFactor + zMin;

                                    anchors = [[((loc_i+i) * x_step*2)+self.x_min, a[3], ((loc_k+j) * z_step*2)+self.z_min, a[0], a[1], a[2], a[4]] for a in self.anchor]

                                    iou = [iou2d(a, lbl) for a in anchors]
                                    if np.max(iou) > maxIou:
                                        maxIou = np.max(iou)
                                        best_a = iou.index(maxIou)

                                    for a in range(self.nb_anchors):
                                        if iou[a] > self.pos_iou:
                                            conf_matrix[loc_i+i, loc_k+j, a, 0] = 1  # - abs(x_v) - abs(z_v) #Implement Probability
                                            class_matrix[loc_i+i, loc_k+j, a, cla] = 1
                                            # print(x_v)
                                            # print(anchors[a][0])
                                            
                                            x_cell = (lbl[0] - anchors[a][0])/ diag[a]
                                            y_cell = (lbl[1] - anchors[a][1])/anchors[a][4]
                                            z_cell = (lbl[2] - anchors[a][2])/ diag[a]

                                            w_cell = np.log(np.clip((lbl[3]/anchors[a][3]),1e-15,1e+15))
                                            h_cell = np.log(np.clip((lbl[4]/anchors[a][4]),1e-15,1e+15))
                                            l_cell = np.log(np.clip((lbl[5]/anchors[a][5]),1e-15,1e+15))

                                            rot_cell = np.sin(lbl[6] - anchors[a][6])

                                            pos_matrix[loc_i+i, loc_k+j, a, :] = [x_cell, y_cell, z_cell]
                                            dim_matrix[loc_i+i, loc_k+j, a, :] = [w_cell, h_cell, l_cell]
                                            rot_matrix[loc_i+i, loc_k+j, a, 0] = rot_cell
                                            velo_matrix[loc_i+i, loc_k+j, a, :] =    

                                        elif iou[a] < self.neg_iou:
                                            conf_matrix[loc_i+i, loc_k+j, a, 0] = 0

                                        else:
                                            conf_matrix[loc_i+i, loc_k+j, a, 0] = 0
                        if maxIou < self.pos_iou:
                            conf_matrix[loc_i, loc_k, best_a, 0] = 1  # - abs(x_v) - abs(z_v) #Implement Probability
                            class_matrix[loc_i, loc_k, best_a, cla] = 1
                            # print(x_v)
                            # print(anchors[a][0])
                            
                            x_cell = (lbl[0] - anchors[best_a][0])/ diag[best_a]
                            y_cell = (lbl[1] - anchors[best_a][1])/anchors[best_a][4]
                            z_cell = (lbl[2] - anchors[best_a][2])/ diag[best_a]

                            w_cell = np.log(np.clip((lbl[3]/anchors[best_a][3]),1e-15,1e+15))
                            h_cell = np.log(np.clip((lbl[4]/anchors[best_a][4]),1e-15,1e+15))
                            l_cell = np.log(np.clip((lbl[5]/anchors[best_a][5]),1e-15,1e+15))

                            rot_cell = np.sin(lbl[6] - anchors[best_a][6])

                            pos_matrix[loc_i, loc_k, best_a, :] = [x_cell, y_cell, z_cell]
                            dim_matrix[loc_i, loc_k, best_a, :] = [w_cell, h_cell, l_cell]
                            rot_matrix[loc_i, loc_k, best_a, 0] = rot_cell

                # print(maxIou)
            else:
                continue

        # conf1 = np.dstack((conf_matrix[:,:,:2,0],np.zeros((X_div, Z_div, 1))))
        # conf2 = np.dstack((conf_matrix[:,:,2:,0],np.zeros((X_div, Z_div, 1))))
        # img = np.hstack([conf1,conf2])

        # plt.imshow(img)
        # plt.show()
        # exit()
        # print(conf_matrix.shape)
        # print(pos_matrix.shape)
        # print(dim_matrix.shape)
        # print(rot_matrix.shape)
        # print(class_matrix.shape)
        output = np.concatenate((conf_matrix, pos_matrix, dim_matrix, rot_matrix, class_matrix), axis=-1)

        # print(output.shape)
        # exit()
        return vox_pillar_L, pos_L, vox_pillar_R, pos_R,img, output

    def data_generation(self, batch_datasets):
        pillar_L = []
        pillar_pos_L = []
        pillar_R = []
        pillar_pos_R = []
        imgs = []
        lbl = []
        
        for dataset in batch_datasets:
            vox_pillar_L, pos_L, vox_pillar_R, pos_R, img, output  = self.read(dataset)

            pillar_L.append(vox_pillar_L)
            pillar_pos_L.append(pos_L)

            pillar_R.append(vox_pillar_R)
            pillar_pos_R.append(pos_R)

            imgs.append(img)

            lbl.append(output)
            

        X_pL = np.array(pillar_L)
        X_posL = np.array(pillar_pos_L)
        X_pR = np.array(pillar_R)
        X_posR = np.array(pillar_pos_R)
        X_imgs = np.array(imgs)

        X = [X_pL, X_posL, X_pR, X_posR, X_imgs]
        
        lbl = np.array(lbl)

        return X, lbl


if __name__ == '__main__':
    # dataset_path = 'C:/Users/Marcelo/Desktop/SCRIPTS/MySCRIPT/Doc_code/data/'
    # dataset_path = 'D:/SCRIPTS/Doc_code/data/'
    dataset_path = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/'
    input_shape = (504, 504, 3)
    batch_size = 1
    train_gen = SequenceData('train', dir=dataset_path, target_size=input_shape, batch_size=batch_size, data_aug=False)
    # train_gen[2]
    print(train_gen[0])

