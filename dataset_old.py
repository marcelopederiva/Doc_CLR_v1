from tensorflow.keras.utils import Sequence
import math
import numpy as np
import os
import config_model as cfg
from utils3 import iou3d, iou2d
import matplotlib.pyplot as plt

from vox_pillar import pillaring

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
        dataset = dataset[:-4]
        # print(dataset)
        # dataset = '000049'
        # dataset = '003007'

        # -----C:/Users/Marcelo/Desktop/SCRIPTS/KITTI/object/training/label_2/
        # label_path = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/label_norm_lidar_v1/'
        # lidar_path = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/input/lidar_crop/'
        # dataset = dataset[:-4]
        # dataset = '000169'
        # dataset = '004283'
        # dataset = '000266'
        # -------------------------- INPUT DATA ----------------------------------------

        cam3d = np.load(lidar_path + dataset + '.npy')  # [0,1,2] -> Z,X,Y
        vox_pillar, pos = pillaring(cam3d)  # (10000,20,7)/ (10000,3)

        # print(pos[:4])
        # pos_ = np.vstack((pos[:,0],pos[:,2]))
        # pos_ =
        # pos = (pos[:,:2]).astype(int) # (1000,3)
        # print(pos[:4])
        # exit()
        # ------------------------- Predicting Tensor ---------------------------------

        # label_matrix = np.zeros([X_div, Z_div, 9])
        # label_matrix[...,1:2] = 0

        class_matrix = np.zeros([X_div, Z_div, self.nb_anchors, self.nb_classes])
        conf_matrix = np.zeros([X_div, Z_div, self.nb_anchors,1])
        pos_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 3])
        dim_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 3])
        rot_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 1])

        # anchors = [[0.5,0.5,0.5,a[0],a[1],a[2],a[3]] for a in self.anchor]
        

        with open(label_path + dataset + '.txt', 'r') as f:
            label = f.readlines()
        
        for l in label:
            l = l.replace('\n', '')
            l = l.split(' ')
            l = np.array(l)
            maxIou = 0
            #######  Normalizing the Data ########
            if l[0] in self.classes_names:
                cla = int(self.classes[l[0]])
                norm_x = (float(l[11]) + abs(self.x_min)) / (
                            self.x_max - self.x_min)  # Center Position X in relation to max X 0-1
                norm_y = (float(l[12]) + abs(self.y_min)) / (
                            self.y_max - self.y_min)  # Center Position Y in relation to max Y 0-1
                norm_z = (float(l[13]) + abs(self.z_min)) / (
                            self.z_max - self.z_min)  # Center Position Z in relation to max Z 0-1

                norm_w = float(l[9]) / (self.x_max - self.x_min)  # Dimension W in relation to max X 0-1
                norm_h = float(l[8]) / (self.y_max - self.y_min)  # Dimension H in relation to max Y 0-1
                norm_l = float(l[10]) / (self.z_max - self.z_min)  # Dimension L in relation to max Z 0-1

                rot = float(l[14])
                # if rot < 0: rot = -rot
                norm_rot = rot / self.rot_max  # Rotation in relation to max Rot in Y axis 0-1

                out_of_size = np.array([norm_x, norm_y, norm_z, norm_w, norm_h, norm_l, norm_rot])
                # print(cla)
                # print(out_of_size)
                # print('\n')
                if np.any(out_of_size > 1):
                    continue
                else:
                    loc = [X_div * norm_x, Z_div * norm_z]

                    loc_i = int(loc[0])
                    # loc_j = int(loc[1])
                    loc_k = int(loc[1])

                    # x_cell = loc[0] - loc_i
                    # y_cell = norm_y
                    # z_cell = loc[1] - loc_k

                    # w_cell = norm_w * X_div
                    # h_cell = norm_h
                    # l_cell = norm_l * Z_div

                    # lbl = [0, 0, 0, float(l[9]), float(l[8]), float(l[10]), norm_rot]
                    #
                    # iou = [iou2d(a, lbl) for a in anchors]
                    # print('Lbl: ',lbl)
                    # print('IoU:',iou,'\n')
                    
                    if conf_matrix[loc_i, loc_k, 0] == 0:
                        # print('Car')
                        # min_i, max_i = np.clip(loc_i - 3, 0, X_div), np.clip(loc_i + 4, 0, X_div)
                        # min_k, max_k = np.clip(loc_k - 3, 0, Z_div), np.clip(loc_k + 4, 0, Z_div)
                        # x_central_real = loc_i*(0.16)* 2 + self.x_min
                        # print(x_central_real)
                        # print(float(l[11]))
                        # exit()
                        # z_central_real = loc_k*(0.16)* 2 + self.z_min

                        x_central_real = float(l[11])
                        z_central_real = float(l[13])


                        anchors = [[x_central_real, a[3], z_central_real, a[0], a[1], a[2], a[4]] for a in self.anchor]
                        diag = [np.sqrt(pow(a[0],2)+pow(a[2],2)) for a in self.anchor]
                        # print(float(l[11]))
                        # print(anchors)
                        # exit()
                        for i in range(-1,2):
                            for j in range(-1,2):
                                if (0 < loc_i + i < X_div) and (0 < loc_k + j < Z_div):
                                    # x_v = (loc_i+i)*(0.16)* 2 + self.x_min # Real --- xId * xStep * downscalingFactor + xMin;
                                    # z_v = (loc_k+j)*(0.16)* 2 + self.z_min # Real --- zId * zStep * downscalingFactor + zMin;

                                    x_v = float(l[11]) + (i*x_step)
                                    z_v = float(l[13]) + (j*z_step)

                                    lbl = [x_v, float(l[12]), z_v, float(l[9]), float(l[8]), float(l[10]), rot]

                                    # print(lbl)
                                    # print(anchors[0])
                                    iou = [iou2d(a, lbl) for a in anchors]
                                    if np.max(iou) > maxIou:
                                        maxIou = np.max(iou)
                                        best_a = iou.index(maxIou)
                                    # print(iou)
                                    # exit()
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
        return vox_pillar, pos, output

    def data_generation(self, batch_datasets):
        pillar = []
        pillar_pos = []
        lbl = []

        for dataset in batch_datasets:
            vox_pillar, pos, output = self.read(dataset)
            pillar.append(vox_pillar)
            pillar_pos.append(pos)

            lbl.append(output)

        X_p = np.array(pillar)
        X_pos = np.array(pillar_pos)
        X = [X_p, X_pos]

        lbl = np.array(lbl)

        # labels = [lbl_conf, lbl_pos, lbl_dim, lbl_rot, lbl_class]

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

