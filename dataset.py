# NUSCENES

from tensorflow.keras.utils import Sequence
import math
import numpy as np
import os
import config_model as cfg
from utils3 import iou3d, iou2d
import matplotlib.pyplot as plt
import cv2
import vox_pillar_l_c_plus  # O módulo C++ compilado
# from nuscenes.nuscenes import NuScenes
# import dataset_input_cy
# import vox_pillar_l_cy
# import vox_pillar_r_cy 
# nusc = NuScenes(version='v1.0-trainval', dataroot=cfg.NUSC_PATH, verbose=True)

X_div = cfg.X_div
# Y_div = cfg.Y_div
Z_div = cfg.Z_div

x_step = cfg.stepx
z_step = cfg.stepz

def load_lidar_points(lidar_path, dataset_id):
    # Carrega a nuvem bruta
    
    lidar = np.load(lidar_path + dataset_id + '.npy')

    cam3d = np.copy(lidar)
    cam3d[..., 0] = lidar[..., 2]
    cam3d[..., 1] = - lidar[..., 0]
    cam3d[..., 2] =  lidar[..., 1]
    cam3d[..., 3] = lidar[..., 3]
    # print('0:',np.max(lidar[:,0]),np.min(lidar[:,0]))
    # print('1:',np.max(lidar[:,1]),np.min(lidar[:,1]))
    # print('2:',np.max(lidar[:,2]),np.min(lidar[:,2]))
    # print('2:',np.max(lidar[:,3]),np.min(lidar[:,3]))
    # exit()
    return cam3d

def load_image(img_path, dataset_id, image_shape):
    # Carrega a imagem normal
    img = cv2.imread(img_path + dataset_id + '.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # se quiser no formato RGB
    img = cv2.resize(img, image_shape)
    img = img / 255.0
    return img

def load_labels_txt(label_path, dataset_id):
    # Lê o arquivo .txt e retorna as anotações brutas (lista de linhas, ou algo processado)
    # Exemplo simples:
    with open(label_path + dataset_id + '.txt', 'r') as f:
        label_lines = f.readlines()
    return label_lines


class SequenceData(Sequence):

    def __init__(self, model, dir, target_size, batch_size, shuffle=True, data_aug=False):
        self.model = model
        self.data_aug = data_aug
        self.datasets = []
        self.nb_anchors = cfg.nb_anchors
        self.nb_classes = cfg.nb_classes
        self.anchor = cfg.anchor
        self.pos_iou = cfg.pos_iou
        self.neg_iou = cfg.neg_iou

        self.TRAIN_TXT = cfg.TRAIN_TXT
        self.VAL_TXT = cfg.VAL_TXT
        self.VAL_MINI_TXT = cfg.VAL_MINI_TXT
        self.NUSC_PATH = cfg.NUSC_PATH
        # self.RADAR_PATH = cfg.RADAR_PATH
        self.LIDAR_PATH = cfg.LIDAR_PATH
        self.IMG_PATH = cfg.IMG_PATH
        self.image_shape = cfg.image_shape
        self.classes = cfg.classes
        self.x_max = cfg.x_max
        self.x_min = cfg.x_min

        self.y_max = cfg.y_max
        self.y_min = cfg.y_min

        self.z_max = cfg.z_max
        self.z_min = cfg.z_min

        self.rot_max = cfg.rot_max

        self.vel_max = cfg.vel_max
        self.vel_min = cfg.vel_min

        self.classes_names = [k for k, v in self.classes.items()]
        


        if self.model == 'train':
            with open(os.path.join(dir, self.TRAIN_TXT), 'r') as f:
                self.datasets = self.datasets + f.readlines()

        elif self.model == 'val':
            with open(os.path.join(dir, self.VAL_TXT), 'r') as f:
                self.datasets = self.datasets + f.readlines()
        
        elif self.model == 'val_mini':
            with open(os.path.join(dir, self.VAL_MINI_TXT), 'r') as f:
                self.datasets = self.datasets + f.readlines()

        self.samples_len = len(self.datasets)

        if self.model == 'train' and self.data_aug:
            self.full_len = 2 * self.samples_len
        else:
            self.full_len = self.samples_len

        self.image_size = target_size[0:2]
        self.batch_size = batch_size
        self.indexes = np.arange(self.full_len)
        # print(self.full_len)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Cada batch terá 'batch_size' itens, e o total de steps cobre 'full_len' amostras
        return len(self.indexes) // self.batch_size  # Descarta o restante

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        pillar_L   = []
        pillar_pos = []
        imgs       = []
        lbl        = []
        for k in batch_indexes:
            if k < self.samples_len: 
                # Versão normal
                dataset_id = self.datasets[k].strip()
                vox_l, pos_l, img, output = self.build_sample(dataset_id, is_flipped=False)
            else:
                # Versão flipada
                dataset_id = self.datasets[k - self.samples_len].strip()
                vox_l, pos_l, img, output = self.build_sample(dataset_id, is_flipped=True)

            pillar_L.append(vox_l)
            pillar_pos.append(pos_l)
            imgs.append(img)
            lbl.append(output)

        # X_pL   = np.array(pillar_L)
        # X_pos  = np.array(pillar_pos)
        # X_imgs = np.array(imgs)
        # Y_lbl  = np.array(lbl)

        X_p = np.array(pillar_L, dtype=np.float16)
        X_pos = np.array(pillar_pos, dtype=np.float32)
        X_imgs = np.array(imgs, dtype=np.float16)
        Y_lbl  = np.array(lbl)

        X = [X_p, X_pos, X_imgs]
        return X, Y_lbl

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def build_label_tensor(self, label_lines, is_flipped=False):
        class_matrix = np.zeros([X_div, Z_div, self.nb_anchors, self.nb_classes])
        conf_matrix = np.zeros([X_div, Z_div, self.nb_anchors,1])
        pos_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 3])
        dim_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 3])
        rot_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 1])

        for l in label_lines:
            l = l.replace('\n','')
            l = l.split(' ')
            l = np.array(l)
            category = str(l[0]) # class --> int
            pos_x = float(l[1]) # Center Position X 
            pos_y = float(l[2]) # Center Position Y 
            pos_z = float(l[3]) # Center Position Z 

            width = float(l[4]) # Dimension W 
            height = float(l[5]) # Dimension H
            lenght = float(l[6]) # Dimension L
            rot = float(l[7]) # Rotation in Y
            maxIou = 0

            if is_flipped:
                # Inverte o X e também o ângulo
                pos_x = -pos_x
                rot   = -rot
        
            #######  Normalizing the Data ########
            if category in self.classes_names:
                cla = int(self.classes[category])

                norm_x = (pos_x + abs(self.x_min)) / (
                            self.x_max - self.x_min)  # Center Position X in relation to max X 0-1
                norm_z = (pos_z + abs(self.z_min)) / (
                            self.z_max - self.z_min)  # Center Position Z in relation to max Z 0-1
                if pos_x > pos_z or pos_x < -pos_z:
                    fov = 2
                else: 
                    fov=0.5 



                out_of_size = np.array([norm_x, norm_z, fov])

                if (np.any(out_of_size > 1)or np.any(out_of_size <0)):
                    continue
                else:
                    loc = [X_div * norm_x, Z_div * norm_z]

                    loc_i = int(loc[0])
                    loc_k = int(loc[1])

                    if conf_matrix[loc_i, loc_k, 0] == 0:
                        
                        x_central_real = pos_x
                        z_central_real = pos_z

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

                                    x_v = x_central_real + (i*x_step*2)
                                    z_v = z_central_real + (j*z_step*2)

                                    lbl = [x_v, pos_y, z_v, width, height, lenght, rot]

                                    # print(lbl)
                                    # print(anchors[0])
                                    iou = [iou2d(a, lbl) for a in anchors]
                                    if np.max(iou) > maxIou:
                                        maxIou = np.max(iou)
                                        best_a = iou.index(maxIou)
                                        best_lbl = lbl
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
                            
                            x_cell = (best_lbl[0] - anchors[best_a][0])/ diag[best_a]
                            y_cell = (best_lbl[1] - anchors[best_a][1])/anchors[best_a][4]
                            z_cell = (best_lbl[2] - anchors[best_a][2])/ diag[best_a]

                            w_cell = np.log(np.clip((best_lbl[3]/anchors[best_a][3]),1e-15,1e+15))
                            h_cell = np.log(np.clip((best_lbl[4]/anchors[best_a][4]),1e-15,1e+15))
                            l_cell = np.log(np.clip((best_lbl[5]/anchors[best_a][5]),1e-15,1e+15))

                            rot_cell = np.sin(best_lbl[6] - anchors[best_a][6])

                            pos_matrix[loc_i, loc_k, best_a, :] = [x_cell, y_cell, z_cell]
                            dim_matrix[loc_i, loc_k, best_a, :] = [w_cell, h_cell, l_cell]
                            rot_matrix[loc_i, loc_k, best_a, 0] = rot_cell
            else:
                continue

        output = np.concatenate((conf_matrix, pos_matrix, dim_matrix, rot_matrix, class_matrix), axis=-1)
        return output
    
    def build_sample(self, dataset_id, is_flipped=False):
        """
        Monta a amostra final (pilares voxelizados, imagem, e label tensor),
        decidindo se será flipada ou não.
        """
        # 1) Carrega a nuvem de pontos bruta
        lidar = load_lidar_points(self.LIDAR_PATH, dataset_id)

        # 2) Se for flip, invertemos o eixo X dos pontos ANTES de voxelizar
        if is_flipped:
            # Supondo que lidar[..., 0] seja o X
            lidar[..., 0] *= -1

        # 3) Faz a voxelização
        vox_pillar_L, pos_L = vox_pillar_l_c_plus.pillaring_l(
        lidar,
        self.image_shape,
        cfg.input_pillar_shape,
        cfg.input_pillar_indices_shape,
        cfg.max_group,
        cfg.max_pillars,
        cfg.x_min, cfg.x_diff,
        cfg.y_min, cfg.y_diff,
        cfg.z_min, cfg.z_diff
        )

        # 4) Carrega a imagem
        img = load_image(self.IMG_PATH, dataset_id, self.image_shape)
        # 5) Flip da imagem (horizontal) se for o caso
        if is_flipped:
            img = np.fliplr(img)

        # 6) Monta o tensor de saída (labels), levando em conta o flip
        #    Primeiro carregamos as anotações brutas.
        label_lines = load_labels_txt(self.NUSC_PATH + 'data_an/', dataset_id)
        output = self.build_label_tensor(label_lines, is_flipped=is_flipped)

        return vox_pillar_L, pos_L, img, output


    # def read(self, dataset):
        dataset = dataset.strip()
        dataset = 'f9bdc7dd40074505bf81c3eef5f13dca'

        Nusc_path = self.NUSC_PATH
        lidar_path = self.LIDAR_PATH
        img_path = self.IMG_PATH


        # -------------------------- INPUT CAM ------------------------------------------
        img = cv2.imread(img_path + dataset+ '.jpg')
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


        # PASSED TO DIFFERENT REFERENCE
        #               y up  z front
        #               ^    ^
        #               |   /
        #               |  /
        #               | /
        #               |/
        # left x ------ 0 ------>  right
        # vox_pillar_L, pos_L =  np.zeros([10000,20,7]), np.zeros([10000,3])
        lidar = np.load(lidar_path + dataset + '.npy')  # [0,1,2] -> Z,X,Y
        # vox_pillar_L, pos_L = vox_pillar_l_cy.pillaring_l(lidar)  # (10000,20,7)/ (10000,3)
        # ------------------------- ANNOTATION Tensor ---------------------------------

        class_matrix = np.zeros([X_div, Z_div, self.nb_anchors, self.nb_classes])
        conf_matrix = np.zeros([X_div, Z_div, self.nb_anchors,1])
        pos_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 3])
        dim_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 3])
        rot_matrix = np.zeros([X_div, Z_div, self.nb_anchors, 1])

        with open(Nusc_path + 'data_an/' + dataset + '.txt', 'r') as doc:
            label = doc.readlines()

        for l in label:
            l = l.replace('\n','')
            l = l.split(' ')
            l = np.array(l)
            category = str(l[0]) # class --> int
            pos_x = float(l[1]) # Center Position X 
            pos_y = float(l[2]) # Center Position Y 
            pos_z = float(l[3]) # Center Position Z 

            width = float(l[4]) # Dimension W 
            height = float(l[5]) # Dimension H
            lenght = float(l[6]) # Dimension L
            rot = float(l[7]) # Rotation in Y
            maxIou = 0
        
            #######  Normalizing the Data ########
            if category in self.classes_names:
                cla = int(self.classes[category])

                norm_x = (pos_x + abs(self.x_min)) / (
                            self.x_max - self.x_min)  # Center Position X in relation to max X 0-1
                norm_y = (pos_y + abs(self.y_min)) / (
                            self.y_max - self.y_min)  # Center Position Y in relation to max Y 0-1
                norm_z = (pos_z + abs(self.z_min)) / (
                            self.z_max - self.z_min)  # Center Position Z in relation to max Z 0-1
                if pos_x > pos_z or pos_x < -pos_z:
                    fov = 2
                else: 
                    fov=0.5 



                out_of_size = np.array([norm_x, norm_z, fov])

                if (np.any(out_of_size > 1)or np.any(out_of_size <0)):
                    continue
                else:
                	# Implementing the grid threshold

                	# norm_x[norm_x==1]=0.9999
                	# norm_x[norm_x==0]=0.0001

                	# norm_z[norm_z==1]=0.9999
                	# norm_z[norm_z==0]=0.0001

                	# velo_x[velo_x==1]=0.9999
                	# velo_x[velo_x==0]=0.0001

                	# velo_z[velo_z==1]=0.9999
                	# velo_z[velo_z==0]=0.0001

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
        # np.save('label_t',velo_matrix)
        # np.save('radar_t',cam_3d)            
        # conf1 = np.dstack((conf_matrix[:,:,:2,0],np.zeros((X_div, Z_div, 1))))
        # conf2 = np.dstack((conf_matrix[:,:,2:,0],np.zeros((X_div, Z_div, 1))))
        # img = np.hstack([conf1,conf2])
        # from occ_radar_ex import tensor_see
        # v = (velo_matrix * (self.vel_max - self.vel_min)) - abs(velo_matrix)
        # tensor_see(conf1,v) 
        # plt.imshow(conf1)
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
        return vox_pillar_L, pos_L, img, output

    # def data_generation(self, batch_datasets):
        pillar_L = []
        pillar_pos_L = []
        imgs = []
        lbl = []
        
        for dataset in batch_datasets:
            vox_pillar_L, pos_L, img, output  = self.read(dataset)

            pillar_L.append(vox_pillar_L)
            pillar_pos_L.append(pos_L)
            imgs.append(img)

            lbl.append(output)
            

        X_pL = np.array(pillar_L)
        X_posL = np.array(pillar_pos_L)
        X_imgs = np.array(imgs)

        X = [X_pL, X_posL, X_imgs]
        
        lbl = np.array(lbl)

        return X, lbl


if __name__ == '__main__':
    # dataset_path = 'C:/Users/Marcelo/Desktop/SCRIPTS/MySCRIPT/Doc_code/data/'
    # dataset_path = 'D:/SCRIPTS/Doc_code/data/'
    dataset_path = 'C:/Users/maped/Documents/Scripts/Nuscenes/'
    input_shape = (512, 512, 3)
    batch_size = 2
    train_gen = SequenceData(
        model='train', 
        dir=dataset_path, 
        target_size=(input_shape[0], input_shape[1], 3),
        batch_size=batch_size,
        shuffle=True,
        data_aug=True  # <--- queremos flip
    )
    # train_gen[2]
    # print(len(train_gen))
    # exit()
    X, y = train_gen[0]

     # X é uma lista com [X_pL, X_posL, X_imgs]
    X_pL, X_posL, X_imgs = X

    print("Shapes do batch:")
    print(" - X_pL  :", X_pL.shape)
    print(" - X_posL:", X_posL.shape)
    print(" - X_imgs:", X_imgs.shape)
    print(" - y     :", y.shape)

    # Só para ver quantas amostras tem de fato no batch:
    print(f"Batch size verificado = {len(X_imgs)} (deveria ser {batch_size}).")

    # Se quiser conferir se a primeira metade é a versão normal e a segunda metade é flipada:
    #   Vamos supor que batch_size=2, então X_imgs[0] é normal, X_imgs[1] deve ser flip
    #   Você pode salvar as imagens para inspecionar manualmente ou imprimir algo.
    img_normal = X_imgs[0]
    img_flip   = X_imgs[1]

    # Exemplo: salva as imagens para ver manualmente
    # (Cuidado, isso requer que img_normal esteja em [0..1]. 
    #  Se estiver, podemos multiplicar por 255 e converter pra uint8)
    import cv2
    cv2.imwrite("debug_img_normal.png",  (img_normal*255).astype('uint8'))
    cv2.imwrite("debug_img_flipped.png", (img_flip*255).astype('uint8'))

    print("Imagens salvas em debug_img_normal.png e debug_img_flipped.png (verifique se a segunda é espelhada).")

