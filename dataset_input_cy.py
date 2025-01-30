import numpy as np
cimport numpy as np
import config_model as cfg
from utils3 import iou2d


x_min = cfg.x_min
x_max = cfg.x_max
x_diff = cfg.x_diff

y_min = cfg.y_min
y_max = cfg.y_max
y_diff = cfg.y_diff

z_min = cfg.z_min
z_max = cfg.z_max
z_diff = cfg.z_diff
X_div = cfg.X_div
# Y_div = cfg.Y_div
Z_div = cfg.Z_div

x_step = cfg.stepx
z_step = cfg.stepz
classes_names = [k for k, v in cfg.classes.items()]
def input_matrix(dataset):
    Nusc_path = cfg.NUSC_PATH
    lidar_path = cfg.LIDAR_PATH
    radar_path = cfg.RADAR_PATH
    img_path = cfg.IMG_PATH
    # dataset = dataset[:-4]


    # my_sample = nusc.get('sample', dataset)

    # RADAR = nusc.get('sample_data', my_sample['data']['RADAR_FRONT'])
    # LIDAR = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
    # CAMERA = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])

    # my_ego_token = LIDAR['ego_pose_token']



    # radar_name = RADAR['filename']
    # radar_name = radar_name.replace('samples/RADAR_FRONT/','')
    # radar_name = radar_name.replace('.pcd','.npy')
    # 'samples/RADAR_FRONT/n015-2018-07-18-11-07-57+0800__RADAR_FRONT__1531883530960489.pcd'

    # lidar_name = LIDAR['filename']
    # lidar_name = lidar_name.replace('samples/LIDAR_TOP/','')
    # lidar_name = lidar_name.replace('.pcd.bin','.npy')
    # 'samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530949817.pcd.bin'

    # camera_name = CAMERA['filename']
    # lidar_name = lidar_name.replace('samples/LIDAR_TOP/','')
    # lidar_name = lidar_name.replace('.pcd.bin','.npy')
    # 'samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530912460.jpg'

    # print(dataset)
    # exit()


    # -------------------------- INPUT CAM ------------------------------------------
    img =  np.zeros([512, 512, 3])
    # img = cv2.imread(img_path + dataset+ '.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img/255.
    # img = cv2.resize(img,(cfg.image_shape))
    # plt.imshow(img)
    # plt.show()

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
    vox_pillar_L, pos_L =  np.zeros([10000,20,7]), np.zeros([10000,3])
    # lidar = np.load(lidar_path + dataset + '.npy')  # [0,1,2] -> Z,X,Y
    # vox_pillar_L, pos_L = vox_pillar_l_cy.pillaring_l(lidar)  # (10000,20,7)/ (10000,3)






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
    # radar = np.load(radar_path + dataset + '.npy')  
    # vox_pillar_R, pos_R = vox_pillar_r_cy.pillaring_r(radar)  # (10,5,5)/ (10,3)
    vox_pillar_R, pos_R =  np.zeros([10,5,5]), np.zeros([10,3])

    # print('Point Shape: ', radar.shape)
    # print('Vox Shape: ', vox_pillar_R.shape)
    # print('Pos Shape: ', pos_R.shape)





    # exit()
    # 
    # ------------------------- ANNOTATION Tensor ---------------------------------

    class_matrix = np.zeros([X_div, Z_div, cfg.nb_anchors, cfg.nb_classes])
    conf_matrix = np.zeros([X_div, Z_div, cfg.nb_anchors,1])
    pos_matrix = np.zeros([X_div, Z_div, cfg.nb_anchors, 3])
    dim_matrix = np.zeros([X_div, Z_div, cfg.nb_anchors, 3])
    rot_matrix = np.zeros([X_div, Z_div, cfg.nb_anchors, 1])
    velo_matrix = np.zeros([X_div, Z_div, cfg.nb_anchors, 2])

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

    # my_ego = nusc.get('ego_pose',my_ego_token)

    # my_pos = my_ego['translation']
    # boxes = nusc.get_sample_data(LIDAR['token'])
    # an = 0
    # for b in boxes[1]:
    #     my_annotation_token = my_sample['anns'][an]
    #     an +=1
    #     category = str(b.name)
    #     pos_x = float(b.center[0])
    #     pos_z = float(b.center[1])
    #     pos_y = float(b.center[2])

    #     width = float(b.wlh[0])
    #     lenght = float(b.wlh[1])
    #     height = float(b.wlh[2])

    #     # axis_x = float(b.orientation.axis[0])
    #     # axis_y = float(b.orientation.axis[1])
    #     axis_z = float(b.orientation.axis[2])
    #     if axis_z < 0:
    #         rot = float(b.orientation.radians) * -1+ np.pi
    #     else:
    #         rot = float(b.orientation.radians)

    #     velo_x = nusc.box_velocity(my_annotation_token)[0]
    #     velo_z = nusc.box_velocity(my_annotation_token)[1]
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

        velo_x = float(l[8]) # Velocity in X
        velo_z = float(l[9]) # Velocity in Z
        maxIou = 0

        #######  Normalizing the Data ########
        if category in classes_names:
            cla = int(cfg.classes[category])

            norm_x = (pos_x + abs(cfg.x_min)) / (
                        cfg.x_max - cfg.x_min)  # Center Position X in relation to max X 0-1
            norm_y = (pos_y + abs(cfg.y_min)) / (
                        cfg.y_max - cfg.y_min)  # Center Position Y in relation to max Y 0-1
            norm_z = (pos_z + abs(cfg.z_min)) / (
                        cfg.z_max - cfg.z_min)  # Center Position Z in relation to max Z 0-1

            norm_vel_x = (velo_x + abs(cfg.vel_min)) / (
                        cfg.vel_max - cfg.vel_min)  # Norm velocity 0-1
            norm_vel_z = (velo_z + abs(cfg.vel_min)) / (
                        cfg.vel_max - cfg.vel_min)  # Norm velocity 0-1
            
            if pos_x > pos_z or pos_x < -pos_z:
                fov = 2
            else: 
                fov=0.5 



            out_of_size = np.array([norm_x, norm_z,norm_vel_x, norm_vel_z, fov])

            if (np.any(out_of_size > 1)or np.any(out_of_size <0)) or (np.isnan(velo_x) or np.isnan(velo_z)):
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
                    
                    diag = [np.sqrt(pow(a[0],2)+pow(a[2],2)) for a in cfg.anchor]
                    for i in range(-1,2):
                        for j in range(-1,2):
                            if (0 < loc_i + i < X_div) and (0 < loc_k + j < Z_div):
                                # x_v = (loc_i+i)*(0.16)* 2 + cfg.x_min # Real --- xId * xStep * downscalingFactor + xMin;
                                # z_v = (loc_k+j)*(0.16)* 2 + cfg.z_min # Real --- zId * zStep * downscalingFactor + zMin;

                                anchors = [[((loc_i+i) * x_step*2)+cfg.x_min, a[3], ((loc_k+j) * z_step*2)+cfg.z_min, a[0], a[1], a[2], a[4]] for a in cfg.anchor]

                                iou = [iou2d(a, lbl) for a in anchors]
                                if np.max(iou) > maxIou:
                                    maxIou = np.max(iou)
                                    best_a = iou.index(maxIou)

                                for a in range(cfg.nb_anchors):
                                    if iou[a] > cfg.pos_iou:
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
                                        velo_matrix[loc_i+i, loc_k+j, a, :] = [norm_vel_x, norm_vel_z]  
                                        

                                    elif iou[a] < cfg.neg_iou:
                                        conf_matrix[loc_i+i, loc_k+j, a, 0] = 0

                                    else:
                                        conf_matrix[loc_i+i, loc_k+j, a, 0] = 0
                    if maxIou < cfg.pos_iou:
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
                        velo_matrix[loc_i, loc_k, best_a, :] = [norm_vel_x, norm_vel_z ]  
                            

            # print(maxIou)
        else:
            continue

    # conf1 = np.dstack((conf_matrix[:,:,:2,0],np.zeros((X_div, Z_div, 1))))
    # conf2 = np.dstack((conf_matrix[:,:,2:,0],np.zeros((X_div, Z_div, 1))))
    # img = np.hstack([conf1,conf2])

    # plt.imshow(conf1)
    # plt.show()
    # exit()
    # print(conf_matrix.shape)
    # print(pos_matrix.shape)
    # print(dim_matrix.shape)
    # print(rot_matrix.shape)
    # print(class_matrix.shape)
    output = np.concatenate((conf_matrix, pos_matrix, dim_matrix, rot_matrix, velo_matrix, class_matrix), axis=-1)
    return vox_pillar_L, pos_L, vox_pillar_R, pos_R, img, output
