import numpy as np
''' **************  SHAPE  *****************  '''
# img_shape = (600,600)
# input_pillar_shape = (10000, 20, 9)
# input_pillar_indices_shape = (10000, 3)
''' **************  SHAPE  *****************  '''
input_pillar_shape = (12000, 100, 9)
input_pillar_indices_shape = (12000, 3)

image_shape = (512,512)

''' **************  DIVISIONS  *****************  '''
X_div = 256
Y_div = 1
Z_div = 256

# X_div = 500
# Y_div = 1
# Z_div = 500

''' **************  PILLAR  *****************  '''
# max_group = 20
# max_pillars = 10000
# nb_channels = 64

''' **************  PILLAR  *****************  '''
max_group = 100
max_pillars = 12000

nb_channels = 64
nb_anchors = 4
# nb_anchors = 2
# nb_classes = 4
# classes = {"Car":               0,
#                "Pedestrian":        1,
#                "Person_sitting":    1,
#                "Cyclist":           2,
#                "Truck":             3,
#                "Van":               3,
#                "Tram":              3,
#                "Misc":              3,
#                }

nb_classes = 2

# classes = {"Car":               0
#                }
# classes = {"vehicle.car":               0
#                }
classes = {"vehicle.car":               0,
           "human.pedestrian.adult":     1,
           "human.pedestrian.child":     1,
           "human.pedestrian.police_officer":     1,
           "human.pedestrian.construction_worker":     1,
               }
#KITTI
# # width,height,lenght,orientation
# anchor = np.array([ [1.6,1.56,3.9,-1, 0],
#                     [1.6,1.56,3.9,-1, 1.57],
#                     [0.6,1.73,0.8,-0.6, 0],
#                     [0.6,1.73,0.8,-0.6, 1.57]], dtype=np.float32).tolist()

# anchor = np.array([ [1.6,1.56,3.9,-1, 0],
#                     [1.6,1.56,3.9,-1, 1.57]], dtype=np.float32).tolist()


#Nuscenes

# anchor = np.array([ [1.97,1.74,4.63, 0, 0],
#                     [1.97,1.74,4.63, 0, 1.57]], dtype=np.float32).tolist()
anchor = np.array([ [1.97,1.74,4.63, 0, 0],
                    [1.97,1.74,4.63, 0, 1.57],
                    [0.67,1.77,0.73, 0, 0],
                    [0.67,1.77,0.73, 0, 1.57]], dtype=np.float32).tolist()
trust_treshold = 0.5
pos_iou = 0.6
neg_iou = 0.4

color_true = { "0": 'limegreen',
                "1": 'cyan',
                "2": 'blue',
               "3": 'pink',
               }


''' **************  PARAMETERS  *****************  '''
BATCH_SIZE = 2
ITERS_TO_DECAY = 100980 # 15*4*ceil(6788/4)  --> every 15 epochs on 6788 samples
LEARNING_RATE = 2e-4
DECAY_RATE = 1e-8

''' **************  PATHS  *****************  '''
# DATASET_PATH = '/home/rtxadmin/Documents/Marcelo/Nuscenes/Nuscenes/'
DATASET_PATH = 'C:/Users/maped/Documents/Scripts/Nuscenes/'

TRAIN_TXT = 'train_80.txt'
VAL_TXT = 'val_10.txt'
VAL_MINI_TXT = 'val_mini.txt'
# TRAIN_TXT = 'train_80.txt'
# VAL_TXT = 'val_20.txt'

# LABEL_PATH = 'D:/SCRIPTS/Doc_code/data/label_2/'
# LIDAR_PATH = 'D:/SCRIPTS/Doc_code/data/input/lidar_crop_mini/'

NUSC_PATH = 'C:/Users/maped/Documents/Scripts/Nuscenes/'
LIDAR_PATH = 'C:/Users/maped/Documents/Scripts/Nuscenes/Lidar/'
RADAR_PATH = 'C:/Users/maped/Documents/Scripts/Nuscenes/Radar_S_fix/Radar_S_fix/'
IMG_PATH = 'C:/Users/maped/Documents/Scripts/Nuscenes/Img_S/'

# NUSC_PATH = '/home/rtxadmin/Documents/Marcelo/Nuscenes/Nuscenes/'
# LIDAR_PATH = '/home/rtxadmin/Documents/Marcelo/Nuscenes/Nuscenes/Lidar/'
# RADAR_PATH = '/home/rtxadmin/Documents/Marcelo/Nuscenes/Nuscenes/Radar_S_fix/'
# IMG_PATH = '/home/rtxadmin/Documents/Marcelo/Nuscenes/Nuscenes/Img_S/'

''' **************  LAMBDA WEIGHTS  *****************  '''

######################### Pillar loss
# lambda_class = 0.5
# lambda_occ = 3.0
# lambda_pos = 2.0
# lambda_dim = 2.0
# lambda_rot = 1.0

######################## Pillar loss
# lambda_class = 0.2
# lambda_occ = 1.0
# lambda_pos = 2.0
# lambda_dim = 2.0
# lambda_rot = 2.0

######################### LW 2
lambda_class = 1
lambda_occ   = 0.5
lambda_pos   = 5.0
lambda_dim   = 5.0
lambda_rot   = 5.0
lambda_velo  = 5.0

# ######################### LW 3
# lambda_class = 0.2
# lambda_occ = 0.1
# lambda_pos = 10.0
# lambda_dim = 10.0
# lambda_rot = 10.0
''' **************  NORMALIZATION  *****************  '''
#Norm Final -- KITTI ---- Best Until the moment

x_min = -30
x_max = 30
x_diff = abs(x_max - x_min)

y_min = -6
y_max = 6
y_diff = abs(y_max - y_min)

z_min = 0
z_max = 60
z_diff = abs(z_max - z_min)

#Norm Final -- NUSCENES ----

# x_min = -80
# x_max = 80
# x_diff = abs(x_max - x_min)

# y_min = -2.5
# y_max = 6
# y_diff = abs(y_max - y_min)

# z_min = 0
# z_max = 100
# z_diff = abs(z_max - z_min)

vel_min = -30
vel_max = 30


# w_max = 3
# h_max = 4.5
# l_max = 10

rot_max = 3.1416 #direction doesent matter

stepx = x_diff/image_shape[0]
stepz = z_diff/image_shape[1]