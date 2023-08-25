import config_model as cfg 
import numpy as np
from utils3 import iou3d,iou2d
import matplotlib.pyplot as plt
from vox_pillar import pillaring

DATASET_PATH =cfg.DATASET_PATH
LABEL_PATH=cfg.LABEL_PATH
LIDAR_PATH=cfg.LIDAR_PATH

X_div = cfg.X_div
Z_div = cfg.Z_div 
nb_anchors= cfg.nb_anchors
nb_classes= cfg.nb_classes
x_max= cfg.x_max
x_min= cfg.x_min
y_max= cfg.y_max
y_min= cfg.y_min
z_max= cfg.z_max
z_min= cfg.z_min
w_max = cfg.w_max
h_max = cfg.h_max
l_max = cfg.l_max
rot_max= cfg.rot_norm
anchor= cfg.anchor
classes= cfg.classes


def lbl(dataset):
	label_path = LABEL_PATH
	lidar_path = LIDAR_PATH
	dataset = dataset[:-5]
	# dataset = '000049'
	# dataset = '000035'

	# -----C:/Users/Marcelo/Desktop/SCRIPTS/KITTI/object/training/label_2/
	# label_path = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/label_norm_lidar_v1/'
	# lidar_path = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/input/lidar_crop/'
	# dataset = dataset[:-4]
	# dataset = '000049'
	dataset = '000435'
	# 002798
	# 003934
	# 000435

	# -------------------------- INPUT DATA ----------------------------------------



	cam3d = np.load(lidar_path + dataset +'.npy')  #[0,1,2] -> Z,X,Y
	vox_pillar,pos = pillaring(cam3d) #(10000,20,7)/ (10000,3)


	# print(pos[:4])
	# pos_ = np.vstack((pos[:,0],pos[:,2]))
	# pos_ = 
	# pos = (pos[:,:2]).astype(int) # (1000,3)
	# print(pos[:4])
	# exit()
	#------------------------- Predicting Tensor ---------------------------------

	# label_matrix = np.zeros([X_div, Z_div, 9])
	# label_matrix[...,1:2] = 0

	class_matrix = -1*np.ones([X_div,Z_div,nb_anchors,nb_classes])
	conf_matrix = np.zeros([X_div,Z_div,nb_anchors])
	pos_matrix = -1*np.ones([X_div,Z_div,nb_anchors, 3])
	dim_matrix = -1*np.ones([X_div,Z_div,nb_anchors, 3])
	rot_matrix = -1*np.ones([X_div,Z_div,nb_anchors])

	anchors = [[0.5,0.5,0.5,a[0],a[1],a[2],a[3]] for a in anchor]


	with open(label_path+dataset+'.txt', 'r') as f:
		label = f.readlines()

	for l in label:
		l = l.replace('\n','')
		l = l.split(' ')
		l = np.array(l)

	    #######  Normalizing the Data ########
		print(l[0])
		if l[0] != 'DontCare':
			cla = int(classes[l[0]])

			norm_x = (float(l[11]) + abs(x_min))/(x_max - x_min)# Center Position X in relation to max X 0-1
			norm_y = (float(l[12]) + abs(y_min))/(y_max - y_min)# Center Position Y in relation to max Y 0-1
			norm_z = (float(l[13]) + abs(z_min))/(z_max - z_min)# Center Position Z in relation to max Z 0-1

			norm_w = float(l[9])/w_max# Dimension W in relation to max W 0-1
			norm_h = float(l[8])/h_max# Dimension H in relation to max H 0-1
			norm_l = float(l[10])/l_max# Dimension L in relation to max Z 0-1

			rot = float(l[14])
			if rot<0: rot = -rot
			norm_rot = rot/rot_max# Rotation in relation to max Rot in Y axis 0-1

			# print(rot)
			# print(norm_rot)


			
			# print(cla)
			# print(out_of_size)
			# print('\n')
			# norm_x = np.clip(norm_x,0.001,0.999)
			# norm_y =np.clip(norm_y,0.001,0.999)
			# norm_z =np.clip(norm_z,0.001,0.999)
			# norm_w =np.clip(norm_w,0.001,0.999)
			# norm_h =np.clip(norm_h,0.001,0.999)
			# norm_l =np.clip(norm_l,0.001,0.999)
			# norm_rot =np.clip(norm_rot,0.001,0.999)

			out_of_size = np.array([norm_x,norm_y,norm_z,norm_w,norm_h,norm_l,norm_rot])
			if np.any(out_of_size>1) or np.any(out_of_size<=0):
				print('out_of_size')
				print(out_of_size)
				continue
			else:
				loc = [X_div * norm_x , Z_div * norm_z]

				loc_i = int(loc[0])
				# loc_j = int(loc[1])
				loc_k = int(loc[1])

				x_cell = loc[0] - loc_i
				y_cell = norm_y
				z_cell = loc[1] - loc_k

				lbl = [x_cell,y_cell,z_cell,float(l[9]),float(l[8]),float(l[10]),norm_rot]

				iou = [iou3d(a,lbl) for a in anchors]         
				print(iou)
				if conf_matrix[loc_i,  loc_k, 0] == 0:
					min_i, max_i = np.clip(loc_i - 3, 0, X_div), np.clip(loc_i + 4, 0, X_div)
					min_k, max_k = np.clip(loc_k - 3, 0, Z_div), np.clip(loc_k + 4, 0, Z_div)

				for i in range(nb_anchors):
					if iou[i] > 0.6:
						conf_matrix[min_i:max_i,  min_k:max_k, i]  =  1
						class_matrix[min_i:max_i,  min_k:max_k, i,cla] = 1
						pos_matrix[min_i:max_i,  min_k:max_k, i,:] = [x_cell,y_cell,z_cell]
						dim_matrix[min_i:max_i,  min_k:max_k, i,:] = [norm_w,norm_h,norm_l]
						rot_matrix[min_i:max_i,  min_k:max_k, i]  = norm_rot
					elif iou[i] < 0.3:
						conf_matrix[min_i:max_i,  min_k:max_k, i]  =  -1
					else:
						conf_matrix[min_i:max_i,  min_k:max_k, i]  =  0.5
						class_matrix[min_i:max_i,  min_k:max_k, i,cla] = 1
						pos_matrix[min_i:max_i,  min_k:max_k, i,:] = [x_cell,y_cell,z_cell]
						dim_matrix[min_i:max_i,  min_k:max_k, i,:] = [norm_w,norm_h,norm_l]
						rot_matrix[min_i:max_i,  min_k:max_k, i]  = norm_rot

		else:
			# print('oi')
			continue
	# exit()
	print(dataset)
	plt.imshow(conf_matrix[:,:,:3])
	plt.show()
	return vox_pillar, pos, conf_matrix, pos_matrix, dim_matrix, rot_matrix, class_matrix

	# label_matrix = [conf_matrix, pos_matrix, dim_matrix, rot_matrix, class_matrix]     
	# plt.imshow(dim_matrix[:,:,1,:])
	# plt.show()
	# exit()


if __name__=='__main__':
	dataset = DATASET_PATH + 'train_80.txt'
	datasets = []
	with open(dataset, 'r') as f:
		datasets = datasets + f.readlines()
	c=0
	for d in datasets:
		lbl(d)
		print(c)
		c+=1