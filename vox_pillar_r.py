import numpy as np
from collections import defaultdict
import config_model as cfg
# import open3d as o3d
image_size = cfg.img_shape
input_pillar_r_shape = cfg.input_pillar_r_shape
input_pillar_r_indices_shape = cfg.input_pillar_r_indices_shape
max_group = cfg.max_group_r
max_pillars = cfg.max_pillars_r


x_min = cfg.x_min
x_max = cfg.x_max
x_diff = cfg.x_diff

y_min = cfg.y_min
y_max = cfg.y_max
y_diff = cfg.y_diff

z_min = cfg.z_min
z_max = cfg.z_max
z_diff = cfg.z_diff

import numpy as np
from collections import defaultdict


# np.set_printoptions(precision=3, suppress= True)


def pillaring_r(cam_3d):
    # Normalizing the data
  # x = -40/40 ----> 80
  # y = -2/6 ----> 8
  # z = 0/80 ----> 80
  # xmin = -40
  # xmax = 40
  # ymin = -2
  # ymax = 6
  # zmin = 0
  # zmax = 80
  # print(cam_3d)
  # exit()
  # pcd = o3d.geometry.PointCloud()
  # pcd.points = o3d.utility.Vector3dVector(cam_3d[...,:3])
  # o3d.visualization.draw_geometries([pcd], width=504, height=504 , 
  #                                 zoom=0.02,
  #                                 front=[ -0.92862581169634939, 0.1699284137233735, 0.32981576078282615 ],
  #                                 lookat=[ 6.613722928178019, -0.21300680832693597, -0.39229345398463189 ],
  #                                 up=[ 0.33417852221359673, -0.0030960494529159504, 0.9425047107409712 ])
  # print(cam_3d.shape)
  # exit()

  cam_3d[:, 1] = - cam_3d[:, 1]
  # cam_3d[:, 2] = - cam_3d[:, 2]

  norm_i =  np.zeros((cam_3d.shape))
  real_3d =  np.zeros((cam_3d.shape))
  norm =  np.zeros((cam_3d.shape[0],2))
  # print(norm.shape)
  for i in range(cam_3d.shape[0]):  
    real_3d[i,0] = cam_3d[i,1]
    real_3d[i,1] = cam_3d[i,2]
    real_3d[i,2] = cam_3d[i,0]



    norm_i[i,0] = norm[i,0] =  (cam_3d[i,1]-x_min)/(x_diff)
    norm_i[i,1] = (cam_3d[i,2]-y_min)/(y_diff)
    norm_i[i,2] = norm[i,1] = (cam_3d[i,0]-z_min)/(z_diff)
    norm_i[i,3] = cam_3d[i,3]
    # norm[i,0] = (cam_3d[i,1]-x_min)/(x_diff)
    # norm[i,1] = (cam_3d[i,0]-z_min)/(z_diff)
  # print(norm)
  # # Developing the grid locations
  
  # for i in range(cam_3d.shape[0]):

  norm[norm>=1] = 0.999 #cutting values out of range
  norm[norm<=0] = 0.001 #cutting values out of range
  norm[:,0] = norm[:,0] * (image_size[0])
  norm[:,1] = norm[:,1] * (image_size[1])

  # print(norm)
  # exit()
  # Definig the position of each detection into grid
  pos = norm.astype(int)
  idx = np.array([[x] for x in range(0,pos.shape[0])])

  # print(pos)
  # print(idx)
  # exit()
  # Implememinting the index of each position
  pos_idx = np.hstack((pos,idx))
  # print(pos_idx)
  # exit()
  # Using Dict to group max 20 detections into the same grid cell 
  dic_pillar = defaultdict(list)
  for line in pos_idx:
    if len(dic_pillar[(line[0],line[1])]) < max_group:
      dic_pillar[(line[0],line[1])].append(line[2])
  # print(dic_pillar)
  # exit()
  # Creating the pillar in each grid cell/ Creating the mean count of each grid cell
  vox_pillar = np.zeros(input_pillar_r_shape)
  vox_pillar_mean = np.zeros(input_pillar_r_indices_shape)
  vox_pillar_indices = np.zeros(input_pillar_r_indices_shape)
  j=0
  # print('norm')
  # print(norm)
  # print('\n norm_i')
  # print(norm_i)
  # print('\n dic')
  # print(pos_idx)
  for key,v in dic_pillar.items():

    k=0 #group

    for id in v:
      # vox_pillar[j,k,:4] = real_3d[id] # copy the X Y Z r from the LIDAR
      # vox_pillar[j,k,7:8] = abs(real_3d[id,0] - ((key[0]*x_diff/image_size[0])+ x_min) ) # Distance of the X point to the middle Pillar
      # vox_pillar[j,k,8:9] = abs(real_3d[id,2] - ((key[1]*z_diff/image_size[1])+ z_min) ) # Distance of the Z point to the middle Pillar

      # vox_pillar_mean[j,0]+= real_3d[id,0:1] # Sum of X in the group
      # vox_pillar_mean[j,1]+= real_3d[id,1:2] # Sum of Y in the group
      # vox_pillar_mean[j,2]+= real_3d[id,2:3] # Sum of Z in the group

      vox_pillar[j,k,:4] = norm_i[id] # copy the X Y Z r from the LIDAR
      vox_pillar[j,k,7:8] = abs((norm[id,0] - key[0]) - 0.5) # Distance of the X point to the middle Pillar
      vox_pillar[j,k,8:9] = abs((norm[id,1] - key[1]) - 0.5) # Distance of the Z point to the middle Pillar
    

      vox_pillar_mean[j,0]+= norm_i[id,0:1] # Sum of X in the group
      vox_pillar_mean[j,1]+= norm_i[id,1:2] # Sum of Y in the group
      vox_pillar_mean[j,2]+= norm_i[id,2:3] # Sum of Z in the group
      k+=1
      if k==vox_pillar.shape[1]:
        break
    
    vox_pillar_mean[j,:] = vox_pillar_mean[j,:]/(k) # Mean of XYZ of the group

    vox_pillar[j,:k,4:5] = abs(vox_pillar[j,:k,0:1] - vox_pillar_mean[j,0:1]) # Distance of X to the X mean Group
    vox_pillar[j,:k,5:6] = abs(vox_pillar[j,:k,1:2] - vox_pillar_mean[j,1:2]) # Distance of Y to the Y mean Group
    vox_pillar[j,:k,6:7] = abs(vox_pillar[j,:k,2:3] - vox_pillar_mean[j,2:3]) # Distance of Z to the Z mean Group


    # vox_pillar[j,:k,4:7] = vox_pillar[j,:k,:3] - vox_pillar_mean[j,:]

    vox_pillar_indices[j,1:2] = key[0]
    vox_pillar_indices[j,2:3] = key[1]

    # vox_pillar_mean = np.zeros(input_pillar_indices_shape)
    j+=1
    if j==max_pillars:
      break
  
  # print('\n', vox_pillar[18][:10],'\n')
  # print(vox_pillar_indices[11])

  # exit()
  # Output of the pilar grouping (10000,20,9) --- 10000 grids, with max 20 detection in each,
  #                                               9 - (X,Y,Z,r,Xo,Yo,Zo,Xp,Zp):
  #                                               X,Y,Z = Lidar norm position
  #                                               r = Lidar reflectance
  #                                               Xo,Yo,Zo = offset of Lidar norm and the mean
  #                                                         of all group detections
  #                                               Xp,Zp = diff from the center of Pillar (0.5,0.5)
  return vox_pillar,vox_pillar_indices

