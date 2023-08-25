import numpy as np
# import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import glob
import sys
from lidar_utils import lidar_on_cam
from collections import defaultdict

def load_data(filepath):
    image_files = sorted(glob.glob( "/home/rtxadmin/Documents/Marcelo/Img_datasets/KITTI_img/*"))
  # print( len(image_files), ' left rectified RGB images are loaded')

    calib_files = sorted(glob.glob( filepath + "/calib/*"))
  # print( len(calib_files), ' calibration files are loaded')

    velo_files = sorted(glob.glob( filepath +  "/velodyne/*"))
  # print( len(velo_pcd_files), ' velodyne(pcd) files are loaded')

    return image_files, calib_files, velo_files

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:,:]    



if __name__ == '__main__':
  filepath = '/home/rtxadmin/Documents/Marcelo/KITTI'
  path = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/input/lidar_crop/'
  image_files, calib_files, velo_files = load_data(filepath)
  from vox_pillar import pillaring
  np.set_printoptions(formatter={'float': lambda x: "{:0.3f}".format(x)})
  for i in range(len(image_files)):
  # i=1
    img = plt.imread(image_files[i])
    # img = cv2.imread(image_files[i])
    print(calib_files[i])
    name = calib_files[i][-10:-4]
    # print(name)
    VeloInCam = lidar_on_cam(calib_files[i])
    velo_points = load_from_bin(velo_files[i])
    
    # Cropping Lidar point, leaving only camera referen                                                      ce view
    cam_3d = VeloInCam.mount_img(velo_points, img) 
    print("X max:" , np.max(cam_3d[:,0])) #Z
    print("X min:" , np.min(cam_3d[:,0])) #Z
    print("Y max:" , np.max(cam_3d[:,1])) #X
    print("Y min:" , np.min(cam_3d[:,1])) #X
    print("Z max:" , np.max(cam_3d[:,2])) #Y
    print("Z min:" , np.min(cam_3d[:,2])) #Y
    exit()

    # np.save(path+name+'.npy',cam_3d)
    # vox_pillar,pos = pillaring(cam_3d) #(10000,20,7)/ (10000,3)
    # print(pos)
    # input()

