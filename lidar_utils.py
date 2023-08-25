import numpy as np
import matplotlib.pyplot as plt

class lidar_on_cam(object):
    def __init__(self,calib_files):
        calib = self.read_calib_file(calib_files)
        self.P = calib['P2']
        self.P = np.reshape(self.P,[3,4]) # Reshaping P2 in 3X4
        self.Velo_to_cam = calib['Tr_velo_to_cam']
        self.Velo_to_cam = np.reshape(self.Velo_to_cam,[3,4]) # Reshaping Tr_velo_to_cam in 3x4
        self.R0 = calib['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3]) # Reshaping R0_rect to 3x3
    
    
    def read_calib_file(self, calib_files):
        file = {}
        with open(calib_files,'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                
                key,value = line.split(':',1)
                try:
                    file[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return file
    
    
    def lidar_to_cam(self, velo_3d):
        
        R0_h = np.vstack([self.R0,[0,0,0]])
        R0_h = np.column_stack([R0_h,[0,0,0,1]])
#         P X R0
        P_R0 = np.dot(self.P,R0_h)
        
#         P X R0 X Velo_to_cam
        P_R0_velo2cam = np.dot(P_R0,np.vstack((self.Velo_to_cam,[0,0,0,1])))
        
        
#         velo_3d  = [x_0 x_1 x_2 ] --> velo_3d_h = [x_0 x_1 x_2  1 ]
#                    [y_0 y_1 y_2 ]                 [y_0 y_1 y_2  1 ]
#                    [z_0 z_1 z_2 ]                 [z_0 z_1 z_2  1 ]
#                    [  .  .   .  ]                 [  .  .   .   . ]
#                    [  .  .   .  ]                 [  .  .   .   . ]
        
        velo_3d_h = np.column_stack([velo_3d, np.ones((velo_3d.shape[0],1))])

#         P X R0 X Velo_to_cam X velo_3d
        velo2cam = np.dot(P_R0_velo2cam,np.transpose(velo_3d_h))
    
        
        velo2cam = np.transpose(velo2cam)
        
        # Saving 3D LIDAR in Camera reference ***
        velo3d_2cam = velo2cam
        # Normalizing by Z axis
        velo2cam[:,0] /= velo2cam[:,2]
        velo2cam[:,1] /= velo2cam[:,2]
#         print(np.max(velo2cam[:,2]))
        # Excluding Z axis, using only X and Y axis for Cam projection
        velo2cam = velo2cam[:,0:2]
        
        return velo2cam,velo3d_2cam
        
    def lidar3d_to_cam(self, pcd_in_imgfov):
        h = np.hstack((pcd_in_imgfov,np.ones((pcd_in_imgfov.shape[0],1))))
        fov_velo2cam = np.dot(h,np.transpose(self.Velo_to_cam))
        cam_3d = np.transpose(np.dot(self.R0,np.transpose(fov_velo2cam)))
        
        return cam_3d

    def mount_img(self, velo_3d, img):
#         LIDAR points in camera referece
        velo2cam,velo3d_2cam = self.lidar_to_cam(velo_3d[:,:3])
        
#         Cropping LIDAR points in camera view (fov)
        x_min = 0
        y_min = 0
        x_max = img.shape[1]
        y_max = img.shape[0]
        
        fov_id = ((velo2cam[:,0] < x_max)&
                   (velo2cam[:,1] < y_max)&
                   (velo2cam[:,0] >= x_min)&
                   (velo2cam[:,1] >= x_min))
        
        # Cutting points closer than 2 meters (X axis LIDAR reference)
        fov_id = fov_id & (velo_3d[:,0]>1) 
        
        pcd_in_imgfov_3D = velo_3d[fov_id,:]

        return pcd_in_imgfov_3D