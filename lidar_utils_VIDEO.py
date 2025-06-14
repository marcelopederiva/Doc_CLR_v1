import numpy as np
# import open3d as o3d
# import cv2
import matplotlib.pyplot as plt

def depth_color(val, min_d=0, max_d=120):
    """ 
    print Color(HSV's H value) corresponding to distance(m) 
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val)  # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def in_h_range_points(points, m, n, fov):
    """ extract horizontal in-range points """
    return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                          np.arctan2(n, m) < (-fov[0] * np.pi / 180))


def in_v_range_points(points, m, n, fov):
    """ extract vertical in-range points """
    return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                          np.arctan2(n, m) > (fov[0] * np.pi / 180))


def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """ filter points based on h,v FOV  """

    if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points

    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[in_v_range_points(points, dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points[in_h_range_points(points, x, y, h_fov)]
    else:
        h_points = in_h_range_points(points, x, y, h_fov)
        v_points = in_v_range_points(points, dist, z, v_fov)
        return points[np.logical_and(h_points, v_points)]


def in_range_points(points, size):
    """ extract in-range points """
    return np.logical_and(points > 0, points < size)


def velo_points_filter(points, v_fov, h_fov):
    """ extract points corresponding to FOV setting """
    # print(points.shape)
    # exit()
    points = points.T
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)

    x_lim = fov_setting(x, x, y, z, dist, h_fov, v_fov)[:, None]
    y_lim = fov_setting(y, x, y, z, dist, h_fov, v_fov)[:, None]
    z_lim = fov_setting(z, x, y, z, dist, h_fov, v_fov)[:, None]
    # print(points.shape)
    if points.shape[0] < 4:
        # int_lim = fov_setting(points[:, 3], x, y, z, dist, h_fov, v_fov)[:, None]
        # Stack arrays in sequence horizontally
        points = points.T
        xyz_ = np.ones((points.shape[0],points.shape[1]+1))
        # xyz_ = xyz_.T
        xyz_[:, 0] = points[: , 0]
        xyz_[:, 1] = points[:, 1]
        xyz_[:, 2] = points[:, 2]
        # print(xyz_)
        # exit()
        xyz_.T
        # xyz_ = np.vstack((points[:, 0], points[:, 1], points[:, 2]))
        # print(xyz_)
        # exit()
        # one_mat = np.full((1, xyz_.shape[1]), 1)
        # # print(one_mat.shape)
        # xyz_ = np.concatenate((xyz_, one_mat),axis = 0)
        # print(xyz_)
        # exit()
    else:
        int_lim = fov_setting(points[:, 3], x, y, z, dist, h_fov, v_fov)[:, None]
        # Stack arrays in sequence horizontally
        xyz_ = np.hstack((x_lim, y_lim, z_lim, int_lim))

    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    # one_mat = np.full((1, xyz_.shape[1]), 1)
    # print(xyz_.shape)
    # print(points[:, 3].shape)
    # xyz_ = np.hstack((xyz_, points[:, 3]))
    # print(xyz_[:30])
    # exit()
    # need dist info for points color
    # dist_lim = fov_setting(dist, x, y, z, dist, h_fov, v_fov)
    # color = depth_color(dist_lim, 0, 70)

    return xyz_, xyz_[:3,:]


def calib_velo2cam(filepath):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
    return R, T


def calib_cam2cam(filepath, mode):
    """
    If your image is 'rectified image' :
        get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)

    in this code, I'll get P matrix since I'm using rectified image
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]
    return P_


def velo3d_2_camera2d_points(points, v_fov, h_fov, vc_path, cc_path, mode='02'):
    """ print velodyne 3D points corresponding to camera 2D image """

    # R_vc = Rotation matrix ( velodyne -> camera )
    # T_vc = Translation matrix ( velodyne -> camera )
    R_vc, T_vc = calib_velo2cam(vc_path)

    # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
    P_ = calib_cam2cam(cc_path, mode)

    """
    xyz_v - 3D velodyne points corresponding to h, v FOV in the velodyne coordinates
    c_    - color value(HSV's Hue) corresponding to distance(m)

             [x_1 , x_2 , .. ]
    xyz_v =  [y_1 , y_2 , .. ]   
             [z_1 , z_2 , .. ]
             [ 1  ,  1  , .. ]
    """
    # print(points[:30])
    # exit()
    # print(points)
    # exit()
    xyz_v, c_ = velo_points_filter(points, v_fov, h_fov)
    # if len(points[0])<4:
    #     xyz_v = points
    # print(np.array(xyz_v).shape)
    # print(np.array(c_).shape)
    # exit()
    """
    RT_ - rotation matrix & translation matrix
        ( velodyne coordinates -> camera coordinates )

            [r_11 , r_12 , r_13 , t_x ]
    RT_  =  [r_21 , r_22 , r_23 , t_y ]   
            [r_31 , r_32 , r_33 , t_z ]
    """
    RT_ = np.concatenate((R_vc, T_vc), axis=1)

    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
    for i in range(xyz_v.shape[1]):
        xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

    """
    xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
             [x_1 , x_2 , .. ]
    xyz_c =  [y_1 , y_2 , .. ]   
             [z_1 , z_2 , .. ]
    """

    xyz_c = np.delete(xyz_v, 3, axis=0)

    # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
    for i in range(xyz_c.shape[1]):
        xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

    """
    xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
    ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
             [s_1*x_1 , s_2*x_2 , .. ]
    xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]  
             [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
    """
    xy_i = xyz_c[::] / xyz_c[::][2]

    ans = np.delete(xy_i, 2, axis=0)


    """
    width = 1242
    height = 375
    w_range = in_range_points(ans[0], width)
    h_range = in_range_points(ans[1], height)

    ans_x = ans[0][np.logical_and(w_range,h_range)][:,None].T
    ans_y = ans[1][np.logical_and(w_range,h_range)][:,None].T
    c_ = c_[np.logical_and(w_range,h_range)]

    ans = np.vstack((ans_x, ans_y))
    """

    return ans, c_, xyz_v
def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:, :4]

def main (velo_file,calib_cam,calib_velo):
    v2c_filepath = calib_velo
    c2c_filepath = calib_cam
    velo_points = load_from_bin(velo_file)
    ans, c_, xyz_v = velo3d_2_camera2d_points(velo_points, v_fov=(-20, 1.0), h_fov=(-40, 40), \
                                              vc_path=v2c_filepath, cc_path=c2c_filepath, mode='02')
    xyz_v = np.array(xyz_v)
    xyz_v = xyz_v.T
    return ans, c_, xyz_v

if __name__ == '__main__':
    filepath_c = 'C:/Users/Marcelo/Desktop/SCRIPTS/KITTI/Path2/calib'
    v2c_filepath = filepath_c + '/calib_velo_to_cam.txt'
    c2c_filepath = filepath_c + '/calib_cam_to_cam.txt'





    # bin file -> numpy array
    velo_points = load_from_bin('C:/Users/Marcelo/Desktop/SCRIPTS/KITTI/Path2/velodyne_points/data/0000000089.bin')

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(velo_points)
    # o3d.visualization.draw_geometries([pcd])

    ans, c_, xyz_v = velo3d_2_camera2d_points(velo_points, v_fov=(-20, 1.0), h_fov=(-40, 40), \
                                       vc_path=v2c_filepath, cc_path=c2c_filepath, mode='02')
    xyz_v = np.array(xyz_v)
    xyz_v = xyz_v.T
    # print(xyz_v.shape)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz_v[:,:3])
    # o3d.visualization.draw_geometries([pcd])
