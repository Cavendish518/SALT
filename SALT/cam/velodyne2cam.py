import os
import numpy as np
from tqdm import tqdm
from utils.utils import *


def get_image(c_ego,intrinsic, pc):
    pc_img = pc.copy()
    pc[:,:3] = transform_point_cloud(pc[:,:3],c_ego)
    pc_img = pc_img[pc[:,2]>=0,:3]
    transformed_points = pc[pc[:,2]>=0, :3]
    
    points_2d_hom = transformed_points @intrinsic.T
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2][:, np.newaxis]
    points_2d = points_2d.astype(np.int32)
    

    unique_points = {}
    for i in range(points_2d.shape[0]):
        x, y = points_2d[i]
        z = transformed_points[i, 2]
        key = (x, y)
        if key not in unique_points or z < unique_points[key][1]:
            unique_points[key] = (points_2d[i], z, pc_img[i])

    filtered_points_2d = np.array([value[0] for value in unique_points.values()]).astype(np.int32)
    depth_values = np.array([value[1] for value in unique_points.values()])
    pc_img = np.array([value[2] for value in unique_points.values()])
    x_img = filtered_points_2d[:, 0]
    y_img = filtered_points_2d[:, 1]
    depth_matrix = np.zeros((480, 848))

    pc_matrix = np.zeros((480, 848,3))
    for i in range(x_img.shape[0]):
        y = x_img[i]
        x = y_img[i]
        p_depth = depth_values[i]
        if (x >= 0 and x < 480 and y >= 0 and y <848):
            pc_matrix[x, y] = pc_img[i]
            depth_matrix[x, y] = p_depth

    return pc_matrix


def velodyne2cam(data_dir,poses,output_camera_dir,cam_num):
    cam1_VLP = np.array(
      [[ 0.00477147, -0.99997976, -0.00425836,  0.0427029 ],
       [ 0.03401572,  0.00441448, -0.99941149,  0.20101945],
       [ 0.9994096 ,  0.00463197,  0.0340358 , -0.19894263],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    cam0_VLP = np.array(
      [[ 8.61011086e-01, -5.08578435e-01,  6.38867837e-04,   1.38573616e-02],
       [ 8.49296731e-03,  1.31174251e-02, -9.99877392e-01,   1.26039675e-01],
       [ 5.08514762e-01,  8.60916235e-01,  1.56168905e-02,  -1.45389431e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,   1.00000000e+00]])

    cam_poses = []
    intrinsic = []
    
    cam_poses.append(cam0_VLP)
    cam_poses.append(cam1_VLP)
    K_Cam1=np.array([[605.772,0.0,424.736],
            [0.0,605.545,236.741],
            [0.0,0.0,1.0]])
    K_Cam0=np.array([[607.575,0.0,432.54],
                [0.0,607.335,243.137],
                [0.0,0.0,1.0]])
    
    intrinsic.append(K_Cam0)
    intrinsic.append(K_Cam1)


    for i in tqdm(range(len(poses))):
        for cam_id in range(cam_num):
            pc = np.fromfile(f"{data_dir}/{i:06d}.bin", dtype=np.float32).reshape(-1, 4)
            pcs_matrix = get_image(cam_poses[cam_id],intrinsic[cam_id], pc)
            if not os.path.exists(f"{output_camera_dir}/{cam_id}"):
                os.makedirs(f"{output_camera_dir}/{cam_id}")
            np.save(f"{output_camera_dir}/{cam_id}/{i:06d}.pcs", pcs_matrix)

    




