import argparse
from utils import config
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata

def get_arg():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        default='config.yaml',
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg

def SAM_Painter(h, s, i):
    """ pseudp color generator

    """
    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)

    # H = 0 (R is max)
    idx = (h >= 0) & (h < 2 * np.pi / 3)
    b[idx] = i[idx] * (1 - s[idx])
    r[idx] = i[idx] * (1 + s[idx] * np.cos(h[idx]) / np.cos(np.pi / 3 - h[idx]))
    g[idx] = 3 * i[idx] - (r[idx] + b[idx])


    # H = 2π/3 (G is max)
    idx = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    r[idx] = i[idx] * (1 - s[idx])
    g[idx] = i[idx] * (1 + s[idx] * np.cos(h[idx] - 2 * np.pi / 3) / np.cos(np.pi - h[idx]))
    b[idx] = 3 * i[idx] - (r[idx] + g[idx])

    # H = 4π/3 (B is max)
    idx = (h >= 4 * np.pi / 3) & (h < 2 * np.pi)
    g[idx] = i[idx] * (1 - s[idx])
    b[idx] = i[idx] * (1 + s[idx] * np.cos(h[idx] - 4 * np.pi / 3) / np.cos(5 * np.pi / 3 - h[idx]))
    r[idx] = 3 * i[idx] - (g[idx] + b[idx])

    # Combine RGB channels
    rgb = np.zeros((*i.shape, 3))
    rgb[:, :, 0] = np.clip(r, 0, 1)  # Red
    rgb[:, :, 1] = np.clip(g, 0, 1)  # Green
    rgb[:, :, 2] = np.clip(b, 0, 1)  # Blue

    return rgb


# depth_dif
def dif_norm(image,K,kernel_size=3, mode="dif_value"):
    pad = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    ave_image = cv2.filter2D(padded_image, -1, kernel)[pad:-pad, pad:-pad]

    if mode == "dif_value":
        result = abs(image - ave_image)
    elif mode == "norm_dif_value":
        result = abs(image - ave_image)
        result = (result - result.min()) / (result.max() - result.min())
    elif mode == "thresh_depth_value":
        result = abs(image - ave_image)
        th_voxel = 0.2
        th_cam = image / max(K[0,0],K[1,1])  # depth/max(fx,fy)
        th = th_voxel + th_cam
        result = np.where(result < th, 0, result)
        # 归一化
        result = (result - result.min()) / (result.max() - result.min())
        # hist eq
        result = cv2.equalizeHist((result * 255).astype(np.uint8))
        # 0.25-0.75
        result = result / 255.0
        result = 0.25 + result / 2.0
    

    return result


def int_norm(intesity_matrix, pc_matrix,mode="norm_hist"):
    result = intesity_matrix
    if mode == "norm_hist":
        result = np.linalg.norm(pc_matrix, axis=-1) * result
        result = (result - result.min()) / (result.max() - result.min())
        result = cv2.equalizeHist((result * 255).astype(np.uint8))
        result = result / 255.0
    result = result * 2 * np.pi 
    return result


def load_poses(poses_file, Tr_matrix):
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.eye(4)
            pose[0, :3] = np.array(values[:3])  # R矩阵
            pose[0, 3] = values[3]  # t向量
            pose[1, :3] = np.array(values[4:7])  # R矩阵
            pose[1, 3] = values[7]  # t向量
            pose[2, :3] = np.array(values[8:11])  # R矩阵
            pose[2, 3] = values[11]  # t向量
            poses.append(np.linalg.inv(Tr_matrix) @ pose @ Tr_matrix)
    return poses
def generate_voxel_pc(pc,voxel_size):
    offsets = np.array([
        [0, 0, 0], [voxel_size[0], 0, 0], [0, voxel_size[1], 0], [0, 0, voxel_size[2]],
        [voxel_size[0], voxel_size[1], 0], [voxel_size[0], 0, voxel_size[2]], [0, voxel_size[1], voxel_size[2]],[voxel_size[0],voxel_size[1],voxel_size[2]]
    ])
    intensity = np.repeat(pc[:, np.newaxis, 3], 8, axis=1)  # (N, 8)

    voxel_corners = np.repeat(pc[:, np.newaxis, :3], 8, axis=1) + offsets[np.newaxis, :, :]

    voxel_corners_with_intensity = np.hstack((voxel_corners.reshape(-1, 3), intensity.reshape(-1, 1)))

    return voxel_corners_with_intensity

def are_points_collinear(points):
    if np.unique(points[:, 0]).shape[0] == 1 or np.unique(points[:, 1]).shape[0] == 1:
        return True
    slopes = []
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        
        if x2 - x1 == 0:
            slope = float('inf') 
        else:
            slope = (y2 - y1) / (x2 - x1)
        
        slopes.append(slope)
    
    return len(np.unique(slopes)) == 1
    


def get_superFrame(sn, idx, data_dir, poses,voxel_size):
    cloud = np.fromfile(f"{data_dir}/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
    voxel_dict = {}

    voxel_indices = np.copy(cloud[:, :3])
    voxel_indices[:, 0] = np.floor(cloud[:, 0] / voxel_size[0]).astype(int)
    voxel_indices[:, 1] = np.floor(cloud[:, 1] / voxel_size[1]).astype(int)
    voxel_indices[:, 2] = np.floor(cloud[:, 2] / voxel_size[2]).astype(int)
    voxel_data = np.hstack((voxel_indices, cloud[:, 3:4]))
    for voxel in voxel_data:
        voxel_key = tuple(voxel[:3])
        intensity = voxel[3]
        if voxel_key in voxel_dict:
            voxel_dict[voxel_key] = max(voxel_dict[voxel_key], intensity)
        else:
            voxel_dict[voxel_key] = intensity

    current_pose = poses[idx]

    p_index = max(0, idx - int(sn / 2))
    q_index = min(len(poses) - 1, idx + int(sn / 2))

    for current_idx in range(p_index, q_index + 1):

        if not os.path.isfile(f"{data_dir}/{current_idx:06d}.bin"):
            continue
        cloud = np.fromfile(f"{data_dir}/{current_idx:06d}.bin", dtype=np.float32).reshape(-1, 4)

        global_cloud = transform_point_cloud(cloud[:, :3], poses[current_idx])
        relative_cloud = transform_point_cloud(global_cloud, np.linalg.inv(current_pose))

        voxel_indices = np.copy(relative_cloud[:, :3])
        voxel_indices[:, 0] = np.floor(relative_cloud[:, 0] / voxel_size[0]).astype(int)
        voxel_indices[:, 1] = np.floor(relative_cloud[:, 1] / voxel_size[1]).astype(int)
        voxel_indices[:, 2] = np.floor(relative_cloud[:, 2] / voxel_size[2]).astype(int)
        voxel_data = np.hstack((voxel_indices, cloud[:, 3:4]))
        for voxel in voxel_data:
            voxel_key = tuple(voxel[:3])
            intensity = voxel[3]
            if voxel_key in voxel_dict:
                voxel_dict[voxel_key] = max(voxel_dict[voxel_key], intensity)
            else:
                voxel_dict[voxel_key] = intensity

    pc = []
    for voxel_key, max_intensity in voxel_dict.items():
        center = (np.array(voxel_key)) * voxel_size
        pc.append(np.append(center, max_intensity))
    pc = np.array(pc)

    pc = generate_voxel_pc(pc,voxel_size)
    return pc


def transform_point_cloud(cloud, transform):
    ones = np.ones((cloud.shape[0], 1))
    cloud_h = np.hstack((cloud, ones))
    transformed_cloud_h = cloud_h @ transform.T 
    return transformed_cloud_h[:, :3]


def angle2RV(camera_angle_ud, camera_angle_rl,tra,rot):
    axis_total = ['x','y','z']
    
    rotation_rot = R.from_euler(axis_total[rot], camera_angle_ud).as_matrix()
    
    rotation_tra = R.from_euler(axis_total[tra], camera_angle_rl).as_matrix()
    rotation_matrix = rotation_rot @ rotation_tra
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    return rotation_vector.ravel()


def get_transformation_matrix(camera_position, camera_orientation):
    R, _ = cv2.Rodrigues(camera_orientation)

    t = np.array(camera_position)

    translation = -R @ t

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = translation

    return transformation_matrix


def get_basicFrame(sn, idx, data_dir, poses,voxel_size):
    cloud = np.fromfile(f"{data_dir}/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
    voxel_dict = {}
    
    voxel_indices = np.copy(cloud[:, :3])
    voxel_indices[:, 0] = np.floor(cloud[:, 0] / voxel_size[0]).astype(int)
    voxel_indices[:, 1] = np.floor(cloud[:, 1] / voxel_size[1]).astype(int)
    voxel_indices[:, 2] = np.floor(cloud[:, 2] / voxel_size[2]).astype(int)
    voxel_data = np.hstack((voxel_indices, cloud[:, 3:4]))
    for voxel in voxel_data:
        voxel_key = tuple(voxel[:3])
        intensity = voxel[3]
        if voxel_key in voxel_dict:
            voxel_dict[voxel_key] = max(voxel_dict[voxel_key], intensity)
        else:
            voxel_dict[voxel_key] = intensity

    current_pose = poses[idx]

    p_index = max(0, idx - int(sn / 2))
    q_index = min(len(poses) - 1, idx + int(sn / 2))

    for current_idx in range(p_index, q_index + 1):

        if not os.path.isfile(f"{data_dir}/{current_idx:06d}.bin"):
            continue
        cloud = np.fromfile(f"{data_dir}/{current_idx:06d}.bin", dtype=np.float32).reshape(-1, 4)

        global_cloud = transform_point_cloud(cloud[:, :3], poses[current_idx])
        relative_cloud = transform_point_cloud(global_cloud, np.linalg.inv(current_pose))

        voxel_indices = np.copy(relative_cloud[:, :3])
        voxel_indices[:, 0] = np.floor(relative_cloud[:, 0] / voxel_size[0]).astype(int)
        voxel_indices[:, 1] = np.floor(relative_cloud[:, 1] / voxel_size[1]).astype(int)
        voxel_indices[:, 2] = np.floor(relative_cloud[:, 2] / voxel_size[2]).astype(int)
        voxel_data = np.hstack((voxel_indices, cloud[:, 3:4]))
        for voxel in voxel_data:
            voxel_key = tuple(voxel[:3])
            intensity = voxel[3]
            if voxel_key in voxel_dict:
                voxel_dict[voxel_key] = max(voxel_dict[voxel_key], intensity)
            else:
                voxel_dict[voxel_key] = intensity

    pc = []
    for voxel_key, max_intensity in voxel_dict.items():
        center = (np.array(voxel_key)) * voxel_size
        pc.append(np.append(center, max_intensity))
    pc = np.array(pc)
    return pc

def gen_groundFrame(sn, idx, data_dir, poses,voxel_size):
    cloud = np.fromfile(f"{data_dir}/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
    voxel_dict = {}
    
    voxel_indices = np.copy(cloud[:, :3])
    voxel_indices[:, 0] = np.floor(cloud[:, 0] / voxel_size[0]).astype(int)
    voxel_indices[:, 1] = np.floor(cloud[:, 1] / voxel_size[1]).astype(int)
    voxel_indices[:, 2] = np.floor(cloud[:, 2] / voxel_size[2]).astype(int)
    voxel_data = np.hstack((voxel_indices, cloud[:, 3:4]))
    for voxel in voxel_data:
        voxel_key = tuple(voxel[:3])
        intensity = voxel[3]
        if voxel_key in voxel_dict:
            voxel_dict[voxel_key] = max(voxel_dict[voxel_key], intensity)
        else:
            voxel_dict[voxel_key] = intensity

    current_pose = poses[idx]

    p_index = max(0, idx - int(sn / 2))
    q_index = min(len(poses) - 1, idx + int(sn / 2))

    for current_idx in range(p_index, q_index + 1):

        if not os.path.isfile(f"{data_dir}/{current_idx:06d}.bin"):
            continue
        cloud = np.fromfile(f"{data_dir}/{current_idx:06d}.bin", dtype=np.float32).reshape(-1, 4)

        global_cloud = transform_point_cloud(cloud[:, :3], poses[current_idx])
        relative_cloud = transform_point_cloud(global_cloud, np.linalg.inv(current_pose))

        voxel_indices = np.copy(relative_cloud[:, :3])
        voxel_indices[:, 0] = np.floor(relative_cloud[:, 0] / voxel_size[0]).astype(int)
        voxel_indices[:, 1] = np.floor(relative_cloud[:, 1] / voxel_size[1]).astype(int)
        voxel_indices[:, 2] = np.floor(relative_cloud[:, 2] / voxel_size[2]).astype(int)
        voxel_data = np.hstack((voxel_indices, cloud[:, 3:4]))
        for voxel in voxel_data:
            voxel_key = tuple(voxel[:3])
            intensity = voxel[3]
            if voxel_key in voxel_dict:
                voxel_dict[voxel_key] = max(voxel_dict[voxel_key], intensity)
            else:
                voxel_dict[voxel_key] = intensity

    pc = []
    for voxel_key, max_intensity in voxel_dict.items():
        center = (np.array(voxel_key)) * voxel_size
        pc.append(np.append(center, max_intensity))
    pc = np.array(pc)
    x_range = np.linspace(int(min(pc[:,0])), int(max(pc[:,0])), int(max(pc[:,0])-min(pc[:,0])))
    y_range = np.linspace(int(min(pc[:,1])), int(max(pc[:,1])), int(max(pc[:,1])-min(pc[:,1]))) 

    x, y = np.meshgrid(x_range, y_range)

    z = np.full_like(x, int(min(pc[:,2])))
    inten = np.full_like(x, 0)
    point_cloud = np.column_stack((x.ravel(), y.ravel(), z.ravel(),inten.ravel()))
    return point_cloud
