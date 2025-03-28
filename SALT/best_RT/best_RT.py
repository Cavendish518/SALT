import cv2
import os
import numpy as np
import random
from model.model import resnet50
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from utils.utils import *


def find_best_score(vec, cluster_centers):
    score = -1
    for i in range(cluster_centers.shape[0]):
        s = np.linalg.norm(vec - cluster_centers[i])
        if score == -1:
            score = s
        else:
            score = min(score, s)
    return score



def find_angle_x(ground_min_height, camera_angle_x, q_angle, p_angle, q, p, main_camera_position, camera_angle_y,
                 ground_pc, K, width, height,tra,rot,Tr_matrix):
    mid_angle = 0
    ground_iter_num = 0
    if ground_min_height >= q and ground_min_height <= p:
        return camera_angle_x
    if ground_min_height < q:
        mid_angle = p_angle
    elif ground_min_height > p:
        mid_angle = q_angle
    while (ground_min_height < q or ground_min_height > p):
        ground_iter_num += 1
        if ground_min_height < q:
            p_angle = mid_angle
            mid_angle = (q_angle + p_angle) / 2
            ground_min_height = get_ground_min_height(main_camera_position, mid_angle, camera_angle_y, ground_pc, K,
                                                      width, height,tra,rot,Tr_matrix)
            if ground_min_height is None:
                return camera_angle_x
        elif ground_min_height > p:
            q_angle = mid_angle
            mid_angle = (q_angle + p_angle) / 2
            ground_min_height = get_ground_min_height(main_camera_position, mid_angle, camera_angle_y, ground_pc, K,
                                                      width, height,tra,rot,Tr_matrix)
            if ground_min_height is None:
                return camera_angle_x
        if ground_iter_num > 30:
            #ground Error
            return camera_angle_x
    return mid_angle


def get_ground_min_height(main_camera_position, camera_angle_x, camera_angle_y, ground_pc1, K, width, height,tra,rot,Tr_matrix):
    ground_pc = ground_pc1.copy()
    main_camera_orientation = angle2RV(camera_angle_x, camera_angle_y,tra,rot)
    ground_pc[:, :3] = transform_point_cloud(ground_pc[:, :3], Tr_matrix)


    transformation_matrix = get_transformation_matrix(main_camera_position, main_camera_orientation)
    transformed_points = transform_point_cloud(ground_pc[:, :3], transformation_matrix)


    transformed_points = transformed_points[transformed_points[:, 2] > 2]

    points_2d_hom = K @ transformed_points.T
    points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]
    points_2d = points_2d.astype(np.int32).T

    unique_points = {}
    for i in range(points_2d.shape[0]):
        x, y = points_2d[i]
        z = points_2d_hom[2, i]
        key = (x, y)
        if key not in unique_points or z < unique_points[key][1]:
            unique_points[key] = (points_2d[i], z)
    filtered_points_2d = np.array([value[0] for value in unique_points.values()]).astype(np.int32)
    depth_values = np.array([value[1] for value in unique_points.values()])
    if filtered_points_2d.shape[0] == 0:
        return None

    x_img = filtered_points_2d[:, 0]
    y_img = filtered_points_2d[:, 1]
    depth_matrix = np.zeros((height, width))

    depth_min_height = height

    voxel = {}

    for i in range(x_img.shape[0]):
        y = x_img[i]
        x = y_img[i]
        p_depth = depth_values[i]
        if (x >= 0 and x < height and y >= 0 and y < width):
            depth_min_height = min(depth_min_height, x)
            depth_matrix[x, y] = p_depth
    return depth_min_height


def get_ground_dpeth(main_camera_position, main_camera_orientation, ground_pc1, K, width, height,img_min,Tr_matrix,tra):
    ground_pc = ground_pc1.copy()
    ground_pc[:, :3] = transform_point_cloud(ground_pc[:, :3], Tr_matrix)

    transformation_matrix = get_transformation_matrix(main_camera_position, main_camera_orientation)
    transformed_points = transform_point_cloud(ground_pc[:, :3], transformation_matrix)
    transformed_points = transformed_points[transformed_points[:, 2] > 0]

    points_2d_hom = K @ transformed_points.T
    points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]
    points_2d = points_2d.astype(np.int32).T

    unique_points = {}
    for i in range(points_2d.shape[0]):
        x, y = points_2d[i]
        z = points_2d_hom[2, i]
        key = (x, y)
        if key not in unique_points or z < unique_points[key][1]:
            unique_points[key] = (points_2d[i], z)
    filtered_points_2d = np.array([value[0] for value in unique_points.values()]).astype(np.int32)
    depth_values = np.array([value[1] for value in unique_points.values()])
    if filtered_points_2d.shape[0] == 0:
        return None,None

    x_img = filtered_points_2d[:, 0]
    y_img = filtered_points_2d[:, 1]

    depth_matrix = np.zeros((height, width))
    ans = 0
    dis = -1
    for i in range(x_img.shape[0]):
        y = x_img[i]
        x = y_img[i]
        p_depth = depth_values[i]
        if (x >= 0 and x < height and y >= 0 and y < width):
            dis_current = (x - height / 2) ** 2 + (y - width / 2) ** 2
            if dis == -1 or dis_current < dis:
                dis = dis_current
                ans = p_depth
            depth_matrix[x, y] = p_depth
    t = 0.1
    if main_camera_position[tra] <0:
        t = -0.1

    max_height = (ans * (height)/K[1, 1] -((t-main_camera_position[tra])/(np.abs(transformation_matrix[:3, :3].T)[tra, 1])))*K[1, 1]/ans-(height)
    if np.isinf(max_height):
        max_height = 0
    else:
        max_height = int(max_height)
    return ans,max_height+height



def get_info(main_camera_position, voxel_size,camera_angle_x, camera_angle_y, pc1, K, width, height, tra,rot,Tr_matrix,super_depth=50):
    pc = pc1.copy()
    main_camera_orientation = angle2RV(camera_angle_x, camera_angle_y,tra,rot)
    pc[:, :3] = transform_point_cloud(pc[:, :3], Tr_matrix)
    transformation_matrix = get_transformation_matrix(main_camera_position, main_camera_orientation)
    transformed_points = transform_point_cloud(pc[:, :3], transformation_matrix)
    pc_img = (pc1[:, :3])[transformed_points[:, 2] > 2]
    transformed_points = transformed_points[transformed_points[:, 2] > 2]

    points_2d_hom = K @ transformed_points.T
    points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]
    points_2d = points_2d.astype(np.int32).T

    unique_points = {}
    for i in range(points_2d.shape[0]):
        x, y = points_2d[i]
        z = points_2d_hom[2, i]
        key = (x, y)

        if key not in unique_points or z < unique_points[key][1]:
            unique_points[key] = (points_2d[i], z, pc_img[i])
    filtered_points_2d = np.array([value[0] for value in unique_points.values()]).astype(np.int32)
    pc_img = np.array([value[2] for value in unique_points.values()])

    x_img = filtered_points_2d[:, 0]
    y_img = filtered_points_2d[:, 1]

    unique_points = {}
    voxel = {}
    voxel_indices = np.copy(pc_img[:, :3])
    voxel_indices[:, 0] = np.floor(pc_img[:, 0] / voxel_size[0]).astype(int)
    voxel_indices[:, 1] = np.floor(pc_img[:, 1] / voxel_size[1]).astype(int)
    voxel_indices[:, 2] = np.floor(pc_img[:, 2] / voxel_size[2]).astype(int)
    for i in range(x_img.shape[0]):
        y = x_img[i]
        x = y_img[i]
        a, b, c = voxel_indices[i]
        if (y >= 0 and y < width):
            if (a, b, c) not in voxel:
                
                voxel[(a, b, c)] = 1
    return len(voxel)



def get_keyFrame_pc(keyFrameFileName):
    return np.fromfile(keyFrameFileName, dtype=np.float32).reshape(-1, 4)
def image2fft(img):
    gray_image = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dft = np.fft.fft2(gray_image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    pil_image = Image.fromarray(magnitude_spectrum)
    rgb_image = pil_image.convert("RGB")
    return rgb_image


def optimal_t(main_camera_position_1, camera_angle_ud, camera_angle_rl, pc_1, ground_pc_1,model,cluster_centers,tra,rot,device,Tr_matrix,K,width,height,step_t=20):
    main_camera_position = [0.0,0.0,0.0]
    main_camera_position[tra] = main_camera_position_1[tra]
    pc = pc_1.copy()
    ground_pc = ground_pc_1.copy()
    
    # t
    main_camera_orientation = angle2RV(camera_angle_ud, camera_angle_rl,tra,rot)
    
    transformed_points = transform_point_cloud(pc[:, :3], Tr_matrix)
    transformation_matrix = get_transformation_matrix(main_camera_position, main_camera_orientation)
    transformed_points = transform_point_cloud(transformed_points, transformation_matrix)
    transformed_points = np.hstack((transformed_points, pc[:, 3].reshape(-1,1)))
    num_voxels = transformed_points.shape[0] // 8
    voxel_indices = np.arange(num_voxels) * 8
    valid_mask = np.all(transformed_points[voxel_indices[:, None] + np.arange(8), 2] > 0, axis=1)

    transformed_points = transformed_points[np.repeat(valid_mask, 8)]
    num_voxels = transformed_points.shape[0] // 8
    pc_img = pc[np.repeat(valid_mask, 8)]
    points_2d_hom = K @ transformed_points[:, :3].T
    points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]
    points_2d = points_2d.astype(np.int32).T
    y_img = points_2d[:, 1]
    img_min = min(0, np.min(y_img))
    ground_z,max_height = get_ground_dpeth(main_camera_position, main_camera_orientation, ground_pc, K, width,
                                    height,img_min,Tr_matrix,tra)
    if ground_z is None:
        return  main_camera_position
    img_max = min(np.max(y_img), max_height+height) + 1

    pseudo_height = max(img_max - img_min, height)
    if pseudo_height>99999:
        return  main_camera_position
    if pseudo_height > height:
        depth_matrix = np.zeros((pseudo_height, width))
        intensity_matrix = np.zeros((pseudo_height, width))
        pc_matrix = np.zeros((pseudo_height, width, 3))
        voxel_matrix = np.zeros((pseudo_height, width))

        for i in range(num_voxels):
            projected_points = points_2d[i * 8:(i + 1) * 8]
            out_of_bounds = np.any((projected_points[:, 0] < 0) | (projected_points[:, 0] >= width))
            if out_of_bounds:
                continue
            projected_points[:,1] =projected_points[:,1] - img_min
            if len(np.unique(projected_points))==1:
                x,y = projected_points[0]
                depth = points_2d_hom[2, i * 8]
                if depth_matrix[y,x] == 0 or depth_matrix[y,x]>depth:
                    intensity = transformed_points[i * 8, 3]
                    pc_values = pc_img[i * 8, :3]
                    depth_matrix[y,x] = depth
                    intensity_matrix[y,x] = intensity
                    pc_matrix[y,x] = pc_values
                continue
            elif are_points_collinear(projected_points):
                y_coords = projected_points[:,1]
                x_coords = projected_points[:,0]
                out_of_bounds = np.any((projected_points[:, 1] < 0) | (projected_points[:, 1] >= pseudo_height))
                if out_of_bounds:
                    continue
                depth = points_2d_hom[2, i * 8]
                intensity = transformed_points[i * 8, 3]
                pc_values = pc_img[i * 8, :3]
            
                update_mask = (depth_matrix[y_coords, x_coords] == 0) | (depth_matrix[y_coords, x_coords] > depth)
            
                depth_matrix[y_coords[update_mask], x_coords[update_mask]] = points_2d_hom[2, i * 8:i * 8+8][update_mask]
                intensity_matrix[y_coords[update_mask], x_coords[update_mask]] = intensity
                pc_matrix[y_coords[update_mask], x_coords[update_mask]] = pc_values
                continue
            hull = ConvexHull(projected_points)
            temp_image = np.zeros(depth_matrix.shape, dtype=np.uint8)
            hull_points = projected_points[hull.vertices]

            hull_points = np.round(hull_points).astype(int)
            
            points = hull_points.reshape((-1, 1, 2))
            cv2.fillPoly(temp_image, [points], color=(255))
            
            x_min, y_min, w, h = cv2.boundingRect(hull_points)

            roi = temp_image[y_min:y_min+h, x_min:x_min+w]
            
            filled_pixels = cv2.findNonZero(roi)
            

            if filled_pixels is not None:
                filled_pixels = filled_pixels.reshape(-1, 2)

                filled_pixels[:, 0] += x_min
                filled_pixels[:, 1] += y_min
                y_coords, x_coords = filled_pixels[:, 1], filled_pixels[:, 0]
            
                depth = points_2d_hom[2, i * 8]
                intensity = transformed_points[i * 8, 3]
                pc_values = pc_img[i * 8, :3]
                update_mask = (depth_matrix[y_coords, x_coords] == 0) | (depth_matrix[y_coords, x_coords] > depth)

                depth_matrix[y_coords[update_mask], x_coords[update_mask]] = depth
                intensity_matrix[y_coords[update_mask], x_coords[update_mask]] = intensity
                pc_matrix[y_coords[update_mask], x_coords[update_mask]] = pc_values
                voxel_matrix[y_coords[update_mask], x_coords[update_mask]] = i+1
        Color_H = int_norm(intensity_matrix, pc_matrix, mode="norm_hist")
        Color_I = dif_norm(depth_matrix, K, kernel_size=3, mode="thresh_depth_value")
        Color_S = np.ones_like(Color_I) * 0.8
        pseudo_img = SAM_Painter(Color_H, Color_S, Color_I)
        score_t_best = -1
        height_t_best = -1
        for img_height in range(height, pseudo_img.shape[0], step_t):
            img_step = pseudo_img[img_height - height:img_height, :]

            zero_threshold = 1e-6

            mask1 = (np.isclose(depth_matrix[img_height - height:img_height, :], 0, atol=zero_threshold)) & (np.arange(height)[:, None] < height / 4)
            img_step[mask1] = [0.53, 0.81, 0.98]

            mask2 = (np.isclose(depth_matrix[img_height - height:img_height, :], 0, atol=zero_threshold)) & (np.arange(height)[:, None] >= height / 4)
            img_step[mask2] = [0.5, 0.5, 0.5]

            voxel_num_matrix = voxel_matrix[img_height - height:img_height, :]
            voxel_num = len(np.unique(voxel_num_matrix[voxel_num_matrix>0]))
            if voxel_num < len(pc_1)/200:
                continue
            fft_image = image2fft(img_step)
            transform = transforms.ToTensor()
            fft_image = transform(fft_image)
            fft_image = fft_image.unsqueeze(0)
            fft_image = fft_image.to(device)
            output = model(fft_image)
            output  = F.normalize(output , dim=1)
            vec = (output.detach().cpu().numpy())[0]

            score_t = find_best_score(vec,cluster_centers)
            if score_t_best == -1:
                score_t_best = score_t
                height_t_best = img_height - height
            elif score_t < score_t_best:

                score_t_best = score_t
                height_t_best = img_height - height
        if height_t_best == -1:
            return main_camera_position

        optimal_y = ground_z * (height) / K[1, 1] - ground_z *  (height_t_best + img_min + height)/K[1, 1]
        move = np.abs(transformation_matrix[:3, :3].T) @ np.array([0.0, optimal_y, 0.0])
        main_camera_position[tra] = main_camera_position[tra] + move[tra]
    return  main_camera_position



def optimal_r(main_camera_position, voxel_size,camera_angle_x, camera_angle_y, ground_pc_1, pc_nosuper_1,
                    start_rot,tra,rot,K,width,height,Tr_matrix,step_r=5, precision=0.01):
    # R
    ground_pc = ground_pc_1.copy()
    ground_min_height_origin = get_ground_min_height(main_camera_position, camera_angle_x, camera_angle_y, ground_pc, K,
                                                     width, height,tra,rot,Tr_matrix)
    if ground_min_height_origin is None:
        return camera_angle_x

    q_angle = 0
    p_angle = 0
    if ground_min_height_origin < height / 4:
        q_angle_origin = start_rot
        p_angle_origin = camera_angle_x
        q_angle = find_angle_x(ground_min_height_origin, camera_angle_x, q_angle_origin, p_angle_origin, height / 4,
                               height / 4 + 30, main_camera_position, camera_angle_y, ground_pc, K, width, height,tra,rot,Tr_matrix)
        p_angle = find_angle_x(ground_min_height_origin, camera_angle_x, q_angle_origin, p_angle_origin,
                               height / 2 - 30, height / 2, main_camera_position, camera_angle_y, ground_pc, K, width,
                               height,tra,rot,Tr_matrix)
    elif ground_min_height_origin > height / 2:

        q_angle_origin = camera_angle_x
        p_angle_origin = start_rot+np.pi
        q_angle = find_angle_x(ground_min_height_origin, camera_angle_x, q_angle_origin, p_angle_origin, height / 4,
                               height / 4 + 30, main_camera_position, camera_angle_y, ground_pc, K, width, height,tra,rot,Tr_matrix)
        p_angle = find_angle_x(ground_min_height_origin, camera_angle_x, q_angle_origin, p_angle_origin,
                               height / 2 - 30, height / 2, main_camera_position, camera_angle_y, ground_pc, K, width,
                               height,tra,rot,Tr_matrix)
    else:
        q_angle_origin = 0
        p_angle_origin = np.pi
        q_angle = find_angle_x(ground_min_height_origin, camera_angle_x, camera_angle_x, p_angle_origin, height / 4,
                               height / 4 + 30, main_camera_position, camera_angle_y, ground_pc, K, width, height,tra,rot,Tr_matrix)
        p_angle = find_angle_x(ground_min_height_origin, camera_angle_x, q_angle_origin, camera_angle_x,
                               height / 2 - 30, height / 2, main_camera_position, camera_angle_y, ground_pc, K, width,
                               height,tra,rot,Tr_matrix)

    if q_angle is None or p_angle is None:
        #R is error
        return camera_angle_x
    pc_nosuper = pc_nosuper_1.copy()
    delta_angle_x = (p_angle - q_angle) / step_r
    p_angle_iter = p_angle
    q_angle_iter = q_angle
    while delta_angle_x > precision:
        delta_angle_x = (p_angle_iter - q_angle_iter) / step_r
        info_num = []
        camera_angle_x_array = []
        for info_i in range(step_r + 1):
            camera_angle_x_iter = q_angle_iter + delta_angle_x * info_i
            info_point = get_info(main_camera_position, voxel_size,camera_angle_x_iter, camera_angle_y, pc_nosuper, K, width,
                                  height,tra,rot,Tr_matrix)
            info_num.append(info_point)
            camera_angle_x_array.append(camera_angle_x_iter)
        max_info = max(info_num)
        p_info = 0
        q_info = len(info_num) - 1
        best_info = 0
        for info_j in range(len(info_num)):
            if info_num[info_j] == max_info:
                m_1 = max(info_j - 1, 0)
                m_2 = info_j
                m_3 = min(info_j + 1, len(info_num) - 1)
                temp_info = info_num[m_1] + info_num[m_2] + info_num[m_3]
                if temp_info > best_info:
                    best_info = temp_info
                    q_angle_iter = camera_angle_x_array[m_1]
                    p_angle_iter = camera_angle_x_array[m_3]
    camera_angle_x = q_angle_iter
    return camera_angle_x

def get_best_RT(nonground_data_dir,ground_data_dir,has_ground,kf_dir,K,rot,tra,start_rot,camera_angle_ud,camera_angle_rl
                ,camera_position,sn,batch_size,Tr_matrix,poses,voxel_size,width,height,cache_dir,device,checkpoint_path):
    #check
    RT_cache_dir = f'{cache_dir}/RT.npy'
    if os.path.exists(RT_cache_dir):
        RT = np.load(RT_cache_dir)
        camera_position=[[0.,0.,0.]]
        camera_position[0][tra] = RT[1]
        return RT[0],camera_position
    
    model = resnet50().to(device)
    cluster_centers=None
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        cluster_centers = checkpoint['cluster_centers']
    model.eval()
    
    kf_ids = np.load(f'{kf_dir}/keyframe_id.npy')
    axis_total = ['x','y','z']
    vis = 3-rot-tra
    vis_camera = [0.,0.,0.]
    vis_camera[vis] = -np.tan((start_rot+np.pi)-camera_angle_ud)*np.abs(camera_position[tra])
    camera_position=(np.array(camera_position)+(R.from_euler(axis_total[tra], np.pi*2 - camera_angle_rl).as_matrix()@vis_camera)).tolist()
    random.shuffle(kf_ids)
    batchs = [kf_ids[i:i + batch_size] for i in range(0, len(kf_ids), batch_size)]
    pre_camera_position = None
    pre_camera_angle_ud = None
    idx = -1
    for batch in batchs:
        idx += 1
        if idx > 1:
            delta_batch_t = np.linalg.norm(pre_camera_position[tra] - camera_position[tra])
            delta_batch_r = abs(pre_camera_angle_ud - camera_angle_ud)
            if delta_batch_t < 5 and delta_batch_r < np.pi/16:
                break
        pre_camera_angle_ud = camera_angle_ud
        pre_camera_position = camera_position
        delta_t = np.inf
        delta_r = np.pi
        while not (delta_t < 5 and delta_r < np.pi/16):
            batch_t = []
            batch_r = []
            for fid in batch:
                pc = get_superFrame(sn, fid, nonground_data_dir, poses,voxel_size)
                ground_pc=None
                if has_ground:
                    ground_pc = get_basicFrame(sn, fid, ground_data_dir, poses,voxel_size)
                else:
                    ground_pc = gen_groundFrame(sn, fid, nonground_data_dir, poses,voxel_size)
                id_t = optimal_t(camera_position,camera_angle_ud,camera_angle_rl, pc, ground_pc,model,cluster_centers,tra,rot,device,Tr_matrix,K,width,height,step_t=20)
                batch_t.append(id_t[tra])
            average_t = sum(batch_t)/len(batch_t)
            batch_position=[0.,0.,0.]
            batch_position[tra] = average_t
            vis_camera = [0.,0.,0.]
            vis_camera[vis] = -np.tan((start_rot+np.pi)-camera_angle_ud)*np.abs(batch_position[tra])
            batch_position_new=(np.array(batch_position)+(R.from_euler(axis_total[tra], np.pi*2 - camera_angle_rl).as_matrix()@vis_camera)).tolist()
            for fid in batch:
                pc_nosuper = get_basicFrame(sn, fid, nonground_data_dir, poses,voxel_size)
                ground_pc=None
                if has_ground:
                    ground_pc = get_basicFrame(sn, fid, ground_data_dir, poses,voxel_size)
                else:
                    ground_pc = gen_groundFrame(sn, fid, nonground_data_dir, poses,voxel_size)
                id_r = optimal_r(batch_position_new, voxel_size,camera_angle_ud,camera_angle_rl, ground_pc,pc_nosuper,start_rot,tra,rot,K,width,height,Tr_matrix,step_r=5, precision=0.01)
                batch_r.append(id_r)
            average_r = sum(batch_r) / len(batch_r)
            delta_t = abs(average_t - camera_position[tra])
            delta_r = abs(average_r - camera_angle_ud)
            camera_angle_ud = average_r
            vis_camera = [0.,0.,0.]
            vis_camera[vis] = -np.tan((start_rot+np.pi)-camera_angle_ud)*np.abs(average_t)
            camera_position=(np.array(batch_position)+(R.from_euler(axis_total[tra], np.pi*2 - camera_angle_rl).as_matrix()@vis_camera)).tolist()
    RT = np.array([camera_angle_ud,average_t])
    np.save(RT_cache_dir,RT)
    camera_position=[[0.,0.,0.]]
    camera_position[0][tra] = average_t
    return camera_angle_ud,camera_position
