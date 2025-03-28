from utils.utils import *
import matplotlib.pyplot as plt
import multiprocessing
from scipy.spatial import ConvexHull
from multiprocessing import Manager



def get_pseudoCameras(n,camera_position,camera_angle_ud,camera_angle_rl,tra,rot,start_rot):

    camera_positions = np.repeat(camera_position, n, axis=0)
    camera_orientations = []
    axis_total = ['x','y','z']
    vis = 3-rot-tra
    vis_camera = [0.,0.,0.]
    vis_camera[vis] = -np.tan((start_rot+np.pi)-camera_angle_ud)*np.abs(camera_position[0][tra])
    for i in range(n):
        angle_y = np.pi * 2 * i / n
        camera_positions[i]=(np.array(camera_positions[i])+(R.from_euler(axis_total[tra], np.pi*2 - angle_y).as_matrix()@vis_camera)).tolist()
        RV = angle2RV(camera_angle_ud, angle_y,tra,rot)
        camera_orientations.append(RV)
    camera_orientations = np.array(camera_orientations)

    return camera_positions,camera_orientations



def get_image(pc, camera_num,camera_positions,camera_orientations,K,Tr_matrix,width, height):
    
    homogeneous_points = transform_point_cloud(pc[:, :3], Tr_matrix)
    images = [] # light,S
    images_1 = [] #light S=(0.5,0.8)
    images_2 = [] #dark 0.8
    images_3 = [] #dark S=(0.5,0.8)
    depth_images = []
    intensity_images = []
    pcs_matrix = []

    for j in range(len(camera_positions)):
        # 计算齐次变换矩阵
        transformation_matrix = get_transformation_matrix(camera_positions[j], camera_orientations[j])
        transformed_points = transform_point_cloud(homogeneous_points, transformation_matrix)
        transformed_points = np.hstack((transformed_points, pc[:, 3].reshape(-1,1)))
        num_voxels = transformed_points.shape[0] // 8  # 计算体素数量
        voxel_indices = np.arange(num_voxels) * 8  # 每个体素的起始索引
        valid_mask = np.all(transformed_points[voxel_indices[:, None] + np.arange(8), 2] > 0, axis=1)
        # 保留符合条件的体素（每 8 个点）
        transformed_points = transformed_points[np.repeat(valid_mask, 8)]
        num_voxels = transformed_points.shape[0] // 8  # 计算体素数量
        pc_img = pc[np.repeat(valid_mask, 8)]
        points_2d_hom = K @ transformed_points[:, :3].T  # Apply intrinsic matrix
        points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]  # Normalize by z?
        points_2d = points_2d.astype(np.int32).T

        depth_matrix = np.zeros((height, width))
        intensity_matrix = np.zeros((height, width))
        pc_matrix = np.zeros((height, width, 3))

        for i in range(num_voxels):
            projected_points = points_2d[i * 8:(i + 1) * 8]
            out_of_bounds = np.any((projected_points[:, 0] < 0) | (projected_points[:, 0] >= width) | (projected_points[:, 1] < 0) | (projected_points[:, 1] >= height))
            if out_of_bounds:
                continue
            if len(np.unique(projected_points))==1:
                x,y = projected_points[0]
                depth = points_2d_hom[2, i * 8]  # 当前体素的深度
                if depth_matrix[y,x] == 0 or depth_matrix[y,x]>depth:
                    intensity = transformed_points[i * 8, 3]  # 当前体素的强度
                    pc_values = pc_img[i * 8, :3]  # 当前体素的原始点云坐标
                    depth_matrix[y,x] = depth
                    intensity_matrix[y,x] = intensity
                    pc_matrix[y,x] = pc_values
                continue
            elif are_points_collinear(projected_points):
                y_coords = projected_points[:,1]
                x_coords = projected_points[:,0]
                depth = points_2d_hom[2, i * 8]  # 当前体素的深度
                intensity = transformed_points[i * 8, 3]  # 当前体素的强度
                pc_values = pc_img[i * 8, :3]  # 当前体素的原始点云坐标
            
                # 批量更新：仅在 note_matrix 为空的位置赋值
                update_mask = (depth_matrix[y_coords, x_coords] == 0) | (depth_matrix[y_coords, x_coords] > depth)
            
                # 仅更新满足条件的点
                depth_matrix[y_coords[update_mask], x_coords[update_mask]] = points_2d_hom[2, i * 8:i * 8+8][update_mask]
                intensity_matrix[y_coords[update_mask], x_coords[update_mask]] = intensity
                pc_matrix[y_coords[update_mask], x_coords[update_mask]] = pc_values
                continue
            hull = ConvexHull(projected_points)
            temp_image = np.zeros(depth_matrix.shape, dtype=np.uint8)
            hull_points = projected_points[hull.vertices]

            # 转换为整数类型
            hull_points = np.round(hull_points).astype(int)
            
            # 使用 OpenCV 的填充功能绘制凸包
            points = hull_points.reshape((-1, 1, 2))
            cv2.fillPoly(temp_image, [points], color=(255))  # 用255填充凸包区域
            
            # 提取被填充的像素坐标（即图像中值为255的坐标）
            x_min, y_min, w, h = cv2.boundingRect(hull_points)

            # 提取 bounding box 内部区域
            roi = temp_image[y_min:y_min+h, x_min:x_min+w]
            
            # 在小区域内查找非零点（坐标相对于 ROI）
            filled_pixels = cv2.findNonZero(roi)
            
            # filled_pixels = cv2.findNonZero(temp_image)

            if filled_pixels is not None:
                filled_pixels = filled_pixels.reshape(-1, 2)  # 变成 (N, 2) 形状
                # 转换为全局坐标
                filled_pixels[:, 0] += x_min
                filled_pixels[:, 1] += y_min
                y_coords, x_coords = filled_pixels[:, 1], filled_pixels[:, 0]  # 提取 y, x 坐标
            
                depth = points_2d_hom[2, i * 8]  # 当前体素的深度
                intensity = transformed_points[i * 8, 3]  # 当前体素的强度
                pc_values = pc_img[i * 8, :3]  # 当前体素的原始点云坐标
            
                # 批量更新：仅在 note_matrix 为空的位置赋值
                update_mask = (depth_matrix[y_coords, x_coords] == 0) | (depth_matrix[y_coords, x_coords] > depth)
            
                # 仅更新满足条件的点
                # 假设你已经计算出了凸包区域或bounding box内的所有需要更新的像素坐标 (y_coords, x_coords)
                # update_pixels = np.column_stack((y_coords[update_mask], x_coords[update_mask]))

                # # 对于每个体素，取出八个顶点的深度
                # depth_values = points_2d_hom[2, i * 8: (i + 1) * 8]  # 8个顶点的深度值

                # # 假设你已经提取出这些8个顶点的对应图像坐标
                # vertex_coords = np.array([
                #     projected_points[0],  # 顶点1的坐标 (x1, y1)
                #     projected_points[1],  # 顶点2的坐标 (x2, y2)
                #     projected_points[2],  # 顶点3的坐标 (x3, y3)
                #     projected_points[3],  # 顶点4的坐标 (x4, y4)
                #     projected_points[4],  # 顶点5的坐标 (x5, y5)
                #     projected_points[5],  # 顶点6的坐标 (x6, y6)
                #     projected_points[6],  # 顶点7的坐标 (x7, y7)
                #     projected_points[7],  # 顶点8的坐标 (x8, y8)
                # ])

                # 使用 griddata 进行线性插值，估算更新区域的深度值
                # interpolated_depth_values = griddata(vertex_coords, depth_values, update_pixels, method='linear')

                # 用插值后的深度值更新 depth_matrix
                depth_matrix[y_coords[update_mask], x_coords[update_mask]] = depth
                intensity_matrix[y_coords[update_mask], x_coords[update_mask]] = intensity
                pc_matrix[y_coords[update_mask], x_coords[update_mask]] = pc_values
        Color_H = int_norm(intensity_matrix, pc_matrix, mode="norm_hist")
        Color_I = dif_norm(depth_matrix, K, kernel_size=3, mode="thresh_depth_value")
        Color_S = np.ones_like(Color_I) * 0.8
        # TODO : no sky and ground
        pseudo_img = SAM_Painter(Color_H, Color_S, Color_I)
        # 设置一个阈值来查找 b 中接近 0 的位置
        zero_threshold = 1e-6  # 可调整为合适的浮点数阈值
        # 创建布尔掩码，找到 b 中接近 0 且 a 行索引小于 120 的位置
        mask1 = (np.isclose(depth_matrix, 0, atol=zero_threshold)) & (np.arange(height)[:, None] < height / 4)
        pseudo_img[mask1] = [0.53, 0.81, 0.98]  # 天蓝色

        # 创建布尔掩码，找到 b 中接近 0 且 a 行索引大于等于 120 的位置
        mask2 = (np.isclose(depth_matrix, 0, atol=zero_threshold)) & (np.arange(height)[:, None] >= height / 4)
        pseudo_img[mask2] = [0.5, 0.5, 0.5]  # 灰色
        images.append(pseudo_img)
        #--------------------------------------------------------------
        # Color_H = int_norm(intensity_matrix, pc_matrix, mode="norm_hist")
        # Color_I = dif_norm(depth_matrix, K, kernel_size=3, mode="thresh_depth_value")
        # Color_S = (Color_I-0.25)*(0.8-0.5)/(0.75-0.25)+0.5
        # pseudo_img = SAM_Painter(Color_H, Color_S, Color_I)
        # pseudo_img[mask1] = [0.53, 0.81, 0.98]  # 天蓝色
        # pseudo_img[mask2] = [0.5, 0.5, 0.5]  # 灰色
        # images_1.append(pseudo_img)

        # Color_H = int_norm(intensity_matrix, pc_matrix, mode="norm_hist")
        # Color_I = dif_norm(depth_matrix, K, kernel_size=3, mode="thresh_depth_value_reverse")
        # Color_S = np.ones_like(Color_I) * 0.8
        # pseudo_img = SAM_Painter(Color_H, Color_S, Color_I)
        # pseudo_img[mask1] = [0.53, 0.81, 0.98]  # 天蓝色
        # pseudo_img[mask2] = [0.5, 0.5, 0.5]  # 灰色
        # images_2.append(pseudo_img)

        # Color_H = int_norm(intensity_matrix, pc_matrix, mode="norm_hist")
        # Color_I = dif_norm(depth_matrix, K, kernel_size=3, mode="thresh_depth_value_reverse")
        # Color_S = (Color_I-0.25)*(0.8-0.5)/(0.75-0.25)+0.5
        # pseudo_img = SAM_Painter(Color_H, Color_S, Color_I)
        # pseudo_img[mask1] = [0.53, 0.81, 0.98]  # 天蓝色
        # pseudo_img[mask2] = [0.5, 0.5, 0.5]  # 灰色
        # images_3.append(pseudo_img)
        #--------------------------------------------------------------

        depth_images.append(depth_matrix)
        
        pcs_matrix.append(pc_matrix)
        intensity_images.append(intensity_matrix)

    return images, depth_images, pcs_matrix, intensity_images,images_1, images_2,images_3

def multi_save_image(i,camera_num,nonground_dir,camera_positions,camera_orientations,poses,voxel_size,output_image_dir,K,Tr_matrix,sn,width, height):
    pc = get_superFrame(sn, i, nonground_dir, poses,voxel_size)
    images, depth_images, pcs_matrix, intensity_images,images_1,images_2,images_3 = get_image(pc, camera_num,camera_positions,camera_orientations,K,Tr_matrix,width, height)
    for m in range(len(images)):
        if not os.path.exists(f"{output_image_dir}/{m}"):
            os.makedirs(f"{output_image_dir}/{m}",exist_ok=True)
        #----------------------------------------------------------
        # if not os.path.exists(f"{output_image_dir}/test_1/{m}"):
        #     os.makedirs(f"{output_image_dir}/test_1/{m}",exist_ok=True)
        # if not os.path.exists(f"{output_image_dir}/test_2/{m}"):
        #     os.makedirs(f"{output_image_dir}/test_2/{m}",exist_ok=True)
        # if not os.path.exists(f"{output_image_dir}/test_3/{m}"):
        #     os.makedirs(f"{output_image_dir}/test_3/{m}",exist_ok=True)
        #----------------------------------------------------------
        # np.save(f"{output_image_dir}/{m}/{i:06d}.image", images[m])
        np.save(f"{output_image_dir}/{m}/{i:06d}.depth", depth_images[m])
        np.save(f"{output_image_dir}/{m}/{i:06d}.pcs", pcs_matrix[m])
        np.save(f"{output_image_dir}/{m}/{i:06d}.intensity", intensity_images[m])
        plt.imsave(f"{output_image_dir}/{m}/{i:06d}.jpg", images[m])
        #----------------------------------------------------------
        # plt.imsave(f"{output_image_dir}/test_1/{m}/{i:06d}.jpg", images_1[m])
        # plt.imsave(f"{output_image_dir}/test_2/{m}/{i:06d}.jpg", images_2[m])
        # plt.imsave(f"{output_image_dir}/test_3/{m}/{i:06d}.jpg", images_3[m])
        #----------------------------------------------------------


def save_image(nonground_dir,output_image_dir,poses,camera_num,camera_position,camera_angle_ud,camera_angle_rl,tra,rot,K,Tr_matrix,
               width, height,sn,multiprocess_num,voxel_size,start_rot,resume=False):
    # multiprocessing.set_start_method('forkserver')
    #check
    #遍历所有文件夹看是否图片数量一致
    if resume:
        check_bool = True
        for cam_idx in range(camera_num):
            if not os.path.exists(f"{output_image_dir}/{cam_idx}"):
                check_bool=False
                break
            img_files = [f for f in os.listdir(f"{output_image_dir}/{cam_idx}") if f.endswith('.jpg')]
            if len(img_files) != len(poses):
                check_bool=False
                break
        if check_bool:
            return
    
    camera_positions,camera_orientations= get_pseudoCameras(camera_num,camera_position,camera_angle_ud,camera_angle_rl,tra,rot,start_rot)
    args = [(key, camera_num,nonground_dir,camera_positions,camera_orientations,poses,voxel_size,output_image_dir,K,Tr_matrix,sn,width, height)
             for key in range(len(poses))]
    with multiprocessing.Pool(processes=multiprocess_num) as pool:
        pool.starmap(multi_save_image, args)

