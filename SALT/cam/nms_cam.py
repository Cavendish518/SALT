import os
import numpy as np
from tqdm import tqdm
from collections import Counter
import pickle
from scipy.spatial import cKDTree
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from fcmeans import FCM
from numpy.typing import NDArray
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

def load_poses(poses_file, Tr_matrix):
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.eye(4)
            pose[0, :3] = np.array(values[:3])
            pose[0, 3] = values[3]
            pose[1, :3] = np.array(values[4:7])
            pose[1, 3] = values[7]
            pose[2, :3] = np.array(values[8:11])
            pose[2, 3] = values[11]
            poses.append(np.linalg.inv(Tr_matrix) @ pose @ Tr_matrix)
    return poses
def transform_point_cloud(cloud, transform):
    ones = np.ones((cloud.shape[0], 1))
    cloud_h = np.hstack((cloud, ones))
    transformed_cloud_h = cloud_h @ transform.T
    return transformed_cloud_h[:, :3]

class CustomFCM(FCM):
    def __init__(self, n_clusters, random_state=None, initial_centers=None):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.initial_centers = initial_centers

    def fit(self, X: NDArray) -> None:
        self.rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        self.u = self.rng.uniform(size=(n_samples, self.n_clusters))
        self.u = self.u / np.tile(
            self.u.sum(axis=1)[np.newaxis].T, self.n_clusters
        )
        init_bool = True 
        for _ in tqdm(
            range(self.max_iter), desc="Training", disable=not self.verbose
        ):
            u_old = self.u.copy()
            if init_bool and self.initial_centers is not None:
                self._centers = self.initial_centers
                init_bool = False
            else:
                self._centers = FCM._next_centers(X, self.u, self.m)
            self.u = self.soft_predict(X)
            if np.linalg.norm(self.u - u_old) < self.error:
                break
        self.trained = True
def histogram_in_window(neighborhood,new_bin_edges):
    hist, _ = np.histogram(neighborhood, bins=new_bin_edges, range=(0, 1))
    hist = hist.astype(np.float64)
    hist /= len(neighborhood)
    return hist
def compute_histograms(grid, has_image,k,new_bin_edges):
    h, w = grid.shape
    num_valid_points = np.sum(has_image)
    histograms = np.zeros((num_valid_points, 10))
    idx_map={}
    idx = 0

    for i in range(h):
        for j in range(w):
            if has_image[i,j]:
                i_min = max(0, i - k)
                i_max = min(h, i + k + 1)
                j_min = max(0, j - k)
                j_max = min(w, j + k + 1)

                neighborhood = grid[i_min:i_max, j_min:j_max]
                has_image_sub = has_image[i_min:i_max, j_min:j_max]

                mask = has_image_sub == 1

                neighborhood = neighborhood[mask]
                histograms[idx] = histogram_in_window(neighborhood,new_bin_edges)
                idx_map[(i,j)] = idx
                idx += 1

    return histograms,idx_map

def histogram(has_image, bev_image):
    num_valid_points = np.sum(has_image)
    data = np.zeros((num_valid_points, 1))
    h, w = bev_image.shape
    idx = 0
    for i in range(h):
        for j in range(w):
            # Define neighborhood boundaries
            if has_image[i,j]:
                data[idx] = bev_image[i,j]
                idx+=1

    original_bins = np.linspace(0, 1, 1001)
    hist, _ = np.histogram(data, bins=original_bins, density=True)

    cumulative_hist = np.cumsum(hist)
    cumulative_hist /= cumulative_hist[-1]

    new_bin_edges = np.interp(np.linspace(0, 1, 11), cumulative_hist, original_bins[1:])
    new_bin_edges[0] = 0.0
    k = 2  
    histograms,idx_map = compute_histograms(bev_image, has_image,k,new_bin_edges)
    return histograms,idx_map

def merge_overlapping_pointclouds(pointcloud_dict, threshold=1e-6, min_overlap=3):
    merged_pointclouds = []
    prev_clouds = None
    
    for key in sorted(pointcloud_dict.keys()):
        current_clouds = pointcloud_dict[key]

        if prev_clouds is None:
            merged_pointclouds.extend(current_clouds)
        else:
            merged = False
            for prev_cloud in prev_clouds:
                prev_tree = KDTree(prev_cloud)
                for curr_cloud in current_clouds:
                    distances, _ = prev_tree.query(curr_cloud, k=1)
                    overlap_count = np.sum(distances < threshold)
                    if overlap_count >= min_overlap:
                        merged_cloud = np.vstack((prev_cloud, curr_cloud))
                        merged_pointclouds.append(merged_cloud)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                merged_pointclouds.extend(current_clouds)
        
        prev_clouds = current_clouds
    
    return merged_pointclouds

def nms_cam(m_dir,data_dir,poses,camera_num,merge_ground,voxel_size,ground_label_max_num,z_grow,kf_dir):
    output_camera_dir = f"{data_dir}/cam"
    cam_dir = output_camera_dir
    mask_out_put_dir = f"{cam_dir}/mask_out"

    output_cam_dir=f"{cam_dir}/predictions"
    velodyne_dir = f"{m_dir}/velodyne"
    ground_dir = f"{data_dir}/ground"
    nonground_dir = f"{data_dir}/nonground"
    input_label_dir = f"{data_dir}/prediction"
    os.makedirs(output_cam_dir, exist_ok=True)
    with open(f'{kf_dir}/obj_dict.pkl', 'rb') as file:
        obj_prompt = pickle.load(file)
    label_max = max(obj_prompt.values())+1
    save_dir=f"{data_dir}/ground_save"
    range_xy = np.load(f"{save_dir}/range_xy.npy")
    x_range = range_xy[0]
    y_range = range_xy[1]
    x_min, x_max = x_range
    y_min, y_max = y_range
    bev_image = np.load(f"{save_dir}/bev_image.npy")
    has_image = np.load(f"{save_dir}/has_image.npy")
    
    bev_label = np.zeros((bev_image.shape), dtype=np.int32)
    label_idx = 1
    dbscan = DBSCAN(eps=z_grow, min_samples=3)
    for i in tqdm(range(len(poses))):
        ground_pc = np.fromfile(f"{ground_dir}/{i:06d}.bin",dtype=np.float32).reshape(-1, 4)
        kdtree = cKDTree(ground_pc[:, :3])
        # pointcloud_dict = {}
        for camera_idx in range(camera_num):
            mask_out_put = f"{mask_out_put_dir}/{camera_idx}"
            try:
                mask_cam = np.load(f"{mask_out_put}/{i:06d}.npy")
            except Exception as e:
                continue
            unique_idxs = np.unique(mask_cam)
            unique_idxs = unique_idxs[unique_idxs != 0]
            pc_img = np.load(f"{cam_dir}/{camera_idx}/{i:06d}.pcs.npy")
            for idx in unique_idxs:
                mask = (mask_cam == idx)
                points = pc_img[mask]
                norms = np.linalg.norm(points, axis=1)
                points = points[norms > 1e-6]
                if len(points) == 0:
                    continue
                distances, indices = kdtree.query(points, k=1)

                overlap_count = np.sum(distances < 1e-6)
                if overlap_count/len(points)>0.7:
                    points = points[distances < 1e-6]
                    global_cloud = transform_point_cloud(points[:, :3], poses[i])
                    relative_cloud = transform_point_cloud(global_cloud[:, :3], np.linalg.inv(poses[0]))
                    x_indices = ((relative_cloud[:, 0] - x_min) / voxel_size[0]).astype(int)
                    y_indices = ((relative_cloud[:, 1] - y_min) / voxel_size[1]).astype(int)
                    label_temp=[]
                    for gi in range(len(relative_cloud)):
                        label_temp.append(bev_label[y_indices[gi], x_indices[gi]])
                    count = Counter(label_temp)
                    if 0 in count:
                        del count[0]
                    if len(count)==0:
                        for gi in range(len(relative_cloud)):
                            bev_label[y_indices[gi], x_indices[gi]] = label_idx
                        label_idx += 1
                    else:
                        max_num, max_count = count.most_common(1)[0]
                        if max_num/len(points)>0.7:
                            for gi in range(len(relative_cloud)):
                                bev_label[y_indices[gi], x_indices[gi]] = max_num
                        else:
                            for gi in range(len(relative_cloud)):
                                bev_label[y_indices[gi], x_indices[gi]] = label_idx
                            label_idx += 1

    histograms,idx_map = histogram(has_image,bev_image)
    label_hist = {}
    for img_i,img_j in idx_map:
        label_t = bev_label[img_i,img_j]
        if label_t != 0:
            if label_t in label_hist:
                label_hist[label_t].append(histograms[idx_map[img_i,img_j]])
            else:
                label_hist[label_t] = [histograms[idx_map[img_i,img_j]]]
    label_center = {}
    center_list = []
    for label in label_hist:
        label_center[label] = np.mean(np.array(label_hist[label]), axis=0)
        label_center[label] = label_center[label] / np.linalg.norm(label_center[label])
        center_list.append(label_center[label])
    label_center_final = {}
    label_mapping = {}
    basis_vectors = np.array(center_list)
    histograms = normalize(histograms, norm='l2', axis=1)
    fcm = FCM(n_clusters=ground_label_max_num, random_state=42)
    fcm.fit(basis_vectors)
    cusfcm = CustomFCM(n_clusters=ground_label_max_num, random_state=42,initial_centers=fcm.centers)
    cusfcm.fit(histograms)
    vis_image = np.zeros((bev_image.shape), dtype=np.int32)
    
    cluster_labels = cusfcm.predict(histograms)
    for img_i,img_j in idx_map:
        vis_image[img_i,img_j] = cluster_labels[idx_map[img_i,img_j]]+1
    for i in tqdm(range(len(poses))):
        nonground_pc = np.fromfile(f"{nonground_dir}/{i:06d}.bin",dtype=np.float32).reshape(-1, 4)
        nonground_pc=nonground_pc[:,:3]
        pc = np.fromfile(f"{velodyne_dir}/{i:06d}.bin",dtype=np.float32).reshape(-1, 4)
        pc=pc[:,:3]
        input_label = np.fromfile(f"{input_label_dir}/{i:06d}.label",dtype=np.int32)
        label_dict = {}
        unique_labels = np.unique(input_label)
        for label in unique_labels:
            if label < 1000:
                continue
            mask = input_label == label
            pointcloud = pc[mask]
            label_dict[label] = {
                "pointcloud": pointcloud,
                "mask": mask
            }
        mask_points = {}
        for camera_idx in range(camera_num):
            mask_points[camera_idx]=[]
            mask_out_put = f"{mask_out_put_dir}/{camera_idx}"
            try:
                mask_cam = np.load(f"{mask_out_put}/{i:06d}.npy")
            except Exception as e:
                continue
            unique_idxs = np.unique(mask_cam)
            unique_idxs = unique_idxs[unique_idxs != 0]
            pc_img = np.load(f"{cam_dir}/{camera_idx}/{i:06d}.pcs.npy")
            for idx in unique_idxs:
                mask = (mask_cam == idx)
                points = pc_img[mask]
                norms = np.linalg.norm(points, axis=1)
                points = points[norms > 1e-6]
                if points.shape[0] > 0:
                    mask_points[camera_idx].append(points)
        merge_list=[]
        need_merge_list=[]
        for camera_idx in range(camera_num-1):
            right_idx = camera_idx-1 if camera_idx-1 >0 else camera_num-1
            prev_clouds = mask_points[right_idx]
            current_clouds = mask_points[camera_idx]
            for prev_idx,prev_cloud in enumerate(prev_clouds):
                prev_tree = KDTree(prev_cloud)
                for curr_idx,curr_cloud in enumerate(current_clouds):
                    distances, _ = prev_tree.query(curr_cloud, k=1)
                    overlap_count = np.sum(distances < 1e-6)
                    if overlap_count >= 3:
                        merge_list.append((right_idx,prev_idx))
                        merge_list.append((camera_idx,curr_idx))
                        need_merge_list.append([(right_idx,prev_idx),(camera_idx,curr_idx)])
        merge_pc=[]
        for camera_idx in range(camera_num):
            current_clouds = mask_points[camera_idx]
            for curr_idx,curr_cloud in enumerate(current_clouds):
                if (camera_idx,curr_idx) in merge_list:
                    continue
                merge_pc.append(curr_cloud)
        for [(right_idx,prev_idx),(camera_idx,curr_idx)] in need_merge_list:
            merge_pc.append(np.vstack((mask_points[right_idx][prev_idx], mask_points[camera_idx][curr_idx])))

        label2merge = {}
        all_kdtree = cKDTree(pc)
        for merge_idx,mask_pointcloud in enumerate(merge_pc):
            distances, indices = all_kdtree.query(mask_pointcloud, k=1)
            nearest_labels = input_label[indices]
            unique_label = np.unique(nearest_labels)
            if len(unique_label) == 1:
                if unique_label[0] in label2merge:
                    label2merge[unique_label[0]].append(merge_idx)
                else:
                    label2merge[unique_label[0]] = [merge_idx]
                continue
            else:
                majority_label = np.bincount(nearest_labels).argmax()
                if majority_label in label2merge:
                    label2merge[majority_label].append(merge_idx)
                else:
                    label2merge[majority_label] = [merge_idx]
        
        for label_del,idxs in label2merge.items():
            if len(idxs) < 2:
                continue
            if label_del < 1000:
                continue
            del_pc = label_dict[label_del]["pointcloud"]
            copy_pc = del_pc.copy()
            new_merge_pc={}
            for m_idx in idxs:
                tree_n = cKDTree(merge_pc[m_idx])
                distances, _ = tree_n.query(copy_pc, k=1)
                mask = distances >= 1e-6
                new_merge_pc[m_idx] = copy_pc[~mask]
                copy_pc = copy_pc[mask]
                
            unique_sorted_list = sorted(idxs, key=lambda k: len(new_merge_pc[k]), reverse=True)
            for m_idx in unique_sorted_list:
                now_points = np.array(new_merge_pc[m_idx],dtype=np.int32).reshape(-1,3)
                z_set = np.unique(now_points[:,2])
                for z in z_set:
                    z_points = copy_pc[copy_pc[:,2] == z]
                    if(z_points.shape[0] == 0):
                        continue
                    copy_pc = copy_pc[copy_pc[:,2] != z]
                    db_labels = dbscan.fit_predict(z_points)
                    cluster_dict = {}
                    for sc_label in set(db_labels):
                        cluster_points = z_points[db_labels == sc_label]
                        cluster_dict[sc_label] = cluster_points.tolist()

                    for db_c in cluster_dict.values():
                        cluster1 = now_points[now_points[:,2] == z]
                        cluster2 = np.array(db_c,dtype=np.int32).reshape(-1,3)
                        distances = cdist(cluster1, cluster2, metric='euclidean')
                        min_distance = np.min(distances)
                        if min_distance < z_grow:
                            new_merge_pc[m_idx]=np.vstack((new_merge_pc[m_idx],cluster2))
                        else:
                            copy_pc = np.concatenate((copy_pc, cluster2), axis=0)
                    if copy_pc.shape[0] == 0:
                        break
                if copy_pc.shape[0] == 0:
                        break
            new_merge_pc[unique_sorted_list[0]]=np.vstack((new_merge_pc[m_idx],copy_pc))
            for _,points in new_merge_pc.items():
                _, indices = all_kdtree.query(points, k=1)
                input_label[indices] = label_max
                label_max +=1
        cloud = np.fromfile(f"{velodyne_dir}/{i:06d}.bin", dtype=np.float32).reshape(-1, 4)
        if merge_ground:
            ground_cloud = np.fromfile(f"{ground_dir}/{i:06d}.bin", dtype=np.float32).reshape(-1, 4)
            global_cloud = transform_point_cloud(ground_cloud[:, :3], poses[i])
            relative_cloud = transform_point_cloud(global_cloud, np.linalg.inv(poses[0]))
            x_indices = ((relative_cloud[:, 0] - x_min) / voxel_size[0]).astype(int)
            y_indices = ((relative_cloud[:, 1] - y_min) / voxel_size[1]).astype(int)
            ground_label =[]
            for gi in range(len(relative_cloud)):
                if y_indices[gi] < vis_image.shape[0] and x_indices[gi] < vis_image.shape[1] and y_indices[gi] >0 and x_indices[gi] > 0:
                    ground_label.append(vis_image[y_indices[gi], x_indices[gi]])
                else:
                    ground_label.append(0)
            kdtree_cloud = KDTree(cloud[:, :3])
            
            _, indices = kdtree_cloud.query(ground_cloud[:, :3], k=1)

            for indices_i, idx in enumerate(indices.flatten()):
                input_label[idx] = ground_label[indices_i]
        input_label.tofile(f"{output_cam_dir}/{i:06d}.label")