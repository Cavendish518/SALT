import os
import numpy as np
from tqdm import tqdm
import pickle
import re
from collections import defaultdict
from collections import Counter
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import itertools
from utils.utils import *
from sklearn.neighbors import KDTree


import multiprocessing



def get_Frame(idx, data_dir,voxel_size,ground_dir,merge_ground,ceiling_folder,indoor):
    cloud = np.fromfile(f"{data_dir}/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
    if merge_ground:
        ground_cloud = np.fromfile(f"{ground_dir}/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
        if ground_cloud.shape[0] > 0:
            kdtree_cloud = KDTree(cloud[:, :3])
            distances, indices = kdtree_cloud.query(ground_cloud[:,:3], k=1)
            delete_indices = indices.flatten()

            cloud = np.delete(cloud, delete_indices, axis=0)
    if indoor:
        ceiling_cloud = np.fromfile(f"{ceiling_folder}/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
        if ceiling_cloud.shape[0] > 0:
            kdtree_cloud = KDTree(cloud[:, :3])
            distances, indices = kdtree_cloud.query(ceiling_cloud[:,:3], k=1)
            delete_indices = indices.flatten()

            cloud = np.delete(cloud, delete_indices, axis=0)
    voxel_dict = {}

    voxel_indices = np.copy(cloud[:, :3])
    voxel_indices[:, 0] = np.floor(cloud[:, 0] / voxel_size[0]).astype(int)
    voxel_indices[:, 1] = np.floor(cloud[:, 1] / voxel_size[1]).astype(int)
    voxel_indices[:, 2] = np.floor(cloud[:, 2] / voxel_size[2]).astype(int)

    for voxel in voxel_indices:
        voxel_key = tuple(voxel[:3])
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key]={}
    return voxel_dict

def mask2point(f_id,camera_num,img_dir,file_dict,outcome_dir):
    if os.path.exists(f'{outcome_dir}/{f_id}.pkl'):
        try:
            with open(f'{outcome_dir}/{f_id}.pkl', 'rb') as f:
                load = pickle.load(f)
            return
        except Exception as e:
            pass
    label_point = {}
    for camera_idx in range(camera_num):
        pc_img = np.load(f"{img_dir}/{camera_idx}/{f_id:06d}.pcs.npy")
        if camera_idx in file_dict[f_id]:
            for idx,filename in file_dict[f_id][camera_idx].items():
                mask = np.load(filename)
                pc_label = pc_img[mask]
                label_point[idx*10+camera_idx]= pc_label

    with open(f'{outcome_dir}/{f_id}.pkl','wb') as f:
        pickle.dump(label_point,f)

def compute_iou(bbox1, bbox2):
    min_coords1, max_coords1 = bbox1
    min_coords2, max_coords2 = bbox2

    # Compute intersection
    inter_min = np.maximum(min_coords1, min_coords2)
    inter_max = np.minimum(max_coords1, max_coords2)

    if np.linalg.norm(inter_min - min_coords2) < 1e-6 and np.linalg.norm(inter_max - max_coords2) < 1e-6:
        return 1
    
    inter_dims = np.maximum(0, inter_max - inter_min + 1)
    inter_volume = np.prod(inter_dims)

    # Compute union
    volume1 = np.prod(max_coords1 - min_coords1 + 1)
    volume2 = np.prod(max_coords2 - min_coords2 + 1)
    union_volume = volume1 + volume2 - inter_volume

    return inter_volume / union_volume if union_volume > 0 else 0


def nms(i,velodyne_dir,voxel_size,poses,sn,outcome_dir,merge_ground,ground_dir,equal_dir,ground_save_dir,kf_dir,output_label_dir,voxel_grow,z_grow,ceiling_folder,indoor):
    if os.path.exists(f"{output_label_dir}/{i:06d}.label"):
        try:
            np.fromfile(f"{output_label_dir}/{i:06d}.label", dtype=np.int32)
            return
        except Exception as e:
            pass
    equal_idxs = {}
    iou_threshold=0.5

    voxel=get_Frame(i,velodyne_dir,voxel_size,ground_dir,merge_ground,ceiling_folder,indoor)
    f2sf={}
    for idx in range(len(poses)):
        p_index = max(0, idx - int(sn / 2))
        q_index = min(len(poses) - 1, idx + int(sn / 2))
        for current_idx in range(p_index, q_index + 1):
            if current_idx in f2sf:
                f2sf[current_idx].append(idx)
            else:
                f2sf[current_idx] = [idx]
    sf_list = f2sf[i]
    sf_idx = -1
    obj_prompt = {}
    with open(f'{kf_dir}/obj_dict.pkl', 'rb') as file:
        obj_prompt = pickle.load(file)
    objid2indice={}
    for k,v in obj_prompt.items():
        objid2indice[v]=k
    for sf_id in sf_list:
        sf_idx += 1
        load = None

        with open(f'{outcome_dir}/{sf_id}.pkl', 'rb') as f:
            load = pickle.load(f)

        if load is not None:
            for pc_id in load.keys():

                if len(objid2indice[pc_id // 10]) == 2:
                    continue
                sf_pc = np.array(load[pc_id])

                distances = np.linalg.norm(sf_pc, axis=1)
                sf_pc = sf_pc[distances > 1e-6]
                pseduo_id = pc_id*100+sf_idx
                if(sf_pc.shape[0] > 0):
                    sf_pc = transform_point_cloud(sf_pc,poses[sf_id])
                    sf_pc = transform_point_cloud(sf_pc, np.linalg.inv(poses[i]))
                    voxel_indices = np.floor(sf_pc[:, :3] / voxel_size).astype(np.int32)
                    unique_voxels = np.unique(voxel_indices, axis=0)
                    for voxel_id in unique_voxels:
                        voxel_key = tuple(voxel_id[:3])
                        if voxel_key in voxel:
                            if pseduo_id in voxel[voxel_key]:
                                voxel[voxel_key][pseduo_id] += 1
                            else:
                                voxel[voxel_key][pseduo_id] = 1
                        

    vid = list(voxel.keys())

    kdtree = cKDTree(vid)
    num_voxels = len(vid)
    visited = np.zeros(num_voxels, dtype=bool)
    clusters_label={}
    unlabeled_clusters = {}
    label2voxel={}
    label2clusterid={}
    cluster_id = 0

    for seed_idx in range(num_voxels):
        if not visited[seed_idx]:
            # Initialize a new cluster
            clusters_label[cluster_id] = {}
            unlabeled_clusters[cluster_id] = []
            label2voxel[cluster_id]={}
            region = [seed_idx]
            visited[seed_idx] = True

            while region:
                current_voxel_idx = region.pop()
                vid_key = vid[current_voxel_idx]
                if len(voxel[vid_key].keys()) > 0:
                    for label in voxel[vid_key]:
                        if label in clusters_label[cluster_id]:
                            clusters_label[cluster_id][label] += voxel[vid_key][label]
                        else:
                            clusters_label[cluster_id][label] = voxel[vid_key][label]

                    for label in voxel[vid_key]:
                        gt_label=label//1000
                        if gt_label in label2voxel[cluster_id]:
                            if label in label2voxel[cluster_id][gt_label]:
                                label2voxel[cluster_id][gt_label][label].append(vid_key)
                            else:
                                label2voxel[cluster_id][gt_label][label] = [vid_key]
                        else:
                            label2voxel[cluster_id][gt_label] = {label: [vid_key]}
                    for label in voxel[vid_key]:
                        gt_label=label//1000
                        if gt_label in label2clusterid:
                            if cluster_id not in label2clusterid[gt_label]:
                                label2clusterid[gt_label].append(cluster_id)
                        else:
                            label2clusterid[gt_label] = [cluster_id]
                else:
                    unlabeled_clusters[cluster_id].append(vid_key)

                # Find neighbors within the distance threshold
                neighbors = kdtree.query_ball_point(vid_key, voxel_grow)

                for neighbor_idx in neighbors:
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        region.append(neighbor_idx)

            cluster_id += 1
    #bleeding
    obj_kfid_xyz = {}
    with open(f'{kf_dir}/obj_kfid_xyz.pkl', 'rb') as file:
        obj_kfid_xyz = pickle.load(file)
    keyframe_id = os.path.join(kf_dir, f"keyframe_id.npy")
    keyframe_id = np.load(keyframe_id)
    for label in list(label2clusterid.keys()):
        if len(label2clusterid[label]) > 1:
            kf_id = list(obj_kfid_xyz[label].keys())[0]
            xyz = np.array(obj_kfid_xyz[label][kf_id]).reshape(-1, 3)
            xyz = transform_point_cloud(xyz, poses[keyframe_id[kf_id]])
            xyz = transform_point_cloud(xyz, np.linalg.inv(poses[i]))
            xyz_indices = np.floor(xyz[:, :3] / voxel_size).astype(int)
            cluster2 = np.array(xyz_indices).reshape(-1, 3)
            min_dist = np.inf
            min_cluster_id = None

            for cluster_id in label2clusterid[label]:
                cluster1 = []
                for vc in label2voxel[cluster_id][label].values():
                    cluster1 = cluster1 + vc
                cluster1 = np.array(cluster1)
                distances = cdist(cluster1, cluster2, metric='euclidean')
                min_distance = np.min(distances)
                if min_distance < min_dist:
                    min_dist = min_distance
                    min_cluster_id = cluster_id
           
            for cluster_id in label2clusterid[label]:
                if cluster_id != min_cluster_id:
                    for pse_label,voxel_ids in label2voxel[cluster_id][label].items():
                        del clusters_label[cluster_id][pse_label]
                        for voxel_id in voxel_ids:
                            del voxel[voxel_id][pse_label]
                            if len(voxel[voxel_id].keys()) == 0:
                                unlabeled_clusters[cluster_id].append(voxel_id)
                    del label2voxel[cluster_id][label]
    
    ans_voxel={}
    dbscan = DBSCAN(eps=z_grow, min_samples=3)

    for cluster_idx in clusters_label.keys():
        values = clusters_label[cluster_idx]
        if len(values.keys()) == 0:
            for v in unlabeled_clusters[cluster_idx]:
                ans_voxel[v] = 0
        else:
            unique_sorted_list = list(values.keys())

            sf2label={}
            sfbbox={}
            equal_temp={}
            for label in unique_sorted_list:
                sf_iidx = label%100
                if sf_iidx in sf2label:
                    sf2label[sf_iidx].add(label)
                else:
                    sf2label[sf_iidx]={label}

            for label_set in sf2label.values():
                for label1,label2 in itertools.combinations(label_set, 2):
                    if label1 // 1000 == label2 // 1000:
                        continue
                    elif label1 % 1000 == label2 % 1000:
                        continue
                    else:
                        bbox1=None
                        bbox2=None
                        if label1 in sfbbox:
                            bbox1 = sfbbox[label1]
                        else:
                            voxel_coords = np.array(label2voxel[cluster_idx][label1//1000][label1])
                            min_coords = voxel_coords.min(axis=0)
                            max_coords = voxel_coords.max(axis=0)

                            bbox1= (min_coords, max_coords)
                            sfbbox[label1] = bbox1
                        if label2 in sfbbox:
                            bbox2 = sfbbox[label2]
                        else:
                            voxel_coords = np.array(label2voxel[cluster_idx][label2//1000][label2])
                            min_coords = voxel_coords.min(axis=0)
                            max_coords = voxel_coords.max(axis=0)

                            bbox2= (min_coords, max_coords)
                            sfbbox[label2] = bbox2
                        # Compute intersection
                        inter_min = np.maximum(bbox1[0], bbox2[0])
                        inter_max = np.minimum(bbox1[1], bbox2[1])
                        inter_dims = np.maximum(0, inter_max - inter_min + 1)
                        inter_volume = np.prod(inter_dims)

                        if inter_volume > 9:
                            gt_label1 = label1 // 1000
                            gt_label2 = label2 // 1000
                            min_label= min(gt_label1,gt_label2)
                            max_label= max(gt_label1,gt_label2)
                            if max_label in equal_temp:
                                while max_label in equal_temp:
                                    max_label = equal_temp[max_label]
                                if max_label > min_label:
                                    equal_temp[max_label] = min_label
                                elif max_label < min_label:
                                    equal_temp[min_label] = max_label
                            else:
                                equal_temp[max_label] = min_label
            

            equal_set={}
            for temp_label in equal_temp.keys():
                t = temp_label
                while t in equal_temp:
                    t = equal_temp[t]
                if t in equal_idxs:
                    equal_idxs[t].add(temp_label)
                else:
                    equal_idxs[t] = {temp_label}
                if t in equal_set:
                    equal_set[t].add(temp_label)
                else:
                    equal_set[t] = {temp_label}

            gtlabel2voxel = {}
            for label in label2voxel[cluster_idx].keys():
                if label in gtlabel2voxel:
                    for vids in label2voxel[cluster_idx][label].values():
                        gtlabel2voxel[label] += vids
                else:
                    gtlabel2voxel[label] = []
                    for vids in label2voxel[cluster_idx][label].values():
                        gtlabel2voxel[label] += vids

            for equal_label in equal_set.keys():
                for temp_label in equal_set[equal_label]:
                    if temp_label in gtlabel2voxel:
                        gtlabel2voxel[equal_label] += gtlabel2voxel[temp_label]
                        gtlabel2voxel.pop(temp_label)
            

            unique_sorted_list = sorted(gtlabel2voxel.keys(), key=lambda k: len(gtlabel2voxel[k]), reverse=True)

            bbox_dict = {}
            
            for label, voxel_keys in gtlabel2voxel.items():
                voxel_coords = np.array(voxel_keys)
                min_coords = voxel_coords.min(axis=0)
                max_coords = voxel_coords.max(axis=0)

                bbox_dict[label] = {
                    'bbox': (min_coords, max_coords)}

            

            while len(unique_sorted_list) > 0:
                max_label = unique_sorted_list[0]
                voxel_coords = np.array(gtlabel2voxel[max_label])
                min_coords = voxel_coords.min(axis=0)
                max_coords = voxel_coords.max(axis=0)
                bbox_dict[max_label] = {
                    'bbox': (min_coords, max_coords)}
                
                if len(unique_sorted_list) == 1:
                    unique_sorted_list.remove(max_label)
                else:
                    is_update = False

                    for rest_label in unique_sorted_list[1:]:
                        iou = compute_iou((min_coords, max_coords), bbox_dict[rest_label]['bbox'])
                        if iou > iou_threshold:
                            is_update = True
                            unique_sorted_list.remove(rest_label)
                            gtlabel2voxel[max_label] += gtlabel2voxel[rest_label]
                            gtlabel2voxel.pop(rest_label)

                            if max_label in equal_idxs:
                                equal_idxs[max_label].add(rest_label)
                            else:
                                equal_idxs[max_label] = {rest_label}
                    if not is_update:
                        unique_sorted_list.remove(max_label)

            unique_sorted_list = sorted(gtlabel2voxel.keys(), key=lambda k: len(gtlabel2voxel[k]), reverse=True)   
            if len(unlabeled_clusters[cluster_idx]) > 0:
                unlabeled_points = np.array(unlabeled_clusters[cluster_idx],dtype=np.int32).reshape(-1,3)
                for label in unique_sorted_list:
                    now_points = np.array(gtlabel2voxel[label],dtype=np.int32).reshape(-1,3)
                    z_set = np.unique(now_points[:,2])
                    for z in z_set:
                        z_points = unlabeled_points[unlabeled_points[:,2] == z]
                        if(z_points.shape[0] == 0):
                            continue
                        unlabeled_points = unlabeled_points[unlabeled_points[:,2] != z]
                        db_labels = dbscan.fit_predict(z_points)
                        cluster_dict = {}
                        for sc_label in set(db_labels):
                            cluster_points = z_points[db_labels == sc_label]
                            cluster_dict[sc_label] = cluster_points.tolist()

                        for db_c in cluster_dict.values():
                            cluster1 = now_points[now_points[:,2] == z]
                            cluster2 = np.array(db_c,dtype=np.int32).reshape(-1,3)
                            distances = cdist(cluster1, cluster2, metric='euclidean')
                            min_distance = np.min(distances)  # 找到最小的距离
                            if min_distance < z_grow:
                                list_of_tuples = [tuple(sublist) for sublist in cluster2.tolist()]
                                gtlabel2voxel[label]+= list_of_tuples
                            else:
                                unlabeled_points = np.concatenate((unlabeled_points, cluster2), axis=0)
                        if unlabeled_points.shape[0] == 0:
                            break
                    if unlabeled_points.shape[0] == 0:
                            break
            unique_sorted_list = sorted(gtlabel2voxel.keys(), key=lambda k: len(gtlabel2voxel[k]))   
            for label in unique_sorted_list:
                for ind in gtlabel2voxel[label]:
                    voxel_key = tuple(ind[:3])
                    ans_voxel[voxel_key] = label
            
    with open(f'{equal_dir}/{i}_equal.pkl', 'wb') as f:
        pickle.dump(equal_idxs, f)
           


    output_label = []
    cloud = np.fromfile(f"{velodyne_dir}/{i:06d}.bin", dtype=np.float32).reshape(-1, 4)

    cloud_indices = np.floor(cloud[:, :3] / voxel_size).astype(int)
    for c_i in cloud_indices:
        voxel_key = tuple(c_i[:3])
        if voxel_key in ans_voxel:
            output_label.append(ans_voxel[voxel_key])
        else:
            output_label.append(0)
    if merge_ground:
        range_xy = np.load(f"{ground_save_dir}/range_xy.npy")
        x_range = range_xy[0]
        y_range = range_xy[1]
        x_min, x_max = x_range
        y_min, y_max = y_range
        vis_image = np.load(f"{ground_save_dir}/ground_seg_image.npy")
        ground_cloud = np.fromfile(f"{ground_dir}/{i:06d}.bin", dtype=np.float32).reshape(-1, 4)
        if ground_cloud.shape[0] > 0:

            global_cloud = transform_point_cloud(ground_cloud[:, :3], poses[i])
            relative_cloud = transform_point_cloud(global_cloud, np.linalg.inv(poses[0]))

            x_indices = ((relative_cloud[:, 0] - x_min) / voxel_size[0]).astype(int)
            y_indices = ((relative_cloud[:, 1] - y_min) / voxel_size[1]).astype(int)
            ground_label =[]

            for gi in range(len(relative_cloud)):
                ground_label.append(vis_image[y_indices[gi], x_indices[gi]])

            kdtree_cloud = KDTree(cloud[:, :3])
            

            _, indices = kdtree_cloud.query(ground_cloud[:, :3], k=1)


            for indices_i, idx in enumerate(indices.flatten()):
                output_label[idx] = ground_label[indices_i]

        if indoor:
            ceiling_cloud = np.fromfile(f"{ceiling_folder}/{i:06d}.bin", dtype=np.float32).reshape(-1, 4)
            if ceiling_cloud.shape[0] > 0:
                kdtree_cloud = KDTree(cloud[:, :3])
                _, indices = kdtree_cloud.query(ceiling_cloud[:, :3], k=1)


                for idx in indices.flatten():
                    output_label[idx] = 999

    output_label = np.array(output_label, dtype=np.int32)
    output_label.tofile(f"{output_label_dir}/{i:06d}.label")



def nms_3d(ground_save_dir,camera_num,mask_out_put,multiprocess_num,outcome_dir,img_dir,velodyne_dir,voxel_size,poses,sn,
           merge_ground,ground_dir,equal_dir,kf_dir,output_label_dir,voxel_grow,z_grow,ceiling_folder,indoor,resume=False):
    file_dict = defaultdict(dict)
    pattern = re.compile(r"(\d+)_(\d+)\.npy")

    for camera_idx in range(camera_num):
        for filename in os.listdir(f"{mask_out_put}/{camera_idx}"):
            match = pattern.match(filename)
            if match:
                f_id = int(match.group(1))
                obj_id = int(match.group(2))
                if camera_idx in file_dict[f_id]:
                    file_dict[f_id][camera_idx][obj_id] = f"{mask_out_put}/{camera_idx}/{filename}"
                else:
                    file_dict[f_id][camera_idx] = {obj_id:f"{mask_out_put}/{camera_idx}/{filename}"}

    sorted_keys = sorted(file_dict.keys())
    #check
    sf_files = [f for f in os.listdir(outcome_dir) if f.endswith('.pkl')]
    if not (resume and (len(sf_files) == len(sorted_keys))):
        args = [(f_id,camera_num,img_dir,file_dict,outcome_dir)
                for f_id in sorted_keys]
        with multiprocessing.Pool(processes=multiprocess_num) as pool:
            pool.starmap(mask2point, args)
    #check
    label_files = [f for f in os.listdir(output_label_dir) if f.endswith('.label')]
    if not (resume and (len(label_files) == len(poses))):
        args = [(key,velodyne_dir,voxel_size,poses,sn,outcome_dir,merge_ground,ground_dir,equal_dir,ground_save_dir,kf_dir,output_label_dir,voxel_grow,z_grow,ceiling_folder,indoor)
                for key in range(len(poses))]
        with multiprocessing.Pool(processes=multiprocess_num) as pool:
                pool.starmap(nms, args)
    



