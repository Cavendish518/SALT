from utils.utils import *
from sklearn.cluster import DBSCAN
import pickle
def cluster_prompt(xyz,eps,min_samples):
    """ DBSCAN prompt point finder
       Inputs:
           xyz: [N,3] numpy array, points to cluster
           eps_list: list of eps values for DBSCAN
           min_samples_list: list of min_samples values for DBSCAN
    """
    
    y_pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz)

    # 计算每个簇的中心点
    cluster_centers = []
    cluster2point={}
    idx = 0

    for cluster_id in set(y_pred):
        if cluster_id == -1: 
            continue

        cluster_points = xyz[y_pred == cluster_id]
        cluster2point[idx] = cluster_points

        center = cluster_points.mean(axis=0)

        distances = np.linalg.norm(cluster_points - center, axis=1)

        nearest_point_idx = np.argmin(distances)

        nearest_point = cluster_points[nearest_point_idx]

        cluster_centers.append(nearest_point)
        idx += 1


    return cluster_centers,cluster2point

def get_kf_prompt(kf_dir,poses,nonground_dir,eps,min_samples,sn,voxel_size,resume=False):
    if resume and os.path.exists(f'{kf_dir}/obj_negative.pkl'):
        return 0
    keyframe_id = os.path.join(kf_dir, f"keyframe_id.npy")
    keyframe_id = np.load(keyframe_id)
    keyframe_id = keyframe_id.tolist()
    negative_dict={}
    for i in range(len(keyframe_id)):
        np_xyz = get_basicFrame(sn, keyframe_id[i], nonground_dir, poses,voxel_size)
        np_xyz=np_xyz[:, :3]
        all_index_cp = []
        all_kp = []


        for idx, (eps_idx, min_samples_idx) in enumerate(zip(eps, min_samples)):
            key_center, cluster2point = cluster_prompt(np_xyz, eps_idx, min_samples_idx)
            kp = np.array(key_center)
            p = kp.shape[0]

            index_cp = np.column_stack((np.full(p, i), np.full(p, idx), np.arange(p)))
            
            all_index_cp.append(index_cp)
            all_kp.append(kp)


        all_index_cp = np.vstack(all_index_cp)
        all_kp = np.vstack(all_kp)
        np.save(f'{kf_dir}/kf_{i}_c.npy', all_kp)
        np.save(f'{kf_dir}/kf_{i}_index_c.npy', all_index_cp)
  
        min_distance=eps[-1]

        for j in range(all_kp.shape[0]):
            point = all_kp[j]
            distances = []

            for k in range(len(cluster2point.keys())):
                cluster = np.array(cluster2point[k])
                dist = np.linalg.norm(cluster - point, axis=1)
                min_dist = np.min(dist)
                if min_dist > min_distance:
                    distances.append((min_dist,tuple(index_cp[k].tolist())))
            distances.sort(key=lambda x: x[0])
            negative_index = [index for _, index in distances[:5]]        
            negative_dict[tuple(all_index_cp[j].tolist())]=negative_index
    
    obj_dict = {}
    obj_kfid_xyz={}
    obj_id = 1000

    for i in range(len(keyframe_id)):
        kcs_load_index = np.load(f'{kf_dir}/kf_{i}_index_c.npy')
        kcs_load = np.load(f'{kf_dir}/kf_{i}_c.npy')
        for j,kc in enumerate(kcs_load_index):
            obj_dict[tuple(kc)] = obj_id
            obj_kfid_xyz[obj_id]={i:kcs_load[j]}
            obj_id += 1
    with open(f'{kf_dir}/obj_dict.pkl', 'wb') as f:
        pickle.dump(obj_dict, f)
    with open(f'{kf_dir}/obj_kfid_xyz.pkl', 'wb') as f:
        pickle.dump(obj_kfid_xyz, f)
    
    save_negative_dict={}
    for k,v in negative_dict.items():
        save_negative_dict[obj_dict[k]]=[]
        for i in v:
            save_negative_dict[obj_dict[k]].append(obj_dict[i])
    with open(f'{kf_dir}/obj_negative.pkl', 'wb') as f:
        pickle.dump(save_negative_dict, f)

