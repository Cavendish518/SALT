import os
import numpy as np
import pickle
import re
from collections import defaultdict
from collections import Counter
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import itertools





def load_sets_from_files(directory):
    all_sets = []

    for filename in os.listdir(directory):
        if filename.endswith("_equal.pkl"):
            file_path = os.path.join(directory, filename)

            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)


                for key, value in data_dict.items():
                    new_set = {key} | value 
                    all_sets.append(new_set)

    return all_sets

def merge_sets(sets):

    merged = True
    while merged:
        merged = False
        result = []
        while sets:
            current_set = sets.pop(0)

            intersecting = [s for s in sets if s & current_set]

            if intersecting:
                for s in intersecting:
                    current_set |= s
                    sets.remove(s)
                merged = True
            result.append(current_set)
        sets = result

    return sets


def create_mapping(final_sets):
    mapping = {}

    for s in final_sets:
        min_value = min(s)
        for num in s:
            mapping[num] = min_value

    return mapping



def nms_4d(sn,kf_dir,equal_dir,camera_num,mask_out_put,output_label_dir,refre_label_dir,resume=False):
    #check
    label_files = [f for f in os.listdir(output_label_dir) if f.endswith('.label')]
    label_files_4d = [f for f in os.listdir(refre_label_dir) if f.endswith('.label')]
    if resume and (len(label_files) == len(label_files_4d)):
        return
    
    obj2kf = {}
    keyframes = np.load(f"{kf_dir}/keyframe_id.npy")
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
    with open(f'{kf_dir}/obj_kfid_xyz.pkl', 'rb') as f:
        obj_kfid_xyz = pickle.load(f)
    if not os.path.exists(f'{equal_dir}/final_mapping.pkl'):
        objid_life={}
        for f_id in file_dict:
            for t in file_dict[f_id]:
                for objid in file_dict[f_id][t]:
                    if objid not in objid_life:
                        objid_life[objid] = [f_id]
                    else:
                        objid_life[objid].append(f_id)
        for objid in objid_life:
            objid_life[objid] = (min(objid_life[objid])-sn//2, max(objid_life[objid])+sn//2)

        if not os.path.exists(f'{equal_dir}/combine_dict.pkl'):
            result_dict = {}
            for filename in os.listdir(equal_dir):
                if filename.endswith("_equal.pkl"):
                    file_path = os.path.join(equal_dir, filename)
                    with open(file_path, 'rb') as f:
                        data_dict = pickle.load(f)
                        f_id = int(filename.split("_")[0])
                        if f_id not in result_dict:
                            result_dict[f_id] = []

                        for key, value in data_dict.items():
                            new_set = {key} | value
                            result_dict[f_id].append(new_set)
            combine_dict = {}
            for f_id in result_dict:
                for set1 in result_dict[f_id]:
                    combinations = list(itertools.combinations(set1, 2))
                    for t in combinations:
                        if t not in combine_dict:
                            combine_dict[t] = [f_id]
                        else:
                            combine_dict[t].append(f_id)

            with open(f'{equal_dir}/combine_dict.pkl', 'wb') as f:
                pickle.dump(combine_dict, f)
            print("combine_dict save")
        else:
            with open(f'{equal_dir}/combine_dict.pkl', 'rb') as f:
                combine_dict = pickle.load(f)

        all_sets = []
        for l1,l2 in combine_dict:
            kf1 = list(obj_kfid_xyz[l1].keys())[0]
            kf2 = list(obj_kfid_xyz[l2].keys())[0]
            nms_set = set(combine_dict[(l1,l2)])

            l1_set = objid_life[l1]
            l2_set = objid_life[l2]
            start_id= max(l1_set[0], l2_set[0])
            end_id = min(l1_set[1], l2_set[1])

            jaccard_index = len(nms_set) / (end_id - start_id + 1)
            if jaccard_index > 0.7:
                all_sets.append(set([l1,l2]))
        final_mapping = create_mapping(all_sets)
        with open(f'{equal_dir}/final_mapping.pkl', 'wb') as f:
            pickle.dump(final_mapping, f)
        print("final_mapping save")
    else:

        with open(f'{equal_dir}/final_mapping.pkl', 'rb') as f:
            final_mapping = pickle.load(f)
    print("final_mapping end")
            
    for label_file in label_files:
        file_path = os.path.join(output_label_dir, label_file)
        label_data = np.fromfile(file_path, dtype=np.int32)
        for ld_idx in range(label_data.shape[0]):
            label_data[ld_idx] = final_mapping.get(label_data[ld_idx], label_data[ld_idx])
        label_data.tofile(os.path.join(refre_label_dir, label_file))
    print("update label end")
    #merge ground