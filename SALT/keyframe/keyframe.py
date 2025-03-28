import os
import numpy as np
from scipy.spatial.transform import Rotation as R


# 关键帧选择函数
def select_keyframes(poses,kf_dir,keyframe_delta_trans,keyframe_delta_angle,MAX_DIS):
    keyframe_index = 0
    accumulated_index = None
    prev_keypose = None
    is_first = True
    keyframes=[]
    count = 0

    for i, pose in enumerate(poses):
        if is_first:
            is_first = False
            prev_keypose = pose
            accumulated_index = i 
            count += 1
            continue


        delta_pose = np.linalg.inv(prev_keypose) @ pose
        dx = np.linalg.norm(delta_pose[:3, 3])
        rotation = R.from_matrix(delta_pose[:3, :3])
        da = rotation.magnitude() 

        if count > MAX_DIS or (not (dx < keyframe_delta_trans and da < keyframe_delta_angle)):
            keyframes.append(accumulated_index)

            prev_keypose = pose
            accumulated_index = i
            count = 1
        else:
            count += 1

    keyframes.append(accumulated_index)
    
    keyframe_id = os.path.join(kf_dir, f"keyframe_id.npy")
    np.save(keyframe_id, np.array(keyframes))

def get_kf(poses,kf_dir,sn,resume=False):
    if resume and os.path.exists(os.path.join(kf_dir, f"keyframe_id.npy")):
        return 0
    keyframe_delta_trans = 10
    keyframe_delta_angle = np.deg2rad(45)
    MAX_DIS = sn-1
    select_keyframes(poses,kf_dir,keyframe_delta_trans,keyframe_delta_angle,MAX_DIS)
