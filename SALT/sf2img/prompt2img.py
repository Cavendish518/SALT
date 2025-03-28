from utils.utils import *
from sf2img.sf2img import get_pseudoCameras


def prompt_tf(kcs_origin, transformation):
    kcs_tf = []
    for m in range(len(kcs_origin)):
        kp = kcs_origin[m]
        kp = transform_point_cloud(kp, transformation)
        kcs_tf.append(kp)
    return kcs_tf


def prompt_zero(kcs_origin, kcs_index):
    kcs_tf = []
    kcs_idx = []
    for m in range(len(kcs_origin)):
        kp = kcs_origin[m]
        mask = kp[:, 2] > 0
        kp = kp[mask]
        kk = kcs_index[m]
        kk = kk[mask]
        kcs_tf.append(kp)
        kcs_idx.append(kk)
    return kcs_tf, kcs_idx


def prompt_K(kcs_origin, K):
    kcs_tf = []
    for m in range(len(kcs_origin)):
        kp = kcs_origin[m]
        kp = kp @ K.T
        kp2 = kp[:, 2]
        kp = kp[:, :] / kp2[:, np.newaxis]
        kp[:, 2] = kp2
        kcs_tf.append(kp)
    return kcs_tf


def get_prompt(kcs_origin, depth_matrix, kcs_index):
    prompt_c = []
    c_index = []
    for m in range(len(kcs_origin)):
        kp = kcs_origin[m]
        for i in range(kp.shape[0]):
            y = int(kp[i, 0])
            x = int(kp[i, 1])
            z = kp[i, 2]
            if (x >= 0 and x < depth_matrix.shape[0] and y >= 0 and y < depth_matrix.shape[1]):
                if (depth_matrix[x, y] > 0 and abs(z - depth_matrix[x, y]) < 2):
                    kk = kcs_index[m]
                    idx = kk[i, :]
                    prompt_c.append([x, y])
                    c_index.append(idx)

    return np.array(prompt_c), c_index

def get_image(output_image_dir, i_idx,camera_positions,camera_orientations,Tr_matrix, K,kcs_tf, kcs_index):

    kcs_tf = prompt_tf(kcs_tf, Tr_matrix)
    prompts_c = []
    prompts_c_idx = []
    

    for j in range(len(camera_positions)):
        depth_matrix = np.load(f"{output_image_dir}/{j}/{i_idx:06d}.depth.npy")

        transformation_matrix = get_transformation_matrix(camera_positions[j], camera_orientations[j])

        kcs_tf_1 = prompt_tf(kcs_tf, transformation_matrix)

        kcs_tf_1, kcs_index_1 = prompt_zero(kcs_tf_1, kcs_index)

        
        kcs_tf_1 = prompt_K(kcs_tf_1, K)

        
        prompt_c, c_idxs = get_prompt(kcs_tf_1, depth_matrix, kcs_index_1)

        
        prompts_c.append(prompt_c)
        prompts_c_idx.append(c_idxs)
        
      
    return prompts_c, prompts_c_idx

def save_prompt(kf_dir,poses,Tr_matrix,K,camera_position,camera_angle_ud,camera_angle_rl,tra,rot,camera_num,output_image_dir,sn,start_rot,resume=False):
    if resume:
        check_bool = True
        for cam_idx in range(camera_num):
            img_files = [f for f in os.listdir(f"{output_image_dir}/{cam_idx}") if f.endswith('.prompteidx')]
            if len(img_files) != len(poses):
                check_bool=False
                break
        if check_bool:
            return
    camera_positions,camera_orientations = get_pseudoCameras(camera_num,camera_position,camera_angle_ud,camera_angle_rl,tra,rot,start_rot)
    keyframe_id = os.path.join(kf_dir, f"keyframe_id.npy")
    keyframe_id = np.load(keyframe_id)
    keyframe_id = keyframe_id.tolist()
    f2k = {}
    for i in range(len(keyframe_id)):
        p_index = max(0, keyframe_id[i] - int(sn / 2))
        q_index = min(len(poses) - 1, keyframe_id[i] + int(sn / 2))
        for v in range(p_index, q_index+1):
            if v in f2k:
                f2k[v].append(i)
            else:
                f2k[v] = [i]
    for i in range(len(poses)):
        kcs_origin = []
        kcs_index = []
        k_repeat = {}
        p_index = max(0, i - int(sn / 2))
        q_index = min(len(poses) - 1, i + int(sn / 2))
        for j in range(p_index, q_index):
            if j in f2k:
                for k in f2k[j]:
                    if k in k_repeat:
                        continue
                    else:
                        k_repeat[k] = k
                    kcs_load = np.load(f'{kf_dir}/kf_{k}_c.npy')
                    kcs_load_index = np.load(f'{kf_dir}/kf_{k}_index_c.npy')
                    kcs_tf = transform_point_cloud(kcs_load, poses[keyframe_id[k]])
                    kcs_tf = transform_point_cloud(kcs_tf, np.linalg.inv(poses[i]))
                    kcs_origin.append(kcs_tf)
                    kcs_index.append(kcs_load_index)

     
        prompt_c, prompt_c_idx = get_image(output_image_dir, i,camera_positions,camera_orientations,Tr_matrix, K,
                                                                                                     kcs_origin,
                                                                                                     kcs_index)
        for m in range(len(prompt_c)):
            if not os.path.exists(f"{output_image_dir}/{m}"):
                print("error")

           
            np.save(f"{output_image_dir}/{m}/{i:06d}.promptc", prompt_c[m])
            np.save(f"{output_image_dir}/{m}/{i:06d}.promptcidx", prompt_c_idx[m])


