from collections import defaultdict
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import sys
from sam2.build_sam import build_sam2_video_predictor
import pickle
import multiprocessing

def multi_img2SAM(device,camera_idx,sn,img_dir,kf_dir,sam2_checkpoint,model_cfg,task_num, all_task_num,resume):
    torch.cuda.set_device(int(device[-1]))
    if device != "cuda:0":
        torch.autocast(device, dtype=torch.bfloat16).__enter__()
    else:
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    batch_group_size = 200
    img_video_dir=f"{img_dir}/{camera_idx}"
    mask_out_put = f"{img_dir}/mask_out/{camera_idx}"

    if not os.path.exists(mask_out_put):
        os.makedirs(mask_out_put,exist_ok=True)
    # select the device for computation

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    frame_names = [
        p for p in os.listdir(img_video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    negative_dict={}
    with open(f'{kf_dir}/obj_negative.pkl', 'rb') as f:
        negative_dict = pickle.load(f)
    
    obj_prompt = {}
    with open(f'{kf_dir}/obj_dict.pkl', 'rb') as file:
        obj_prompt = pickle.load(file)
    
    obj2kf = {}
    
    keyframe_id = os.path.join(kf_dir, f"keyframe_id.npy")
    keyframes = np.load(keyframe_id)

    
    obj_dict = defaultdict(dict)
    neg_obj_dict = defaultdict(dict)

    for frame_id in range(len(frame_names)):
        prompt_c = np.load(f"{img_video_dir}/{frame_id:06d}.promptc.npy")
        prompt_c_idx = np.load(f"{img_video_dir}/{frame_id:06d}.promptcidx.npy")
        for j in range(prompt_c_idx.shape[0]):
            x, y, z = prompt_c_idx[j]
            key = (x, y,z)
            points = np.array([[prompt_c[j][1], prompt_c[j][0]]], dtype=np.float32)
            if y==0:
                obj_dict[obj_prompt[key]][frame_id] = points
                obj2kf[obj_prompt[key]] = x
            if y==1:
                neg_obj_dict[obj_prompt[key]][frame_id] = points
    
    sorted_keys = sorted(obj_dict.keys())
    half =None
    if all_task_num != 0:
        half_length = len(sorted_keys) // all_task_num
        if task_num != all_task_num:
            half = sorted_keys[(task_num-1) * half_length:task_num * half_length]
        else:
            half = sorted_keys[(task_num-1) * half_length:]
    else:
        half = sorted_keys

    for obj_idx in half:
        check_obj =False
        if resume:
            for f_id in obj_dict[obj_idx].keys():
                if os.path.exists(f"{mask_out_put}/{f_id}_{obj_idx}.npy"):
                    check_obj = True
                    try:
                        np.load(f"{mask_out_put}/{f_id}_{obj_idx}.npy")
                    except Exception as e:
                        check_obj=False
                        break
            if check_obj:
                continue
        kf1 = obj2kf[obj_idx]
        neg_obj_idxs = negative_dict[obj_idx]

        f_min = min(obj_dict[obj_idx].keys())
        f_max = max(obj_dict[obj_idx].keys())
        batches = [(i, min(i + batch_group_size - 1, f_max)) for i in range(f_min, f_max + 1, batch_group_size)]
        for batch in batches:
            frame_sam = []
            for f_id in range(batch[0], batch[1] + 1):
                frame_sam.append(frame_names[f_id])
            inference_state = predictor.init_state(video_path=frame_sam, video_dir=img_video_dir)
            for f_id in range(batch[0], batch[1] + 1):
                labels = np.array([], np.int32)
                click_points = np.array([]).reshape(-1, 2)
                if f_id in obj_dict[obj_idx].keys():
                    click_points = np.concatenate((click_points, obj_dict[obj_idx][f_id]), axis=0)
                    labels = np.concatenate((labels, [1]), dtype=np.int32, axis=0)
                    for obj_idx2 in neg_obj_idxs:
                        if obj_idx2 in neg_obj_dict:
                            if f_id in neg_obj_dict[obj_idx2].keys():
                                click_points = np.concatenate((click_points, neg_obj_dict[obj_idx2][f_id]), axis=0)
                                labels = np.concatenate((labels, [0]), dtype=np.int32, axis=0)
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=f_id-batch[0],
                        obj_id=obj_idx,
                        points=click_points,
                        labels=labels,
                    )
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            for f_id in range(batch[0], batch[1] + 1):
                if f_id in obj_dict[obj_idx].keys():
                    mask = video_segments[f_id-batch[0]][obj_idx][0]
                    if mask.any():
                        np.save(f"{mask_out_put}/{f_id}_{obj_idx}.npy",mask)

            predictor.reset_state(inference_state)
            torch.cuda.empty_cache()
    print(f"Info: img2SAM pseduo camera {camera_idx} done")
    sys.stdout.flush()

def img2SAM(GPU_num,GPU_prework_num,cam_num,sam2_checkpoint,model_cfg,kf_dir,sn,img_dir,resume=False):
    if resume and os.path.exists(f"{img_dir}/check.npy"):
        return
    all_task_num = ((GPU_num * GPU_prework_num) // cam_num)
    if all_task_num == 0:
        cuda_task_list=[]
        GPU_idx = 0
        for cam_idx in range(cam_num):
            cuda_task_list.append((f"cuda:{GPU_idx//GPU_prework_num}",cam_idx,0,all_task_num))
            GPU_idx+=1
            if GPU_idx == GPU_num * GPU_prework_num:
                GPU_idx = 0
        args = [(key[0],key[1],sn,img_dir,kf_dir,sam2_checkpoint,model_cfg,key[2],key[3],resume)
                for key in cuda_task_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(GPU_num *GPU_prework_num) as pool:
            pool.starmap(multi_img2SAM, args)
    else:
        cuda_task_list=[]
        GPU_idx = 0
        for cam_idx in range(cam_num):
            for task_num in range(all_task_num):
                cuda_task_list.append((f"cuda:{GPU_idx//GPU_prework_num}",cam_idx,task_num+1,all_task_num))
                GPU_idx+=1
        args = [(key[0],key[1],sn,img_dir,kf_dir,sam2_checkpoint,model_cfg,key[2],key[3],resume)
                for key in cuda_task_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=len(args)) as pool:
            pool.starmap(multi_img2SAM, args)
    check_array=np.array([1])
    np.save(f"{img_dir}/check.npy",check_array)

