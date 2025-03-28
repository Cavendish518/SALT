from segGround.seg_ground_nonground import seg_ground_nonground
from best_RT.best_RT import get_best_RT
from utils.utils import *
from keyframe.keyframe import get_kf
from keyframe.kf2prompt import get_kf_prompt
from sf2img.sf2img import save_image
from sf2img.prompt2img import save_prompt
from sf2sam.img2sam import img2SAM
from segGround.ground_kmeans import seg_semantic_ground
from getLabel.nms_3d import nms_3d
from getLabel.nms_4d import nms_4d
from map.map import labelmap
from map.map import SALT_label
from cam.velodyne2cam import velodyne2cam
from cam.cam2sam import cam2sam
from cam.nms_cam import nms_cam
import numpy as np
import torch
import random
import sys
import shutil

if __name__=='__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    args = get_arg()
    data_dir = args.data_dir
    
    resume_model = args.resume
    cache_dir = args.cache_dir
    seg_ground = args.seg_ground
    K = np.array(args.K)
    camera_model = args.camera_model
    real_cam_num = args.real_cam_num
    rot = args.rot
    tra = args.tra
    start_rot = args.start_rot
    camera_angle_ud = args.camera_angle_ud
    camera_angle_rl = args.camera_angle_rl
    camera_position = args.camera_position
    sn = args.sn
    RT_batch_size = args.RT_batch_size
    voxel_size = args.voxel_size
    width = args.width
    height = args.height
    device = args.device
    checkpoint_path = "best_model.pth"
    eps = args.eps
    min_samples = args.min_samples
    multiprocess_num = args.multiprocess_num
    camera_num = args.pseduo_camera_num
    GPU_num = args.GPU_num
    GPU_prework_num = args.GPU_prework_num
    sam2_checkpoint = args.sam2_checkpoint
    model_cfg = args.model_cfg
    ground_num = args.ground_num
    voxel_grow = args.voxel_grow
    z_grow = args.z_grow
    indoor = args.indoor

    velodyne_folder = f"{data_dir}/velodyne"
    ground_folder = f"{cache_dir}/ground"
    nonground_folder = f"{cache_dir}/nonground"
    ceiling_folder=None
    if indoor:
        ceiling_folder = f"{cache_dir}/ceiling"
    kf_dir = f"{cache_dir}/kf"
    output_image_dir = f"{cache_dir}/img_seq"
    mask_out_dir = f"{output_image_dir}/mask_out"
    ground_save_dir = f"{cache_dir}/ground_save"
    equal_dir = f"{cache_dir}/equal"
    outcome_dir = f"{cache_dir}/outcome"
    output_3d_label_dir = f"{cache_dir}/prediction_3d"
    output_4d_label_dir = f"{cache_dir}/prediction"
    output_map_label_dir = f"{cache_dir}/prediction_map"
    output_camera_dir = f"{cache_dir}/cam"
    output_cam_label_dir=f"{output_camera_dir}/predictions"
    output_cam_label_map_dir=f"{output_camera_dir}/prediction_map"

    os.makedirs(ground_folder, exist_ok=True)
    os.makedirs(nonground_folder, exist_ok=True)
    if indoor:
        os.makedirs(ceiling_folder, exist_ok=True)
    os.makedirs(kf_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(ground_save_dir, exist_ok=True)
    os.makedirs(outcome_dir, exist_ok=True)
    os.makedirs(equal_dir, exist_ok=True)
    os.makedirs(output_3d_label_dir, exist_ok=True)
    os.makedirs(output_4d_label_dir, exist_ok=True)
    os.makedirs(output_map_label_dir, exist_ok=True)
    os.makedirs(output_camera_dir, exist_ok=True)
    os.makedirs(output_cam_label_dir, exist_ok=True)
    os.makedirs(output_cam_label_map_dir, exist_ok=True)
    poses_file = f'{data_dir}/poses.txt'
    calib_file = f'{data_dir}/calib.txt'
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('Tr:'):
                values = line.split()[1:]
                Tr_matrix = np.array(values, dtype=np.float32).reshape(3, 4)
                break

    if Tr_matrix is not None:
        Tr_matrix = np.vstack((Tr_matrix, [0, 0, 0, 1]))
    poses = load_poses(poses_file, Tr_matrix)
    if not seg_ground:
        nonground_folder = velodyne_folder
    #1.SegGround 5%---------------------------------------------------------------------------------------------------------
    print("Info: ground seg start")
    sys.stdout.flush()

    if seg_ground:
        seg_ground_nonground(data_dir,ground_folder,nonground_folder,len(poses),ceiling_folder,indoor,resume=resume_model)

    print(f"Progress: 5%")
    sys.stdout.flush() 
    print("Info: ground over")
    sys.stdout.flush()
    #2.select keyframes 7%----------------------------------------------------------------------------------------------------
    print("Info: select keyframes over")
    sys.stdout.flush() 

    get_kf(poses,kf_dir,sn,resume=resume_model)

    print(f"Progress: 7%")
    sys.stdout.flush() 
    print("Info: select keyframes over")
    sys.stdout.flush()
    #3.choose RT 10%-------------------------------------------------------------------------------------------------------------
    print("Info: calculate RT start")
    sys.stdout.flush() 

    camera_angle_ud,camera_position = get_best_RT(nonground_folder,ground_folder,seg_ground,kf_dir,K,rot,tra,start_rot,camera_angle_ud,camera_angle_rl
                ,camera_position[0],sn,RT_batch_size,Tr_matrix,poses,voxel_size,width,height,cache_dir,device,checkpoint_path)

    print(f"Progress: 10%")
    sys.stdout.flush() 
    print(f"Info: calculate RT over camera_angle:{camera_angle_ud} camera_position:{camera_position}")
    sys.stdout.flush() 
    #4.get prompt 15%---------------------------------------------------------------------------------------------------------------
    print("Info: get prompt start")
    sys.stdout.flush() 

    get_kf_prompt(kf_dir,poses,nonground_folder,eps,min_samples,sn,voxel_size,resume=resume_model)

    print(f"Progress: 15%")
    sys.stdout.flush() 
    print("get prompt over")
    sys.stdout.flush() 
    #5.get image and prompt 25%----------------------------------------------------------------------------------------------------------
    print("Info: rendering image")
    sys.stdout.flush() 
    
    save_image(nonground_folder,output_image_dir,poses,camera_num,camera_position,camera_angle_ud,camera_angle_rl,tra,rot,K,Tr_matrix,
               width, height,sn,multiprocess_num,voxel_size,start_rot,resume=resume_model)
    
    print(f"Progress: 25%")
    sys.stdout.flush() 
    print("Info: rendering image over")
    sys.stdout.flush() 
    # 30%------------------------------------------------------------------------------------------------------------------------------------
    print("Info: project prompt")
    sys.stdout.flush() 
    
    save_prompt(kf_dir,poses,Tr_matrix,K,camera_position,camera_angle_ud,camera_angle_rl,tra,rot,camera_num,output_image_dir,sn,start_rot,resume=resume_model)
    
    print(f"Progress: 30%")
    sys.stdout.flush() 
    print("Info: project prompt over")
    sys.stdout.flush() 
    #6.SAM2 50%------------------------------------------------------------------------------------------------------------------------
    print("Info: img2SAM start")
    sys.stdout.flush()

    img2SAM(GPU_num,GPU_prework_num,camera_num,sam2_checkpoint,model_cfg,kf_dir,sn,output_image_dir,resume=resume_model)
    
    print(f"Progress: 50%")
    sys.stdout.flush() 
    print("Info: img2SAM over")
    sys.stdout.flush()
    #7.seg Ground 60%------------------------------------------------------------------------------------------------------------------------
    print("Info: semantic segment ground start")
    sys.stdout.flush()

    if seg_ground:
        seg_semantic_ground(ground_folder,poses,ground_save_dir,ground_num,voxel_size,resume=resume_model)
    
    print(f"Progress: 60%")
    sys.stdout.flush() 
    print("Info: semantic segment ground over")
    sys.stdout.flush()
    #8.get label 80%------------------------------------------------------------------------------------------------------------------------
    print("Info: nms3d start")
    sys.stdout.flush()

    nms_3d(ground_save_dir,camera_num,mask_out_dir,multiprocess_num,outcome_dir,output_image_dir,velodyne_folder,voxel_size,poses,sn,
           seg_ground,ground_folder,equal_dir,kf_dir,output_3d_label_dir,voxel_grow,z_grow,ceiling_folder,indoor,resume=resume_model)
    
    print(f"Progress: 80%")
    sys.stdout.flush() 
    print("Info: nms3d over")
    sys.stdout.flush()
    #9.4d label 90%------------------------------------------------------------------------------------------------------------------------
    print("Info: nms4d start")
    sys.stdout.flush()

    nms_4d(sn,kf_dir,equal_dir,camera_num,mask_out_dir,output_3d_label_dir,output_4d_label_dir,resume=resume_model)

    print(f"Progress: 90%")
    sys.stdout.flush() 
    print("Info: nms4d over")
    sys.stdout.flush()
    #10.map 95%------------------------------------------------------------------------------------------------------------------------
    print("Info: smooth start")
    sys.stdout.flush()
    
    labelmap(poses,velodyne_folder,output_4d_label_dir,output_map_label_dir,resume=resume_model)

    print(f"Progress: 95%")
    sys.stdout.flush() 
    print("Info: smooth over")
    sys.stdout.flush()
    if camera_model:
        #camera velo2cam 96%------------------------------------------------------------------------------------------------------------------------
        print("Info: camera: velodyne2cam start")
        sys.stdout.flush()
        
        velodyne2cam(velodyne_folder,poses,output_camera_dir,real_cam_num)

        print(f"Progress: 96%")
        sys.stdout.flush() 
        print("Info: camera: velodyne2cam over")
        sys.stdout.flush()
        #camera cam2sam 98%------------------------------------------------------------------------------------------------------------------------
        print("Info: camera: cam2sam start")
        sys.stdout.flush()
        
        cam2sam(data_dir,real_cam_num,sam2_checkpoint,model_cfg)

        print(f"Progress: 98%")
        sys.stdout.flush() 
        print("Info: camera: cam2 over")
        sys.stdout.flush()
        #camera camnms 99%------------------------------------------------------------------------------------------------------------------------
        print("Info: camera: cam nms start")
        sys.stdout.flush()
        
        nms_cam(data_dir,cache_dir,poses,real_cam_num,seg_ground,voxel_size,ground_num,z_grow,kf_dir)

        print(f"Progress: 99%")
        sys.stdout.flush() 
        print("Info: camera: cam nms over")
        sys.stdout.flush()

        print("Info: camera: smooth start")
        sys.stdout.flush()
        labelmap(poses,velodyne_folder,output_cam_label_dir,output_cam_label_map_dir,resume=resume_model)
        print("Info: camera: smooth over")
        sys.stdout.flush()
    #11.SALT_label 100%------------------------------------------------------------------------------------------------------------------------
    print("Info: output file...")
    sys.stdout.flush()
    SALT_label(poses,output_map_label_dir, data_dir,camera_model,output_cam_label_map_dir)
    shutil.rmtree(cache_dir)
    print(f"Progress: 100%")
    sys.stdout.flush() 
    print("Info: complete!")
    sys.stdout.flush()
   #----------------------------------------------------------------------------------------------------------------------------------------
