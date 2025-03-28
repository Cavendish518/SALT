import numpy as np
import os
import pypatchworkpp

def seg_ground_nonground(data_dir,ground_folder,nonground_folder,num_frames,ceiling_folder,indoor,resume=False):
    #check
    check_bool = False
    if not os.path.exists(ground_folder):
        os.makedirs(ground_folder)
        check_bool = True
    if not os.path.exists(nonground_folder):
        os.makedirs(nonground_folder)
        check_bool = True
    if indoor:
        if not os.path.exists(ceiling_folder):
            os.makedirs(ceiling_folder)
            check_bool = True
    if not check_bool and resume:
        for i in range(num_frames):
            if not os.path.exists(f"{ground_folder}/{i:06d}.bin"):
                check_bool = True
                break
            if not os.path.exists(f"{nonground_folder}/{i:06d}.bin"):
                check_bool = True
                break
            if indoor:
                if not os.path.exists(f"{ceiling_folder}/{i:06d}.bin"):
                    check_bool = True
                    break
    if not resume or (resume and check_bool):
        for i in range(num_frames):
            params = pypatchworkpp.Parameters()
            params.verbose = True
            PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
            pointcloud = np.fromfile(f"{data_dir}/velodyne/{i:06d}.bin", dtype=np.float32).reshape((-1, 4))
            PatchworkPLUSPLUS.estimateGround(pointcloud)
            ground_idx = PatchworkPLUSPLUS.getGroundIndices()
            nonground_idx   = PatchworkPLUSPLUS.getNongroundIndices()
            # Get Ground and Nonground
            ground      = pointcloud[ground_idx]
            nonground   = pointcloud[nonground_idx]
            ground.tofile(f"{ground_folder}/{i:06d}.bin")
            if not indoor:
            # Save
                nonground.tofile(f"{nonground_folder}/{i:06d}.bin")
            else:
                temp_pc = nonground.copy()
                temp_pc[:,2] = -temp_pc[:,2]
                PatchworkPLUSPLUS.estimateGround(temp_pc)
                ceiling_idx = PatchworkPLUSPLUS.getGroundIndices()
                nonground_idx   = PatchworkPLUSPLUS.getNongroundIndices()
                ceiling      = nonground[ceiling_idx]
                nonground   = nonground[nonground_idx]
                ceiling.tofile(f"{ceiling_folder}/{i:06d}.bin")
                nonground.tofile(f"{nonground_folder}/{i:06d}.bin")