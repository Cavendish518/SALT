from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from tqdm import tqdm
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
def merge_masks(anns):
    if len(anns) == 0:
        return None
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    h, w = sorted_anns[0]['segmentation'].shape
    merged_mask = np.zeros((h, w), dtype=np.int32)

    for idx, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        merged_mask[m] = idx+1

    return merged_mask
def cam2sam(m_dir,camera_num,sam2_checkpoint,model_cfg):
    cam_name = []
    for i in range(camera_num):
        cam_name.append(f'camera{i}')
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(model=sam2,
    points_per_side=32,
    pred_iou_thresh=0.84,
    stability_score_thresh=0.86,
    min_mask_region_area=100,)
    for camera_idx in range(camera_num):
        mask_out_put = f"{m_dir}/cache/cam/mask_out/{camera_idx}"
        if not os.path.exists(mask_out_put):
            os.makedirs(mask_out_put)
        img_video_dir=f"{m_dir}/{cam_name[camera_idx]}"
        frame_names = [
            p for p in os.listdir(img_video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        for frame_name in tqdm(frame_names):
            img_path = os.path.join(img_video_dir, frame_name)
            image = Image.open(img_path)
            image = np.array(image.convert("RGB"))
            masks = mask_generator.generate(image)
            save_mask = merge_masks(masks)
            np.save(f"{mask_out_put}/{frame_name[:-4]}.npy",save_mask)

    