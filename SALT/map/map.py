from utils.utils import *
from scipy.spatial import KDTree
def group_points_by_label(points: np.ndarray, labels: np.ndarray, min_label):
    label_dict = {}
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label < min_label:
            continue 
        
        mask = labels == label
        pointcloud = points[mask]
        
        x_min, y_min, z_min = pointcloud.min(axis=0)
        x_max, y_max, z_max = pointcloud.max(axis=0)
        bbox = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
        
        bbox_center = (bbox[:3] + bbox[3:]) / 2.0
        
        label_dict[label] = {
            "pointcloud": pointcloud,
            "mask": mask,
            "bbox": bbox,
            "bbox_center": bbox_center
        }
    
    return label_dict

def compute_iou_3d(bbox1, bbox2):

    x_min_inter = max(bbox1[0], bbox2[0])
    y_min_inter = max(bbox1[1], bbox2[1])
    z_min_inter = max(bbox1[2], bbox2[2])
    x_max_inter = min(bbox1[3], bbox2[3])
    y_max_inter = min(bbox1[4], bbox2[4])
    z_max_inter = min(bbox1[5], bbox2[5])

    inter_w = max(0, x_max_inter - x_min_inter)
    inter_h = max(0, y_max_inter - y_min_inter)
    inter_d = max(0, z_max_inter - z_min_inter)
    inter_volume = inter_w * inter_h * inter_d

    volume1 = (bbox1[3] - bbox1[0]) * (bbox1[4] - bbox1[1]) * (bbox1[5] - bbox1[2])
    volume2 = (bbox2[3] - bbox2[0]) * (bbox2[4] - bbox2[1]) * (bbox2[5] - bbox2[2])

    union_volume = volume1 + volume2 - inter_volume
    iou = inter_volume / union_volume if union_volume > 0 else 0.0

    return iou

def labelmap(poses,velodyne_dir,label_dir,output_label_dir,resume=False):
    BBOX_dict = {}
    min_label=1000
    for current_idx in range(0,len(poses)):
        cloud = np.fromfile(f"{velodyne_dir}/{current_idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
        cloud = transform_point_cloud(cloud[:, :3], poses[current_idx])
        cloud = transform_point_cloud(cloud, np.linalg.inv(poses[0]))
        current_label = np.fromfile(f"{label_dir}/{current_idx:06d}.label", dtype=np.int32)
        output_label = current_label.copy()
        current_label_dict = group_points_by_label(cloud,current_label,min_label)
        if current_idx == 0:
            output_label = output_label.astype(np.int32)
            output_label.tofile(f"{output_label_dir}/{current_idx:06d}.label")
            continue
        pre_cloud=np.fromfile(f"{velodyne_dir}/{(current_idx-1):06d}.bin", dtype=np.float32).reshape(-1, 4)
        pre_cloud = transform_point_cloud(pre_cloud[:, :3], poses[current_idx-1])
        pre_cloud = transform_point_cloud(pre_cloud, np.linalg.inv(poses[0]))
        pre_label= np.fromfile(f"{output_label_dir}/{(current_idx-1):06d}.label", dtype=np.int32)
        pre_label_dict = group_points_by_label(pre_cloud,pre_label,min_label)

        centers = []
        labels = []
        for label, data in pre_label_dict.items():
            centers.append(data["bbox_center"])
            labels.append(label)
        
        centers = np.array(centers)
        kdtree = KDTree(centers)
        for label,data in current_label_dict.items(): 
            distances, idx = kdtree.query(data["bbox_center"])
            # update_bool = False

            if distances < 2:
                pre_bbox = pre_label_dict[labels[idx]]["bbox"]
                curr_box = current_label_dict[label]["bbox"]
                size1 = pre_bbox[3:] - pre_bbox[:3]
                size2 = curr_box[3:] - curr_box[:3]
                ratio = np.minimum(size1 / size2, size2 / size1)
                if ratio[0]>0.8 and ratio[1] > 0.8 and ratio[2] > 0.8:
                    output_label[current_label_dict[label]["mask"]]=labels[idx]
        output_label = output_label.astype(np.int32)
        output_label.tofile(f"{output_label_dir}/{current_idx:06d}.label")
    
def SALT_label(poses,input_dir, output_dir,camera_model,input_cam_label_map_dir):
    label_dict = {0:0}
    min_label = 1025
    os.makedirs(f"{output_dir}/labels",exist_ok=True)
    os.makedirs(f"{output_dir}/SALT_labels",exist_ok=True)
    for i in range(len(poses)):
        curr_label = np.fromfile(f"{input_dir}/{i:06d}.label", dtype=np.int32)
        for j in range(len(curr_label)):
            if curr_label[j] not in label_dict:
                label_dict[curr_label[j]]=min_label
                min_label+=1
            curr_label[j]=label_dict[curr_label[j]]
        curr_label = curr_label.astype(np.int32)
        curr_label.tofile(f"{output_dir}/labels/{i:06d}.label")
        curr_label.tofile(f"{output_dir}/SALT_labels/{i:06d}.label")
    if camera_model:
        os.makedirs(f"{output_dir}/SALT_cam_labels",exist_ok=True)
        label_dict = {0:0}
        min_label = 1025
        for i in range(len(poses)):
            curr_label = np.fromfile(f"{input_cam_label_map_dir}/{i:06d}.label", dtype=np.int32)
            for j in range(len(curr_label)):
                if curr_label[j] not in label_dict:
                    label_dict[curr_label[j]]=min_label
                    min_label+=1
                curr_label[j]=label_dict[curr_label[j]]
            curr_label = curr_label.astype(np.int32)
            curr_label.tofile(f"{output_dir}/SALT_cam_labels/{i:06d}.label")

