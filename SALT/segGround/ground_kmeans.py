from utils.utils import *
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
from fcmeans import FCM  # 引入Fuzzy C-Means库

def get_allFrame(data_dir, poses):
    idx = 0
    cloud = np.fromfile(f"{data_dir}/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
    cloud[:,3] = cloud[:, 3]*(np.linalg.norm(cloud[:, :3],axis=1))
    pc = cloud.copy()

    current_pose = poses[idx]

    for current_idx in range(1, len(poses)):

        if not os.path.isfile(f"{data_dir}/{current_idx:06d}.bin"):
            continue
        cloud = np.fromfile(f"{data_dir}/{current_idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
        intensity = (cloud[:, 3]*(np.linalg.norm(cloud[:, :3],axis=1))).reshape(-1,1)

        global_cloud = transform_point_cloud(cloud[:, :3], poses[current_idx])
        relative_cloud = transform_point_cloud(global_cloud, np.linalg.inv(current_pose))
        relative_cloud = np.hstack((relative_cloud,intensity))
        pc = np.vstack((pc, relative_cloud))
    return pc


def project_point_cloud_to_bev_no_limit(point_cloud, resolution):

    # Compute the range of the point cloud
    x_min, x_max = point_cloud[:, 0].min(), point_cloud[:, 0].max()
    y_min, y_max = point_cloud[:, 1].min(), point_cloud[:, 1].max()

    # Compute BEV image dimensions
    bev_width = int(np.ceil((x_max - x_min) / resolution[0]))
    bev_height = int(np.ceil((y_max - y_min) / resolution[1]))

    # Initialize the BEV intensity image
    bev_image = np.zeros((bev_height, bev_width), dtype=np.float32)
    has_image = np.zeros((bev_height, bev_width), dtype=np.int32)

    # Compute pixel indices for each point
    x_indices = ((point_cloud[:, 0] - x_min) / resolution[0]).astype(int)
    y_indices = ((point_cloud[:, 1] - y_min) / resolution[1]).astype(int)

    # Normalize intensities to the range [0, 255] (optional)
    intensities = point_cloud[:, 3]
    intensities = (intensities-intensities.min())/(intensities.max()-intensities.min())
    # intensity_dict = {}

    # Fill the BEV image with intensity values
    for i in range(len(point_cloud)):
        x_idx = x_indices[i]
        y_idx = y_indices[i]
        has_image[y_idx, x_idx] = 1
        bev_image[y_idx, x_idx] = max(bev_image[y_idx, x_idx], intensities[i]) # Use max intensity for overlapping points
    return bev_image, has_image,(x_min, x_max), (y_min, y_max)

def compute_histogram_and_cluster(grid, has_image,k, new_bin_edges,method="K-means", **kwargs):
    def histogram_in_window(neighborhood,new_bin_edges):
        hist, _ = np.histogram(neighborhood, bins=new_bin_edges, range=(0, 1))  # Compute histogram
        hist = hist.astype(np.float64)
        hist /= len(neighborhood)
        return hist  # Return histogram array
    def compute_histograms(grid, has_image,k,new_bin_edges):
        h, w = grid.shape
        num_valid_points = np.sum(has_image)  # Count the number of valid points
        histograms = np.zeros((num_valid_points, 10))  # Preallocate histogram array
        idx_map={}
        idx = 0
    
        # Iterate through each grid point
        for i in range(h):
            for j in range(w):
                # Define neighborhood boundaries
                if has_image[i,j]:
                    i_min = max(0, i - k)
                    i_max = min(h, i + k + 1)
                    j_min = max(0, j - k)
                    j_max = min(w, j + k + 1)
    
                    # Extract the neighborhood
                    neighborhood = grid[i_min:i_max, j_min:j_max]
                    has_image_sub = has_image[i_min:i_max, j_min:j_max]
                    mask = has_image_sub == 1
                    neighborhood = neighborhood[mask]
                    histograms[idx] = histogram_in_window(neighborhood,new_bin_edges)
                    idx_map[idx] = (i,j)
                    idx += 1
    
        return histograms,idx_map

    # Apply generic_filter to compute histogram for each grid point
    histograms,idx_map = compute_histograms(grid, has_image,k,new_bin_edges)

    data_for_clustering = normalize(histograms, norm='l2', axis=1)
    # data_for_clustering = histograms

    # Perform clustering
    if method == "K-means":
        n_clusters = kwargs.get("n_clusters", 5)  # Default number of clusters is 5
        fcm = FCM(n_clusters=n_clusters, random_state=42)
        fcm.fit(data_for_clustering)
        cluster_labels = fcm.predict(data_for_clustering)

    cluster_centers = fcm.centers

    return cluster_labels,idx_map,cluster_centers

def seg_semantic_ground(ground_dir,poses,ground_save_dir,ground_num,voxel_size,resume=False):
    #check
    if resume and os.path.exists(f"{ground_save_dir}/ground_seg_image.npy"):
        return
    np.random.seed(42)
    resolution = voxel_size
    # Project to BEV
    if not os.path.exists(f'{ground_save_dir}/bev_image.npy'):
        P2 = get_allFrame(ground_dir, poses)
        bev_image, has_image,x_range, y_range = project_point_cloud_to_bev_no_limit(P2, resolution)
        range_xy = np.array((x_range, y_range))
        np.save(f"{ground_save_dir}/range_xy.npy", range_xy)
        np.save(f"{ground_save_dir}/bev_image.npy", bev_image)
        np.save(f"{ground_save_dir}/has_image.npy", has_image)
    else:
        bev_image = np.load(f"{ground_save_dir}/bev_image.npy")
        has_image = np.load(f"{ground_save_dir}/has_image.npy")
        range_xy = np.load(f"{ground_save_dir}/range_xy.npy")
        x_range = range_xy[0]
        y_range = range_xy[1]
    num_valid_points = np.sum(has_image)  # Count the number of valid points
    data = np.zeros((num_valid_points, 1))  # Preallocate histogram array
    h, w = bev_image.shape
    idx = 0
    for i in range(h):
        for j in range(w):
            # Define neighborhood boundaries
            if has_image[i,j]:
                data[idx] = bev_image[i,j]
                idx+=1

    original_bins = np.linspace(0, 1, 1001) 
    hist, _ = np.histogram(data, bins=original_bins, density=True)

    cumulative_hist = np.cumsum(hist)
    cumulative_hist /= cumulative_hist[-1]

    new_bins = np.linspace(0, 1, 11)
    new_bin_edges = np.interp(np.linspace(0, 1, 11), cumulative_hist, original_bins[1:])
    new_bin_edges[0] = 0.0
    grid = bev_image
    k = 2  
    clusters_kmeans,idx_map,cluster_centers = compute_histogram_and_cluster(grid, has_image,k, new_bin_edges,method="K-means", n_clusters=ground_num)
    vis_image = np.zeros((bev_image.shape))
    for i in range(len(idx_map.keys())):
        vis_image[idx_map[i]] = clusters_kmeans[i]+1
    
    np.save(f"{ground_save_dir}/ground_seg_image.npy", vis_image)
    np.save(f"{ground_save_dir}/cluster_centers.npy", np.array(cluster_centers))
    