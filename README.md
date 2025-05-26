<div align="center">
<img src="fig/salt.png" width ="200" alt="celebration"/>


# SALT: A Flexible Semi-Automatic Labeling Tool for General LiDAR Point Clouds with Cross-Scene Adaptability and 4D Consistency

This repo is the official project repository of **[\[SALT\]](https://arxiv.org/abs/2503.23980)**.

<div align="left">

## 1. Overview
We introduce SALT, a flexible semi-automatic labeling tool for general LiDAR point clouds, featuring adaptive cross-scene and 4D consistency. 
SALT demonstrates exceptional zero-shot adaptability across various sensors, scenes, and motion conditions, greatly enhancing annotation efficiency.

![image](fig/overview.jpg)

## 2. Environment

### 2.1 Docker-based
This setup allows you to run `SALT` in an isolated and reproducible Docker container, without having to install dependencies manually.


#### Step 1: Run the Container

To run the container with access to your project files and enable GPU:

```bash
docker run -it \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/SALT:/workspace/SALT \
  -w /workspace/SALT \
  cytt/salt:latest /bin/bash
```

#### Step 2: Run SALT Inside Docker

Once inside the container:

```bash
cd ~/SALT/bin
./SALT
```

The GUI labeling tool should now launch inside Docker.

---

### 2.2 Non-Docker
The dependencies can be installed from the package manager (tested On Ubuntu 18.04):
```
sudo apt-get install g++ build-essential libeigen3-dev python3-pip \
 python3-dev cmake git libboost-all-dev qtbase5-dev libglew-dev libyaml-cpp-dev -y 
```
Then, ensure [Anaconda](https://www.anaconda.com/download/) is installed in your system. Create and activate a virtual environment named `SALT`.

```
conda create -n SALT python==3.10
conda activate SALT
```
`SALT` depends on [Patchwork++](https://github.com/url-kaist/patchwork-plusplus), a real-time ground segmentation library. Install it by running:
```
cd 3rdparty
git clone https://github.com/url-kaist/patchwork-plusplus.git && cd patchwork-plusplus
make pyinstall
cd ..
```
[SAM 2](https://github.com/facebookresearch/sam2) is required for the segmentation module. The code requires  `torch>=2.5.1` and `torchvision>=0.20.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM 2 on a GPU machine using:
```
cd sam2
pip install -e .
cd checkpoints && ./download_ckpts.sh
cd ../../..
```
or individually download `checkpoint` from [SAM 2](https://github.com/facebookresearch/sam2).

Install `SALT` dependencies and download `Data Alignment Model`  checkpoints.

```
pip install -r requirements.txt
cd ..
```
Meanwhile, please manually download the [Data_Alignment_Model](https://drive.google.com/file/d/1pnnW2JQsc8syDMyQiXGdBtSIjWlzfVjD/view?usp=sharing) and place it in the `~/SALT/SALT/` directory.

The project folder structure should look like this:
<pre>
SALT folder
├── 3rdparty
├── SALT 
   ├── best_model.pth    -- Data_Alignment_Model
   ├── ...
├── src   
└── ...
</pre>

Install system-level dependencies and build the C++ backend of the labeler tool:

```
mkdir build && cd build
cmake ..
make -j5
```
Now the project root directory (e.g. `~/SALT`) should contain a `bin` directory containing the labeler.In the bin directory, just run `./SALT` to start the labeling tool.

## 3. User Manual 
### 3.1 Automatic Segmentation of Whole sequence
After loading the raw point cloud sequence data, the user
 only needs to click the "SALT" button once to obtain the pre-segmentation results 
 for the entire sequence. After clicking the "SALT" button, users can modify the config file according to their own data characteristics. 
 Once the progress bar (zero-shot segmentation step) is complete, the pre-segmentation
  results are automatically saved for subsequent semantic and instance labeling. The pre-segmentation results are also 
  automatically displayed in the user interface with different colors.

<img src="fig/example1.GIF" width="600" />
  
### 3.2 Manual Assignment and Refinement for Semantic Annotation
The user is free to define as many semantic classes appear in the
sequence. Users can assign custom semantic labels to the pre-segmentation results based 
on their needs. By simply clicking on a predefined color button
 representing a specific semantic category and then selecting a point cloud with a particular ID, all 
 points with that ID will be assigned the chosen label and updated to the corresponding color. This 
 operation is as intuitive and effortless as a coloring game. Please note that the colors used to display 
 the pre-segmentation results are designed to avoid overlapping with user-defined semantic label colors. 
 If users are not satisfied with the pre-annotated results, they can modify them using the polygon tool. 
 Inherited from LABELER, SALT supports the option to hide other classes, making manual annotation adjustments 
 more convenient.

<img src="fig/example2.GIF" width="600" />
 
### 3.3 Auto Ordering and Manual Refinement for Instance Annotation
Once users are satisfied with the semantic labeling results, they can simply click the "Auto Instance" button
 to automatically assign instance IDs to all semantic categories. Users
  can then further refine the results by splitting or merging instance IDs within each category.

## 4. Quick Demo

We provide a simple demo dataset to quickly test the functionality of `SALT`.

You can download the demo data from [here](https://drive.google.com/file/d/1xNRPgNdNcdeueLMZurA4o1yjW_bQci8C/view?usp=drive_link).This is a sample subset from the [SemanticKITTI](https://www.semantic-kitti.org/).


After downloading and extracting the data, you will find a `config.yaml` file inside the demo directory. You can use this configuration to quickly get started by simply updating a few path fields in the config file.

### Step 1: Configure `config.yaml`

Open the file at:

```
<SALT_path>/SALT/config.yaml
```

Update the following fields to point to your local environment:

- `conda_sh_path`: Path to your `conda.sh`, e.g., `/home/user/anaconda3/etc/profile.d/conda.sh`
- `data_dir`: Path to the demo data folder, e.g., `/home/user/demo_data`
- `cache_dir`: Path to the cache output folder, e.g., `/home/user/demo_data/cache`
- `sam2_checkpoint`: Path to the SAM2 checkpoint, e.g., `/home/user/SALT/3rdparty/sam2/checkpoints/sam2.1_hiera_large.pt`
- `model_cfg`: Path to the SAM2 model configuration file, e.g., `//home/user/SALT/3rdparty/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml`
- `Computing device configuration`: depends on the machine you are using.

You can refer to the provided `config.yaml` as a template.

### Step 2: Launch the GUI Tool

Once your paths are configured, simply launch the GUI and click the **"SALT"** button in the top toolbar to start labeling.

```bash
cd bin
./SALT
```

This will load the demo dataset using your customized `config.yaml` and bring up the annotation interface.

Enjoy labeling!

## 5. Acknowledgements
We would like to thank all the pioneers [SemanticKITTI_LABLER](https://github.com/jbehley/point_labeler), [SAM2](https://github.com/facebookresearch/sam2). 

## 6. Citation
If your like our projects, please cite us and give this repo a star.
```
@article{wang2025salt,
  title={SALT: A Flexible Semi-Automatic Labeling Tool for General LiDAR Point Clouds with Cross-Scene Adaptability and 4D Consistency},
  author={Wang, Yanbo and Chen, Yongtao and Cao, Chuan and Deng, Tianchen and Zhao, Wentao and Wang, Jingchuan and Chen, Weidong},
  journal={arXiv preprint arXiv:2503.23980},
  year={2025}
}
```
