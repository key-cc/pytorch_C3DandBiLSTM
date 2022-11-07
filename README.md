# pytorch-video-recognition

## Introduction
This repo contains several models for video action recognition,
including ConvLSTM, C3D inplemented using PyTorch .
These models are trained on UCF101 and HMDB51 datasets.

## Installation

0. Clone the repo:
    ```Shell
    git clone https://github.com/key-cc/pytorch_C3DandBiLSTM.git
    cd pytorch-video-recognition
    ```

1. Install dependencies:
    ```Shell
    conda install opencv
    pip install tqdm
    pip install scikit-learn 
    pip install tensorboardX
    ```

2. Download pretrained model for C3D from [BaiduYun](https://pan.baidu.com/s/1saNqGBkzZHwZpG-A5RDLVw) or 
[GoogleDrive](https://drive.google.com/file/d/19NWziHWh1LgCcHU34geoKwYezAogv9fX/view?usp=sharing).

3. Configure your dataset and pretrained model path in
[mypath.py](https://github.com/key-cc/pytorch_C3DandBiLSTM/blob/main/mypath.py).

4. You can choose different models and datasets in
[train.py](https://github.com/jfzhang95/pytorch-video-recognition/blob/main/train.py).
you also need to change the line 10 in the file [dataset.py](https://github.com/key-cc/pytorch_C3DandBiLSTM/blob/main/dataloaders/dataset.py) to choose the type of the model.

    To train the model, please do:
    ```Shell
    python train.py
    ```

## Datasets:

I used two different datasets: UCF101 and HMDB.

Dataset directory tree is shown below

- **UCF101**
Make sure to put the files as the following structure:
  ```
  UCF-101
  ├── ApplyEyeMakeup
  │   ├── v_ApplyEyeMakeup_g01_c01.avi
  │   └── ...
  ├── ApplyLipstick
  │   ├── v_ApplyLipstick_g01_c01.avi
  │   └── ...
  └── Archery
  │   ├── v_Archery_g01_c01.avi
  │   └── ...
  ```
After pre-processing, the output dir's structure is as follows:
  ```
  ucf101
  ├── ApplyEyeMakeup
  │   ├── v_ApplyEyeMakeup_g01_c01
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  ├── ApplyLipstick
  │   ├── v_ApplyLipstick_g01_c01
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  └── Archery
  │   ├── v_Archery_g01_c01
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  ```

Note: HMDB dataset's directory tree is similar to UCF101 dataset's.
