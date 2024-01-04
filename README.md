<div align="center">
  

</div>


<div align="center">

# FENet: Context Enhancement Network for Lane Detection with Lane Direction Informed-IoU

</div>



Pytorch implementation of the paper "[FENet: Context Enhancement Network for Lane Detection with Lane Direction Informed-IoU](https://arxiv.org/abs/2203.10350)" (Not yet Accepted).

## Introduction
![Arch](.github/main.png)
![Arch](.github/figure2.png)
- FENet propose a new network structure (FENet) based on the significance of global semantic information in lane detection to adequately exploit and compensate for the lack of global information in the FPN network and strengthen the contextual information in high-level and low-level features.
- FENet introduce a strategy anchored in the Gaze-inspired Concentrated Sample paradigm that emphasizes attention to the image’s distal components through a limited set of sampling points
- FENet propose the Lane Direction Informed IoU (LDI-IoU) Loss, explicitly designed for the iterative prediction of lanes.
- FENet achieves mF1 SOTA result on CULane, and LLAMAS datasets.

## Installation

### Prerequisites
Only test on Ubuntu18.04 and 20.04 with:
- Python >= 3.8 (tested with Python3.8)
- PyTorch >= 1.11 (tested with Pytorch1.11)
- CUDA (tested with cuda11.3)
- Other dependencies described in `requirements.txt`

### Clone this repository
Clone this code to your workspace. 
We call this directory as `FENET_ROOT`
```Shell
git clone ...
```

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n fenet python=3.8 -y
conda activate fenet
```

### Install dependencies
```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# Or you can install via pip
pip install torch==1.11.0 torchvision==0.12.0

# Install python packages
python setup.py build develop
```

### Data preparation
#### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd CENET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```

#### LLAMAS
Dowload [LLAMAS](https://unsupervised-llamas.com/llamas/). Then extract them to `$LLAMASROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $LLAMASROOT data/llamas
```

Unzip both files (`color_images.zip` and `labels.zip`) into the same directory (e.g., `data/llamas/`), which will be the dataset's root. For LLAMAS, you should have structure like this:
```
$LLAMASROOT/color_images/train # data folders
$LLAMASROOT/color_images/test # data folders
$LLAMASROOT/color_images/valid # data folders
$LLAMASROOT/labels/train # labels folders
$LLAMASROOT/labels/valid # labels folders
```

## Getting Started
### Training
For training, run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/fenet/CE_dla34_culane.py --gpus 0
```

### Validation
For testing, run
```Shell
python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/fenet/CE_dla34_culane.py --validate --load_from ./checkpoint/fenet_culane_dla34.pth --gpus 0
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/visualization`.


## Results

[assets]: https://github.com/turoad/CLRNet/releases

### CULane

|   Model | Backbone  |  mF1 | F1@50  | F1@75 | GFlops |
| :---     |  :---:   |  :---:   |   :---:    | :---:| :---:|
| [LaneATT]   |[ResNet-122]   | 51.48  |  77.02   | 57.50 | 70.5 |
| [CLRNet]   |[ResNet-18]   | 55.23  |  79.58   | 62.21 | 11.9 |
| [CLRNet]    |[DLA-34]     | 55.64 | 80.47  | 62.78 | 18.5 |
| [FENet]   |[DLA-34]    | 56.17 |  80.19   | 63.50 | 18.8 |


### LLAMAS
|   Model |   Backbone    |  <center>  valid <br><center> &nbsp; mF1 &nbsp; &nbsp;  &nbsp;F1@50 &nbsp; F1@75      |
|  :---:  |    :---:    |        :---:|
| [CLRNet] | [DLA-34]  |  <center> 71.57 &nbsp; &nbsp;  97.06  &nbsp; &nbsp; 85.43 | 
| [FENet] | [DLA-34]   |  <center> 71.85 &nbsp; &nbsp;  96.97  &nbsp; &nbsp; 85.63  |   

“F1@50” refers to the official metric, i.e., F1 score when IoU threshold is 0.5 between the gt and prediction. "F1@75" is the F1 score when IoU threshold is 0.75.

## Citation
```
@InProceedings{
    
}
```

## Acknowledgement
<!--ts-->
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [Turoad/CLRNet](https://github.com/Turoad/CLRNet)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
<!--te-->