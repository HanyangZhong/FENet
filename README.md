# FENet: Focusing Enhanced Network for Lane Detection

Pytorch implementation of the paper "FENet: Focusing Enhanced Network for Lane Detection" 

The paper is available at [arXiv](https://arxiv.org/abs/2312.17163).

## Introduction
![Arch](img/Figure3.png)


This research contributes four innovations: 

1. Focusing Sampling: a training strategy prioritizing small and distant lane details
2. Partial Field of View Evaluation: new metrics for accuracy in forward road sections critical for real-world applications
3. An enhanced FPN architecture that incorporates either positional non-local blocks or standard non-local blocks, depending on the requirement (PEFPN & FEFPN)
4. Directional IoU Loss: a novel regression loss that addresses directional discrepancies in distant lanes


FENetV1, employing positional non-local blocks, achieves state-of-the-art results on conventional metrics by concentrating on perspective-dependent semantics. 

FENetV2, which integrates coordinate modelling into the 'Directional IoU Loss', excels in boundary localization accuracy for distant lanes.

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
git clone https://github.com/HanyangZhong/FENet.git
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
python main.py configs/fenet/FENetV1_dla34_culane.py --gpus 0
```

### Validation
For testing, run
```Shell
python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/fenet/FENetV2_dla34_culane.py --validate --load_from ./checkpoint/fenetv2_culane_dla34.pth --gpus 0
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
| [FENetV1]   |[DLA-34]    | 56.17 |  80.19   | 63.50 | 18.8 |
| [FENetV2]   |[DLA-34]    | 56.17 |  80.19   | 63.50 | 18.8 |

### LLAMAS
|   Model |   Backbone    |  <center>  valid <br><center> &nbsp; mF1 &nbsp; &nbsp;  &nbsp;F1@50 &nbsp; F1@75      |
|  :---:  |    :---:    |        :---:|
| [CLRNet] | [DLA-34]  |  <center> 71.57 &nbsp; &nbsp;  97.06  &nbsp; &nbsp; 85.43 | 
| [FENetV2] | [DLA-34]   |  <center> 71.85 &nbsp; &nbsp;  96.97  &nbsp; &nbsp; 85.63  |   

“F1@50” refers to the official metric, i.e., F1 score when IoU threshold is 0.5 between the gt and prediction. "F1@75" is the F1 score when IoU threshold is 0.75.

## Result comparation
![Arch](img/Figure6.png)

## Citation
```
@misc{wang2024fenet,
      title={FENet: Focusing Enhanced Network for Lane Detection}, 
      author={Liman Wang and Hanyang Zhong},
      year={2024},
      eprint={2312.17163},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
<!--ts-->
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [Turoad/CLRNet](https://github.com/Turoad/CLRNet)
<!--te-->