# CV Instance Segmentation

Code for instance segmentation adapting MaskRCNN.
 
 
## Environment
- Ubuntu 16.04 LTS

## Outline
1. [Installation](#Installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Training](#Training)


## Installation

### clone necessary repo

First, clone our [instance_seg repo](https://github.com/osinoyan/instance_seg)
Then, clone the [Mask_RCNN repo](https://github.com/matterport/Mask_RCNN) inside *instance_seg/*


```
$ https://github.com/osinoyan/instance_seg
$ cd instance_seg
$ git clone https://github.com/matterport/Mask_RCNN
```

### environment installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n cv_img_seg python=3.6
source activate cv_img_seg
pip install -r requirements.txt
```
:::danger
Make sure you have installed the correct version of packages:
`tensorflow=1.15.2`
`keras=2.2.5`
:::

## Dataset Preparation

### PASCAL VOC Dataset
The tiny PASCAL VOC dataset (in Coco data format) contains merely over 1,000 training images and 100 test images with 20 object classes, and it should be inside the directory *dataset/* and you do not have to do any preparation.
```
    dataset/
    +- pascal_train.json
    +- test.json
    +- training_images/
    |   +- 2007_000033.png
    |   +- ...
    +- test_images/
    |   +- 2007_000629.png
    |   +- ...
```


## Training
We briefly provide the instructions to train and test the model on PASCAL VOC (in Coco format)
For more information, see [Mask_RCNN repo](https://github.com/matterport/Mask_RCNN).

### train models
To train MaskRCNN model which was pre-trained on ImageNet, run the  following commands.
```
python3 coco_pascal.py train --dataset=dataset/ --model=imagenet
```
### test
After training, try using the detector to test the prepared testing  data. The result will be writen to *submmision.json*
```
python3 coco_pascal.py pascal_test --dataset=dataset/ --model=last --limit=100 
```
