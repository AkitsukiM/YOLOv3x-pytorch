# YOLOv3.x with PyTorch, PASCAL VOC and MS COCO

不会写英文，非常抱歉

不会写这个README.md，非常抱歉（所以 > http://www.mdeditor.com/ ？）

## 文件说明

.../  
├─ Dataset  
········├─ MSCOCO2017  
················├─ train2017  
················├─ val2017  
················└─ annotations  
········└─ VOCdevkit  
················├─ VOC2007  
················└─ VOC2012  
└─ YOLOv3  
········├─ config  
················├─ \_\_init\_\_.py  
················└─ yolov3\_config.py  
········├─ eval  
················├─ \_\_init\_\_.py  
················└─ evaluator.py  
········├─ model  
················├─ backbones  
························├─ \_\_init\_\_.py  
························├─ cspdarknet53.py  
························└─ darknet53.py  
················├─ head  
························├─ \_\_init\_\_.py  
························└─ yolo\_head.py  
················├─ layers  
························├─ \_\_init\_\_.py  
························└─ blocks\_module.py  
················├─ loss  
························├─ \_\_init\_\_.py  
························└─ yolo\_loss.py  
················├─ necks  
························├─ \_\_init\_\_.py  
························└─ yolo\_fpn.py  
················├─ \_\_init\_\_.py  
················└─ yolov3.py  
········├─ utils  
················├─ \_\_init\_\_.py  
················├─ datasets.py  
················├─ gpu.py  
················├─ tools.py  
················└─ visualize.py  
········├─ coco.py  
········├─ test.py  
········├─ train.py  
········├─ voc.py  
········├─ data  
········└─ weight  
················└─ darknet53\_448.weights  

## Dependency

- [x] torch
- [x] torchvision
- [x] numpy
- [x] matplotlib
- [x] opencv-python
- [x] tqdm
- [x] argparse

## 数据集下载

### PASCAL VOC数据集下载

```shell
# .../Dataset/VOCdevkit/VOC2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# .../Dataset/VOCdevkit/VOC2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
# .../Dataset/VOCdevkit/VOC2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

您也可以使用PyTorch提供的函数下载，详见 https://pytorch.org/docs/stable/torchvision/datasets.html?highlight=voc#torchvision.datasets.VOCDetection

### MS COCO数据集下载

```shell
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
wget http://images.cocodataset.org/zips/unlabeled2017.zip
wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
```

您也可以使用百度网盘离线下载（保存到网盘，再从网盘下载，非常好用）

### MS COCO API下载

Linux (Ubuntu 20.04.1 LTS):

```shell
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

Windows 10 + Visual Studio 2019:

```
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

## 使用说明

* 下载该代码
```shell
git clone https://github.com/AkitsukiM/YOLOv3x-pytorch.git
```
* 下载weights文件到weight目录
```shell
wget https://pjreddie.com/media/files/darknet53_448.weights
```
* 在config/yolov3_config.py中修改PATH变量
* 选用VOC数据集需先运行voc.py以创建data/[anno_type]_annotation.txt；选用COCO数据集则无需该步操作
* 请运行train.py以训练（注意指定数据集）
* 请运行test.py以测试（注意指定数据集）

## YOLOv4

- [ ] Backbone
    - [x] CSPDarknet53
        - [ ] enabled
- [ ] Neck
    - [ ] SPP
    - [x] PAN
- [x] Head
    - [x] YOLOv3
- [ ] Bag of Freebies (BoF) for backbone
    - [x] CutMix and Mosaic data augmentation
    - [ ] DropBlock regularization
    - [x] Class label smoothing
- [ ] Bag of Specials (BoS) for backbone
    - [x] Mish activation
        - [ ] enabled
    - [x] Cross-stage partial connections (CSP)
        - [ ] enabled
    - [ ] Multi-input weighted residual connections (MiWRC)
- [ ] Bag of Freebies (BoF) for detector
    - [x] CIoU-loss
    - [ ] CmBN
    - [ ] DropBlock regularization
    - [x] Mosaic data augmentation
    - [ ] Self-Adversarial Training
    - [ ] Eliminate grid sensitivity
    - [x] Using multiple anchors for a single ground truth
    - [x] Cosine annealing scheduler
    - [ ] Optimal hyperparameters
    - [ ] Random training shapes
- [ ] Bag of Specials (BoS) for detector
    - [x] Mish activation
        - [ ] enabled
    - [ ] SPP-block
    - [ ] SAM-block
    - [x] PAN path-aggregation block
    - [x] DIoU-NMS
        - [ ] enabled

## 测试结果

### PASCAL VOC数据集

* YOLOv3 (last updated)
* 使用单个NVIDIA RTX2080Ti
* 数据集位于固态硬盘

> 使用固态硬盘与机械硬盘存在很大区别  

* 训练时长约12小时，测试时长约2分钟
* 截至目前共训练-测试1次
```
# TEST_IMG_SIZE = 416
 mAP ∈ {0.8418}
```

> 对于残差层，用conv-bn-relu-conv-bn-(+resi)-relu代替conv-bn-relu-conv-bn-relu-(+resi)会导致mAP下降约0.10  
> 对于损失函数，用BCE代替Focal Loss会导致mAP下降约0.03  
> 尝试实现Cosine annealing scheduler，mAP提升了0.006  
> 尝试实现CutMix and Mosaic data augmentation，mAP提升了0.008  
> 尝试实现CIoU-loss，mAP提升了0.002  
> 尝试实现DIoU-NMS，mAP下降了0.001  
> 尝试实现CSPDarknet53+Mish activation，mAP下降了0.20，失败！  
> 尝试实现PANet，mAP提升了0.006  

### MS COCO数据集

* YOLOv3 (last updated)
* 使用单个NVIDIA RTX2080Ti
* 数据集位于固态硬盘
* 训练时长约99小时，测试时长约2分钟
* 截至目前共训练-测试1次
```
# TEST_IMG_SIZE = 416
 FPS: ~40 (v3: 35; v4: 38)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.304 (v3: 0.310; v4: 0.412)
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.536 (v3: 0.553; v4: 0.628)
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308 (v3: 0.323; v4: 0.443)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.123 (v3: 0.152; v4: 0.204)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.326 (v3: 0.332; v4: 0.444)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.463 (v3: 0.428; v4: 0.560)
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.395
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.422
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
```

## 参考代码

[1] https://github.com/Peterisfar/YOLOV3

[2] https://github.com/argusswift/YOLOv4-pytorch

-----

Copyright (c) 2020 Marina Akitsuki. All rights reserved.

Date modified: 2020/12/11

