# [Efficient Convolutional Neural Networks on Raspberry Pi for Image Classification](https://arxiv.org/abs/2204.00943)
<p align="center">
  <img src="Img/architecture.jpg" width="640" title="architecture">
</p>

## Citation
If you find TripleNet useful in your research, please consider citing:

	@article{TripleNet,
	 title={Efficient Convolutional Neural Networks on Raspberry Pi for Image Classification},
	 author={Rui-Yang Ju, Ting-Yu Lin, Jia-Hao Jian, Jen-Shiun Chiang},
	 year={2022}
	 }
	 
## Contents
1. [Introduction](#introduction)
2. [Usage](#Usage)
3. [Model](#Model)
4. [Results](#Results)
5. [Requirements](#Requirements)
6. [Config](#Config)
7. [References](#References)

## Introduction
TripleNet is adopted from the concept of block connections in ThreshNet, it compresses and accelerates the network model, reduces the amount of parameters of the network, and shortens the inference time of each image while ensuring the accuracy. TripleNet and other state-of-the-art (SOTA) neural networks perform image classification experiments with the CIFAR-10 and SVHN datasets on Raspberry Pi. The experimental results show that, compared with GhostNet, MobileNet, ThreshNet, EfficientNet, and HarDNet, the inference time of TripleNet per image is shortened by 15%, 16%, 17%, 24%, and 30%, respectively.

 <img src="Img/conv_layers.jpg" width="640" title="conv_layers">

## Usage
```bash
python3 main.py
```
optional arguments:

    --lr                default=1e-3    learning rate
    --epoch             default=200     number of epochs tp train for
    --trainBatchSize    default=100     training batch size
    --testBatchSize     default=100     test batch size

## Model
| **Model** | **Layer** | **Channel** | **Growth Rate** |
| :---: | :---: | :---: | :---: |
| TripleNet-S | 6, 16, 16, 16, 2 | 128, 192, 256, 320, 720 | 32, 16, 20, 40, 160 |
| TripleNet-B | 6, 16, 16, 16, 3 | 128, 192, 256, 320, 1080 | 32, 16, 20, 40, 160 |

## Results
| Name | Raspberry Pi 4 Time (ms) | C10 Error (%) | FLOPs (G) | MAdd (G) | Memory (MB) | #Params (M) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TripleNet-S** | 38.7 | 6.18 | 4.17 | 8.32 | 90.25 | 9.67 |
| SqueezeNet | 0.36 | 14.25 | 2.69 | 5.32 | 211.42 | 0.78 |
| MobileNet | 0.38 | 16.12 | 2.34 | 4.63 | 230.84 | 3.32 |
| **ThreshNet79** | 0.42 | 13.66  | 3.46 | 6.90 | 109.68  | 14.31 |
| HarDNet68 | 0.44 | 14.66 | 4.26 | 8.51 | 49.28 | 17.57 |
| MobileNetV2 | 0.46 | 14.06 | 2.42 | 4.75 | 384.78 | 2.37 |
| **ThreshNet95** | 0.46 | 13.31 | 4.07 | 8.12 | 132.34 | 16.19 | 
| HarDNet85 | 0.50 | 13.89 | 9.10 | 18.18 | 74.65 | 36.67 |

\* Raspberry Pi Time is the inference time per image on Raspberry Pi 4

## Requirements
#### Raspberry Pi 4 Model B 4GB
* python3 - 3.9.2
* torch - 1.11.0
* torchvision - 0.12
* numpy - 1.22.3

## Config
###### Optimizer 
__Adam Optimizer__
###### Learning Rate
__1e-3__ for [1,74] epochs <br>
__5e-4__ for [75,149] epochs <br>
__2.5e-4__ for [150,200) epochs <br>


## References
* [torchstat](https://github.com/Swall0w/torchstat)
* [pytorch-cifar10](https://github.com/soapisnotfat/pytorch-cifar10)
* [HarDNet](https://github.com/PingoLH/Pytorch-HarDNet)
