# MedicalSeg
Medical Image Segmentation Models，集成各种医学图像分割模型，主要是3D，持续更新...

## 支持3D图像预处理

1. 随机旋转 RandomRotate
2. 随机翻转 RandomFlip
3. 高斯噪声 AdditiveGaussianNoise
4. 泊松噪声 AdditivePoissonNoise
5. 随机旋转90度 RandomRotate90
## 支持3D图像裁剪，两种方式

由于3D图像通常较大，因此一个3D图像数据需要裁剪某几块进行训练

1. RandCropByPosNegLabel 根据输入前景背景比例随机剪裁
2. CenterSpatialCrop 中心剪裁

## 支持分patch进行推理

1. SlidingWindowInferer 滑窗推理

## 支持的模型

1. UNet
2. deeplabv3
3. hrnet
4. kiunet
5. MlpMixer 模块
6. swin_transformer 模块
7. transformer 模块
8. TransBTS
9. pra_net
10. segresnet
11. VNet
12. U2Net
13. SENet
14. UNet++

部分模型还没经过测试，欢迎测试使用。

## examples
1. spleen 分割 200 epoch dice->0.88 iou->0.80

## 声明

本仓库代码不完全是本人开发，有些模型使用了github中别人贡献的代码，在此仓库作为统一整理，部分代码借鉴了Monai框架 https://github.com/Project-MONAI/MONAI/tree/dev/monai 非常感谢。

