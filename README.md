# MedicalSeg
Medical Image Segmentation Models，集成各种医学图像分割模型，主要是3D，持续更新...
1. 目前例子有BraTS2020数据集的训练代码，如果切换别的数据集，只需要写好自定义Dataset里面的 ```_load_cache_item```函数即可。
2. 目前训练与验证方式采用标准的3D医学图像处理方式，训练为随机Crop N个patch作为输入，进行训练；验证使用滑窗推理进行模型的预测。
3. 数据增强方式具体看example代码，支持多种数据增强。

## 支持3D图像数据增强

1. 随机旋转 RandomRotate
2. 随机翻转 RandomFlip
3. 高斯噪声 AdditiveGaussianNoise
4. 泊松噪声 AdditivePoissonNoise
5. 随机旋转90度 RandomRotate90
6. 随机弹性形变 Elastic
7. Gamma增强 GammaTransformer
8. 镜像变换 MirrorTransformer
9. 对比度增强 ContrastAugmentationTransform
10. 亮度增强 BrightnessMultiplicativeTransform
11. 高斯噪声 GaussianNoiseTransform
12. 高斯模糊 GaussianBlurTransform
13. 重采样增强

## 支持3D图像裁剪

由于3D图像通常较大，因此一个3D图像数据需要裁剪某几块进行训练

1. RandCropByPosNegLabel 根据输入前景背景比例随机剪裁
2. CenterSpatialCrop 中心剪裁

## 支持分patch进行推理

1. SlidingWindowInferer 滑窗推理

## 支持的模型（验证有效果的）

1. UNet
2. deeplabv3
5. MlpMixer 模块
6. swin_transformer 模块
7. transformer 模块
8. TransBTS
10. segresnet
11. VNet
14. UNet++
15. MultiResNet
16. SwinUNet https://arxiv.org/pdf/2105.05537.pdf 预训练参数地址：https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing
17. UNETR https://arxiv.org/pdf/2103.10504.pdf
18. PoolFormer https://arxiv.org/pdf/2111.11418.pdf

部分模型还没经过测试，欢迎测试使用。

## examples
1. BraTS2020 segrensnet 300epoch WT-Dice:0.93 TC-Dice:0.80 ET-Dice:0.67

## 声明

本仓库代码不完全是本人开发，有些模型使用了github中别人贡献的代码，在此仓库作为统一整理，部分代码借鉴了Monai框架 https://github.com/Project-MONAI/MONAI/tree/dev/monai 非常感谢。

