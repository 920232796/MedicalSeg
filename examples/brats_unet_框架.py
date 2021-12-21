## 数据 brats 2020 数据 引入 自定义Dataset类
import glob
import os
from random import random, randrange 
import SimpleITK as sitk
import numpy as np
import torch 
from torch import optim
import setproctitle
import torch.nn as nn
import SimpleITK as sitk
from torch.optim import lr_scheduler
from monai.utils import set_determinism
from medical_seg.networks import BasicUNet
from medical_seg.loss.dice_loss import DiceLoss
from medical_seg.dataset.dataset import DataLoader3D, Dataset3D
from medical_seg.transformer import RandomRotate, RandCropByPosNegLabel, RandomFlip, \
                                    AdditiveGaussianNoise, AdditivePoissonNoise, Standardize, \
                                    CenterSpatialCrop, Elatic, GammaTransformer, MirrorTransform,\
                                    ResampleImage, Normalization
from medical_seg.transformer.noisy_transformers import GaussianNoiseTransform, GaussianBlurTransform
from medical_seg.transformer.color_transformers import ContrastAugmentationTransform, BrightnessMultiplicativeTransform
from medical_seg.transformer.resample_transformers import SimulateLowResolutionTransform
from medical_seg.networks import BasicUNet
from medical_seg.utils import set_seed
from medical_seg.inferer import SlidingWindowInferer
from medical_seg.evaluation import average_metric
from medical_seg.evaluation import Metric
from medical_seg.utils import get_kfold_data
from medical_seg.transformer.transforms import resample_patient
from medical_seg.trainer.trainer import Trainer

data_paths = sorted(glob.glob("./data/MICCAI_BraTS2020_TrainingData/*"))[:-2]
train_paths = data_paths[:315]
val_paths = data_paths[315:]
# train_paths = data_paths[:10]
# val_paths = data_paths[360:]

print(f"数据共：{len(data_paths)}例")

## 一些必要参数
seed = 3213214325
in_channels = 4
out_channels = 4
random_state = np.random.RandomState(seed)
set_seed(seed)
# set_determinism(seed=seed)
lr = 1e-4
spatial_size = (128, 128, 128)
print(f"spatial size is {spatial_size}")
sample_size = 2
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 300
val_internal = 50
start_val = 100
print(f"val internal is {val_internal}")
model_name = "brats_unet_noaug"

model_save_dir = "./state_dict/" + model_name + "/"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

class Transform:
    def __init__(self, random_state) -> None:
        self.random_state = random_state

        self.rf = RandomFlip(self.random_state, spatial_axis=[0], execution_probability=0.4)
        self.rf_2 = RandomFlip(self.random_state, spatial_axis=[1], execution_probability=0.4)
        self.rf_3 = RandomFlip(self.random_state, spatial_axis=[2], execution_probability=0.4)
        self.rr = RandomRotate(self.random_state, angle_spectrum=60, execution_probability=0.4)
        self.elastic = Elatic(self.random_state, alpha=(0, 900), sigma=(9, 13), 
                                scale=(0.85, 1.25), order_seg=0, order_data=3,
                                execution_probability=0.2)

        self.gamma = GammaTransformer(self.random_state, gamma_range=(0.5, 2), execution_probability=0.2)
        self.noisy_trans = GaussianNoiseTransform(self.random_state, noise_variance=(0, 0.1), execution_probability=0.15)
        self.blur_trans = GaussianBlurTransform(self.random_state, execution_probability=0.2, p_per_channel=0.5)
        self.bright_trans = BrightnessMultiplicativeTransform(self.random_state, multiplier_range=(0.75, 1.25), execution_probability=0.15)
        self.contrast_trans = ContrastAugmentationTransform(self.random_state, contrast_range=(0.75, 1.25), execution_probability=0.15)
        self.resample_trans = SimulateLowResolutionTransform(self.random_state, zoom_range=(0.5, 1.0), per_channel=True, p_per_channel=0.5, 
                                        order_downsample=1, order_upsample=3, execution_probability=0.25)
    def __call__(self, image, label):
        image, label = self.rf(image, label)
        image, label = self.rf_2(image, label)
        image, label = self.rf_3(image, label)
        # image, label = self.rr(image, label)

        # image = self.gamma(image)
        # image = self.noisy_trans(image)
        # image = self.blur_trans(image)
        # image = self.bright_trans(image)
        # image = self.resample_trans(image)
    
        return image, label   

class BraTSDataset(Dataset3D):
    """
    """
    def __init__(
            self,
            paths,
            trans_func=None,
            crop_func=None,
            train=False,
            
    ) -> None:

        super().__init__(paths, trans_func=trans_func, crop_func=crop_func, train=train)

    def _load_cache_item(self, d_path):
        
        images = [] 
        label = None
        paths = sorted(glob.glob(d_path + "/*.nii"))
        # print(paths)
        spacing = None 
        for p in paths:
            if "_seg.nii" in p:
                # 找到seg文件
                label = sitk.ReadImage(p)
                label = sitk.GetArrayFromImage(label)
                label[label == 4] = 3
            else :
                image = sitk.ReadImage(p)
                if spacing is None :
                    spacing = image.GetSpacing()[:-1]
                    spacing = (spacing[-1], ) + spacing[:2]
                image = sitk.GetArrayFromImage(image)
                images.append(image)
        
        images = np.array(images)

        images, labels = resample_patient(images, label, spacing, (1.0, 1.0, 1.0),
                                                3, 1,force_separate_z=None, order_z_data=3, 
                                                order_z_seg=1,
                                                separate_z_anisotropy_threshold=3)
        # print(np.unique(images))
        return images, labels

class Loss:
    def __init__(self) -> None:
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
        self.cropy_loss = nn.CrossEntropyLoss()

    def __call__(self, pred, label):
        return self.dice_loss(pred, label) + self.cropy_loss(pred, label) 

## 定义一些必要方法
metric = Metric(class_name=["WT", "TC", "ET"], voxel_spacing=(1, 1, 1, 1), nan_for_nonexisting=True)
sliding_window_infer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=1, overlap=0.25)
# trans_func = Transform(random_state=random_state)
trans_func = None
crop_func = RandCropByPosNegLabel(spatial_size=spatial_size, pos=1, neg=1, num_samples=sample_size)


def main_train(net_1, train_data_paths, test_data_paths, k_fold):

    train_ds = BraTSDataset(train_data_paths, trans_func=trans_func, crop_func=crop_func, train=True)
    train_loader = DataLoader3D(train_ds, batch_size=1, shuffle=True)
    optimizer_1 = optim.AdamW(net_1.parameters(), lr=lr, weight_decay=1e-5)
    net_1.to(device)

    loss_func = Loss()
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=epochs)
    lr_scheduler = None 
    val_ds = BraTSDataset(test_data_paths, train=False)
    val_loader = DataLoader3D(val_ds, batch_size=1, shuffle=False)

    trainer = Trainer(network=net_1, train_loader=train_loader, val_loader=val_loader, 
                        optimizer=optimizer_1, loss_func=loss_func, metric=metric, 
                        sliding_window_infer=sliding_window_infer, val_internal=val_internal, 
                        epochs=300, k_fold=k_fold, is_show_image=False, device=device, model_save_dir=model_save_dir,
                        lr_scheduler=lr_scheduler, start_val=start_val, task_name=model_name)

    end_metric, best_metric = trainer.train()

    return end_metric, best_metric 

if __name__ == "__main__":
    res_metric = []
    best_metric = []
    fold = 0
    if os.path.exists(f"{model_save_dir}epoch_res.txt") :
        os.remove(f"{model_save_dir}epoch_res.txt")
        print("删除log成功。")
    if os.path.exists(f"{model_save_dir}res.txt"):
        os.remove(f"{model_save_dir}res.txt")
        print("删除res log成功。")

    model = BasicUNet(dimensions=3, in_channels=in_channels, out_channels=out_channels, features=[16, 16, 32, 64, 128, 16],
                            pool_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)])
    
    metric_fold, best_metric_fold = main_train(model, train_data_paths=train_paths, 
                                test_data_paths=val_paths, k_fold=fold)
    res_metric.append(metric_fold)
    best_metric.append(best_metric_fold)

    with open(model_save_dir + "res.txt", "a+") as f:
        f.write(f"res fold {fold} is {metric_fold}")
        f.write("\n")
        f.write(f"best res fold {fold} is {best_metric_fold}")
        f.write("\n")

    res_metric = average_metric(res_metric)
    best_metric = average_metric(best_metric)

    print(f"res metric is {res_metric} \n best metric is {best_metric}")

    with open(model_save_dir + "res.txt", "a+") as f:
            f.write(f"res is {res_metric}")
            f.write("\n")
            f.write(f"best res is {best_metric}")
            f.write("\n")