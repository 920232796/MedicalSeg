import torch
import numpy as np
## 数据 brats 2020 数据
import glob
from medical_seg.utils.enums import BlendMode
import os
import SimpleITK as sitk
from sklearn.model_selection import KFold  ## K折交叉验证
from torch import optim
import setproctitle
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from medical_seg.transformer import RandomRotate, RandCropByPosNegLabel, RandomFlip, \
    AdditiveGaussianNoise, AdditivePoissonNoise, Standardize, \
    CenterSpatialCrop, Elatic, GammaTransformer, MirrorTransform
from medical_seg.networks import BasicUNet
from medical_seg.utils import set_seed
from medical_seg.dataset import collate_fn
from tqdm import tqdm
from medical_seg.inferer import SlidingWindowInferer
from medical_seg.utils import segmenation_metric, resample_image_array_size
from medical_seg.evaluation import evaluate_BraTS_case, average_metric

data_paths = sorted(glob.glob("./data/MICCAI_BraTS2020_TrainingData/*"))[:-2]
print(data_paths)
train_paths = data_paths[:315]
val_paths = data_paths[315:]

spatial_size = (128, 128, 128)
sliding_window_infer = SlidingWindowInferer(roi_size=spatial_size, sw_batch_size=2, overlap=0.5)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_name = "brats_unet"

seed = 3213214325
in_channels = 4
out_channels = 4
sample_size = 1
random_state = np.random.RandomState(seed)
set_seed(seed)

model_save_dir = "./state_dict/" + model_name + "/"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

class Transform:
    def __init__(self, random_state) -> None:
        self.random_state = random_state

        self.rf = RandomFlip(self.random_state, execution_probability=0.2)
        self.rr = RandomRotate(self.random_state, angle_spectrum=30, execution_probability=0.2)
        self.elastic = Elatic(self.random_state, alpha=(0, 900), sigma=(9, 13),
                              scale=(0.85, 1.25), order_seg=0, order_data=3,
                              execution_probability=0.2)
        self.gamma = GammaTransformer(self.random_state, gamma_range=(0.5, 2), execution_probability=0.2)
        self.mirror = MirrorTransform(self.random_state, axes=(0, 1, 2), execution_probability=0.2)

    def __call__(self, image, label):
        image, label = self.rf(image, label)
        image, label = self.rr(image, label)
        # start = time.time()
        # image, label = self.elastic(m=image, seg=label)
        # end = time.time()
        # print(f"elastic spend {end - start}")
        image = self.gamma(image)
        image, label = self.mirror(image, seg=label)

        return image, label

class BraTSDataset(Dataset):
    def __init__(self, paths, train=True) -> None:
        super().__init__()
        self.paths = paths
        self.train = train
        if train:
            self.transform = Transform(random_state=random_state)
            self.random_crop = RandCropByPosNegLabel(spatial_size=spatial_size, pos=1, neg=1,
                                                     num_samples=sample_size, image=None, image_threshold=0,
                                                     random_state=random_state)
        else :
            self.transform = None
            self.random_crop = None
        self.cached_image = []
        self.cached_label = []
        for p in tqdm(self.paths, total=len(self.paths), desc="loading training data........"):
            image, label = self._read_image(p)
            sd = Standardize(a_min=image.min(), a_max=image.max(), b_min=0, b_max=1, clip=True)
            image = sd(image)
            self.cached_image.append(image)
            self.cached_label.append(label)

    def __getitem__(self, i):
        image, label = self.cached_image[i], self.cached_label[i]

        if self.train:
            image_patchs = self.random_crop(image, label=label)
            label_patchs = self.random_crop(label, label=label, is_label=True)
            for i, imla in enumerate(zip(image_patchs, label_patchs)):
                image_patchs[i], label_patchs[i] = self.transform(imla[0], imla[1])
        else :

            assert len(image.shape) == 4, "image shape is must be 4."
            assert len(label.shape) == 3, "label shape is must be 3."
            image = [image]
            label = [label]

        if self.train:
            return {
                "image": image_patchs,
                "label": label_patchs
            }

        return {
            "image": image,
            "label": label,
        }

    def __len__(self):
        return len(self.paths)

    def _read_image(self, image_path):
        images = []
        label = None
        paths = sorted(glob.glob(image_path + "/*.nii"))
        for p in paths:
            if "_seg.nii" in p:
                # 找到seg文件
                label = sitk.ReadImage(p)
                label = sitk.GetArrayFromImage(label)
                label[label == 4] = 3
            else :
                image = sitk.ReadImage(p)
                image = sitk.GetArrayFromImage(image)
                images.append(image)

        images = np.array(images)

        return images, label

def test_model(net_1, val_loader):
    # 训练完毕进行测试。
    net_1.eval()
    end_metric = []
    for image, label in tqdm(val_loader, total=len(val_loader)):

        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred_1 = sliding_window_infer(image, network=net_1)
            pred_1 = pred_1.argmax(dim=1)
            res = evaluate_BraTS_case(pred_1.cpu().numpy(), label.cpu().numpy())
            print(f"single metric is {res}")
            end_metric.append(res)

    end_metric = average_metric(end_metric)

    return end_metric



if __name__ == '__main__':
    model = BasicUNet(dimensions=3, in_channels=in_channels, out_channels=out_channels, features=[16, 16, 32, 32, 64, 16])
    model.load_state_dict(torch.load(model_save_dir + "/model_1_epoch_300.bin", map_location=device))

    val_ds = BraTSDataset(val_paths, train=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    res = test_model(model, val_loader)
    print(f"res metric is {res}")



