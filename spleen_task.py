## 数据 http://medicaldecathlon.com/ 里面的spleen 分割 数据为3D size为（51， 512， 512）  5折交叉验证
import glob
from medical_seg.utils.enums import BlendMode

from matplotlib import cm
from medical_seg.utils.misc import dtype_numpy_to_torch
from operator import pos 
import os 
import matplotlib.pyplot as plt 
import SimpleITK as sitk
from numpy.lib.type_check import imag 
from sklearn.model_selection import KFold  ## K折交叉验证
import numpy as np 
import torch 
from torch import optim
import setproctitle
from torch._C import dtype
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from medical_seg.transformer import RandomRotate, RandCropByPosNegLabel, RandomFlip, \
                                    AdditiveGaussianNoise, AdditivePoissonNoise, Standardize
from medical_seg.networks import BasicUNet
from medical_seg.utils import set_seed
from tqdm import tqdm
from medical_seg.inferer import SlidingWindowInferer
from utils import segmenation_metric, resample_image_array_size

spleen_image_paths = glob.glob("./data/Task09_Spleen/imagesTr/*")
spleen_label_paths = glob.glob("./data/Task09_Spleen/labelsTr/*")

data_paths = [{"image": image, "label": label} for image, label in zip(spleen_image_paths, spleen_label_paths)]

seed = 3213214325
in_channels = 1
out_channels = 2
batch_size = 1
sample_size = 2
random_state = np.random.RandomState(seed)
set_seed(seed)
lr = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 200
model_name = "spleen_unet"
spatial_size = (32, 256, 256)

sliding_window_infer = SlidingWindowInferer(roi_size=spatial_size, sw_batch_size=2, overlap=0.5)


class Transform:
    def __init__(self, random_state) -> None:
        self.random_state = random_state
        
        self.rf = RandomFlip(self.random_state, execution_probability=0.2)
        self.rr = RandomRotate(self.random_state, angle_spectrum=30)
        self.ag = AdditiveGaussianNoise(self.random_state, scale=(0, 0.2), execution_probability=0.2)
        self.ap = AdditivePoissonNoise(self.random_state, lam=(0, 0.01), execution_probability=0.2)

    def __call__(self, image, label):
        image, label = self.rf(image, label)
        image, label = self.rr(image, label)
        image = self.ag(image)
        image = self.ap(image)
        return image, label   

class MyDataset(Dataset):
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

    def __getitem__(self, i):
        get_path = self.paths[i]
        image, label = self._read_image(get_path)
        if image.shape[1] < 32:
            image_shape = image.shape
            out_size = (image_shape[0], 32, 256, 256)
            image = resample_image_array_size(image, out_size, order=3)
            label = resample_image_array_size(label, out_size=(32, 256, 256), order=0)


        sd = Standardize(a_min=image.min(), a_max=image.max(), b_min=0, b_max=1, clip=True)
        image = sd(image)

        if self.train:
            image, label = self.transform(image, label)
        
            if len(label.shape) == 3:
                label = np.expand_dims(label, axis=0)
            image = self.random_crop(image, label=label)
            label = self.random_crop(label, label=label, is_label=True)
        else :
            if len(image.shape) == 3:
                image = [[ image ]]
                label = [[ label ]]
            elif len(image.shape) == 4:
                image = [image]
                label = [[ label ]]

        return {
            "image": image, 
            "label": label,
        }

    def __len__(self):
        return len(self.paths)

    def _read_image(self, image_path):
        image = sitk.ReadImage(image_path["image"])
        label = sitk.ReadImage(image_path["label"])
        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        return image, label

def collate_fn(batch):
    assert len(batch) == 1, "随机crop时，请设置sample size，而batch size只能为1"
    batch = batch[0]
    image = batch["image"]
    label = batch["label"]

    image = np.array(image, dtype=np.float32)
    label = np.array(label, dtype=np.int16)

    return torch.from_numpy(image), torch.from_numpy(label)


def main_train(net_1, train_data_paths, test_data_paths, k_fold):
    train_ds = MyDataset(train_data_paths, train=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer_1 = optim.Adam(net_1.parameters(), lr=lr, weight_decay=1e-5)
    net_1.to(device)
    criterion_1 = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8], device=device, dtype=torch.float32))

    val_ds = MyDataset(test_data_paths, train=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for epoch in range(epochs):
        setproctitle.setproctitle("{}_{}".format(model_name, epoch))
        print("epoch is " + str(epoch))
        net_1.train()

        epoch_loss_1 = 0.0
        for image, label in tqdm(train_loader, total=len(train_loader)):

            optimizer_1.zero_grad()

            image = image.to(device)
            label = label.to(device)
            pred_1 = net_1(image)
            label = label.squeeze(dim=1).long()
            hard_loss_1 = criterion_1(pred_1, label)
            epoch_loss_1 += hard_loss_1.item()
            hard_loss_1.backward()
            optimizer_1.step()

        if (epoch + 1) % 50 == 0:
            for image, label in tqdm(val_loader, total=len(val_loader)):
                # image = nn.functional.interpolate(image, size=(32, 256, 256), mode="trilinear", align_corners=False)
                # label = torch.unsqueeze(label, dim=1)
                # label = nn.functional.interpolate(label, size=(32, 256, 256), mode="nearest")
                # label = torch.squeeze(label, dim=1).long()

                image = image.to(device)
                label = label.to(device)
                label = label.squeeze(dim=1).long()
                with torch.no_grad():
                    pred_1 = sliding_window_infer(image, network=net_1)

                    print(f"pred_1 is {pred_1.shape}")

                    metric = segmenation_metric(pred_1, label)
                    print(f"metric is {metric}")


if __name__ == "__main__":

    X = np.arange(len(data_paths))
    kfold = KFold(n_splits=4, shuffle=False)  ## kfold为KFolf类的一个对象
    fold = 0
    for a, b in kfold.split(X):  ## .split(X)方法返回迭代器，迭代器每次产生两个元素，1、训练数据集的索引；2. 测试集索引
        # print(a, b)
        fold += 1
        train_paths = []
        val_paths = []
        for train_indice in a:
            train_paths.append(data_paths[train_indice])
        for val_indice in b :
            val_paths.append(data_paths[val_indice])
        
        print(f"fold is {fold} \n train_set is {a} \n test_set is {b}")

        model = BasicUNet(dimensions=3, in_channels=in_channels, out_channels=out_channels, features=[16, 16, 32, 64, 128, 16])

        main_train(model, train_data_paths=train_paths, test_data_paths=val_paths, k_fold=fold)


    # print(spleen_image_paths)
    # for image_path, label_path in zip(spleen_image_paths, spleen_label_paths):
    #     image = sitk.ReadImage(image_path)
    #     label = sitk.ReadImage(label_path)
    #     # print(image.shape)
    #     image = sitk.GetArrayFromImage(image)
    #     label = sitk.GetArrayFromImage(label)
    #     print(image.shape)
    #     print(label.shape)
    #     print(label.min())
    #     print(label.max())
    #     print(image.min())
    #     print(image.max())
    #     image = (image - image.min()) / (image.max() - image.min())
    #     print(image.min())
    #     print(image.max())
    #     for i in range(50):
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(image[i], cmap="gray")
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(label[i], cmap="gray")
    #         plt.show()
    #     break 
    ##  # print(image.shape)
            # print(label.shape)
            # import matplotlib.pyplot as plt 
            # for i in range(15, 30):
            #     plt.subplot(1, 2, 1)
            #     img = image[0, 0, i]
            #     plt.imshow(img, cmap="gray")
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(label[0, 0, i], cmap="gray")
            #     plt.show()
                
            # continue