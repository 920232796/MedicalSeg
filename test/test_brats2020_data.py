from functools import total_ordering
import matplotlib.pyplot as plt 

import glob 
from tqdm import tqdm
import numpy as np 
import torch 
import SimpleITK as sitk 
from medical_seg.transformer import RandomRotate, RandCropByPosNegLabel, RandomFlip, \
                                    AdditiveGaussianNoise, AdditivePoissonNoise, Standardize, \
                                    CenterSpatialCrop, Elatic, MirrorTransform, GammaTransformer
from torch.utils.data import DataLoader, Dataset
import time 

seed = 3213214325
sample_size = 2
random_state = np.random.RandomState(seed)
spatial_size = (128, 128, 128)


class Transform:
    def __init__(self, random_state) -> None:
        self.random_state = random_state
        
        self.rf = RandomFlip(self.random_state, execution_probability=1)
        self.rr = RandomRotate(self.random_state, angle_spectrum=30, execution_probability=1)
        self.elastic = Elatic(self.random_state, alpha=(0, 900), sigma=(9, 13), 
                                scale=(0.85, 1.25), order_seg=0, order_data=3,
                                execution_probability=1)
        self.gamma = GammaTransformer(self.random_state, gamma_range=(0.5, 2), execution_probability=1)
        self.mirror = MirrorTransform(self.random_state, axes=(0, 1, 2), execution_probability=1)

    def __call__(self, image, label):
        start = time.time()
        image, label = self.rf(image, label)
        end = time.time()
        print(f"rf spend {end - start}")
        start = time.time()
        image, label = self.rr(image, label)
        end = time.time()
        print(f"rr spend {end - start}")

        # start = time.time()
        # image, label = self.elastic(m=image, seg=label)
        # end = time.time()
        # print(f"elastic spend {end - start}")

        start = time.time()
        # image, label = self.elastic(m=image, seg=label)
        image = self.gamma(image)
        end = time.time()
        print(f"gamma spend {end - start}")
        start = time.time()
        image, label = self.mirror(image, seg=label)
        end = time.time()
        print(f"mirror spend {end - start}")
       
        return image, label   

class BraTSDataset(Dataset):
    def __init__(self, paths, train=True) -> None:
        super().__init__()
        self.paths = paths[:4]
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
            else :
                image = sitk.ReadImage(p)
                image = sitk.GetArrayFromImage(image)
                images.append(image)
        
        images = np.array(images)
       
        return images, label

def collate_fn(batch):
    assert len(batch) == 1, "随机crop时，请设置sample size，而batch size只能为1"
    batch = batch[0]
    image = batch["image"]
    label = batch["label"]

    image = np.array(image, dtype=np.float32)
    label = np.array(label, dtype=np.int16)

    return torch.from_numpy(image), torch.from_numpy(label)


if __name__ == "__main__":
    data_paths = sorted(glob.glob("./data/MICCAI_BraTS2020_TrainingData/*"))[:-2]
    print(data_paths)
    train_paths = data_paths[:315]
    val_paths = data_paths[315:]
    ## test dataloader
    ds = BraTSDataset(paths=data_paths, train=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    import matplotlib.pyplot as plt 
    for image, label in dl:
        print(image.shape)
        print(label.shape)
        plt.subplot(1, 5, 1)
        plt.imshow(image[0, 0, 60], cmap="gray")
        plt.subplot(1, 5, 2)
        plt.imshow(image[0, 1, 60], cmap="gray")
        plt.subplot(1, 5, 3)
        plt.imshow(image[0, 2, 60], cmap="gray")
        plt.subplot(1, 5, 4)
        plt.imshow(image[0, 3, 60], cmap="gray")
        plt.subplot(1, 5, 5)
        plt.imshow(label[0, 60], cmap="gray")
        plt.show()