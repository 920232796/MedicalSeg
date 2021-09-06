import matplotlib.pyplot as plt 

import glob 
import numpy as np 
import torch 
import SimpleITK as sitk 
from medical_seg.transformer import RandomRotate, RandCropByPosNegLabel, RandomFlip, \
                                    AdditiveGaussianNoise, AdditivePoissonNoise, Standardize, CenterSpatialCrop
from torch.utils.data import DataLoader, Dataset


seed = 3213214325
sample_size = 2
random_state = np.random.RandomState(seed)
spatial_size = (128, 128, 128)

class Transform:
    def __init__(self, random_state) -> None:
        self.random_state = random_state
        
        self.rf = RandomFlip(self.random_state, execution_probability=0.1)
        self.rr = RandomRotate(self.random_state, angle_spectrum=30)
        self.ag = AdditiveGaussianNoise(self.random_state, scale=(0, 0.1), execution_probability=0.1)
        self.ap = AdditivePoissonNoise(self.random_state, lam=(0, 0.005), execution_probability=0.1)

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
    ds = MyDataset(paths=data_paths, train=False)
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
        plt.imshow(label[0, 0, 60], cmap="gray")
        plt.show()