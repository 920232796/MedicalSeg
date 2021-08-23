## 数据 http://medicaldecathlon.com/ 里面的spleen 分割 数据为3D size为（51， 512， 512）  5折交叉验证
import glob 
import matplotlib.pyplot as plt 
import SimpleITK as sitk 
from sklearn.model_selection import KFold  ## K折交叉验证
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from medical_seg.transformer import RandomRotate, RandCropByPosNegLabel, RandomFlip, \
                                    AdditiveGaussianNoise, AdditivePoissonNoise
from medical_seg.networks import BasicUNet

spleen_image_paths = glob.glob("./data/Task09_Spleen/imagesTr/*")
spleen_label_paths = glob.glob("./data/Task09_Spleen/labelsTr/*")

data_paths = [{"image": image, "label": label} for image, label in zip(spleen_image_paths, spleen_label_paths)]

class MyDataset(Dataset):
    def __init__(self, paths, is_train=True) -> None:
        super().__init__()
        self.paths = paths
        self.is_train = is_train

    def __getitem__(self, i):
        get_path = self.paths[i]

    def __len__(self):
        return self.paths

    def _read_image(self, image_path):
        image = sitk.ReadImage(image_path["image"])
        label = sitk.ReadImage(image_path["label"])
        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)


if __name__ == "__main__":

    X = np.arange(51)
    kfold = KFold(n_splits=5, shuffle=False)  ## kfold为KFolf类的一个对象
    fold = 0
    for a, b in kfold.split(X):  ## .split(X)方法返回迭代器，迭代器每次产生两个元素，1、训练数据集的索引；2. 测试集索引
        print(a, b)
        train_paths = [], val_paths = []
        for train_indice in a:
            train_paths = train_paths.append(data_paths[train_indice])
        for val_indice in b :
            val_paths = val_paths.append(data_paths[val_indice])
        


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