## 数据 brats 2020 数据
import glob

from matplotlib import cm
from medical_seg.utils.enums import BlendMode
import os 
import SimpleITK as sitk
from sklearn.model_selection import KFold  ## K折交叉验证
import numpy as np
import torch 
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

seed = 3213214325
in_channels = 4
out_channels = 4
sample_size = 2
random_state = np.random.RandomState(seed)
set_seed(seed)
lr = 0.0001
# device = torch.device("cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
epochs = 500
model_name = "brats_unet"
spatial_size = (128, 128, 128)

sliding_window_infer = SlidingWindowInferer(roi_size=spatial_size, sw_batch_size=2, overlap=0.5)

if not os.path.exists("./state_dict/"):
    os.mkdir("./state_dict/")
    print("新建state_dict文件夹，用于存放保存的模型与结果...")

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
            end_metric.append(evaluate_BraTS_case(pred_1.cpu().numpy(), label.cpu().numpy()))
        
    end_metric = average_metric(end_metric)
   
    return end_metric

def main_train(net_1, train_data_paths, test_data_paths, k_fold):
    train_ds = BraTSDataset(train_data_paths, train=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    optimizer_1 = optim.Adam(net_1.parameters(), lr=lr, weight_decay=1e-5)
    net_1.to(device)
    criterion_1 = nn.CrossEntropyLoss()

    val_ds = BraTSDataset(test_data_paths, train=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for epoch in range(epochs):
        setproctitle.setproctitle("{}_{}".format(model_name, k_fold))
        print(f"epoch is {epoch}, k_fold is {k_fold}")
        net_1.train()

        epoch_loss_1 = 0.0
        for image, label in tqdm(train_loader, total=len(train_loader)):
            # print(np.unique(label.numpy()))
            optimizer_1.zero_grad()

            image = image.to(device)
            label = label.to(device)
            pred_1 = net_1(image)
            hard_loss_1 = criterion_1(pred_1, label)
            epoch_loss_1 += hard_loss_1.item()
            hard_loss_1.backward()
            optimizer_1.step()

        print(f"epoch_loss_1 is {epoch_loss_1}")
        if (epoch + 1) % 1 == 0:
              # 保存模型
            save_path = f"{model_save_dir}model_{k_fold}_epoch_{epoch}.bin"
            torch.save(net_1.state_dict(), save_path)
            print(f"模型保存成功: {save_path}")
            metric = test_model(net_1=net_1, val_loader=val_loader)
            print(f"metric is {metric}")
    metric = test_model(net_1=net_1, val_loader=val_loader)

    return metric 

if __name__ == "__main__":
   
    model = BasicUNet(dimensions=3, in_channels=in_channels, out_channels=out_channels, features=[16, 16, 32, 32, 64, 16])
    
    metric_fold = main_train(model, train_data_paths=train_paths, test_data_paths=val_paths, k_fold=1)
    with open(model_save_dir + "res.txt", "a+") as f:
        f.write(f"res is {metric_fold}")
        f.write("\n")
    

    