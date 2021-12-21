import numpy as np
import torch
import math
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import argparse
import os
from medical_seg.networks.nets.swin_unet.model import SwinUnet
from medical_seg.networks import BasicUNet
from medical_seg.networks.nets.segresnet import SegResNet
from medical_seg.networks.spatial_fusion_net import SpatialTransNetV3
from medical_seg.evaluation import Metric
from medical_seg.networks.nets.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
import copy
import setproctitle
import h5py
from sklearn.model_selection import KFold  ## K折交叉验证
import os
import SimpleITK as sitk
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import random
import glob


## 2.5d 模型改成2d 多通道
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

huanhu_transbts = "meni_transbts"
huanhu_swin_unet = "meni_swin_unet"
huanhu_unet = "meni_unet"
huanhu_segresnet = "meni_segresnet"
huanhu_ours = "meni_multi_att_net_v3_cpc_data_aug"

model_save_dir = "./state_dict/"

batch_size = 1
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("device is " + str(device))
in_channels = 2
out_channels = 3

metric = Metric(class_name=["tumor", "shuizhong"], voxel_spacing=(1, 1, 1, 1), nan_for_nonexisting=True)


data_paths = glob.glob("./data/Meningiomas/*.h5")
print("数据共：{} 例".format(len(data_paths)))

train_two_fold = []
test_two_fold = []
train_paths = []
test_paths = []
for i in range(78):
    train_paths.append(data_paths[i])
#
for i in range(78, len(data_paths)):
    test_paths.append(data_paths[i])


train_two_fold.append(train_paths)
test_two_fold.append(test_paths)


# ## 第二折
train_paths = []
test_paths = []
for i in range(78, len(data_paths)):
    train_paths.append(data_paths[i])

for i in range(78):
    test_paths.append(data_paths[i])

train_two_fold.append(train_paths)
test_two_fold.append(test_paths)


class Dataset3d(Dataset):
    """
    """
    def __init__(
            self,
            paths,
            train=True,
            is_mean=True,
    ) -> None:

        super(Dataset3d, self).__init__()
        self.train = train
        self.is_mean = is_mean
        self._cache_image = []

        self._cache_label = []

        for i in range(len(paths)):
            image, label = self._load_cache_item(paths[i])
            if image is not None :
                self._cache_image.append(image)
                self._cache_label.append(label)


    def get_labels(self, label):
        labels = np.zeros(label.shape[1:])
        labels[label[0] == 1] = 1
        labels[label[1] == 1] = 2
        return labels

    def _load_cache_item(self, d_path):
        h5_image = h5py.File(d_path, "r")
        image = h5_image["image"][()]
        label = h5_image["label"][()]
        h5_image.close()

        labels = self.get_labels(label)

        if len(np.unique(labels)) == 2:
            return None, None 
        
        return image, labels

    def __getitem__(self, index):
        image = self._cache_image[index]
        

        if self.is_mean:
            image = (image - image.mean()) / image.std()
        else :
            image = (image - image.min()) / (image.max() - image.min())

        label = self._cache_label[index]

        # if self.train:
        #     image, label = self.trans(image, label)

        return image.astype(np.float32), label.astype(np.float32)

    def __len__(self):
        return len(self._cache_image)



def save_prediction(pred, image, label, save_path, model_name, dice, prefix=""):
    ## 保存第几折 第几个

    pred = pred.argmax(dim=1).int()
    pred = pred.squeeze(0)
    image = image.squeeze(0).squeeze(0).float()
    label = label.squeeze(0).int()

    torch.save(image, save_path + "/image" + ".image")
    torch.save(pred, save_path + "/pred-" + dice + "-" + model_name + "_" + str(prefix))
    torch.save(label, save_path + "/label" + ".label")

def test_single_model(net, test_data_paths, model_name, k_fold, is_2d=False, is_swin=False, is_mean=True):

    net.eval()

    val_ds = Dataset3d(test_data_paths, is_mean=is_mean)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    ii = 0
    dices = []

    save_path = "./plot_huanhu_case/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for image, label in tqdm(val_loader, total=len(val_loader)):
        ii += 1

        save_path = "./plot_huanhu_case/" + str(k_fold) + "_" + str(ii) + "/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if is_swin:
            image = nn.functional.interpolate(image, size=(32, 224, 224), mode="trilinear", align_corners=False)
        

        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = net(image)
            if is_swin:
                pred = nn.functional.interpolate(pred, size=(32, 256, 256), mode="trilinear", align_corners=False)
            
            pred_arg = pred.argmax(dim=1)
            metric_res = metric.run(pred=pred_arg.cpu().numpy(), label=label.cpu().numpy())
            print(f"metric is {metric_res}")
            dice = str(metric_res["tumor_Dice"])[:4] + "_" + str(metric_res["shuizhong_Dice"])[:4]

            save_prediction(pred, image, label, save_path, model_name=model_name, dice=dice, prefix="single")

        print("save single model success...")

if __name__ == "__main__":

    for i in range(2):

        val_paths = test_two_fold[i]

        ## transbts model
        _, transbts_model = TransBTS(image_size=(32, 256, 256), patch_dim = (4, 32, 32), base_channel=16,
                            in_channels=2, out_channels=3, _conv_repr=True, 
                            _pe_type="learned", embedding_dim=128, hidden_dim=128, num_layers=3)
        
        if i == 0:
            checkpoint = torch.load(f"{model_save_dir}{huanhu_transbts}/model_{i}_epoch_350.bin", map_location=device)
        else :
            checkpoint = torch.load(f"{model_save_dir}{huanhu_transbts}/model_{i}_epoch_230.bin", map_location=device)
        
        transbts_model.load_state_dict(checkpoint)
        transbts_model.to(device)

        ## swin unet model 
        swin_unet_model = SwinUnet(img_size=224, in_channel=in_channels, num_classes=out_channels)

        swin_unet_model.load_state_dict(
            torch.load(f"{model_save_dir}{huanhu_swin_unet}/model_{i}_epoch_85.bin", map_location=device))
        swin_unet_model.to(device)

        unet_model = BasicUNet(dimensions=3, in_channels=in_channels, out_channels=out_channels, features=[16, 16, 32, 64, 128, 16])
        unet_model.load_state_dict(
            torch.load(f"{model_save_dir}{huanhu_unet}/model_{i}_epoch_target.bin", map_location=device))
        unet_model.to(device)

        segresnet_model = SegResNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels, init_filters=16)
        segresnet_model.load_state_dict(
            torch.load(f"{model_save_dir}{huanhu_segresnet}/model_{i}_epoch_95.bin", map_location=device))
        segresnet_model.to(device)


        ours_model = SpatialTransNetV3(model_num=in_channels, 
                                        out_channels=out_channels, 
                                        image_size=(2, 16, 16), 
                                        cpc_layer_num=2)
        ours_model.load_state_dict(
            torch.load(f"{model_save_dir}{huanhu_ours}/model_best_{i}.bin", map_location=device))
        ours_model.to(device)

        test_single_model(swin_unet_model, test_data_paths=val_paths, model_name=huanhu_swin_unet, k_fold=i, is_swin=True)

        test_single_model(transbts_model, test_data_paths=val_paths, model_name=huanhu_transbts, k_fold=i)

        test_single_model(unet_model, test_data_paths=val_paths, model_name=huanhu_unet, k_fold=i)
        test_single_model(segresnet_model, test_data_paths=val_paths, model_name=huanhu_segresnet, k_fold=i)
        test_single_model(ours_model, test_data_paths=val_paths, model_name=huanhu_ours, k_fold=i, is_mean=False)

        print("测试结束。")