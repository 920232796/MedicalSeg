
import torch
import random
import numpy as np
import torch.nn.functional as f
import os
import torch.nn as nn
from scipy import ndimage
from sklearn.model_selection import KFold  ## K折交叉验证
import warnings

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def resample_tensor_image_label(image, label, size):
 
    if len(image.shape) != 5 or len(label.shape) != 4: 
        print("image or label shape is  err ")
        return
    label = label.float()
    image = nn.functional.interpolate(image, size=size, mode="trilinear", align_corners=False)
    label = torch.unsqueeze(label, dim=1)
    label = nn.functional.interpolate(label, size=size, mode="nearest")
    label = torch.squeeze(label, dim=1).long()
    return image, label

def resample_tensor(tensor_data, size, is_label=False):
    
    if is_label:
        assert len(tensor_data.shape) == 4, "label shape len is 4"
        tensor_data = tensor_data.float()
        tensor_data = torch.unsqueeze(tensor_data, dim=1)
        tensor_data = nn.functional.interpolate(tensor_data, size=size, mode="nearest")
        tensor_data = torch.squeeze(tensor_data, dim=1).long()
    else:
        assert len(tensor_data.shape) == 5, 'image shape len is 5'
        tensor_data = nn.functional.interpolate(tensor_data, size=size, mode="trilinear", align_corners=False)

    
    return tensor_data


def get_kfold_data(data_paths, n_splits, total_num, shuffle=False):
    X = np.arange(total_num)
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)  ## kfold为KFolf类的一个对象
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res

def write_res_line(w_path, content):
    with open(w_path, "a+", encoding="utf-8") as f:
        if type(content) is list:
            for c in content:
                f.write("写入结果为：")
                f.write(str(c))
                f.write("\n")
        else :
            f.write("写入结果为：")
            f.write(str(content))
            f.write("\n")

def softmin(x, dim=-1, t=1):
    m = nn.Softmin(dim)

    return  m(x / t)

def compute_orthogonal_loss(net_1, net_2):
    device = next(net_1.parameters()).device
    orthogonal_loss = 0.0
    n = 0
    for name, _ in net_1.named_parameters():
        weight_1 = net_1.state_dict()[name]
        weight_2 = net_2.state_dict()[name]

        if len(weight_1.shape) != 5:
            # print("非卷积核权重，跳过")
            continue
        n += 1
        kernel_size = weight_1.shape[0]
        weight_1 = weight_1.view((kernel_size, -1))
        weight_2 = weight_2.view((kernel_size, -1))

        weight_1 = f.normalize(weight_1, dim=-1, p=2)
        weight_2 = f.normalize(weight_2, dim=-1, p=2)
        sim_matrix = torch.matmul(weight_1, weight_2.t())

        err = 0.5 * (sim_matrix ** 2).sum()
        # mask = torch.eye(*sim_matrix.shape, device=device)
        # err = (sim_matrix**2 * mask).sum()
        # print(f"err is {err}")
        # print(f"sim matrix is {sim_matrix * mask}")

        orthogonal_loss += err
    # orthogonal_loss = orthogonal_loss / n # 求平均
    return orthogonal_loss


def set_seed(seed: int, use_deterministic_algorithms=None):
   

    seed = int(seed)
    torch.manual_seed(seed)

    global _seed
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
   
    if use_deterministic_algorithms is not None:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(use_deterministic_algorithms)
        elif hasattr(torch, "set_deterministic"):
            torch.set_deterministic(use_deterministic_algorithms)  # type: ignore
        else:
            warnings.warn("use_deterministic_algorithms=True, but PyTorch version is too old to set the mode.")


def compute_dice(input, target):
    eps = 0.0001
    # input 是经过了sigmoid 之后的输出。
    input = (input > 0.5).float()
    target = (target > 0.5).float()
    if target.sum() == 0:
        return 1.0

    inter = torch.sum(target.view(-1) * input.view(-1)) + eps

    # print(self.inter)
    union = torch.sum(input) + torch.sum(target) + eps

    t = (2 * inter.float()) / union.float()
    if int(t.item()) == 2:
        return 1.0
    return t.item()


def compute_dice_muti_class(input, target):

    ## 此时是多个类别应该得分开算dice
    class_num = input.shape[1]
    input = input.argmax(dim=1)
    # print(input.shape)
    dice = []
    for i in range(class_num):
        if i == 0:
            continue
        label = torch.zeros_like(target)
        input_single = torch.zeros_like(target)
        label[target == i] = 1
        input_single[input == i] = 1
        dice.append(compute_dice(input_single, label))
    return dice


def compute_iou(pred, label):
    # 计算iou loss
    inter = (pred * label).sum()
    union = torch.logical_or(pred, label)
    union = union.sum()

    return ((inter + 1e-6) /(union + 1e-6)).item()


def compute_3d_metric(pred_3d, label):
    ## 计算多个指标 3d
    # pred_3d_sig = (pred_3d_sig > 0.5).float()
    seg_inv, gt_inv = torch.logical_not(pred_3d), torch.logical_not(label)
    true_pos = float(torch.logical_and(pred_3d, label).sum())  # float for division
    true_neg = torch.logical_and(seg_inv, gt_inv).sum()
    false_pos = torch.logical_and(pred_3d, gt_inv).sum()
    false_neg = torch.logical_and(seg_inv, label).sum()

    # 然后根据公式分别计算出这几种指标
    prec = true_pos / (true_pos + false_pos + 1e-6)
    rec = true_pos / (true_pos + false_neg + 1e-6)
    specificity = true_neg / (true_neg + false_neg + 1e-6)

    return prec.item(), rec.item(), specificity.item()


def segmenation_metric(pred, target):
    class_num = pred.shape[1]
    pred = pred.argmax(dim=1)
    # print(input.shape)
    metric = []

    for i in range(class_num):
        if i == 0:
            continue
        label = torch.zeros_like(target)
        input_single = torch.zeros_like(target)
        label[target == i] = 1
        input_single[pred == i] = 1
        dice = compute_dice(input_single, label)
        prec, rec, specificity = compute_3d_metric(input_single, label)
        iou = compute_iou(input_single, label)

        metric.append([dice, prec, rec, specificity, iou])
    # print(metric)
    return metric


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def resample_image_array_size(image_array, out_size, order=3):
    #Bilinear interpolation would be order=1,
    # nearest is order=0,
    # and cubic is the default (order=3).
    real_resize = np.array(out_size) / image_array.shape

    new_volume = ndimage.zoom(image_array, zoom=real_resize, order=order)
    return new_volume