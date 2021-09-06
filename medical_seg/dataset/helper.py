
import SimpleITK as sitk 
import numpy as np 
import torch 


def read_nii_file(image_path):
    ## image_path: {"image": image_nii, "label": label_nii}
    ## return image:(batch, channel, d, w, h)
    ## label: (batch, d, w, h)
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

