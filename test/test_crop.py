
from medical_seg.transformer.transforms import RandCropByPosNegLabel, CenterSpatialCrop
import torch.nn as nn 
import numpy as np 
import torch 

if __name__ == '__main__':
    random_state = np.random.RandomState(666)

    label = np.array([[[1, 2], [3, 4]], [[5, 6], [0, 0]]])
    label = np.expand_dims(label, axis=0)
    image = label
    print(image.shape)
    print(label.shape)
    pos_neg_crop = RandCropByPosNegLabel(spatial_size=(1, 1, 1), label=label, 
    pos=2, neg=1, num_samples=2, image=None, image_threshold=None, random_state=random_state)

    out = pos_neg_crop(image)

    print(out)


    center_crop = CenterSpatialCrop(roi_size=(2, 2, 1))
    out = center_crop(image)
    print(out)


    # label_flat = np.any(label, axis=0).ravel()  # in case label has multiple dimensions
    # fg_indices = np.nonzero(label_flat)[0]
    # print(fg_indices)