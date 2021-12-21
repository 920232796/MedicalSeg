
from typing import Union
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from tqdm import tqdm
from medical_seg.transformer import  RandCropByPosNegLabel, Normalization, Standardize
from medical_seg.dataset import collate_fn

import os 

class Dataset3D(Dataset):
    """
    """
    def __init__(
            self,
            paths,
            trans_func=None,
            crop_func=None,
            train=False,
           
    ) -> None:

        super().__init__()
       
        # norm = Normalization(channel_wise=True)     
       
        self.train = train
        if trans_func is not None :
            self.trans = trans_func
        else :
            self.trans = None 

        self._cache_image = []
        self._cache_label = []

        if train and crop_func is None :
            #不行
            print("训练时必须有crop func")
            os._exit(0)
        self.random_crop = crop_func

        for i in tqdm(range(len(paths)), total=len(paths)):
            image, label = self._load_cache_item(paths[i])

            ## norm
            if image is not None :
                image = image.astype(np.float32)
                norm = Standardize(a_min=image.min(), a_max=image.max(), b_min=0, b_max=1)
                image = norm(image)
                self._cache_image.append(image)
                self._cache_label.append(label)

    def _load_cache_item(self, d_path):
        ## d_path:  {"image": image_path, "label": label_path }

        raise NotImplementedError

    def __getitem__(self, index):
        
        image = self._cache_image[index]
        label = self._cache_label[index]

        assert len(image.shape) == 4 and len(label.shape) == 3, "image shape must be 4 and label shape must be 3."

        if self.train:
            # 先数据增强 再去进行crop
            if self.trans is not None:
                image, label = self.trans(image, label)

            image_patchs = self.random_crop(image, label=label)
            label_patchs = self.random_crop(label, label=label, is_label=True)
            
        else :
            
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
        return len(self._cache_image)


def DataLoader3D(dataset: Dataset3D, batch_size=1, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)    