

import numpy as np 
import h5py
from medical_seg.transformer import Normalize, RandomRotate90
import matplotlib.pyplot as plt 


if __name__ == "__main__":
    pass
    ##
    # a1 = np.array([[1, 3, 4], [5, 6, 7]])
    # norm_1 = Normalize(a1.min(), a1.max())

    # print(norm_1(a1))

    rr90 = RandomRotate90(np.random.RandomState(777))
    index = 10
    image = h5py.File("./data/test.h5", "r")
    single_model_image = image["image"][:1]
    label = image["label"][0]
    print(f"label shape is {label.shape}")
    print(single_model_image.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(single_model_image[0, index], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(label[index], cmap="gray")
    plt.show()

    image_rr, label = rr90(single_model_image, label)
    plt.subplot(1, 2, 1)
    plt.imshow(image_rr[0, index], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(label[index], cmap="gray")
    plt.show()