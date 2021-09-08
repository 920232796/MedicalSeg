

import numpy as np 
import h5py
from medical_seg.transformer import Normalize, RandomRotate90, Elatic, GammaTransformer
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    
    # image = h5py.File("./data/test.h5", "r")
    # single_model_image = image["image"][:1]
    # # single_model_image = np.expand_dims(single_model_image, axis=0)
    # single_model_image = (single_model_image - single_model_image.min()) / (single_model_image.max() - single_model_image.min())
    # label = image["label"][0]
    # label = np.expand_dims(label, axis=0)
    # # label = np.expand_dims(label, axis=0)
    # print(f"label shape is {label.shape}")
    # print(single_model_image.shape)

    # alpha=(0., 900.)
    # sigma=(9., 13.)
    # a = np.random.uniform(alpha[0], alpha[1])
    # s = np.random.uniform(sigma[0], sigma[1])

    # patch_size = (24, 256, 256)
    # coords = create_zero_centered_coordinate_mesh(patch_size)
    # print(coords.shape)
    # # plt.imshow(coords[0, 10], cmap="gray")
    # # plt.show()
    # coords = elastic_deform_coordinates(coords, a, s)
    # # print(coords)
    # dim = 3
    # seg = label
    # seg_result = np.zeros((1, patch_size[0], patch_size[1], patch_size[2]),
    #                               dtype=np.float32)
    # data_result = np.zeros((single_model_image.shape[0], patch_size[0], patch_size[1], patch_size[2]),
    #                            dtype=np.float32)
    # for d in range(dim):
      
    #     ctr = single_model_image.shape[d + 1] / 2. - 0.5
    #     coords[d] += ctr

    # scale = (0.85, 1.25)
    # order_seg = 1
    # order_data = 3
    # border_mode_seg = "constant"
    # border_cval_seg = 0
    # if scale[0] < 1:
    #     sc = np.random.uniform(scale[0], 1)
    # else :
    #     sc = np.random.uniform(max(scale[0], 1), scale[1])
    # coords = scale_coords(coords, sc)

    # for channel_id in range(single_model_image.shape[0]):
    #     data_result[channel_id] = interpolate_img(single_model_image[channel_id], coords, order_data,
    #                                                         cval=0.0, is_seg=False)
    # if seg is not None:
    #     for channel_id in range(seg.shape[0]):
    #         seg_result[channel_id] = interpolate_img(seg[channel_id], coords, order_seg,
    #                                                                     border_mode_seg, cval=border_cval_seg,
    #                                                                     is_seg=True)

    # print(data_result.shape)      
    # slices = 10 
    # # data_result = (data_result - data_result.min()) / (data_result.max() - data_result.min())
    # # single_model_image = (single_model_image - single_model_image.min()) / (single_model_image.max() - single_model_image.min())
    # plt.subplot(2, 2, 1)
    # plt.imshow(data_result[0, slices], cmap="gray")
    # # plt.show()
    # plt.subplot(2, 2, 2)
    # plt.imshow(single_model_image[0, slices], cmap="gray")

    # plt.subplot(2, 2, 3)
    # plt.imshow(seg_result[0, slices], cmap="gray")
    # plt.subplot(2, 2, 4)
    # plt.imshow(label[0, slices], cmap="gray")
    # plt.show()
    #                                           
    # plt.imshow(coords[0, 10], cmap="gray")
    # plt.show()

    rs = np.random.RandomState(777)
    rr90 = RandomRotate90(rs)
    index = 10
    image = h5py.File("./data/test.h5", "r")
    single_model_image = image["image"][:1]
    single_model_image = (single_model_image - single_model_image.min()) / (single_model_image.max() - single_model_image.min())

    label = image["label"][0]
    print(f"label shape is {label.shape}")
    print(single_model_image.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(single_model_image[0, index], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(label[index], cmap="gray")
    plt.show()

    image, label = rr90(single_model_image, label)
    plt.subplot(1, 2, 1)
    plt.imshow(image[0, index], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(label[index], cmap="gray")
    plt.show()
    print(f"image shape is {image.shape}")
    elastic = Elatic(rs, order_seg=0)
    image, label = elastic(image, seg=label)
    plt.subplot(1, 2, 1)
    plt.imshow(image[0, index], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(label[index], cmap="gray")
    plt.show()

    gt = GammaTransformer()
    image = gt(image)
    plt.subplot(1, 2, 1)
    plt.imshow(image[0, index], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(label[index], cmap="gray")
    plt.show()


