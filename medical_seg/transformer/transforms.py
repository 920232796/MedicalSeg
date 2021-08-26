from functools import total_ordering
from operator import is_
import numpy as np
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
import h5py
import matplotlib.pyplot as plt 
import torch 
from .utils import generate_pos_neg_label_crop_centers

class Random:
    def __init__(self, seed) -> None:
        self.seed = seed
        self.R = np.random.RandomState(seed)

    def do_transform(self, prob):
        ## 随机一个概率，当这个概率小于prob的时候，便去进行变换。
        prob = min(max(prob, 0.0), 1.0)
        return self.R.rand() < prob


def map_spatial_axes(
    img_ndim: int,
    spatial_axes=None,
    channel_first=True,
) -> List[int]:
    """
    Utility to map the spatial axes to real axes in channel first/last shape.
    For example:
    If `channel_first` is True, and `img` has 3 spatial dims, map spatial axes to real axes as below:
    None -> [1, 2, 3]
    [0, 1] -> [1, 2]
    [0, -1] -> [1, -1]
    If `channel_first` is False, and `img` has 3 spatial dims, map spatial axes to real axes as below:
    None -> [0, 1, 2]
    [0, 1] -> [0, 1]
    [0, -1] -> [0, -2]

    Args:
        img_ndim: dimension number of the target image.
        spatial_axes: spatial axes to be converted, default is None.
            The default `None` will convert to all the spatial axes of the image.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints.
        channel_first: the image data is channel first or channel last, default to channel first.

    """
    if spatial_axes is None:
        spatial_axes_ = list(range(1, img_ndim) if channel_first else range(img_ndim - 1))

    else:
        spatial_axes_ = []
        for a in spatial_axes:
            if channel_first:
                spatial_axes_.append(a if a < 0 else a + 1)
            else:
                spatial_axes_.append(a - 1 if a < 0 else a)

    return spatial_axes_

class RandomFlip():
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``np.flip`` in practice. See numpy.flip for additional details:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html.

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.

    """

    def __init__(self, random_state, spatial_axis = None, execution_probability=0.2):
        self.spatial_axis = spatial_axis
        self.random_state = random_state
        self.execution_probability = execution_probability
    def __call__(self, img: np.ndarray, label: np.ndarray = None) -> np.ndarray:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        if self.random_state.rand() > self.execution_probability:
            ## 不去做变换
            return img, label

        result: np.ndarray = np.flip(img, map_spatial_axes(img.ndim, self.spatial_axis))
        if label is not None :
            if len(label.shape) == 3:
                # 说明通道维度没有
                label = np.expand_dims(label, axis=0)
                label = np.flip(label, map_spatial_axes(label.ndim, self.spatial_axis))
                label = np.squeeze(label, axis=0)
            elif len(label.shape) == 4:
                label = np.flip(label, map_spatial_axes(label.ndim, self.spatial_axis))
            else :
                raise "label shape err"
            return result.astype(img.dtype), label.astype(label.dtype)

        return result.astype(img.dtype)

class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='reflect', order=0):
        if axes is None:
            axes = [[2, 1]] # 这样就是以后两个维度为平面进行旋转。 第一个维度是深度
    
        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m, label=None):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)
        assert m.ndim == 4, "输入必须为3d图像，第一个维度为channel"
        channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                    in range(m.shape[0])]
        m = np.stack(channels, axis=0)

        if label is not None :
            assert label.ndim == 3, "label shape 必须为三维"
            label = rotate(label, angle, axes=axis, reshape=False, order=self.order, mode="nearest", cval=-1)

        return m, label


class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    """

    def __init__(self, a_min, a_max, b_min=0, b_max=1, eps=1e-6, clip=True):
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.eps = eps
        self.clip = clip

    def __call__(self, m):
        img = (m - self.a_min) / (self.a_max - self.a_min)

        if self.clip:
            img = np.clip(img, self.b_min, self.b_max)

        return img

class AdditiveGaussianNoise:
    def __init__(self, random_state, scale=(0.0, 0.2), execution_probability=0.2):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.scale = scale


    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            std = self.random_state.uniform(self.scale[0], self.scale[1])
            gaussian_noise = self.random_state.normal(0, std, size=m.shape)
            return m + gaussian_noise

        return m

class AdditivePoissonNoise:
    def __init__(self, random_state, lam=(0.0, 0.2), execution_probability=0.2):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.lam = lam

    def __call__(self, m):
        if self.random_state.rand() < self.execution_probability:
            lam = self.random_state.uniform(self.lam[0], self.lam[1])
            poisson_noise = self.random_state.poisson(lam, size=m.shape)
            return m + poisson_noise
        return m
    

class SpatialCrop:
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial (channel-first) data.
    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (allows for use of -ve indexing and `None`)
        - a spatial center and size
        - the start and end coordinates of the ROI
    """

    def __init__(
        self,
        roi_center: Union[Sequence[int], np.ndarray, None] = None,
        roi_size: Union[Sequence[int], np.ndarray, None] = None,
        roi_start: Union[Sequence[int], np.ndarray, None] = None,
        roi_end: Union[Sequence[int], np.ndarray, None] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
        """
        
        if roi_center is not None and roi_size is not None:
            roi_center = np.asarray(roi_center, dtype=np.int16)
            roi_size = np.asarray(roi_size, dtype=np.int16)
            roi_start_np = np.maximum(roi_center - np.floor_divide(roi_size, 2), 0)
            roi_end_np = np.maximum(roi_start_np + roi_size, roi_start_np)
        else:
            if roi_start is None or roi_end is None:
                raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
            roi_start_np = np.maximum(np.asarray(roi_start, dtype=np.int16), 0)
            roi_end_np = np.maximum(np.asarray(roi_end, dtype=np.int16), roi_start_np)
        # Allow for 1D by converting back to np.array (since np.maximum will convert to int)
        roi_start_np = roi_start_np if isinstance(roi_start_np, np.ndarray) else np.array([roi_start_np])
        roi_end_np = roi_end_np if isinstance(roi_end_np, np.ndarray) else np.array([roi_end_np])
        # convert to slices
        self.slices = [slice(s, e) for s, e in zip(roi_start_np, roi_end_np)]

    def __call__(self, img: Union[np.ndarray, torch.Tensor]):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        sd = min(len(self.slices), len(img.shape[1:]))  # spatial dims
        slices = [slice(None)] + self.slices[:sd]
        return img[tuple(slices)]

class CenterSpatialCrop:
    """
    Crop at the center of image with specified ROI size.
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
    """

    def __init__(self, roi_size: Union[Sequence[int], int]) -> None:
        self.roi_size = roi_size

    def __call__(self, img: np.ndarray):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        assert img.ndim == 4, "img ndim 必须为4， (channel, W, H, D)"
        center = [i // 2 for i in img.shape[1:]]
        cropper = SpatialCrop(roi_center=center, roi_size=self.roi_size)
        return cropper(img)

def map_binary_to_indices(
    label: np.ndarray,
    image: Optional[np.ndarray] = None,
    image_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the foreground and background of input label data, return the indices after fattening.
    For example:
    ``label = np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]])``
    ``foreground indices = np.array([1, 2, 3, 5, 6, 7])`` and ``background indices = np.array([0, 4, 8])``
    Args:
        label: use the label data to get the foreground/background information.
        image: if image is not None, use ``label = 0 & image > image_threshold``
            to define background. so the output items will not map to all the voxels in the label.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.
    """
    # Prepare fg/bg indices
    if label.shape[0] > 1:
        label = label[1:]  # for One-Hot format data, remove the background channel
    label_flat = np.any(label, axis=0).ravel()  # in case label has multiple dimensions
    fg_indices = np.nonzero(label_flat)[0]
    if image is not None:
        img_flat = np.any(image > image_threshold, axis=0).ravel()
        bg_indices = np.nonzero(np.logical_and(img_flat, ~label_flat))[0]
    else:
        bg_indices = np.nonzero(~label_flat)[0]

    return fg_indices, bg_indices

class RandCropByPosNegLabel:
    """
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::
        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]
    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.
    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.
    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        label: Optional[np.ndarray] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: Optional[np.ndarray] = None,
        image_threshold: float = 0.0,
        random_state: np.random.RandomState = None,
    ) -> None:
        self.spatial_size = spatial_size
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.random_state = random_state
       
    def randomize(
        self,
        label: np.ndarray,
       
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = self.spatial_size
        
       
        fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
       
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_, bg_indices_, rand_state=self.random_state
        )

    def __call__(
        self,
        img: np.ndarray,
        label: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        is_label = False,
    ) -> List[np.ndarray]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
        """
        if label is None:
            label = self.label
        if label is None:
            raise ValueError("label should be provided.")
        if image is None:
            image = self.image

        if not is_label:
            self.randomize(label, image)
        results: List[np.ndarray] = []
        if self.centers is not None:
            for center in self.centers:
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
                r = cropper(img)
                results.append(r)
                

        return results

class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value, max_value):
        assert max_value > min_value
        self.min_value = min_value
        self.value_range = max_value - min_value

    def __call__(self, m):
        norm_0_1 = (m - self.min_value) / self.value_range
        return np.clip(2 * norm_0_1 - 1, -1, 1)

    
# if __name__ == "__main__":
    # csc = CenterSpatialCrop(roi_size=(2, 2, 2))

    # t1 = torch.rand(1, 4, 4, 4)
    # print(t1)
    # out = csc(t1)
    # print(out)
    

# if __name__ == "__main__":
#     print("数据增强函数测试")
#     r = Random(seed=8)
#     print(r.do_transform(0.5))
#     print(r.do_transform(0.5))
#     print(r.do_transform(0.5))
#     print(r.do_transform(0.5))

#     f = RandomFlip(r.R)
#     image = h5py.File("./BAI_YUE_BIN_data.h5", "r")
#     single_model_image = image["image"][:1]
#     label = image["label"][0]
#     print(f"label shape is {label.shape}")
#     print(single_model_image.shape)


#     sd = Standardize(a_min=single_model_image.min(), a_max=single_model_image.max())
#     single_model_image = sd(single_model_image)
#     print("归一化变换")
#     plot_3d(single_model_image)
#     plot_3d_label(label)

#     # print("随机翻转变换")
#     # single_model_image, label = f(single_model_image, label)
#     # plot_3d(single_model_image)
#     # plot_3d_label(label)

#     # print("随机旋转变换")
#     # ro = RandomRotate(random_state=r.R)
#     # single_model_image, label = ro(single_model_image, label)
#     # print(single_model_image.shape)
#     # plot_3d(single_model_image)
#     # plot_3d_label(label)

#     # print("添加高斯噪声")
#     # gn = AdditiveGaussianNoise(r.R)
#     # single_model_image = gn(single_model_image)
#     # plot_3d(single_model_image)

#     print("添加柏松噪声")

#     pn = AdditivePoissonNoise(r.R)
#     single_model_image = pn(single_model_image)
#     plot_3d(single_model_image)

