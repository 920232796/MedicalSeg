from functools import total_ordering
from operator import is_
import numpy as np
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
from numpy import random
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
import h5py
from itertools import chain

from batchgenerators.augmentations.utils import resize_segmentation
import matplotlib.pyplot as plt 
import torch

from .utils import generate_pos_neg_label_crop_centers, \
                    create_zero_centered_coordinate_mesh, \
                    elastic_deform_coordinates, \
                    interpolate_img, scale_coords,\
                    augment_gamma, augment_mirroring, is_positive, generate_spatial_bounding_box,\
                    Pad

from medical_seg.utils import resample_image_array_size
from .utils import resample_data_or_seg


RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3

def get_do_separate_z(spacing, anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis

def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    """
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """
    assert not ((data is None) and (seg is None))
    
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        if len(seg.shape) == 3:
            seg = np.expand_dims(seg, axis=0)

        assert len(seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z,
                                             order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, order_z=order_z_seg)
    else:
        seg_reshaped = None
    
    if len(seg_reshaped.shape) == 4:
        seg_reshaped = np.squeeze(seg_reshaped, axis=0)

    return data_reshaped, seg_reshaped


class ResampleImage:

    def __init__(self, resample_size, order=[3, 0]) -> None:
        self.rsize = resample_size

        self.order = order

    def __call__(self, image, label=None):

        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        c = image.shape[0]

        image = resample_image_array_size(image, out_size=(c,) + self.rsize, order=self.order[0])
        if label is not None:
            label = resample_image_array_size(label, out_size=self.rsize, order=self.order[1])

        return image, label

class CropForegroundImageLabel:
    def __init__(self,
        select_fn: Callable = is_positive,
        channel_indices = None,
        margin = 0,
        mode = ["constant"]
    ):
        pass 
        self.cropper = CropForeground(
            select_fn=select_fn, channel_indices=channel_indices, margin=margin
        )
        self.mode = mode
    def __call__(self, image, label=None):

        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        box_start, box_end = self.cropper.compute_bounding_box(image)
        print(box_start, box_end)
        # d[self.start_coord_key] = box_start
        # d[self.end_coord_key] = box_end
        # for key, m in self.key_iterator(d, self.mode):
            # self.push_transform(d, key, extra_info={"box_start": box_start, "box_end": box_end})
        image = self.cropper.crop_pad(img=image, box_start=box_start, box_end=box_end, mode=self.mode[0])
        if label is not None :
            if len(label.shape) == 3:
                label = np.expand_dims(label, axis=0)
            label = self.cropper.crop_pad(img=label, box_start=box_start, box_end=box_end, mode=self.mode[1])
            if len(label.shape) == 4:
                label = np.squeeze(label, axis=0)

        return image, label
        

class CropForeground():
    """
    Crop an image using a bounding box. The bounding box is generated by selecting foreground using select_fn
    at channels channel_indices. margin is added in each spatial dimension of the bounding box.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box of foreground object.
    For example:

    .. code-block:: python

        image = np.array(
            [[[0, 0, 0, 0, 0],
              [0, 1, 2, 1, 0],
              [0, 1, 3, 2, 0],
              [0, 1, 2, 1, 0],
              [0, 0, 0, 0, 0]]])  # 1x5x5, single channel 5x5 image


        def threshold_at_one(x):
            # threshold at 1
            return x > 1


        cropper = CropForeground(select_fn=threshold_at_one, margin=0)
        print(cropper(image))
        [[[2, 1],
          [3, 2],
          [2, 1]]]

    """

    def __init__(
        self,
        select_fn: Callable = is_positive,
        channel_indices = None,
        margin: Union[Sequence[int], int] = 0,
        return_coords: bool = False,
        mode: str = "constant",
        **np_kwargs,
    ) -> None:
        """
        Args:
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            return_coords: whether return the coordinates of spatial bounding box for foreground.
            k_divisible: make each spatial dimension to be divisible by k, default to 1.
                if `k_divisible` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            np_kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
                more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

        """
        self.select_fn = select_fn
        self.channel_indices = channel_indices
        self.margin = margin
        self.return_coords = return_coords
        self.mode = mode
        self.np_kwargs = np_kwargs

    def compute_bounding_box(self, img):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = generate_spatial_bounding_box(img, self.select_fn, self.channel_indices, self.margin)
        # box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        # box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        # print(box_start)
        # print(box_end)
        box_start = np.array(box_start)
        box_end = np.array(box_end)
        orig_spatial_size = box_end - box_start
        # make the spatial size divisible by `k`
        spatial_size = np.array(orig_spatial_size)
        # spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=self.k_divisible))
        # update box_start and box_end
        box_start_ = box_start - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start + spatial_size
        return box_start_, box_end_

    def crop_pad(
        self,
        img,
        box_start: np.ndarray,
        box_end: np.ndarray,
        mode = None,
    ):
        """
        Crop and pad based on the bounding box.

        """
        cropped = SpatialCrop(roi_start=box_start, roi_end=box_end)(img)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(box_end - np.asarray(img.shape[1:]), 0)
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        return BorderPad(spatial_border=pad, mode=mode or self.mode, **self.np_kwargs)(cropped)

    def __call__(self, img, mode = None):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        box_start, box_end = self.compute_bounding_box(img)
        cropped = self.crop_pad(img, box_start, box_end, mode)

        if self.return_coords:
            return cropped, box_start, box_end
        return cropped

class Random:
    def __init__(self, seed) -> None:
        self.seed = seed
        self.R = np.random.RandomState(seed)

    def do_transform(self, prob):
        ## 随机一个概率，当这个概率小于prob的时候，便去进行变换。
        prob = min(max(prob, 0.0), 1.0)
        return self.R.rand() < prob

class BorderPad:
    """
    Pad the input data by adding specified borders to every dimension.

    Args:
        spatial_border: specified size for every spatial border. Any -ve values will be set to 0. It can be 3 shapes:

            - single int number, pad all the borders with the same size.
            - length equals the length of image shape, pad every spatial dimension separately.
              for example, image shape(CHW) is [1, 4, 4], spatial_border is [2, 1],
              pad every border of H dim with 2, pad every border of W dim with 1, result shape is [1, 8, 6].
            - length equals 2 x (length of image shape), pad every border of every dimension separately.
              for example, image shape(CHW) is [1, 4, 4], spatial_border is [1, 2, 3, 4], pad top of H dim with 1,
              pad bottom of H dim with 2, pad left of W dim with 3, pad right of W dim with 4.
              the result shape is [1, 7, 11].
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    def __init__(
        self,
        spatial_border: Union[Sequence[int], int],
        mode = "constant",
        **kwargs,
    ) -> None:
        self.spatial_border = spatial_border
        self.mode = mode
        self.kwargs = kwargs

    def __call__(
        self, img, mode = None
    ):
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to `self.mode`.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

        Raises:
            ValueError: When ``self.spatial_border`` does not contain ints.
            ValueError: When ``self.spatial_border`` length is not one of
                [1, len(spatial_shape), 2*len(spatial_shape)].

        """
        spatial_shape = img.shape[1:]
        spatial_border = self.spatial_border
        if not all(isinstance(b, int) for b in spatial_border):
            raise ValueError(f"self.spatial_border must contain only ints, got {spatial_border}.")
        spatial_border = tuple(max(0, b) for b in spatial_border)

        if len(spatial_border) == 1:
            data_pad_width = [(spatial_border[0], spatial_border[0]) for _ in spatial_shape]
        elif len(spatial_border) == len(spatial_shape):
            data_pad_width = [(sp, sp) for sp in spatial_border[: len(spatial_shape)]]
        elif len(spatial_border) == len(spatial_shape) * 2:
            data_pad_width = [(spatial_border[2 * i], spatial_border[2 * i + 1]) for i in range(len(spatial_shape))]
        else:
            raise ValueError(
                f"Unsupported spatial_border length: {len(spatial_border)}, available options are "
                f"[1, len(spatial_shape)={len(spatial_shape)}, 2*len(spatial_shape)={2*len(spatial_shape)}]."
            )

        all_pad_width = [(0, 0)] + data_pad_width
        padder = Pad(all_pad_width, mode or self.mode, **self.kwargs)
        return padder(img)

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
        if self.random_state.uniform() > self.execution_probability:
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

class RandomRotate90:
    def __init__(self, random_state, execution_probability=0.2):
        self.random_state = random_state
        self.axis = (1, 2)
        self.execution_probability = execution_probability

    def __call__(self, m, label=None):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        assert m.ndim == 4, "输入必须为3d图像，第一个维度为channel"
        if self.random_state.uniform() < self.execution_probability:

            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)
            if label is not None :
                assert label.ndim == 3, "label shape 必须为三维"
                label = np.rot90(label, k, self.axis)

        return m, label

class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='reflect', order=0, execution_probability=0.2):
        if axes is None:
            axes = [[2, 1]] # 这样就是以后两个维度为平面进行旋转。 第一个维度是深度
    
        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.execution_probability = execution_probability
        self.mode = mode
        self.order = order

    def __call__(self, m, label=None):
        if self.random_state.uniform() < self.execution_probability:

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

class Elatic:
    def __init__(self, random_state, alpha=(0., 900.), sigma=(9., 13.), scale=(0.85, 1.25), 
                        order_seg=1, order_data=3, border_mode_seg="constant", 
                        border_cval_seg=0, execution_probability=0.2) -> None:
        self.random_state = random_state
        self.alpha = alpha
        self.sigma = sigma 
        self.scale = scale 
        self.order_seg = order_seg
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.execution_probability = execution_probability


    
    def _do_elastic(self, m, seg=None):
        a = self.random_state.uniform(self.alpha[0], self.alpha[1])
        s = self.random_state.uniform(self.sigma[0], self.sigma[1])

        patch_size = m.shape[1:]
        coords = create_zero_centered_coordinate_mesh(patch_size)
        coords = elastic_deform_coordinates(coords, a, s, self.random_state)
        dim = 3
        seg_result = None 
        if seg is not None:
            seg_result = np.zeros((patch_size[0], patch_size[1], patch_size[2]),
                                        dtype=np.float32)
                            
        data_result = np.zeros((m.shape[0], patch_size[0], patch_size[1], patch_size[2]),
                                dtype=np.float32)
        for d in range(dim):
        
            ctr = m.shape[d + 1] / 2. - 0.5
            coords[d] += ctr
        
        if self.scale[0] < 1:
            sc = self.random_state.uniform(self.scale[0], 1)
        else :
            sc = self.random_state.uniform(max(self.scale[0], 1), self.scale[1])
        coords = scale_coords(coords, sc)

        for channel_id in range(m.shape[0]):
            data_result[channel_id] = interpolate_img(m[channel_id], coords, self.order_data,
                                                                cval=0.0, is_seg=False)
        if seg is not None:
            
            seg_result = interpolate_img(seg, coords, self.order_seg,
                                                                    self.border_mode_seg, 
                                                                    cval=self.border_cval_seg,
                                                                    is_seg=True)
        return data_result, seg_result
    def __call__(self, m, seg=None):
        assert len(m.shape) == 4, "image dim 必须为4"
        
        if self.random_state.uniform() < self.execution_probability:
            m, seg = self._do_elastic(m, seg=seg)

        if seg is not None :
            return m, seg
        else :
            return m


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

class Normalization():
    def __init__(self, channel_wise=False):
        pass
        self.channel_wise = channel_wise

    def __call__(self, m):
        
        assert len(m.shape) == 4, "image shape err"
        if not self.channel_wise:
            m = (m - m.mean()) / m.std()
        else :
            for i, d in enumerate(m):
                
                slices = d != 0
                _sub = d[slices].mean()
                _div = d[slices].std()

                m[i][slices] = (m[i][slices] - _sub) / (_div+1e-8)

        return m

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
        if len(label.shape) == 3:
            label = np.expand_dims(label, axis=0)
        if image is None:
            image = self.image

        if not is_label:
            self.randomize(label, image)
        else :
            if len(img.shape) == 3:
                img = np.expand_dims(img, axis=0)
        results: List[np.ndarray] = []
        if self.centers is not None:
            for center in self.centers:
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
                r = cropper(img)
                if is_label:
                    if len(r.shape) == 4:
                        r = np.squeeze(r, axis=0)
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


class GammaTransformer:
    def __init__(self, random_state, gamma_range=(0.5, 2), epsilon=1e-7, per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]] = False, execution_probability=0.2) -> None:
        self.gamma_range = gamma_range
        self.epsilon = epsilon
        self.per_channel = per_channel
        self.retain_stats = retain_stats
        self.execution_probability = execution_probability
        self.random_state = random_state
    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:

            m = augment_gamma(m, gamma_range=self.gamma_range, epsilon=self.epsilon, 
                                per_channel=self.per_channel, retain_stats=self.retain_stats)
        
        return m
    
class MirrorTransform:
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, random_state, axes=(0, 1, 2), execution_probability=0.2):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, data, seg=None):
        if self.random_state.uniform() < self.execution_probability:
            
            ret_val = augment_mirroring(data, self.random_state, sample_seg=seg, axes=self.axes)
            data = ret_val[0]
            if seg is not None:
                seg = ret_val[1]
        
        return data, seg   

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

