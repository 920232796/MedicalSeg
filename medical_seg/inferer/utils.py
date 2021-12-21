
from typing import Callable, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import math
import numpy as np

from medical_seg.utils import BlendMode, PytorchPadMode
from medical_seg.utils import fall_back_tuple, ensure_tuple_rep, ensure_tuple_size, first

from medical_seg.networks.layers.simplelayers import GaussianFilter

def get_valid_patch_size(image_size: Sequence[int], patch_size: Union[Sequence[int], int]) -> Tuple[int, ...]:
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))

def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
) -> List[Tuple[slice, ...]]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval

    Returns:
        a list of slice objects defining each patch

    """
    num_spatial_dims = len(image_size)
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    slices = [tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]
    return slices

def compute_importance_map(
    patch_size: Tuple[int, ...],
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    device: Union[torch.device, int, str] = "cpu",
) -> torch.Tensor:
    """Get importance map for different weight modes.

    Args:
        patch_size: Size of the required importance map. This should be either H, W [,D].
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: Sigma_scale to calculate sigma for each dimension
            (sigma = sigma_scale * dim_size). Used for gaussian mode only.
        device: Device to put importance map on.

    Raises:
        ValueError: When ``mode`` is not one of ["constant", "gaussian"].

    Returns:
        Tensor of size patch_size.

    """
    mode = BlendMode(mode)
    device = torch.device(device)
    if mode == BlendMode.CONSTANT:
        importance_map = torch.ones(patch_size, device=device).float()
    elif mode == BlendMode.GAUSSIAN:
        center_coords = [i // 2 for i in patch_size]
        sigma_scale = ensure_tuple_rep(sigma_scale, len(patch_size))
        sigmas = [i * sigma_s for i, sigma_s in zip(patch_size, sigma_scale)]

        importance_map = torch.zeros(patch_size, device=device)
        importance_map[tuple(center_coords)] = 1
        pt_gaussian = GaussianFilter(len(patch_size), sigmas).to(device=device, dtype=torch.float)
        importance_map = pt_gaussian(importance_map.unsqueeze(0).unsqueeze(0))
        importance_map = importance_map.squeeze(0).squeeze(0)
        importance_map = importance_map / torch.max(importance_map)
        importance_map = importance_map.float()

        # importance_map cannot be 0, otherwise we may end up with nans!
        min_non_zero = importance_map[importance_map != 0].min().item()
        importance_map = torch.clamp(importance_map, min=min_non_zero)
    else:
        raise ValueError(
            f"Unsupported mode: {mode}, available options are [{BlendMode.CONSTANT}, {BlendMode.CONSTANT}]."
        )

    return importance_map


def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    pred_type: str ="forward",
    do_mirror=False,
    mirror_axes=None,
) -> torch.Tensor:
    """
    """

    num_spatial_dims = len(inputs.shape) - 2
    assert 0 <= overlap < 1, "overlap must be >= 0 and < 1."

    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    # print("device is " + str(device))
    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    # print(slices)
    num_win = len(slices)  # number of windows per image
    # print("num win is " + str(num_win))
    total_slices = num_win * batch_size  # total number of windows
    # print("total slices is " + str(total_slices))
    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )
    # print("importance map is " + str(importance_map.shape))

    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False

    index = 0

    for slice_g in range(0, total_slices, sw_batch_size):
        index += 1
        # print("index is " + str(index))
        # print("slice g is " + str(slice_g))
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        # print(unravel_slice)

        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        # print("data is " + str(window_data.shape))
        if pred_type == "forward":
            if not do_mirror:
                seg_prob = predictor(window_data).to(device)  # batched patch segmentation
            else :
                mirror_idx = 8
                if mirror_axes is None:
                    print("do_mirror，则必须传入mirror_axes")
                    import os
                    os._exit(0)
                num_results = 2 ** len(mirror_axes)
                for m in range(mirror_idx):
                    if m == 0:
                        pred = predictor(window_data).to(device)
                        seg_prob = 1 / num_results * pred

                    if m == 1 and (2 in mirror_axes):
                        pred = predictor(torch.flip(window_data, (4, )))
                        seg_prob += 1 / num_results * torch.flip(pred, (4,))

                    if m == 2 and (1 in mirror_axes):
                        pred = predictor(torch.flip(window_data, (3, )))
                        seg_prob += 1 / num_results * torch.flip(pred, (3,))

                    if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                        pred = predictor(torch.flip(window_data, (4, 3)))
                        seg_prob += 1 / num_results * torch.flip(pred, (4, 3))

                    if m == 4 and (0 in mirror_axes):
                        pred = predictor(torch.flip(window_data, (2, )))
                        seg_prob += 1 / num_results * torch.flip(pred, (2,))

                    if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                        pred = predictor(torch.flip(window_data, (4, 2)))
                        seg_prob += 1 / num_results * torch.flip(pred, (4, 2))

                    if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                        pred = predictor(torch.flip(window_data, (3, 2)))
                        seg_prob += 1 / num_results * torch.flip(pred, (3, 2))

                    if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                        pred = predictor(torch.flip(window_data, (4, 3, 2)))
                        seg_prob += 1 / num_results * torch.flip(pred, (4, 3, 2))
                        

        elif pred_type == "uncer":
            seg_prob = predictor.uncer(window_data).to(device)

        else :
            raise Exception(f"no support pred_type: {pred_type}")

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            count_map[original_idx] += importance_map

    # account for any overlapping sections
    output_image = output_image / count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    return output_image[final_slicing]


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)