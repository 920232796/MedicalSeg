
import numpy as np 
from typing import Optional, Union, Sequence, List


def correct_crop_centers(
    centers: List[np.ndarray], spatial_size: Union[Sequence[int], int], label_spatial_shape: Sequence[int]
) -> List[np.ndarray]:
    """
    Utility to correct the crop center if the crop size is bigger than the image size.
    Args:
        ceters: pre-computed crop centers, will correct based on the valid region.
        spatial_size: spatial size of the ROIs to be sampled.
        label_spatial_shape: spatial shape of the original label data to compare with ROI.
    """
    spatial_size = spatial_size
    default=label_spatial_shape
    if not (np.subtract(label_spatial_shape, spatial_size) >= 0).all():
        raise ValueError("The size of the proposed random crop ROI is larger than the image size.")

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i, valid_s in enumerate(valid_start):
        # need this because np.random.randint does not work with same start and end
        if valid_s == valid_end[i]:
            valid_end[i] += 1

    for i, c in enumerate(centers):
        center_i = c
        if c < valid_start[i]:
            center_i = valid_start[i]
        if c >= valid_end[i]:
            center_i = valid_end[i] - 1
        centers[i] = center_i

    return centers

def generate_pos_neg_label_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: np.ndarray,
    bg_indices: np.ndarray,
    rand_state: Optional[np.random.RandomState] = None,
) -> List[List[np.ndarray]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]
    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.
    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.
    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    centers = []
    fg_indices, bg_indices = np.asarray(fg_indices), np.asarray(bg_indices)
    if fg_indices.size == 0 and bg_indices.size == 0:
        raise ValueError("No sampling location available.")

    if fg_indices.size == 0 or bg_indices.size == 0:
        print(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if fg_indices.size == 0 else 1

    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        center = np.unravel_index(indices_to_use[random_int], label_spatial_shape)
        # shift center to range of valid centers
        center_ori = list(center)
        centers.append(correct_crop_centers(center_ori, spatial_size, label_spatial_shape))

    return centers
