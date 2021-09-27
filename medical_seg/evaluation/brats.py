

from medpy.metric import dc, hd95
import numpy as np 


def compute_BraTS_dice(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    :param ref:
    :param gt:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 1
        else:
            return 0
    else:
        return dc(pred, ref)

def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref, (1, 1, 1, 1))


def evaluate_BraTS_case(arr: np.ndarray, arr_gt: np.ndarray):
    """
    attempting to reimplement the brats evaluation scheme
    assumes edema=1, non_enh=2, enh=3
    :param arr:
    :param arr_gt:
    :return:
    """
    # whole tumor
    mask_gt = (arr_gt != 0).astype(int)
    mask_pred = (arr != 0).astype(int)
    dc_whole = compute_BraTS_dice(mask_gt, mask_pred)
    hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # tumor core
    mask_gt = (arr_gt > 1).astype(int)
    mask_pred = (arr > 1).astype(int)
    dc_core = compute_BraTS_dice(mask_gt, mask_pred)
    hd95_core = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # enhancing
    mask_gt = (arr_gt == 3).astype(int)
    mask_pred = (arr == 3).astype(int)
    dc_enh = compute_BraTS_dice(mask_gt, mask_pred)
    hd95_enh = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return {"dc_whole":dc_whole, "dc_core": dc_core, "dc_enh": dc_enh, "hd95_whole": hd95_whole,
             "hd95_core": hd95_core, "hd95_enh": hd95_enh}