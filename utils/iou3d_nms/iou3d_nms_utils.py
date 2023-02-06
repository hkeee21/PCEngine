import torch
import numpy as np

from torch.utils.cpp_extension import load
import os

pd = os.path.dirname(__file__)
load_list = [
    os.path.join(pd, 'src', 'iou3d_cpu.cpp'),
    os.path.join(pd, 'src', 'iou3d_nms_kernel.cu'),
]
iou3d_nms_cuda = load('iou3d_nms', load_list, verbose=False)


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]
    Returns:
    """
    boxes_a, is_numpy = check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(),
                                     boxes_b.contiguous(), ans_iou)
    return ans_iou.numpy() if is_numpy else ans_iou
