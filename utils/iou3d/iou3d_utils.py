import os
import numpy as np
import torch
from torch.utils.cpp_extension import load

from .. import kitti

pd = os.path.dirname(__file__)
load_list = [
    os.path.join(pd, 'src', 'iou3d.cpp'),
    os.path.join(pd, 'src', 'iou3d_kernel.cu')
]
iou3d_cuda = load('iou3d_cuda', load_list, verbose=False)


def box_transform_for_iou_calculation(_bboxes):
    bboxes = _bboxes[:, :8].clone()
    bboxes[:, -1] = _bboxes[:, -1]
    bboxes[:, [4, 5]] = torch.where(
        (torch.abs(bboxes[:, -2] % np.pi - np.pi / 2) <
            np.pi / 4).unsqueeze(-1), bboxes[:, [4, 5]],
        bboxes[:, [5, 4]])
    bboxes[:, -2] = np.pi / 2.
    return bboxes


def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 6)
    :param boxes_b: (N, 6)
    :return:
        ans_iou: (M, N)
    """

    boxes_a_bev = kitti.boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = kitti.boxes3d_to_bev_torch(boxes_b)

    ans_iou = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a_bev.contiguous(),
                                 boxes_b_bev.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 8) [x, y, z, h, w, l, ry, batch_id]
    :param boxes_b: (M, 8) [x, y, z, h, w, l, ry, batch_id]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_3d = kitti.boxes3d_to_3d_torch(boxes_a)
    boxes_b_3d = kitti.boxes3d_to_3d_torch(boxes_b)

    # 3d overlap
    iou3d = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_iou_3d_gpu(boxes_a_3d.contiguous(),
                                boxes_b_3d.contiguous(), iou3d)
    return iou3d


def correspond_boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 8) [x, y, z, h, w, l, ry, batch_id]
    :param boxes_b: (N, 8) [x, y, z, h, w, l, ry, batch_id]
    :return:
        ans_iou: (N)
    """
    boxes_a_3d = kitti.boxes3d_to_3d_torch(boxes_a)
    boxes_b_3d = kitti.boxes3d_to_3d_torch(boxes_b)

    # 3d overlap
    iou3d = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], ))).zero_()  # (N)
    iou3d_cuda.correspond_boxes_iou_3d_gpu(boxes_a_3d.contiguous(),
                                boxes_b_3d.contiguous(), iou3d)
    return iou3d


def nms_gpu_with_threshold(boxes, scores, score_thresh, 
                           nms_thresh, before_nms_max, after_nms_max, ious=None):
    # Note: ious in [-1, 1]
    keep_idx = torch.arange(len(boxes), device=boxes.device)
    valid_flag = scores >= score_thresh
    cls_preds = scores[valid_flag]
    box_preds = boxes[valid_flag]
    keep_idx = keep_idx[valid_flag]
    
    if len(keep_idx) == 0:
        return keep_idx
    
    if ious is not None:
        cls_preds *= torch.pow(ious[valid_flag], 2)

    if cls_preds.shape[0] > before_nms_max:
        _, selected = torch.topk(cls_preds, before_nms_max, largest=True)
        cls_preds = cls_preds[selected]
        box_preds = box_preds[selected]
        keep_idx = keep_idx[selected]
    
    selected = nms_gpu(
        box_preds[:, [0, 1, 2, 3, 4, 5, 6, -1]],
        cls_preds,
        nms_thresh
    )
    selected = selected[:after_nms_max]
    keep_idx = keep_idx[selected]
    if ious is None:
        return keep_idx
    else:
        return keep_idx, cls_preds[selected]



def multi_class_nms_gpu_with_threshold(boxes, scores, score_thresh, 
                                       nms_thresh, before_nms_max, after_nms_max, ious=None):
    # Note: ious in [-1, 1]
    pred_scores, pred_labels, pred_boxes = [], [], []
    # class separate
    for k in range(scores.shape[1]):
        valid_flag = scores[:, k] >= score_thresh
        cls_preds = scores[:, k][valid_flag]
        if len(cls_preds) == 0:
            valid_flag[scores[:, k].argmax()] = 1
        cls_preds = scores[:, k][valid_flag]
        box_preds = boxes[valid_flag]
        label_preds = torch.ones_like(cls_preds) * k
        
        if len(box_preds) == 0:
            continue

        if ious is not None:
            cls_preds *= torch.pow(ious[valid_flag], 2)

        if cls_preds.shape[0] > before_nms_max:
            _, selected = torch.topk(cls_preds, before_nms_max, largest=True)
            cls_preds = cls_preds[selected]
            box_preds = box_preds[selected]
            label_preds = label_preds[selected]

        selected = nms_gpu(
            box_preds[:, [0, 1, 2, 3, 4, 5, 6, -1]],
            cls_preds,
            nms_thresh
        )
        selected = selected[:after_nms_max]
        cls_preds = cls_preds[selected]
        box_preds = box_preds[selected]
        label_preds = label_preds[selected]
        
        pred_scores.append(cls_preds)
        pred_labels.append(label_preds)
        pred_boxes.append(box_preds)
    
    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)
        
    return pred_scores, pred_labels, pred_boxes


def nms_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 8) [x, y, z, h, w, l, ry, batch_id]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = kitti.boxes3d_to_bev_torch(boxes)
    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


def nms_3d_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 7) [x1, y1, x2, y2, ry, x3, y3]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_3d_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


def nms_normal_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()
