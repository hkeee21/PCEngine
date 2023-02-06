import random
import numpy as np
import torch
import utils.kitti as kitti_utils
from datasets.transform import DataTransform

__all__ = [
    'RandomRotation', 'RandomScale', 'RandomTranslation', 'RandomFlip',
    'RandomSample', 'Crop'
]


class RandomRotation(DataTransform):
    def __init__(self, rot_range: np.ndarray, p: float = 1.0, coeff: float = 1.0):
        super().__init__()
        self.rot_range = rot_range
        self.p = p
        self.coeff = coeff

    def apply(self, inputs: dict):
        pts_rect = inputs['pts_rect']
        gt_boxes3d = inputs['gt_boxes3d']

        angle = random.uniform(self.rot_range[0], self.rot_range[1])
        pts_rect = kitti_utils.rotate_pc_along_y(pts_rect, rot_angle=angle)

        for c in gt_boxes3d.keys():
            # xyz change, hwl unchange
            gt_boxes3d[c] = kitti_utils.rotate_pc_along_y(gt_boxes3d[c],
                                                          rot_angle=angle)
            # calculate the ry after rotation
            gt_boxes3d[c][:, 6] = gt_boxes3d[c][:, 6] + angle * self.coeff
            # calculate the rotated velocity
            if gt_boxes3d[c].shape[1] > 7:
                velo = np.hstack(
                    [gt_boxes3d[c][:, 7:8], np.zeros_like(gt_boxes3d[c][:, 7:8]), gt_boxes3d[c][:, 8:9]]
                )
                velo_trans = kitti_utils.rotate_pc_along_y(velo, rot_angle=angle)
                gt_boxes3d[c][:, 7] = velo_trans[:, 0]
                gt_boxes3d[c][:, 8] = velo_trans[:, 2]

        inputs.update(pts_rect=pts_rect)
        inputs.update(gt_boxes3d=gt_boxes3d)
        return inputs


class RandomScale(DataTransform):
    def __init__(self, scale_range: np.ndarray, p: float = 1.0):
        super().__init__()
        self.scale_range = scale_range
        self.p = p

    def apply(self, inputs):
        pts_rect = inputs['pts_rect']
        gt_boxes3d = inputs['gt_boxes3d']

        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        pts_rect = pts_rect * scale

        for c in gt_boxes3d.keys():
            gt_boxes3d[c][:, 0:6] = gt_boxes3d[c][:, 0:6] * scale
            # velocity scaling
            if gt_boxes3d[c].shape[1] > 7:
                gt_boxes3d[c][:, 8:10] = gt_boxes3d[c][:, 8:10] * scale

        inputs.update(pts_rect=pts_rect)
        inputs.update(gt_boxes3d=gt_boxes3d)
        return inputs


class RandomTranslation(DataTransform):
    def __init__(self, trans_std: float, p: float = 1.0):
        super().__init__()
        self.trans_std = trans_std
        self.p = p

    def apply(self, inputs):
        pts_rect = inputs['pts_rect']
        gt_boxes3d = inputs['gt_boxes3d']

        trans = [
            random.gauss(0, self.trans_std)
            for i in range(3)
        ]

        pts_rect[:, :3] = pts_rect[:, :3] + trans

        for c in gt_boxes3d.keys():
            gt_boxes3d[c][:, 0:3] = gt_boxes3d[c][:, 0:3] + trans

        inputs.update(pts_rect=pts_rect)
        inputs.update(gt_boxes3d=gt_boxes3d)
        return inputs


class RandomFlip(DataTransform):
    def __init__(self, p: float = 0.5, flip_dim: int = 0):
        super().__init__()
        self.p = p
        self.flip_dim = flip_dim

    def apply(self, inputs):
        pts_rect = inputs['pts_rect']
        gt_boxes3d = inputs['gt_boxes3d']
        flip_dim = self.flip_dim
        
        if flip_dim == 0:
            pts_rect[:, flip_dim] = -pts_rect[:, flip_dim]
        else:
            pts_rect[:, 2] = -pts_rect[:, 2]
        for c in gt_boxes3d.keys():
            if flip_dim == 0:
                gt_boxes3d[c][:, flip_dim] = -gt_boxes3d[c][:, flip_dim]
            else:
                gt_boxes3d[c][:, 2] = -gt_boxes3d[c][:, 2]
            # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
            if flip_dim == 0:
                gt_boxes3d[c][:, 6] = np.pi - gt_boxes3d[c][:, 6]
            else:
                gt_boxes3d[c][:, 6] = - gt_boxes3d[c][:, 6]

            if gt_boxes3d[c].shape[1] > 7:
                gt_boxes3d[c][:, 7 + flip_dim] = -gt_boxes3d[c][:, 7 + flip_dim]

        inputs.update(pts_rect=pts_rect)
        inputs.update(gt_boxes3d=gt_boxes3d)
        return inputs


class RandomSample(DataTransform):
    def __init__(self,
                 max_points,
                 uniform_sample: bool = False,
                 near_threshold: float = 40.0,
                 p: float = 1.0):
        super().__init__()
        self.max_points = max_points
        self.uniform_sample = uniform_sample
        self.near_threshold = near_threshold
        self.p = p

    def apply(self, inputs):
        pts_rect = inputs['pts_rect']
        feats = inputs['feats']
        locs = inputs['locs']

        
        if self.max_points is not None and len(pts_rect) > self.max_points:
            pts_depth = pts_rect[:, 2]
            if self.uniform_sample:
                near_idxs_choice = random.sample(list(range(len(pts_rect))),
                                                 self.max_points)
            else:
                pts_near_flag = pts_depth < self.near_threshold
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = random.sample(
                    list(near_idxs), self.max_points - len(far_idxs_choice))
                choice = np.concatenate(
                    (np.array(near_idxs_choice), far_idxs_choice),
                    axis=0) if len(far_idxs_choice) > 0 else near_idxs_choice
        else:
            choice = np.arange(0, len(pts_rect), dtype=np.int32)
        random.shuffle(choice)

        feats = feats[choice, :]
        locs = locs[choice, :]
        pts_rect = pts_rect[choice, :]
        inputs.update(feats=feats, locs=locs, pts_rect=pts_rect)
        return inputs


class Crop(DataTransform):
    def __init__(self, loc_range: np.ndarray):
        self.p = 1
        self.loc_range = loc_range

    def apply(self, inputs):
        feats = inputs['feats']
        locs = inputs['locs']
        pts_rect = inputs['pts_rect']

        locs_valid_flag = np.all(np.logical_and(locs >= 0,
                                                locs < self.loc_range),
                                 axis=-1)
        feats = feats[locs_valid_flag]
        locs = locs[locs_valid_flag]
        pts_rect = pts_rect[locs_valid_flag]

        inputs.update(feats=feats, locs=locs, pts_rect=pts_rect)
        return inputs
