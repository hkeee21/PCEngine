import numpy as np
from abc import abstractmethod
import torch
# from torchsparse import SparseTensor
# from torchsparse.utils.collate import sparse_collate
from script.sptensor import spTensor
from script.utils import sparse_collate

from utils import classproperty

__all__ = ['DetectionDataset']


class DetectionDataset:
    def __init__(self):

        self.reduce_by_range = True

    def get_image_shape(self, idx):
        raise NotImplementedError

    def get_lidar(self, idx):
        raise NotImplementedError

    def get_calib(self, idx):
        raise NotImplementedError

    def get_label(self, idx):
        raise NotImplementedError

    def get_road_plane(self, idx):
        raise NotImplementedError

    @classproperty
    def voxel_size(self):
        return np.array([0.05, 0.05, 0.05], 'float32')

    @classproperty
    def loc_range(self):
        return np.array(
            np.round((self.pc_area_scope[:, 1] - self.pc_area_scope[:, 0]) /
                     self.voxel_size), 'int32')

    @classproperty
    def loc_min(self):
        return dict([
            (c,
             np.array(
                 np.round(
                     (self.pc_area_scope[c][:, 0] - self.pc_area_scope[:, 0]) /
                     self.voxel_size), 'int32')) for c in self.classes
        ])

    @classproperty
    def loc_max(self):
        return dict([
            (c,
             np.array(
                 np.round(
                     (self.pc_area_scope[c][:, 1] - self.pc_area_scope[:, 0]) /
                     self.voxel_size), 'int32')) for c in self.classes
        ])

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def pc_area_scope(self):
        raise NotImplementedError


    @property
    def mean_size(self):
        raise NotImplementedError

    @property
    def mean_center_z(self):
        raise NotImplementedError

    #@property
    #def gt_aug_enabled(self):
    #    raise NotImplementedError

    @property
    def gt_aug_num_range(self):
        raise NotImplementedError

    @property
    def gt_aug_hard_ratio(self):
        raise NotImplementedError

    def collate_fn(self, batch):
        ans_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], spTensor):
                ans_dict[key] = sparse_collate(
                    [sample[key] for sample in batch])
            elif isinstance(batch[0][key], torch.Tensor):
                ans_dict[key] = torch.cat(
                    [sample[key][np.newaxis, ...] for sample in batch], axis=0)
            elif isinstance(batch[0][key], dict):
                ans_dict[key] = self.collate_fn(
                    [sample[key] for sample in batch])
            else:
                ans_dict[key] = [sample[key] for sample in batch]
        return ans_dict
