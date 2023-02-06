import os
import numpy as np
import pickle
import torch
from script.sptensor import spTensor
from utils import classproperty
from .nuscenes import NuScenes
from script.utils import sparse_quantize

__all__ = ['NuScenesLiDARSeg']


class NuScenesLiDARSeg(NuScenes):
    def __init__(self, 
            root_dir='data/nuscenes',
            max_points=60000,
            **kwargs):
        super().__init__(root_dir, max_points, **kwargs)
        self.voxel_size = (0.1, 0.1, 0.1)
        

    def __getitem__(self, index):
        points = self.get_lidar_with_sweeps(index, self.max_sweeps)
        points[:, :3] -= points[:, :3].min(0)
        coords, inds = sparse_quantize(points[:, :3], self.voxel_size, return_index=True)
        # subsample
        return {'pts_input': spTensor(points[inds], coords, buffer=None)}

