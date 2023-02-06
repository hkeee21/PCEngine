import os
import numpy as np
import pickle
import torch
from script.sptensor import spTensor
from utils import classproperty
from datasets.detection_dataset import DetectionDataset
from script.utils import sparse_quantize

__all__ = ['SemanticKITTI']


class SemanticKITTI(DetectionDataset):
    def __init__(self, 
            root_dir='data/semantic-kitti',
            max_points=60000,
            **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.fns = sorted(os.listdir(os.path.join(root_dir, '08', 'velodyne')))
        self.voxel_size = (0.05, 0.05, 0.05)


    def __len__(self):
        return len(self.fns)
        

    def __getitem__(self, index):
        fn = os.path.join(self.root_dir, '08', 'velodyne', self.fns[index])
        points = np.fromfile(fn, dtype=np.float32).reshape(-1, 4)
        # print("points:", points.shape[0])
        points[:, :3] -= points[:, :3].min(0)
        coords, inds = sparse_quantize(points[:, :3], self.voxel_size, return_index=True)
        # subsample
        return {'pts_input': spTensor(points[inds], coords, buffer=None)}


