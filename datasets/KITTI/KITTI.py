from os import pathsep
import numpy as np
from script.utils import sparse_quantize
import torch
from script.sptensor import spTensor

class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, path, size, transform=None):

        self.path = path
        self.size = size
        
        files = dict()
        for i in range(size): # size < 100
            if i < 10:
                file = (path + '00000%s.bin' % i)
            elif i < 100:
                file = (path + '0000%s.bin' % i)
            elif i < 1000:
                file = (path + '000%s.bin' % i)
            elif i < 10000:
                file = (path + '00%s.bin' % i)
            data = np.fromfile(str(file), dtype=np.float32).reshape(-1, 4)
            files[i] = data
        self.files = files

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        data = self.files[idx]
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)

        data = (data - data_min) / (data_max - data_min)
        coord = data[:, :3]
        feat = data
        # Get coords
        coords, inds = sparse_quantize(coord, voxel_size=0.002, return_index=True)
        # Use color or other features if available
        coords = torch.as_tensor(coords, dtype=torch.int)
        feats = torch.as_tensor(feat[inds], dtype=torch.float)
        input = spTensor(coords=coords, feats=feats, buffer=None)

        return {'input': input}