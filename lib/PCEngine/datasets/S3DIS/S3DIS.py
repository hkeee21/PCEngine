import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from ...script.utils import sparse_quantize
import torch
from ...script.sptensor import spTensor


class S3DISDataset(Dataset):
    def __init__(self, data_root='trainval_fullarea', num_point=4096, test_area=1, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size # 1.0
        self.transform = transform
        rooms = sorted(os.listdir(data_root))  #   data_root = 'data/s3dis/stanford_indoor3d/'
        rooms = [room for room in rooms if 'Area_' in room] # 'Area_1_WC_1.npy' # 'Area_1_conferenceRoom_1.npy'

        rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for room_name in rooms_split:
            
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)
            points, labels = room_data[:, 0:6], room_data[:, 6] 
           
            tmp, _ = np.histogram(labels, range(14))
          
            labelweights += tmp 
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3] 
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
       
       
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point) 
        room_idxs = [] 
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        # print("Totally {} samples in the set.".format(len(self.room_idxs)))

    def __getitem__(self, idx):

        np.random.seed(idx)
        
        room_idx = self.room_idxs[idx]  #
        points = self.room_points[room_idx]   # N * 6 
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True): 
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > self.num_point:
                break
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True) #

        # normalize
        selected_points = points[selected_point_idxs, :] 
        current_points = np.zeros((self.num_point, 9))
        current_points[:, 6] = (selected_points[:, 0] - self.room_coord_min[room_idx][0]) / \
            (self.room_coord_max[room_idx][0] - self.room_coord_min[room_idx][0])
        current_points[:, 7] = (selected_points[:, 1] - self.room_coord_min[room_idx][1]) / \
            (self.room_coord_max[room_idx][1] - self.room_coord_min[room_idx][1])
        current_points[:, 8] = (selected_points[:, 2] - self.room_coord_min[room_idx][2]) / \
            (self.room_coord_max[room_idx][2] - self.room_coord_min[room_idx][2])
        selected_points[:, 3:6] /= 255.0 
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        
        coords, feats = current_points[:, 6:9], current_points[:, 2:6]
        coords -= np.min(coords, axis=0, keepdims=True)
        coords, inds = sparse_quantize(coords, voxel_size=0.005, return_index=True)
        coords = torch.as_tensor(coords, dtype=torch.int)
        feats = torch.as_tensor(feats[inds], dtype=torch.float)
        label = torch.as_tensor(current_labels[inds], dtype=torch.int)
        input = spTensor(coords=coords, feats=feats, buffer=None, coords_max=None, coords_min=None)

        return {'input': input, 'label': label}

    def __len__(self):
        return len(self.room_idxs)