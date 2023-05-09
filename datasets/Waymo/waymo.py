import os
import numpy as np
import pickle
import torch
from script.sptensor import spTensor
from utils import classproperty
from datasets.detection_dataset import DetectionDataset
from datasets.transform import Crop, RandomSample, ToKITTICoords


class Waymo(DetectionDataset):
    def __init__(self, 
            root_dir='data/waymo',
            max_sweeps=1,
            max_points=150000,
            **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.max_sweeps = max_sweeps
        self.max_points = max_points
        info_file = os.path.join(root_dir, f'infos_waymo_{max_sweeps}.pkl')
        with open(info_file, 'rb') as f:
            infos = pickle.load(f)
        self.infos = infos
        self.croper = Crop(self.loc_range)
        self.augmentor = ToKITTICoords()
        self.subsampler = RandomSample(self.max_points)

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        points = self.get_lidar_with_sweeps(index, self.max_sweeps)
        points = self.augmentor(dict(points=points, pts_rect=points))['points']

        # print("points:", points.shape[0])

        locs = np.floor((points[:, :3] - self.pc_area_scope[:, 0]) /
                        self.voxel_size).astype('int32')
        # crop
        input_dict = dict(feats=points, locs=locs, pts_rect=points)
        updated_input_dict = self.croper(input_dict)
        points = updated_input_dict['feats']
        locs = updated_input_dict['locs']

        features_with_cnt = np.concatenate(
            [np.ones_like(points[:, :1]), points], axis=-1)
        flatten = torch.sparse.FloatTensor(
            torch.from_numpy(locs).t().long(),
            torch.from_numpy(features_with_cnt)).coalesce()
        locs = flatten.indices().t().int().numpy()
        features_with_cnt = flatten.values().numpy()
        feats = features_with_cnt[:, 1:] / features_with_cnt[:, :1]
        input_dict = dict(feats=feats, locs=locs, pts_rect=feats)
        updated_input_dict = self.subsampler(input_dict)
        locs = updated_input_dict['locs']
        feats = updated_input_dict['feats']
        # subsample
        return {'pts_input': spTensor(feats, locs, buffer=None, coords_min=None, coords_max=None)}

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) &
                     (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = os.path.join('/home/nfs_data/yangshang19/smr/torchsparse-data', sweep_info['lidar_path'])
        with open(lidar_path, 'rb') as f:
          point_dict = pickle.load(f)
        points_sweep = np.hstack([point_dict['lidars']['points_xyz'], point_dict['lidars']['points_feature']])
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones(
            (1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T


    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = os.path.join('/home/nfs_data/yangshang19/smr/torchsparse-data', info['lidar_path'])
        # logger.info(lidar_path)
        with open(lidar_path, 'rb') as f:
            point_dict = pickle.load(f)
        points = np.hstack([point_dict['lidars']['points_xyz'], point_dict['lidars']['points_feature']])

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']),
                                  max_sweeps - 1,
                                  replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        points[:, 3] = np.tanh(points[:, 3])
        return points


    @classproperty
    def classes(self):
        return ['vehicle', 'pedestrian', 'cyclist']

    @classproperty
    def voxel_size(self):
        return np.array([.1, .15, .1], 'float32')

    @classproperty
    def pc_area_scope(self):
        return np.array([[-75.2, 75.2], [-4, 2], [-75.2, 75.2]], 'float32')

    @classproperty
    def mean_size(self):
        return {
            self.classes[0]: np.array([1.7, 2.1, 4.7], 'float32'),
            self.classes[1]: np.array([1.73, 0.86, 0.91], 'float32'),
            self.classes[2]: np.array([1.78, 0.84, 1.78], 'float32')
        }

    @classproperty
    def mean_center_z(self):
        return {
            self.classes[0]: 0.85,
            self.classes[1]: 0.865,
            self.classes[2]: 0.89
        }
