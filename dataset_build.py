from typing import Callable
import numpy as np
import torch
import torch.optim
from torch import nn
from configs.config import Config
from script.utils import sparse_collate_fn

def make_dataset(configs: Config, n_sample=-1):
    if configs.dataset.name == 'kitti':
        from datasets.KITTI import KITTI
        dataset = KITTI(root=configs.dataset.root,
                        max_points=configs.dataset.max_points)
    elif configs.dataset.name == 'nuscenes':
        from datasets.NuScenes import NuScenes
        voxel_size = configs.dataset.get(
            'voxel_size', [0.1, 0.2, 0.1]
        )
        pc_area_scope = configs.dataset.get(
            'pc_area_scope', [[-51.2, 51.2], [-3, 5], [-51.2, 51.2]]
        )
        rotation_bound = configs.dataset.get(
            'rotation_bound', [-0.3925, 0.3925]
        )
        scale_bound = configs.dataset.get(
            'scale_bound', [0.95, 1.05]
        )
        translation_std = configs.dataset.get(
            'translation_std', 0.5
        )
        cbgs = configs.dataset.get('cbgs', False)
        dataset = NuScenes(
            configs.dataset.root,
            max_sweeps=configs.dataset.max_sweeps,
            train_max_points=configs.dataset.train_max_points,
            max_points=configs.dataset.val_max_points,
            val_max_points=configs.dataset.val_max_points,
            voxel_size=voxel_size,
            pc_area_scope=pc_area_scope,
            rotation_bound=rotation_bound,
            scale_bound=scale_bound,
            translation_std=translation_std,
            cbgs=cbgs
        )
        if n_sample > 0 and n_sample < len(dataset):
            dataset.infos = dataset.infos[:n_sample]
    elif configs.dataset.name == 'waymo':
        from datasets.Waymo import Waymo
        voxel_size = configs.dataset.get(
            'voxel_size', [0.1, 0.15, 0.1]
        )
        pc_area_scope = configs.dataset.get(
            'pc_area_scope', [[-75.2, 75.2], [-4, 2], [-75.2, 75.2]]
        )
        rotation_bound = configs.dataset.get(
            'rotation_bound', [-0.78539816, 0.78539816]
        )
        scale_bound = configs.dataset.get(
            'scale_bound', [0.95, 1.05]
        )
        translation_std = configs.dataset.get(
            'translation_std', 0.0
        )
        sample_stride = configs.dataset.get(
            'sample_stride', 1
        )
        dataset = Waymo(
            configs.dataset.root,
            max_sweeps=configs.dataset.max_sweeps,
            train_max_points=configs.dataset.train_max_points,
            val_max_points=configs.dataset.val_max_points,
            max_points=configs.dataset.val_max_points,
            voxel_size=voxel_size,
            pc_area_scope=pc_area_scope,
            rotation_bound=rotation_bound,
            scale_bound=scale_bound,
            translation_std=translation_std,
            sample_stride=sample_stride
        )
        if n_sample > 0 and n_sample < len(dataset):
            dataset.infos = dataset.infos[:n_sample]
    elif configs.dataset.name == 'nuscenes-lidarseg':
        from datasets.NuScenes import NuScenesLiDARSeg
        dataset = NuScenesLiDARSeg(
            root_dir=configs.dataset.root,
            max_points=configs.dataset.max_points,
            max_sweeps=configs.dataset.get('max_sweeps', 1)
        )
        if n_sample > 0 and n_sample < len(dataset):
            dataset.infos = dataset.infos[:n_sample]
    elif configs.dataset.name == 'semantic-kitti':
        from datasets.SemanticKITTI import SemanticKITTI
        dataset = SemanticKITTI(
            root_dir=configs.dataset.root,
            max_points=configs.dataset.max_points,
            subset=False
        )
        if n_sample > 0 and n_sample < len(dataset):
            dataset.fns = dataset.fns[:n_sample]
    else:
        raise NotImplementedError(configs.dataset.name)
    return dataset


if __name__ == '__main__': 
    
    configs = Config()

    '''
    ##### nuscenes #####

    configs.load('configs/default/nuscenes/centerpoint_default.yaml', recursive=True)
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/nuscenes'
    dataset = make_dataset(configs, -1)

    print('Dataset NuScenes Checked. Length: %d' % (dataset.__len__()))

    ###### nuscenes-lidarseg #####

    configs.load('configs/default/nuscenes_lidarseg/default.yaml', recursive=True)
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/nuscenes'
    dataset = make_dataset(configs, -1)

    print('Dataset NuScenes-Lidarseg Checked. Length: %d' % (dataset.__len__()))'''

    ###### semantic-kitti #####

    configs.load('configs/default/semantic_kitti/default.yaml', recursive=True)
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/semantic-kitti'
    dataset = make_dataset(configs, -1)
    dataflow = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    # num_workers=configs.workers_per_gpu,
                    pin_memory=False,
                    collate_fn=sparse_collate_fn,
                    shuffle=False
                )

    nnz = 0
    sparsity = list()
    for i, batch in enumerate(dataflow):
        temp_nnz = batch["pts_input"].C.shape[0]
        nnz += temp_nnz

        x_range = torch.max(batch["pts_input"].C[:, 1], dim=0)[0] - torch.min(batch["pts_input"].C[:, 1], dim=0)[0]
        y_range = torch.max(batch["pts_input"].C[:, 2], dim=0)[0] - torch.min(batch["pts_input"].C[:, 2], dim=0)[0]
        z_range = torch.max(batch["pts_input"].C[:, 3], dim=0)[0] - torch.min(batch["pts_input"].C[:, 3], dim=0)[0]

        # volume = x_range * y_range * z_range

        sparsity.append(temp_nnz / x_range / y_range / z_range * 100)

        print("%d-th sample, nnz: %d, sparsity: %.4f" % (i, temp_nnz, (temp_nnz / x_range / y_range / z_range * 100)))

    print('Dataset SemanticKITTI Checked. Length: %d' % (dataset.__len__()))

    print('Dataset SemanticKITTI Average NNZs: %d' % (nnz //dataset.__len__()))

    print('Dataset SemanticKITTI Average Density: %.4f' % (np.mean(sparsity)))

    '''
    ###### waymo #####

    configs.load('configs/default/waymo/centerpoint_default.yaml', recursive=True)
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/waymo'
    dataset = make_dataset(configs, -1)

    print('Dataset Waymo Checked. Length: %d' % (dataset.__len__()))'''
