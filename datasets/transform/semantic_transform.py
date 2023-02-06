import pickle
import random

import utils.kitti as kitti_utils
import numpy as np
import torch
from utils.roipool3d import roipool3d_utils
from utils.iou3d_nms import iou3d_nms_utils
import utils.object3d as object3d

from .transform import DataTransform

__all__ = ['RandomGTAugment', 'NuscenesGTAugment', 'WaymoGTAugment']


class RandomGTAugment(DataTransform):
    def __init__(self,
                 gt_database: dict,
                 gt_augment_num_range: dict,
                 gt_augment_hard_ratio: dict,
                 pc_area_scope: np.ndarray,
                 p: float = 1.0):
        super().__init__()
        self.gt_database = gt_database
        self.gt_augment_num_range = gt_augment_num_range
        self.gt_augment_hard_ratio = gt_augment_hard_ratio
        self.pc_area_scope = pc_area_scope

        self.p = p

    def get_valid_flag(self, pts_rect):
        x_range, y_range, z_range = self.pc_area_scope
        pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
        range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
        return range_flag

    def __call__(self, inputs):
        if self.p < 1.:
            cur_p = random.uniform(0, 1.)
            if cur_p > self.p:
                return inputs

        road_plane = inputs['road_plane']
        pts_rect = inputs['pts_rect']
        pts_intensity = inputs['pts_intensity']
        gt_boxes3d = inputs['gt_boxes'] if 'gt_boxes' in inputs else inputs['gt_boxes3d']

        cur_gt_boxes3d = np.concatenate(list(gt_boxes3d.values()), axis=0)
        # avoid too nearby boxes
        #cur_gt_boxes3d[:, 4] += 0.5
        #cur_gt_boxes3d[:, 5] += 0.5
        cur_gt_corners = kitti_utils.boxes3d_to_corners3d(cur_gt_boxes3d)

        new_pts_list, new_pts_intensity_list = [], []
        src_pts_flag = np.ones(pts_rect.shape[0], dtype=np.int32)

        for c in gt_boxes3d.keys():
            gt_database = self.gt_database[c]
            extra_gt_num = random.randint(*self.gt_augment_num_range[c])
            try_times = 100
            cnt = 0
            extra_gt_boxes3d_list = []

            while try_times > 0:
                if cnt > extra_gt_num:
                    break

                try_times -= 1
                if self.gt_augment_hard_ratio[c] is not None:
                    p = random.random()
                    if p > self.gt_augment_hard_ratio[c]:
                        # use easy sample
                        rand_idx = random.randint(0, len(gt_database[0]) - 1)
                        new_gt_dict = gt_database[0][rand_idx]
                    else:
                        # use hard sample
                        rand_idx = random.randint(0, len(gt_database[1]) - 1)
                        new_gt_dict = gt_database[1][rand_idx]
                else:
                    rand_idx = random.randint(0, gt_database.__len__() - 1)
                    new_gt_dict = gt_database[rand_idx]

                new_gt_box3d = new_gt_dict['gt_box3d'].copy()
                new_gt_points = new_gt_dict['points'].copy()
                new_gt_intensity = new_gt_dict['intensity'].copy()
                center = new_gt_box3d[0:3]

                if self.get_valid_flag(center[np.newaxis])[0] is False:
                    continue

                if new_gt_points.__len__() < 5:  # too few points
                    continue

                # put it on the road plane
                ra, rb, rc, rd = road_plane
                cur_height = (-rd - ra * center[0] - rc * center[2]) / rb
                move_height = new_gt_box3d[1] - cur_height
                new_gt_box3d[1] -= move_height
                new_gt_points[:, 1] -= move_height

                new_enlarged_box3d = new_gt_box3d.copy()
                # enlarge new added box to avoid too nearby boxes
                new_enlarged_box3d[4] += 0.5
                new_enlarged_box3d[5] += 0.5

                cnt += 1
                new_corners = kitti_utils.boxes3d_to_corners3d(
                    new_enlarged_box3d[np.newaxis])
                if cur_gt_corners.shape[0] > 0:
                    _, iou_bev = kitti_utils.get_iou3d(new_corners,
                                                       cur_gt_corners,
                                                       need_bev=True)
                    valid_flag = iou_bev.max() < 1e-8
                    if not valid_flag:
                        continue

                enlarged_box3d = new_gt_box3d.copy()
                enlarged_box3d[3] += 2  # remove the points below the object
                boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(
                    torch.from_numpy(pts_rect),
                    torch.from_numpy(enlarged_box3d[np.newaxis]))
                pt_mask_flag = (boxes_pts_mask_list[0].numpy() == 1)
                src_pts_flag[
                    pt_mask_flag] = 0  # remove the original points which are inside the new box

                new_pts_list.append(new_gt_points)
                new_pts_intensity_list.append(new_gt_intensity)
                cur_gt_boxes3d = np.concatenate(
                    (cur_gt_boxes3d, new_enlarged_box3d[np.newaxis]), axis=0)
                cur_gt_corners = np.concatenate((cur_gt_corners, new_corners),
                                                axis=0)
                extra_gt_boxes3d_list.append(new_gt_box3d[np.newaxis])
            gt_boxes3d[c] = np.concatenate([gt_boxes3d[c]] +
                                           extra_gt_boxes3d_list,
                                           axis=0)
        if new_pts_list.__len__() == 0:
            inputs.update(pts_rect=pts_rect)
            inputs.update(pts_intensity=pts_intensity)
            inputs.update(gt_boxes3d=gt_boxes3d)
            return inputs

        # remove original points and add new points
        pts_rect = pts_rect[src_pts_flag == 1]
        pts_intensity = pts_intensity[src_pts_flag == 1]
        new_pts_rect = np.concatenate(new_pts_list, axis=0)
        new_pts_intensity = np.concatenate(new_pts_intensity_list, axis=0)
        pts_rect = np.concatenate((pts_rect, new_pts_rect), axis=0)
        pts_intensity = np.concatenate((pts_intensity, new_pts_intensity),
                                       axis=0)

        inputs.update(pts_rect=pts_rect)
        inputs.update(pts_intensity=pts_intensity)
        inputs.update(gt_boxes3d=gt_boxes3d)
        return inputs



class GTAugmentTemplate(DataTransform):
    def __init__(
        self, 
        root_path, 
        class_names,
        p=1,
        **kwargs
    ):

        super().__init__()
        self.root_path = root_path
        self.class_names = class_names
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []


        for db_info_path in kwargs['db_info_path']:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [
                    self.db_infos[cur_class].extend(infos[cur_class])
                    for cur_class in class_names
                ]

        FILTER_BY_MIN_POINTS = kwargs['filter_cfg']
        self.db_infos = self.filter_by_min_points(self.db_infos,
                                                  FILTER_BY_MIN_POINTS)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = True

        SAMPLE_GROUPS = kwargs['sample_group_cfg']
        for class_name, sample_num in SAMPLE_GROUPS:
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

        self.num_point_features = kwargs.get('num_point_features', 5)
        self.box_dimensions = kwargs.get('box_dimensions', 7)

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name, min_num in min_gt_points_list:
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                #logger.info('Database filter by min points %s: %d => %d' %
                #            (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:
        """
        sample_num, pointer, indices = int(
            sample_group['sample_num']
        ), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [
            self.db_infos[class_name][idx]
            for idx in indices[pointer:pointer + sample_num]
        ]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes,
                                   total_valid_sampled_dict):
        # logger.info('Sampled dict: ' + total_valid_sampled_dict)
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['pts_rect']

        obj_points_list = []
        NUM_POINT_FEATURES = self.num_point_features
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array(
            [x['name'] for x in total_valid_sampled_dict])

        # large_sampled_gt_boxes = box_utils.enlarge_box3d(
        #     sampled_gt_boxes[:, 0:7],
        #     extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH)
        points = object3d.remove_points_in_boxes3d(points,
                                                   sampled_gt_boxes[:, 0:7])
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['pts_rect'] = points
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
        Returns:
        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(
                    int(self.sample_class_num[class_name]) - num_gt)

            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(
                    class_name, sample_group)

                sampled_boxes = np.stack(
                    [x['box3d_lidar'] for x in sampled_dict],
                    axis=0).astype(np.float32)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(
                    sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(
                    sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]),
                     range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) +
                               iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate(
                    (existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(
                data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        data_dict['gt_boxes3d'] = dict()
        for c in self.class_names:
            indices = np.array(
                [i for i, x in enumerate(data_dict['gt_names']) if x == c])
            if len(indices) == 0:
                data_dict['gt_boxes3d'][c] = np.zeros((0, self.box_dimensions))
            else:
                class_gt_boxes = data_dict['gt_boxes'][indices]
                data_dict['gt_boxes3d'][
                    c] = class_gt_boxes[:, np.arange(self.box_dimensions).tolist()]

        data_dict.pop('gt_boxes_mask')
        return data_dict


class NuscenesGTAugment(GTAugmentTemplate):
    def __init__(self, root_path, class_names, p=1, num_sweeps=10):
        super().__init__(
            root_path=root_path,
            class_names=class_names,
            p=p,
            db_info_path=['nuscenes_dbinfos_{}sweeps_withvelo.pkl'.format(num_sweeps)],
            filter_cfg=[('car', 5), ('truck', 5),
                        ('construction_vehicle', 5), ('bus', 5),
                        ('trailer', 5), ('barrier', 5),
                        ('motorcycle', 5), ('pedestrian', 5),
                        ('traffic_cone', 5)],
            sample_group_cfg=[('car', 2), ('truck', 3), ('construction_vehicle', 7),
                        ('bus', 4), ('trailer', 6), ('barrier', 2),
                        ('motorcycle', 6), ('bicycle', 6), ('pedestrian', 2),
                        ('traffic_cone', 2)],
            num_point_features=5,
            box_dimensions=9
        )


class WaymoGTAugment(GTAugmentTemplate):
    def __init__(self, root_path, class_names, p=1, num_sweeps=1):
        super().__init__(
            root_path=root_path,
            class_names=class_names,
            p=p,
            db_info_path=['waymo_dbinfos_{}sweeps.pkl'.format(num_sweeps)],
            filter_cfg=[('vehicle', 5), ('pedestrian', 5),
                        ('cyclist', 5)],
            sample_group_cfg=[('vehicle', 15), ('pedestrian', 10), ('cyclist', 10)],
            num_point_features=6,
            box_dimensions=7
        )
