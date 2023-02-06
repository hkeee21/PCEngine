import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, point_cloud_range=None):
        super().__init__()
        self.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp']
        self.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp']
        self.point_cloud_range = point_cloud_range

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        data_dict['points'], use_lead_xyz = self.absolute_coordinates_encoding(
            data_dict['points'])
        data_dict['use_lead_xyz'] = use_lead_xyz
        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx + 1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True


class KITTIBoxGenerator(object):
    def __init__(self, classes, code_size=9):
        super().__init__()
        self.classes = classes
        self.code_size = code_size

    def __call__(self, data_dict):
        data_dict['gt_boxes3d'] = dict()
        for c in self.classes:
            indices = np.array(
                [i for i, x in enumerate(data_dict['gt_names']) if x == c])
            if len(indices) == 0:
                data_dict['gt_boxes3d'][c] = np.zeros((0, self.code_size))
            else:
                class_gt_boxes = data_dict['gt_boxes'][indices]
                ## Switch to KITTI format
                if self.code_size == 9:
                    data_dict['gt_boxes3d'][
                        c] = class_gt_boxes[:, [0, 2, 1, 5, 4, 3, 6, 7, 8]]
                else:
                    data_dict['gt_boxes3d'][
                        c] = class_gt_boxes[:, [0, 2, 1, 5, 4, 3, 6]]

                data_dict['gt_boxes3d'][c][:, 1] *= -1
                data_dict['gt_boxes3d'][
                    c][:, 1] += data_dict['gt_boxes3d'][c][:, 3] / 2
        return data_dict


class ToKITTICoords(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data_dict):
        ## Should refactor this
        for key in ['pts_rect', 'points']:
            value = np.copy(data_dict[key])
            data_dict.pop(key)
            value[:, [1, 2]] = value[:, [2, 1]]
            value[:, 1] *= -1
            data_dict.update({key: value})        
        return data_dict
