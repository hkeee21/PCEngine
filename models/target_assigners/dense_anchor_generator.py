import torch
import numpy as np
from typing import List, Tuple, Union, Optional

__all__ = ['DenseAnchorGenerator']

class DenseAnchorGenerator(object):
    def __init__(self, 
                 grid_size: Union[List[float], Tuple[float, ...], np.ndarray],
                 pc_area_scope: Union[List[List[float]], np.ndarray],
                 mean_size: dict,
                 mean_center_z: dict,
                 rotations: Optional[Union[List[float], Tuple[float, ...]]] = [0, np.pi / 2.]
    ):                 
        super().__init__()
        self.grid_size = grid_size
        self.anchor_range = pc_area_scope
        self.mean_size = mean_size
        self.mean_center_z = mean_center_z
        self.rotations = rotations

    def generate_anchors(self):
        all_anchors = []
        num_anchors_per_location = []
        for c in self.mean_size.keys():
            num_anchors_per_location.append(len(self.rotations))
            
            device = 'cuda'
            grid_size = self.grid_size
            H, W = grid_size
            
            x_stride = (self.anchor_range[0][1] - self.anchor_range[0][0]) / H
            y_stride = (self.anchor_range[2][1] - self.anchor_range[2][0]) / W
            x_offset, y_offset = x_stride / 2, y_stride / 2
            
            
            x_shifts = torch.arange(self.anchor_range[0][0] + x_offset, 
                                    self.anchor_range[0][1] + 1e-5,
                                    step=x_stride).to(device)
            y_shifts = x_shifts.new_tensor(self.mean_center_z[c]).view(1)  
            z_shifts = torch.arange(self.anchor_range[2][0] + y_offset, 
                                    self.anchor_range[2][1] + 1e-5,
                                    step=y_stride).to(device)
            anchors = torch.stack(torch.meshgrid([x_shifts, y_shifts, z_shifts]))
            # H x W x 1 x 3
            anchors = anchors.permute(1, 3, 2, 0).contiguous()
            # 3
            anchor_size = x_shifts.new(self.mean_size[c].to(device))
            # 1 x 1 x 1 x 3
            anchor_size = anchor_size.view(1, 1, 1, 3)
            # H x W x 1 x 6
            anchors = torch.cat([anchors, anchor_size.repeat(H, W, 1, 1)], -1)
            # 2
            anchor_rotation = torch.tensor(self.rotations).to(device)
            # 1 x 1 x 2 x 1
            anchor_rotation = anchor_rotation.view(1, 1, -1, 1)
            # H x W x 2 x 6
            anchors = anchors.repeat(1, 1, anchor_rotation.shape[-2], 1)
            # H x W x 2 x 7
            anchors = torch.cat([anchors, anchor_rotation.repeat(H, W, 1, 1)], -1)
            anchors = anchors.unsqueeze(2)
            # previous: + gt_boxes[c][:, 3] / 2. done in target assigner
            # now: following openpcdet
            anchors[..., 1] = self.mean_center_z[c] + anchors[..., 3] / 2.
            all_anchors.append(anchors)
        
        return all_anchors, num_anchors_per_location
    
    def __call__(self, inputs):
        return self.generate_anchors(inputs)


if __name__ == '__main__':
    from easydict import EasyDict
    A = DenseAnchorGenerator()
    #import pdb
    #pdb.set_trace()
    anchors = A.generate_anchors({
        'Car': torch.randn(2, 200, 176, 16).cuda(),
        'Pedestrian': torch.randn(2, 200, 240, 16).cuda(),
        'Cyclist': torch.randn(2, 100, 120, 16).cuda()
    })
    for c in anchors:
        print(anchors[c].shape)
