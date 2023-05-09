from typing import Optional, Tuple

from torch import nn
from script.sptensor import spTensor

import torch

__all__ = ['SparseCrop']

# hk: modified to fit PCEngine
def spcrop(input: spTensor,
           coords_min: Optional[Tuple[int, ...]] = None,
           coords_max: Optional[Tuple[int, ...]] = None) -> spTensor:
    coords, feats, stride = input.coords, input.feats, input.stride

    mask = torch.ones((coords.shape[0], 3),
                      dtype=torch.bool,
                      device=coords.device)
    if coords_min is not None:
        coords_min = torch.tensor(coords_min,
                                  dtype=torch.int,
                                  device=coords.device).unsqueeze(dim=0)
        mask &= (coords[:, 0:3] >= coords_min)
    if coords_max is not None:
        coords_max = torch.tensor(coords_max,
                                  dtype=torch.int,
                                  device=coords.device).unsqueeze(dim=0)
        # Using "<" instead of "<=" is for the backward compatability (in
        # some existing detection codebase). We might need to reflect this
        # in the document or change it back to "<=" in the future.
        mask &= (coords[:, 0:3] < coords_max)

    mask = torch.all(mask, dim=1)
    coords, feats = coords[mask], feats[mask]
    # coords_min = [0, 0, 0]
    # coords_max = ((torch.max(coords[:, 1:], dim=0).values).cpu().numpy()).tolist()
    # hk: wonder if buffer is necessary
    output = spTensor(coords=coords, feats=feats, 
                      stride=stride, buffer=None, coords_max=None, coords_min=None)
    return output


class SparseCrop(nn.Module):

    def __init__(self,
                 coords_min: Optional[Tuple[int, ...]] = None,
                 coords_max: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        self.coords_min = coords_min
        self.coords_max = coords_max

    def forward(self, input: spTensor) -> spTensor:
        return spcrop(input, self.coords_min, self.coords_max)
