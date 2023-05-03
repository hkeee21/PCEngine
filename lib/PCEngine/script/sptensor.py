import torch
from typing import Dict, Union, Tuple, List

class spTensor:

    def __init__(self,
                 feats: torch.Tensor,
                 coords: torch.Tensor,
                 buffer: torch.Tensor, 
                 coords_min: list,
                 coords_max: list, 
                 batchsize: int=1, 
                 stride: int=94585,
                 init_tag: list=['end']
                 ) -> None:
        self.feats = feats
        self.coords = coords
        self.stride = stride
        self.batchsize = batchsize
        self.buffer = buffer
        self.kmaps: Dict[Tuple[int, int]] = {}
        self.cbook: Dict[int] = {}
        self.init_tag = init_tag
        self.coords_min = coords_min
        self.coords_max = coords_max
    
    @property
    def F(self) -> torch.Tensor:
        return self.feats

    @F.setter
    def F(self, feats: torch.Tensor) -> None:
        self.feats = feats

    @property
    def C(self) -> torch.Tensor:
        return self.coords

    @C.setter
    def C(self, coords: torch.Tensor) -> None:
        self.coords = coords

    def cpu(self):
        self.coords = self.coords.cpu()
        self.feats = self.feats.cpu()
        return self

    def cuda(self):
        self.coords = self.coords.cuda()
        self.feats = self.feats.cuda()
        return self

    def detach(self):
        self.coords = self.coords.detach()
        self.feats = self.feats.detach()
        return self

    def to(self, device: str, non_blocking: bool = True):
        self.coords = self.coords.to(device, non_blocking=non_blocking)
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        return self

    def build_buffer(self, cinfo: Dict, device: str):
        max_c_in = max(cinfo["in"])
        max_c_out = max(cinfo["out"])
        max_ks = max(cinfo["kernel"])
        nnz = self.coords.size(0)

        if max_ks == 3:
            buffer_size = (max_c_in + max_c_out) * nnz * 16
        elif max_ks == 5:
            buffer_size = (max_c_in + max_c_out) * nnz * 48
        elif max_ks == 2:
            buffer_size = (max_c_in + max_c_out) * nnz * 8
        else:
            buffer_size = (max_c_in + max_c_out) * nnz * (max_ks ** 3)

        self.buffer = torch.zeros((buffer_size,), dtype=torch.float, device=device)
    
    def __add__(self, other):
        output = spTensor(coords=self.coords, 
                              feats=self.feats + other.feats, 
                              batchsize=self.batchsize, 
                              stride=self.stride, 
                              init_tag=self.init_tag, 
                              buffer=self.buffer, 
                              coords_max=self.coords_max,
                              coords_min=self.coords_min
                              )
        output.cbook = self.cbook
        output.kmaps = self.kmaps
        return output



