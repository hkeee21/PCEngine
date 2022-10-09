import torch
from typing import Dict, Union, Tuple, List

class spTensor:

    def __init__(self,
                 feats: torch.FloatTensor,
                 coords: torch.IntTensor,
                 buffer: torch.Tensor, 
                 stride: int=329
                 ) -> None:
        self.feats = feats
        # TODO: coords can be added to cbook
        self.coords = coords
        self.stride = stride
        self.buffer = buffer
        self.kmaps: Dict[Tuple[int, int]] = {}
        self.cbook: Dict[int] = {}
    
    @property
    def F(self) -> torch.FloatTensor:
        return self.feats

    @F.setter
    def F(self, feats: torch.FloatTensor) -> None:
        self.feats = feats

    @property
    def C(self) -> torch.IntTensor:
        return self.coords

    @C.setter
    def C(self, coords: torch.IntTensor) -> None:
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
                              stride=self.stride,
                              buffer=self.buffer)
        output.cbook = self.cbook
        output.kmaps = self.kmaps
        return output



