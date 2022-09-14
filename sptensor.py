import torch
from typing import Dict

class spTensor:

    def __init__(self,
                 feats: torch.FloatTensor,
                 coords: torch.IntTensor
                 ) -> None:
        self.feats = feats
        self.coords = coords
        self.mappat = list()
        self.snnz: Dict[int] = {}
        self.imap: Dict[int] = {}
        self.omap: Dict[int] = {}
        self.icsr: Dict[int] = {}
        self.ocsr: Dict[int] = {}
        self.knnz: Dict[int] = {}
        self.kpos: Dict[int] = {}
    
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

