from typing import Union, Tuple, List
import torch
from torch import nn
from .backbone2d_template import Backbone2DTemplate
from .crop import SparseCrop
from script.sptensor import spTensor
from script.utils import conv_info_decoder

__all__ = ['ToBEVConvolutionBlock']

'''
class ToBEVHeightCompression(nn.Module):
    """Converts a SparseTensor to a flattened volumetric tensor.

    Args:
        channels: Number of input channels
        (Note: output channels = channels x #unique z values)
        shape: Shape of BEV map
        dim: Dimension index for z (default: 1 for KITTI coords)
    """

    def __init__(self,
                 channels: int,
                 shape: Union[List[int], Tuple[int, int, int], torch.Tensor],
                 offset: Tuple[int, int, int] = (0, 0, 0),
                 dim: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.register_buffer('offset', torch.IntTensor([[0] + list(offset)]))
        # self.offset = [0, min_x, min_y, min_z]
        if isinstance(shape, torch.Tensor):
            self.register_buffer('shape', shape.int())
        else:
            self.register_buffer('shape', torch.IntTensor(shape))
        # self.shape = [shape_x, shape_y, shape_z]
        self.dim = dim 
        # self.dim = 1
        self.bev_dims = [i for i in range(3) if i != self.dim]
        # self.bev_dims = [0, 2]
        self.bev_shape = self.shape[self.bev_dims]
        # self.bev_shape = [shape_x, shape_z]

    def extra_repr(self) -> str:
        return f'channels={self.channels}'

    def forward(self, input: spTensor) -> torch.Tensor:
        coords, feats, stride = input.coords, input.feats, input.stride
        stride = conv_info_decoder(stride)
        stride = torch.tensor(stride).unsqueeze(dim=0).to(coords.device)
        assert isinstance(stride, torch.Tensor), type(stride)

        # [b, x, y, z]
        # ([x, y, z, b] - [min_x, min_y, min_z, 0]).t()  -> (4, N) [3, 0, 2, 1]
        # -> [b, x, z, y]
        # ([b, x, y, z] - [0, min_x, min_y, min_z]).t() -> (4, N) [0, 1, 3, 2]
        # -> [b, x, z, y]
        coords_dim = self.dim + 1
        coords_bev_dims = [(i + 1) for i in self.bev_dims]
        coords = (coords - self.offset).t()[[0] + coords_bev_dims
                                            + [coords_dim]].long()
        shape = self.shape[self.bev_dims + [self.dim]]
        # shape = [shape_x, shape_z, shape_y]
        # now stride must be torch.Tensor since input.s is tuple.
        stride = stride[:, self.bev_dims + [self.dim]].t()
        # stride = [stride_x, stride_z, stride_y]
        coords[1:] = (coords[1:] // stride).long()
        coords[-1] = torch.clamp(coords[-1], 0, shape[-1] - 1)
        indices = coords[0] * int(shape.prod()) + coords[1] * int(
            shape[1:].prod()) + coords[2] * int(shape[2]) + coords[3]
        batch_size = coords[0].max().item() + 1
        output = torch.sparse_coo_tensor(
            indices.unsqueeze(dim=0),
            feats,
            torch.Size([batch_size * int(self.shape.prod()),
                        feats.size(-1)]),
        ).to_dense()
        output = output.view(batch_size, *self.bev_shape.cpu().numpy(), -1)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output


class ToBEVConvolutionBlock(Backbone2DTemplate):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 loc_min: torch.Tensor, 
                 loc_max: torch.Tensor,
                 proposal_stride) -> None:
        super().__init__()

        self.to_bev = nn.Sequential(
            SparseCrop(coords_min=loc_min, coords_max=loc_max),
            ToBEVHeightCompression(
                in_channels,
                shape=(loc_max - loc_min) // proposal_stride,
                offset=loc_min
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        return self.to_bev(x)'''

class ToBEVHeightCompression(nn.Module):
    """Converts a SparseTensor to a flattened volumetric tensor.

    Args:
        channels: Number of input channels
        (Note: output channels = channels x #unique z values)
        shape: Shape of BEV map
        dim: Dimension index for z (default: 1 for KITTI coords)
    """

    def __init__(self,
                 channels: int,
                 shape: Union[List[int], Tuple[int, int, int], torch.Tensor],
                 offset: Tuple[int, int, int] = (0, 0, 0),
                 dim: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.register_buffer('offset', torch.IntTensor([list(offset) + [0]]))
        if isinstance(shape, torch.Tensor):
            self.register_buffer('shape', shape.int())
        else:
            self.register_buffer('shape', torch.IntTensor(shape))
        self.dim = dim
        self.bev_dims = [i for i in range(3) if i != self.dim]
        self.bev_shape = self.shape[self.bev_dims]

    def extra_repr(self) -> str:
        return f'channels={self.channels}'

    def forward(self, input: spTensor) -> torch.Tensor:
        coords, feats, stride = input.coords, input.feats, input.stride
        # print(coords.shape)
        stride = conv_info_decoder(stride)
        stride = torch.tensor(stride).unsqueeze(dim=0).to(coords.device)
        assert isinstance(stride, torch.Tensor), type(stride)

        # [b, x, y, z]
        coords = (coords - self.offset).t()[[3] + self.bev_dims
                                            + [self.dim]].long()
        shape = self.shape[self.bev_dims + [self.dim]]

        # now stride must be torch.Tensor since input.s is tuple.
        stride = stride[:, self.bev_dims + [self.dim]].t()

        coords[1:] = (coords[1:] // stride).long()
        coords[-1] = torch.clamp(coords[-1], 0, shape[-1] - 1)
        indices = coords[0] * int(shape.prod()) + coords[1] * int(
            shape[1:].prod()) + coords[2] * int(shape[2]) + coords[3]
        batch_size = coords[0].max().item() + 1
        # print(stride, coords.shape, indices.shape, feats.shape)
        output = torch.sparse_coo_tensor(
            indices.unsqueeze(dim=0),
            feats,
            torch.Size([batch_size * int(self.shape.prod()),
                        feats.size(-1)]),
        ).to_dense()
        output = output.view(batch_size, *self.bev_shape.cpu().numpy(), -1)
        output = output.permute(0, 3, 1, 2).contiguous()

        # print(output.shape)
        return output


class ToBEVConvolutionBlock(Backbone2DTemplate):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 loc_min: torch.Tensor, 
                 loc_max: torch.Tensor,
                 proposal_stride,
                 backend='torchsparse') -> None:
        super().__init__()

        self.backend = backend
        self.to_bev = nn.Sequential(
            SparseCrop(coords_min=loc_min, coords_max=loc_max),
            ToBEVHeightCompression(
                in_channels,
                shape=(loc_max - loc_min) // proposal_stride,
                offset=loc_min
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("before:", x.coords.shape)
        x.coords = x.coords[:, [1, 2, 3, 0]]
        # print("after:", x.coords.shape)
    
        return self.to_bev(x)
        