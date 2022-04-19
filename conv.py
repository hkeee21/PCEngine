from typing import Optional, Tuple, Union
from torch import nn
import torch
from torch.autograd import Function
import math
import numpy as np
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.cpp_extension import load

# import torchsparse.backend as backend
from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple

__all__ = ['conv3d', 'Conv3d']


backend = load(name="convolution_forward_cuda",
                   sources=["/home/hongke21/nfs/code/TS-Test/backend_EF/pybind_cuda.cpp", 
                   "/home/hongke21/nfs/code/TS-Test/backend_EF/convolution/convolution_ef.cu"],
                   verbose=True)

backend = load(name="convolution_backward_cuda",
                   sources=["/home/hongke21/nfs/code/TS-Test/backend_EF/pybind_cuda.cpp", 
                   "/home/hongke21/nfs/code/TS-Test/backend_EF/convolution/convolution_ef.cu"],
                   verbose=True)
                   


class Conv3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transposed: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = dilation
        self.transposed = transposed

        self.kernel_volume = int(np.prod(self.kernel_size))
        if self.kernel_volume > 1:
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, in_channels, out_channels))
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        if self.transposed:
            s += ', transposed=True'
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume)
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return conv3d(input,
                        self.kernel,
                        kernel_size=self.kernel_size,
                        bias=self.bias,
                        stride=self.stride,
                        dilation=self.dilation,
                        transposed=self.transposed)



class ConvolutionFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        nbmaps: torch.Tensor,
        nbsizes: torch.Tensor,
        sizes: Tuple[int, int],
        transposed: bool = False,
    ) -> torch.Tensor:
        input = input.contiguous()
        weight = weight.contiguous()
        nbmaps = nbmaps.int().contiguous()
        nbsizes = nbsizes.int().contiguous()

        if not transposed:
            output = torch.zeros(
                sizes[1],
                weight.size(-1),
                dtype=input.dtype,
                device=input.device,
            )
        else:
            # TODO(Haotian): ensure the original, upsampled size to be the same.
            output = torch.zeros(
                sizes[0],
                weight.size(-1),
                dtype=input.dtype,
                device=input.device,
            )
        
        # TODO: Currently we only consider GPUs, perhaps extension to other architectures
        backend.convolution_forward_cuda(
                input,
                output,
                weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
            )

        ctx.for_backwards = (input, weight, nbmaps, nbsizes, transposed)
        return output

    @staticmethod
    @custom_bwd
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        input, weight, nbmaps, nbsizes, transposed = ctx.for_backwards

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)

        # TODO: Currently we only consider GPUs, perhaps extension to other architectures
        backend.convolution_backward_cuda(
                input,
                grad_input,
                grad_output.contiguous(),
                weight,
                grad_weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
            )
        
        return grad_input, grad_weight, None, None, None, None



def conv3d(
    input: SparseTensor,
    weight: torch.Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
    transposed: bool = False,
) -> SparseTensor:
    kernel_size = make_ntuple(kernel_size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    if (kernel_size == make_ntuple(1, ndim=3)
            and stride == make_ntuple(1, ndim=3)
            and dilation == make_ntuple(1, ndim=3)):
        output_stride = input.stride
        output_coords = input.coords
        output_feats = input.feats.matmul(weight)
    elif not transposed:
        output_stride = tuple(input.stride[k] * stride[k] for k in range(3))

        if output_stride in input.cmaps:
            output_coords = input.cmaps[output_stride]
        elif all(stride[k] == 1 for k in range(3)):
            output_coords = input.coords
        else:
            output_coords = F.spdownsample(
                input.coords,
                stride,
                kernel_size,
                input.stride,
            )

        if (input.stride, kernel_size, stride, dilation) not in input.kmaps:
            offsets = get_kernel_offsets(
                kernel_size,
                stride=input.stride,
                dilation=dilation,
                device=input.feats.device,
            )

            references = F.sphash(input.coords)
            queries = F.sphash(output_coords, offsets)
            results = F.sphashquery(queries, references)

            nbsizes = torch.sum(results != -1, dim=1)
            nbmaps = torch.nonzero(results != -1)

            indices = nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]
            nbmaps[:, 0] = results.view(-1)[indices]

            input.kmaps[(input.stride, kernel_size, stride, dilation)] = [
                nbmaps, nbsizes, (input.coords.shape[0], output_coords.shape[0])
            ]

        output_feats = ConvolutionFunction.apply(
            input.feats,
            weight,
            *input.kmaps[(input.stride, kernel_size, stride, dilation)],
            transposed,
        )
    else:
        output_stride = tuple(input.stride[k] // stride[k] for k in range(3))
        output_coords = input.cmaps[output_stride]
        output_feats = ConvolutionFunction.apply(
            input.feats,
            weight,
            *input.kmaps[(output_stride, kernel_size, stride, dilation)],
            transposed,
        )

    if bias is not None:
        output_feats += bias

    output = SparseTensor(
        coords=output_coords,
        feats=output_feats,
        stride=output_stride,
    )
    output.cmaps = input.cmaps
    output.cmaps.setdefault(output_stride, output_coords)
    output.kmaps = input.kmaps
    return output
