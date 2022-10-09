import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import torch
import math
import numpy as np
from .sptensor import spTensor
from typing import Dict, List, Tuple, Union
from dgCloud.backend import mapping_cuda, conv_fwd_cuda, conv_bwd_cuda
from .utils import output_stride_compute, conv_info_encoder


class conv3d(nn.Module):
    
    def __init__(self,
                 in_channels: int = 16,
                 out_channels: int = 32,
                 kernel_size: Union[int, List[int]] = 3,
                 stride: Union[int, List[int]] = 1,
                 bias: bool = False, 
                 transposed: bool = False, 
                 tc_mode_16f: bool = 0
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transposed = transposed

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size, kernel_size]
        elif isinstance(kernel_size, list):
            self.kernel_size = kernel_size
        else:
            raise NotImplementedError
        self.kernel_size_code = conv_info_encoder(self.kernel_size)
        self.kernel_volume = np.prod(self.kernel_size, dtype=int)
        
        if isinstance(stride, int):
            self.stride = [stride, stride, stride]
        elif isinstance(stride, list):
            self.stride = stride
        else:
            raise NotImplementedError
        self.stride_code = conv_info_encoder(self.stride)

        self.tensorcore16F = tc_mode_16f
        
        if self.kernel_volume > 1:
            self.kernel = nn.Parameter(
                torch.zeros((self.kernel_volume, in_channels, out_channels), \
                    dtype=torch.float))
        else:
            self.kernel = nn.Parameter(torch.zeros((in_channels, out_channels), \
                    dtype=torch.float))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_channels), dtype=torch.float))
        else:
            self.bias = None

        self.reset_parameters()
    

    def extra_repr(self) -> str:
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.stride_code != conv_info_encoder([1, 1, 1]):
            s += ', stride={stride}'
        if self.bias is None:
            s += ', bias=False'
        if self.transposed:
            s += ', transposed=True'
        return s.format(**self.__dict__)
    

    def forward(self, input: spTensor) -> spTensor:
        
        return conv_func(input=input,
                        in_channel=self.in_channels, 
                        out_channel=self.out_channels,
                        kernel=self.kernel,
                        kernel_size_code=self.kernel_size_code,
                        stride_code=self.stride_code,
                        bias=self.bias, 
                        transposed=self.transposed, 
                        tensorcore16F=self.tensorcore16F
                        )
    

    def reset_parameters(self):
        n = self.out_channels * self.kernel_volume
        stdv = 1.0 / math.sqrt(n)
        self.kernel.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    

class convF(Function):
    
    @staticmethod
    @custom_fwd
    def forward(ctx,
                in_feats: torch.Tensor,
                knnz: torch.Tensor, 
                kpos: torch.Tensor, 
                imap: torch.Tensor, 
                omap: torch.Tensor,
                icsr: torch.Tensor,
                ocsr: torch.Tensor, 
                in_channel: int,
                out_channel: int, 
                kernel: torch.Tensor, 
                kernel_size_code: int, 
                sum_nnz: int, 
                nnz: Tuple[int, int], 
                buffer: torch.Tensor, 
                transposed: bool, 
                tensorcore16F: bool) -> torch.Tensor:
        
        in_feats = in_feats.contiguous()
        kernel = kernel.contiguous()
        imap = imap.contiguous()
        omap = omap.contiguous()
        icsr = icsr.contiguous()
        ocsr = ocsr.contiguous()
        buffer = buffer.contiguous()

        separate_mid = (nnz[0] == nnz[1])

        if transposed:
            output_feats = torch.zeros((nnz[0], out_channel), \
                dtype=torch.float, device=in_feats.device)

            conv_fwd_cuda(
                in_feats,
                kernel,
                kernel_size_code, 
                sum_nnz, 
                output_feats,
                knnz,
                kpos, 
                omap,
                imap,
                ocsr,
                icsr, 
                buffer, 
                separate_mid, 
                tensorcore16F
            )
        else:
            output_feats = torch.zeros((nnz[1], out_channel), \
                dtype=torch.float, device=in_feats.device)

            conv_fwd_cuda(
                in_feats,
                kernel,
                kernel_size_code, 
                sum_nnz, 
                output_feats,
                knnz,
                kpos, 
                imap,
                omap,
                icsr,
                ocsr, 
                buffer, 
                separate_mid, 
                tensorcore16F
            )

    
        # TODO: replace in_feats with gathered features in in_buffer
        ctx.for_backwards = (in_feats, kernel, knnz, imap, omap, \
            buffer, tensorcore16F)
        
        return output_feats

    
    @staticmethod
    @custom_bwd
    def backward(ctx, out_feats_grad: torch.Tensor):

        in_feats, kernel, kernel_size, knnz, kpos, imap, omap, \
            in_buffer, out_buffer, tensorcore16F = ctx.for_backwards
        
        input_size = in_feats.size(0)
        input_channel = in_feats.size(1)
        output_channel = kernel.size(-1)
        kernel_volume = kernel.size(0)

        in_feats_grad = torch.zeros((input_size, input_channel), dtype=torch.float, \
            device=in_feats.device)
        weight_grad = torch.zeros((kernel_volume, input_channel, output_channel), \
            dtype=torch.float, device=in_feats.device)

        conv_bwd_cuda(
            out_feats_grad, 
            in_feats,
            kernel,
            kernel_size,
            in_feats_grad, 
            weight_grad, 
            knnz,
            kpos, 
            imap,
            omap,
            in_buffer, 
            out_buffer,  
            tensorcore16F
        )

        return in_feats_grad, None, None, None, None, None, \
            weight_grad, None, None, None, None


def conv_func(input, in_channel, out_channel, kernel, 
    kernel_size_code, stride_code, bias, transposed, tensorcore16F):
    
    input_size = input.coords.size(0)
    kernel_volume = kernel.size(0)

    if kernel_size_code == conv_info_encoder([1, 1, 1]) \
        and stride_code == conv_info_encoder([1, 1, 1]):
        out_feats = input.feats.matmul(kernel)
        if bias is not None:
            out_feats += bias
        output = spTensor(coords=input.coords, feats=out_feats, \
            buffer=input.buffer, stride=input.stride)
    
    elif transposed: 
        output_stride = output_stride_compute(stride_code, input.stride, 1)
        kmap = input.kmaps.get((kernel_size_code, output_stride, stride_code))
        out_coords = input.cbook[output_stride]

        out_feats = convF.apply(
            input.feats,
            kmap[4], 
            kmap[5], 
            kmap[0],
            kmap[1],
            kmap[2],
            kmap[3],
            in_channel, 
            out_channel,
            kernel, 
            kernel_size_code, 
            kmap[6],  
            kmap[7], 
            input.buffer,
            transposed, 
            tensorcore16F
        )
        if bias is not None:
            out_feats += bias
        output = spTensor(out_feats, out_coords, input.buffer, output_stride)

    else:
        output_stride = output_stride_compute(stride_code, input.stride, 0)
        kmap = input.kmaps.get((kernel_size_code, input.stride, stride_code)) 
        out_coords = input.coords
 
        if kmap is None:

            imap = - torch.ones((input_size * kernel_volume), \
                dtype=torch.int, device=input.coords.device)
            omap = - torch.ones((input_size * kernel_volume), \
                dtype=torch.int, device=input.coords.device)
            knnz = torch.zeros((kernel_volume), dtype=torch.int, \
                device=input.coords.device)
            kpos = torch.zeros((kernel_volume), dtype=torch.int, \
                device=input.coords.device)
            icsr = torch.zeros((input_size + 2), dtype=torch.int, \
                device=input.coords.device)
            ocsr = torch.zeros((input_size + 2), dtype=torch.int, \
                device=input.coords.device)
    
            separate_mid = (stride_code == conv_info_encoder([1, 1, 1]))

            out_coords = mapping_cuda(
                input.coords,
                kernel_size_code, 
                kernel_volume, 
                in_channel, 
                out_channel, 
                stride_code, 
                input.stride, 
                imap,
                omap, 
                icsr, 
                ocsr,  
                knnz,
                kpos,
                separate_mid
            )

            knnz = knnz.cpu()
            sum_nnz = knnz.sum().int()
            out_nnz = out_coords.size(0)
        
            if input.buffer.size(0) < sum_nnz * (in_channel + out_channel):
                print("Newly allocated buffer.")
                input.buffer = torch.zeros((sum_nnz, (in_channel + out_channel)), \
                    dtype=torch.float, device=input.feats.device)

            kmap = [imap, omap, icsr, ocsr, knnz, kpos, sum_nnz, (input_size, out_nnz)]
            input.kmaps[(kernel_size_code, input.stride, stride_code)] = kmap
            input.cbook[output_stride] = out_coords


        out_feats = convF.apply(
            input.feats,
            kmap[4], 
            kmap[5], 
            kmap[0],
            kmap[1],
            kmap[2],
            kmap[3],
            in_channel, 
            out_channel,
            kernel, 
            kernel_size_code, 
            kmap[6],  
            kmap[7], 
            input.buffer,
            transposed, 
            tensorcore16F
        )
        if bias is not None:
            out_feats += bias
        output = spTensor(out_feats, out_coords, input.buffer, output_stride)
    
    output.kmaps = input.kmaps
    output.cbook = input.cbook
    
    return output
