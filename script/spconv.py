import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import torch
import math
import numpy as np
import time
from .sptensor import spTensor
from typing import Dict, List, Tuple, Union
from PCEngine.backend import mapping_cuda, mapping_simple_cuda, \
    conv_fwd_cuda, conv_fwd_simple_cuda, conv_bwd_cuda
from .utils import output_stride_compute, conv_info_encoder


class conv3d(nn.Module):
    
    def __init__(self,
                 in_channels: int = 16,
                 out_channels: int = 32,
                 kernel_size: Union[int, List[int]] = 3,
                 stride: Union[int, List[int]] = 1,
                 bias: bool = False, 
                 transposed: bool = False, 
                 tc_mode: bool = True
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

        self.tensorcore = tc_mode
        
        if self.kernel_volume > 1:
            self.kernel = nn.Parameter(
                torch.zeros((self.kernel_volume, in_channels, out_channels)))
        else:
            self.kernel = nn.Parameter(torch.zeros((in_channels, out_channels)))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_channels)))
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
                        tensorcore=self.tensorcore
                        )
    

    def reset_parameters(self):
        n = self.out_channels * self.kernel_volume
        stdv = 1.0 / math.sqrt(n)
        self.kernel.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    

class conv_d1(Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx,
                in_feats: torch.Tensor,
                kpos: torch.Tensor, 
                qkpos: torch.Tensor, 
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
                tensorcore: bool) -> torch.Tensor:
        
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
                dtype=in_feats.dtype, device=in_feats.device)

            '''kernel_volume = kernel.shape[0]
            kernel_x = kernel_size_code // 311
            kernel_y = (kernel_size_code - kernel_x * 311) // 17
            kernel_z = (kernel_size_code - kernel_x * 311 - kernel_y * 17)
            mid_weight = (kernel_x - 1) // 2 * kernel_y * kernel_z + \
                (kernel_y - 1) // 2 * kernel_z + (kernel_z - 1) // 2
            print("%d %d %d %d" % (nnz[1], kernel.shape[1], kernel.shape[2], kernel_volume), end=" ")
            for i in range(kernel_volume):
                if i == mid_weight and separate_mid and knnz[i] == 0:
                    print("%d" % nnz[0], end=" ")
                elif i == kernel_volume - 1:
                    print("%d" % knnz[i])
                else:
                    print("%d" % knnz[i], end=" ")'''

            conv_fwd_cuda(
                in_feats,
                kernel,
                kernel_size_code, 
                sum_nnz, 
                output_feats,
                kpos,
                qkpos, 
                omap,
                imap,
                ocsr,
                icsr, 
                buffer, 
                separate_mid, 
                tensorcore
            )
        else:
            output_feats = torch.zeros((nnz[1], out_channel), \
                dtype=in_feats.dtype, device=in_feats.device)

            '''kernel_volume = kernel.shape[0]
            kernel_x = kernel_size_code // 311
            kernel_y = (kernel_size_code - kernel_x * 311) // 17
            kernel_z = (kernel_size_code - kernel_x * 311 - kernel_y * 17)
            mid_weight = (kernel_x - 1) // 2 * kernel_y * kernel_z + \
                (kernel_y - 1) // 2 * kernel_z + (kernel_z - 1) // 2
            print("%d %d %d %d" % (nnz[1], kernel.shape[1], kernel.shape[2], kernel_volume), end=" ")
            for i in range(kernel_volume):
                if i == mid_weight and separate_mid and knnz[i] == 0:
                    print("%d" % nnz[0], end=" ")
                elif i == kernel_volume - 1:
                    print("%d" % knnz[i])
                else:
                    print("%d" % knnz[i], end=" ")'''

            conv_fwd_cuda(
                in_feats,
                kernel,
                kernel_size_code, 
                sum_nnz, 
                output_feats,
                kpos,
                qkpos, 
                imap,
                omap,
                icsr,
                ocsr, 
                buffer, 
                separate_mid, 
                tensorcore
            )

            # print("completed.")
            # print(output_feats)
    
        # TODO: replace in_feats with gathered features in in_buffer
        ctx.for_backwards = (in_feats, kernel, kpos, imap, omap, \
            buffer, tensorcore)
        
        return output_feats

    
    @staticmethod
    @custom_bwd
    def backward(ctx, out_feats_grad: torch.Tensor):

        in_feats, kernel, kernel_size_code, sum_nnz, knnz, kpos, imap, omap, \
            icsr, ocsr, buffer, transposed, tensorcore = ctx.for_backwards
        
        input_size = in_feats.size(0)
        input_channel = in_feats.size(1)
        output_channel = kernel.size(-1)
        kernel_volume = kernel.size(0)

        in_feats_grad = torch.zeros((input_size, input_channel), dtype=torch.float, \
            device=in_feats.device)
        weight_grad = torch.zeros((kernel_volume, input_channel, output_channel), \
            dtype=torch.float, device=in_feats.device)

        if (transposed):
            conv_bwd_cuda(
                out_feats_grad, in_feats, kernel, kernel_size_code, sum_nnz, 
                in_feats_grad, weight_grad, knnz, kpos, omap, imap, ocsr, icsr,  
                buffer, tensorcore
            )
        else:
            conv_bwd_cuda(
                out_feats_grad, in_feats, kernel, kernel_size_code, sum_nnz, 
                in_feats_grad, weight_grad, knnz, kpos, imap, omap, icsr, ocsr,  
                buffer, tensorcore
            )

        return in_feats_grad, None, None, None, None, None, None, None, None, \
            weight_grad, None, None, None, None, None, None


class conv_d2(Function):
    
    @staticmethod
    @custom_fwd
    def forward(ctx,
                in_feats: torch.Tensor,
                kpos: torch.Tensor, 
                qkpos: torch.Tensor, 
                imap: torch.Tensor, 
                omap: torch.Tensor,
                in_channel: int,
                out_channel: int, 
                kernel: torch.Tensor, 
                kernel_size_code: int, 
                qsum_nnz: int, 
                nnz: Tuple[int, int], 
                transposed: bool, 
                tf32: bool,
                ) -> torch.Tensor:
        
        in_feats = in_feats.contiguous()
        kernel = kernel.contiguous()
        imap = imap.contiguous()
        omap = omap.contiguous()

        separate_mid = (nnz[0] == nnz[1])

        # check if the device supports tf32
        device_capability = torch.cuda.get_device_capability()
        device_capability = device_capability[0] * 100 + device_capability[1] * 10
        tf32 = device_capability >= 800

        if transposed:
            output_feats = torch.zeros((nnz[0], out_channel), \
                dtype=in_feats.dtype, device=in_feats.device)

            '''conv_data = dict()
            conv_data['in_feats'] = in_feats
            conv_data['kernel'] = kernel
            conv_data['ksize_code'] = kernel_size_code
            conv_data['sum_nnz'] = sum_nnz
            conv_data['knnz'] = knnz
            conv_data['kpos'] = kpos
            conv_data['imap'] = omap
            conv_data['omap'] = imap
            conv_data['out_nnz'] = nnz[0]
            torch.save(conv_data, '/home/eva_data/hongke21/conv_data/MK-SKh-FP16/' + str(time.time()) + '.pth')'''

            conv_fwd_simple_cuda(
                in_feats,
                kernel,
                kernel_size_code, 
                qsum_nnz, 
                output_feats,
                kpos, 
                qkpos, 
                omap,
                imap, 
                separate_mid, 
                tf32
            )
        else:
            output_feats = torch.zeros((nnz[1], out_channel), \
                dtype=in_feats.dtype, device=in_feats.device)

            '''conv_data = dict()
            conv_data['in_feats'] = in_feats
            conv_data['kernel'] = kernel
            conv_data['ksize_code'] = kernel_size_code
            conv_data['sum_nnz'] = sum_nnz
            conv_data['knnz'] = knnz
            conv_data['kpos'] = kpos
            conv_data['imap'] = imap
            conv_data['omap'] = omap
            conv_data['out_nnz'] = nnz[1]
            torch.save(conv_data, '/home/eva_data/hongke21/conv_data/MK-SKh-FP16/' + str(time.time()) + '.pth')'''

            conv_fwd_simple_cuda(
                in_feats,
                kernel,
                kernel_size_code, 
                qsum_nnz, 
                output_feats,
                kpos, 
                qkpos,
                imap,
                omap,
                separate_mid, 
                tf32
            )

    
        # TODO: replace in_feats with gathered features in in_buffer
        ctx.for_backwards = (in_feats, kernel, kpos, imap, omap, tf32)
        
        return output_feats

    
    @staticmethod
    @custom_bwd
    def backward(ctx, out_feats_grad: torch.Tensor):

        in_feats, kernel, kernel_size, knnz, kpos, \
            imap, omap, tensorcore = ctx.for_backwards
        
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
            tensorcore
        )

        return in_feats_grad, None, None, None, None, None, \
            weight_grad, None, None, None, None


def conv_func(input, in_channel, out_channel, kernel, 
    kernel_size_code, stride_code, bias, transposed, tensorcore):
    
    input_size = input.coords.size(0)
    kernel_volume = kernel.size(0)

    if kernel_size_code == conv_info_encoder([1, 1, 1]) \
        and stride_code == conv_info_encoder([1, 1, 1]):
        out_feats = input.feats.matmul(kernel)
        if bias is not None:
            out_feats += bias
        output = spTensor(coords=input.coords, feats=out_feats, buffer=input.buffer, \
            batchsize=input.batchsize, stride=input.stride, init_tag=input.init_tag)
    
    elif transposed: 
        output_stride = output_stride_compute(stride_code, input.stride, 1)
        kmap = input.kmaps.get((kernel_size_code, output_stride, stride_code))
        out_coords = input.cbook[output_stride]

        if len(kmap) == 8:
            out_feats = conv_d1.apply(
                input.feats, kmap[4], kmap[5], kmap[0], kmap[1], kmap[2], kmap[3],
                in_channel, out_channel, kernel, kernel_size_code, kmap[6], kmap[7], 
                input.buffer, transposed, 1
            )
        elif len(kmap) == 6:
            out_feats = conv_d2.apply(
                input.feats, kmap[2], kmap[3], kmap[0], kmap[1], in_channel, out_channel, 
                kernel, kernel_size_code, kmap[4], kmap[5], transposed, tensorcore
            )
        else:
            raise NotImplementedError
        
        if bias is not None:
            out_feats += bias
        output = spTensor(feats=out_feats, coords=out_coords, buffer=input.buffer, 
            batchsize=input.batchsize, stride=output_stride, init_tag=input.init_tag)

    else:
        output_stride = output_stride_compute(stride_code, input.stride, 0)
        kmap = input.kmaps.get((kernel_size_code, input.stride, stride_code)) 
        out_coords = input.coords

        separate_mid = (stride_code == conv_info_encoder([1, 1, 1]))
 
        if kmap is None:

            if input.init_tag[0] == 'simple':

                imap = - torch.ones((kernel_volume * input_size), \
                    dtype=torch.int, device=input.coords.device)
                knnz = torch.zeros((kernel_volume), dtype=torch.int, \
                    device=input.coords.device)
                kpos = torch.zeros((kernel_volume + 1), dtype=torch.int, \
                    device=input.coords.device)
                qkpos = torch.zeros((kernel_volume + 1), dtype=torch.int, \
                    device=input.coords.device)
                
                out_coords = mapping_simple_cuda(
                    input.coords, input.batchsize, kernel_size_code, kernel_volume, 
                    in_channel, out_channel, stride_code, input.stride, 
                    imap, knnz, kpos, qkpos, separate_mid
                )

                nonzero_idx = torch.nonzero(imap != -1)
                omap = imap[nonzero_idx]
                imap = (nonzero_idx % input_size).int()

                # knnz = knnz.cpu()
                # sum_nnz = knnz.sum().int()
                qsum_nnz = qkpos[-1].cpu().int()
                # print('sum nnz: %d' % sum_nnz)
                out_nnz = out_coords.size(0)
                # print('out nnz: %d' % out_nnz)
                # print("output coords max: ", out_coords.max(0))

                kmap = [imap, omap, kpos, qkpos, qsum_nnz, (input_size, out_nnz)]
                input.kmaps[(kernel_size_code, input.stride, stride_code)] = kmap
                input.cbook[output_stride] = out_coords
                input.init_tag = input.init_tag[1:]


            elif input.init_tag[0] == 'whole':

                imap = - torch.ones((input_size * kernel_volume), \
                    dtype=torch.int, device=input.coords.device)
                omap = - torch.ones((input_size * kernel_volume * 8), \
                    dtype=torch.int, device=input.coords.device)
                knnz = torch.zeros((kernel_volume), dtype=torch.int, \
                    device=input.coords.device)
                kpos = torch.zeros((kernel_volume), dtype=torch.int, \
                    device=input.coords.device)
                qkpos = torch.zeros((kernel_volume + 1), dtype=torch.int, \
                    device=input.coords.device)
                icsr = torch.zeros((input_size + 2), dtype=torch.int, \
                    device=input.coords.device)
                ocsr = torch.zeros(((input_size + 2) * 8), dtype=torch.int, \
                    device=input.coords.device)

                out_coords = mapping_cuda(
                    input.coords, kernel_size_code, kernel_volume, 
                    in_channel, out_channel, stride_code, input.stride, 
                    imap, omap, icsr, ocsr, knnz, kpos, qkpos, separate_mid
                )

                # knnz = knnz.cpu()
                # sum_nnz = knnz.sum().int()
                # print("sum nnz: %d" % sum_nnz)
                qsum_nnz = qkpos[-1].cpu().int()
                out_nnz = out_coords.size(0)
                # print('out nnz: %d' % out_nnz)
                # print("output coords max: ", out_coords.max(0))
        
                if input.buffer.size(0) < qsum_nnz * (in_channel + out_channel):
                    print("Newly allocated buffer.")
                    input.buffer = torch.zeros((qsum_nnz, (in_channel + out_channel)), \
                        dtype=input.feats.dtype, device=input.feats.device)

                kmap = [imap, omap, icsr, ocsr, kpos, qkpos, qsum_nnz, (input_size, out_nnz)]
                input.kmaps[(kernel_size_code, input.stride, stride_code)] = kmap
                input.cbook[output_stride] = out_coords
                input.init_tag = input.init_tag[1:]

                # j = 0
                # for i in range(input_size):
                #     if(icsr[i] < icsr[i + 1]):
                #         # print("%d-%d" % (imap[j] // 1186111, imap[j] % 1186111))
                #         j += 1
                # print("%.4f=%d/%d" % (sum_nnz/input_size, sum_nnz, input_size))
                # print("-------------")
            
            else:
                raise NotImplementedError

        if len(kmap) == 8:
            out_feats = conv_d1.apply(
                input.feats, kmap[4], kmap[5], kmap[0], kmap[1], kmap[2], kmap[3],
                in_channel, out_channel, kernel, kernel_size_code, kmap[6], kmap[7], 
                input.buffer, transposed, 1
            )
        elif len(kmap) == 6:
            out_feats = conv_d2.apply(
                input.feats, kmap[2], kmap[3], kmap[0], kmap[1], in_channel, out_channel, 
                kernel, kernel_size_code, kmap[4], kmap[5], transposed, tensorcore
            )
        else:
            raise NotImplementedError
        
        if bias is not None:
            out_feats += bias
        output = spTensor(feats=out_feats, coords=out_coords, buffer=input.buffer, 
            batchsize=input.batchsize, stride=output_stride, init_tag=input.init_tag)
    
    output.kmaps = input.kmaps
    output.cbook = input.cbook

    # print("(%d, %d) = %s" % (in_channel, out_channel, output.init_tag))
    
    return output
