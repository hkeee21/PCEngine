import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import torch
import math
import numpy as np
import time
from .sptensor import spTensor
from typing import Dict, List, Tuple, Union
from PCEngine.backend import mapping_cuda, mapping_simple_cuda, mapping_coded_csr_cuda, \
    conv_fwd_cuda, conv_fwd_simple_cuda, conv_fwd_naive_cuda, conv_bwd_cuda, \
    gather_coded_CSR_cuda, scatter_coded_CSR_cuda, \
    gather_vanilla_cuda, scatter_vanilla_cuda, map_to_matrix_cuda, \
    torchsparse_gather_cuda, torchsparse_scatter_cuda
from .utils import output_stride_compute, conv_info_encoder, conv_info_decoder


class conv3d(nn.Module):
    
    def __init__(self,
                 in_channels: int = 16,
                 out_channels: int = 32,
                 kernel_size: Union[int, List[int]] = 3,
                 stride: Union[int, List[int]] = 1,
                 padding: Union[int, List[int]] = 0,
                 bias: bool = False, 
                 transposed: bool = False, 
                 tc_mode: bool = True,
                 coded_csr_flag: int = 0,
                 heuristics_flag: int = 0
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transposed = transposed
        self.coded_csr_flag = coded_csr_flag
        self.heuristics_flag = heuristics_flag

        # kernel size coding
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size, kernel_size]
        elif isinstance(kernel_size, list):
            self.kernel_size = kernel_size
        else:
            raise NotImplementedError
        self.kernel_size_code = conv_info_encoder(self.kernel_size)
        self.kernel_volume = np.prod(self.kernel_size, dtype=int)
        
        # stride coding
        if isinstance(stride, int):
            self.stride = [stride, stride, stride]
        elif isinstance(stride, list):
            self.stride = stride
        else:
            raise NotImplementedError
        self.stride_code = conv_info_encoder(self.stride)

        # padding coding
        if isinstance(padding, int):
            self.padding = [padding, padding, padding]
        elif isinstance(padding, list):
            self.padding = padding
        else:
            raise NotImplementedError
        self.padding_code = conv_info_encoder(self.padding)

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
                        padding_code=self.padding_code, 
                        bias=self.bias, 
                        transposed=self.transposed, 
                        tensorcore=self.tensorcore,
                        coded_csr_flag=self.coded_csr_flag,
                        heuristics_flag=self.heuristics_flag
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

            conv_fwd_cuda(
                in_feats, kernel, kernel_size_code, 
                sum_nnz, output_feats, knnz, kpos, 
                omap, imap, ocsr, icsr, buffer, 
                separate_mid, tensorcore
            )
        else:
            output_feats = torch.zeros((nnz[1], out_channel), \
                dtype=in_feats.dtype, device=in_feats.device)

            conv_fwd_cuda(
                in_feats, kernel, kernel_size_code, 
                sum_nnz, output_feats, knnz, kpos, 
                imap, omap, icsr, ocsr, buffer, 
                separate_mid, tensorcore
            )
    
        # TODO: replace in_feats with gathered features in in_buffer
        ctx.for_backwards = (in_feats, kernel, knnz, imap, omap, \
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
                knnz: torch.Tensor, 
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
                tensorcore: bool,
                heuristics_flag: int
                ) -> torch.Tensor:
        
        in_feats = in_feats.contiguous()
        kernel = kernel.contiguous()
        imap = imap.contiguous()
        omap = omap.contiguous()

        separate_mid = (nnz[0] == nnz[1])

        if transposed:
            output_feats = torch.zeros((nnz[0], out_channel), \
                dtype=in_feats.dtype, device=in_feats.device)
            
            if heuristics_flag > 0:
                knnz = knnz.cpu()
                sum_nnz = knnz.sum().int()
                conv_fwd_naive_cuda(
                    in_feats, kernel, kernel_size_code, 
                    sum_nnz, output_feats, knnz, kpos, 
                    omap, imap, False, False
                )
            
            else:
                conv_fwd_simple_cuda(
                    in_feats, kernel, kernel_size_code, 
                    qsum_nnz, output_feats, kpos, qkpos, 
                    omap, imap, separate_mid, True
                )

        else:
            output_feats = torch.zeros((nnz[1], out_channel), \
                dtype=in_feats.dtype, device=in_feats.device)

            if heuristics_flag > 0:
                knnz = knnz.cpu()
                sum_nnz = knnz.sum().int()
                conv_fwd_naive_cuda(
                    in_feats, kernel, kernel_size_code, 
                    sum_nnz, output_feats, knnz, kpos,
                    imap, omap, False, False
                )
            
            else:
                conv_fwd_simple_cuda(
                    in_feats, kernel, kernel_size_code, 
                    qsum_nnz, output_feats, kpos, qkpos,
                    imap, omap, separate_mid, True
                )

    
        # TODO: replace in_feats with gathered features in in_buffer
        ctx.for_backwards = (in_feats, kernel, kpos, imap, omap, tensorcore)
        
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
            out_feats_grad, in_feats, kernel, kernel_size,
            in_feats_grad, weight_grad, knnz, kpos, imap, omap, 
            tensorcore
        )

        return in_feats_grad, None, None, None, None, None, \
            weight_grad, None, None, None, None
    

class conv_coded_csr_test(Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
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
                tensorcore: bool, 
                coded_csr_flag: int) -> torch.Tensor:
        
        in_feats = in_feats.contiguous()
        kernel = kernel.contiguous()
        imap = imap.contiguous()
        omap = omap.contiguous()
        icsr = icsr.contiguous()
        ocsr = ocsr.contiguous()
        buffer = buffer.contiguous()

        k_vol = kernel.size(0)
        precompute_mid = nnz[0] == nnz[1]

        results_dict = np.load('coded-csr-intermediate-data.npy', allow_pickle=True).item()

        if transposed:
            output_feats = torch.zeros((nnz[0], out_channel), \
                dtype=in_feats.dtype, device=in_feats.device)
            
            if coded_csr_flag == 1:

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    gather_coded_CSR_cuda(in_feats, kpos, omap, ocsr, buffer)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['coded-csr-gather'].append(((end - start) * 100000))

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    scatter_coded_CSR_cuda(sum_nnz * in_channel, output_feats, kpos, imap, icsr, buffer)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['coded-csr-scatter'].append(((end - start) * 100000))

            elif coded_csr_flag == 2:

                imap_matrix = - torch.ones((output_feats.size(0), k_vol), 
                    dtype=torch.int, device=imap.device)
                omap_matrix = - torch.ones((in_feats.size(0), k_vol), 
                    dtype=torch.int, device=omap.device)
                
                map_to_matrix_cuda(nnz[0], k_vol, icsr, imap, imap_matrix)
                map_to_matrix_cuda(nnz[1], k_vol, ocsr, omap, omap_matrix)

                assert torch.sum(imap_matrix >= 0) == sum_nnz
                assert torch.sum(omap_matrix >= 0) == sum_nnz

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    gather_vanilla_cuda(k_vol, in_feats, kpos, omap_matrix, buffer)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['vanilla-gather'].append(((end - start) * 100000))

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    scatter_vanilla_cuda(sum_nnz * in_channel, k_vol, output_feats, kpos, imap_matrix, buffer)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['vanilla-scatter'].append(((end - start) * 100000))
            
            elif coded_csr_flag == 3:

                imap_matrix = - torch.ones((output_feats.size(0), k_vol), 
                    dtype=torch.int, device=imap.device)
                omap_matrix = - torch.ones((in_feats.size(0), k_vol), 
                    dtype=torch.int, device=omap.device)
                
                map_to_matrix_cuda(nnz[0], k_vol, icsr, imap, imap_matrix)
                map_to_matrix_cuda(nnz[1], k_vol, ocsr, omap, omap_matrix)

                assert torch.sum(imap_matrix >= 0) == sum_nnz
                assert torch.sum(omap_matrix >= 0) == sum_nnz

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    torchsparse_gather_cuda(in_feats, buffer, k_vol, kpos, 
                                        imap_matrix, omap_matrix, transposed, precompute_mid)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['torchsparse-gather'].append(((end - start) * 100000))

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    torchsparse_scatter_cuda(output_feats, buffer, sum_nnz * in_channel, k_vol, kpos, 
                                        imap_matrix, omap_matrix, transposed, precompute_mid)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['torchsparse-scatter'].append(((end - start) * 100000))

            else:
                raise NotImplementedError

        else:
            output_feats = torch.zeros((nnz[1], out_channel), \
                dtype=in_feats.dtype, device=in_feats.device)

            if coded_csr_flag == 1:

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    gather_coded_CSR_cuda(in_feats, kpos, imap, icsr, buffer)
                torch.cuda.synchronize()
                end=time.time()
                # print('gather duration: %.4f us.' % ((end - start) * 1000000))
                results_dict['coded-csr-gather'].append(((end - start) * 100000))

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    scatter_coded_CSR_cuda(sum_nnz * in_channel, output_feats, kpos, omap, ocsr, buffer)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['coded-csr-scatter'].append(((end - start) * 100000))

            elif coded_csr_flag == 2:

                imap_matrix = - torch.ones((in_feats.size(0), k_vol), 
                    dtype=torch.int, device=imap.device)
                omap_matrix = - torch.ones((output_feats.size(0), k_vol), 
                    dtype=torch.int, device=omap.device)
                
                map_to_matrix_cuda(nnz[0], k_vol, icsr, imap, imap_matrix)
                map_to_matrix_cuda(nnz[1], k_vol, ocsr, omap, omap_matrix)

                assert torch.sum(imap_matrix >= 0) == sum_nnz
                assert torch.sum(omap_matrix >= 0) == sum_nnz

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    gather_vanilla_cuda(k_vol, in_feats, kpos, imap_matrix, buffer)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['vanilla-gather'].append(((end - start) * 100000))

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    scatter_vanilla_cuda(sum_nnz * in_channel, k_vol, output_feats, kpos, omap_matrix, buffer)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['vanilla-scatter'].append(((end - start) * 100000))

            elif coded_csr_flag == 3:

                imap_matrix = - torch.ones((in_feats.size(0), k_vol), 
                    dtype=torch.int, device=imap.device)
                omap_matrix = - torch.ones((output_feats.size(0), k_vol), 
                    dtype=torch.int, device=omap.device)
                
                map_to_matrix_cuda(nnz[0], k_vol, icsr, imap, imap_matrix)
                map_to_matrix_cuda(nnz[1], k_vol, ocsr, omap, omap_matrix)

                assert torch.sum(imap_matrix >= 0) == sum_nnz
                assert torch.sum(omap_matrix >= 0) == sum_nnz

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    torchsparse_gather_cuda(in_feats, buffer, k_vol, kpos, 
                                        imap_matrix, omap_matrix, transposed, precompute_mid)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['torchsparse-gather'].append(((end - start) * 100000))

                torch.cuda.synchronize()
                start=time.time()
                for _ in range(10):
                    torchsparse_scatter_cuda(output_feats, buffer, sum_nnz * in_channel, k_vol, kpos, 
                                        imap_matrix, omap_matrix, transposed, precompute_mid)
                torch.cuda.synchronize()
                end=time.time()
                results_dict['torchsparse-scatter'].append(((end - start) * 100000))

            else:
                raise NotImplementedError
    
        np.save('coded-csr-intermediate-data.npy', results_dict)

        ctx.for_backwards = (in_feats, kernel, knnz, imap, omap, \
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


def conv_func(input, in_channel, out_channel, kernel, 
    kernel_size_code, stride_code, padding_code, bias, 
    transposed, tensorcore, coded_csr_flag, heuristics_flag):
    
    input_size = input.coords.size(0)
    kernel_volume = kernel.size(0)

    if kernel_size_code == conv_info_encoder([1, 1, 1]) \
        and stride_code == conv_info_encoder([1, 1, 1]):
        out_feats = input.feats.matmul(kernel)
        if bias is not None:
            out_feats += bias
        output = spTensor(coords=input.coords, feats=out_feats, buffer=input.buffer, \
            batchsize=input.batchsize, stride=input.stride, init_tag=input.init_tag, \
            coords_max=input.coords_max, coords_min=input.coords_min)
    
    elif transposed: 
        output_stride = output_stride_compute(stride_code, input.stride, 1)
        kmap = input.kmaps.get((kernel_size_code, output_stride, stride_code))
        out_coords = input.cbook[output_stride]

        if coded_csr_flag == 0:
            if len(kmap) == 8:
                out_feats = conv_d1.apply(
                    input.feats, kmap[4], kmap[5], kmap[0], kmap[1], kmap[2], kmap[3],
                    in_channel, out_channel, kernel, kernel_size_code, kmap[6], kmap[7], 
                    input.buffer, transposed, tensorcore
                )
            elif len(kmap) == 7:
                out_feats = conv_d2.apply(
                    input.feats, kmap[6], kmap[2], kmap[3], kmap[0], kmap[1], in_channel, out_channel, 
                    kernel, kernel_size_code, kmap[4], kmap[5], transposed, tensorcore, heuristics_flag
                )
            else:
                raise NotImplementedError
        elif coded_csr_flag == 1 or coded_csr_flag == 2 or coded_csr_flag == 3:
            out_feats = conv_coded_csr_test.apply(
                input.feats, kmap[4], kmap[5], kmap[0], kmap[1], kmap[2], kmap[3],
                in_channel, out_channel, kernel, kernel_size_code, kmap[6], kmap[7], 
                input.buffer, transposed, tensorcore, coded_csr_flag
            )
        else:
            raise NotImplementedError
        
        if bias is not None:
            out_feats += bias
        output = spTensor(feats=out_feats, coords=out_coords, buffer=input.buffer, \
            batchsize=input.batchsize, stride=output_stride, init_tag=input.init_tag, \
            coords_max=input.coords_max, coords_min=input.coords_min)

    else:
        output_stride = output_stride_compute(stride_code, input.stride, 0)
        kmap = input.kmaps.get((kernel_size_code, input.stride, stride_code)) 
        out_coords = input.coords

        separate_mid = (stride_code == conv_info_encoder([1, 1, 1]))
 
        if kmap is None:

            if input.init_tag[0] == 'simple':

                imap = - torch.ones((kernel_volume * input_size), \
                    dtype=torch.int, device=input.coords.device)
                knnz = torch.zeros((kernel_volume + 1), dtype=torch.int, \
                    device=input.coords.device)
                kpos = torch.zeros((kernel_volume + 1), dtype=torch.int, \
                    device=input.coords.device)
                qkpos = torch.zeros((kernel_volume + 1), dtype=torch.int, \
                    device=input.coords.device)
                
                # coords_max = input.coords_max
                # kernel_size = conv_info_decoder(kernel_size_code)
                # stride = conv_info_decoder(stride_code)
                # padding = conv_info_decoder(padding_code)
                # input.coords_max[0] = (coords_max[0] + 2 * padding[0] - (kernel_size[0] - 1)) // stride[0]
                # input.coords_max[1] = (coords_max[1] + 2 * padding[1] - (kernel_size[1] - 1)) // stride[1]
                # input.coords_max[2] = (coords_max[2] + 2 * padding[2] - (kernel_size[2] - 1)) // stride[2]
                
                out_coords = mapping_simple_cuda(
                    input.coords, input.batchsize, kernel_size_code, kernel_volume, 
                    in_channel, out_channel, stride_code, input.stride, padding_code, 
                    input.coords_min[0], input.coords_min[1], input.coords_min[2], 
                    input.coords_max[0], input.coords_max[1], input.coords_max[2],  
                    imap, knnz, kpos, qkpos, separate_mid
                )

                nonzero_idx = torch.nonzero(imap != -1)
                omap = imap[nonzero_idx]
                imap = (nonzero_idx % input_size).int()

                # knnz = knnz.cpu()
                qsum_nnz = qkpos[-1].cpu().int()
                # print('sum nnz: %d' % sum_nnz)
                out_nnz = out_coords.size(0)
                # print('out nnz: %d' % out_nnz)
                # print("output coords max: ", out_coords.max(0))

                kmap = [imap, omap, kpos, qkpos, qsum_nnz, (input_size, out_nnz), knnz]
                input.kmaps[(kernel_size_code, input.stride, stride_code)] = kmap
                input.cbook[output_stride] = out_coords
                input.init_tag = input.init_tag[1:]


            elif input.init_tag[0] == 'whole':

                if coded_csr_flag == 0:

                    imap = - torch.ones((input_size * kernel_volume), \
                        dtype=torch.int, device=input.coords.device)
                    omap = - torch.ones((input_size * kernel_volume * 4), \
                        dtype=torch.int, device=input.coords.device)
                    knnz = torch.zeros((kernel_volume), dtype=torch.int, \
                        device=input.coords.device)
                    kpos = torch.zeros((kernel_volume), dtype=torch.int, \
                        device=input.coords.device)
                    icsr = torch.zeros((input_size + 2), dtype=torch.int, \
                        device=input.coords.device)
                    ocsr = torch.zeros(((input_size + 2) * 4), dtype=torch.int, \
                        device=input.coords.device)

                    out_coords = mapping_cuda(
                        input.coords, kernel_size_code, kernel_volume, in_channel, 
                        out_channel, stride_code, input.stride, padding_code, 
                        input.coords_min[0], input.coords_min[1], input.coords_min[2], 
                        input.coords_max[0], input.coords_max[1], input.coords_max[2], 
                        imap, omap, icsr, ocsr, knnz, kpos, separate_mid
                    )

                    knnz = knnz.cpu()
                    sum_nnz = knnz.sum().int()
                    # print("sum nnz: %d" % sum_nnz)
                    out_nnz = out_coords.size(0)
                    # print('out nnz: %d' % out_nnz)
                    # print("output coords max: ", out_coords.max(0))
        
                    if input.buffer.size(0) < sum_nnz * (in_channel + out_channel):
                        print("Newly allocated buffer.")
                        input.buffer = torch.zeros((sum_nnz, (in_channel + out_channel)), \
                            dtype=input.feats.dtype, device=input.feats.device)

                    kmap = [imap, omap, icsr, ocsr, knnz, kpos, sum_nnz, (input_size, out_nnz)]
                    input.kmaps[(kernel_size_code, input.stride, stride_code)] = kmap
                    input.cbook[output_stride] = out_coords
                    input.init_tag = input.init_tag[1:]
                
                else:
                    
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
                    ocsr = torch.zeros(((input_size + 2)), dtype=torch.int, \
                        device=input.coords.device)

                    out_coords = mapping_coded_csr_cuda(
                        input.coords, kernel_size_code, kernel_volume, in_channel, 
                        out_channel, stride_code, input.stride, padding_code, 
                        input.coords_min[0], input.coords_min[1], input.coords_min[2], 
                        input.coords_max[0], input.coords_max[1], input.coords_max[2], 
                        imap, omap, icsr, ocsr, knnz, kpos, separate_mid
                    )

                    knnz = knnz.cpu()
                    sum_nnz = knnz.sum().int()
                    # print("sum nnz: %d" % sum_nnz)
                    out_nnz = out_coords.size(0)
                    # print('out nnz: %d' % out_nnz)
                    # print("output coords max: ", out_coords.max(0))
        
                    if input.buffer.size(0) < sum_nnz * (in_channel + out_channel):
                        print("Newly allocated buffer.")
                        input.buffer = torch.zeros((sum_nnz, (in_channel + out_channel)), \
                            dtype=input.feats.dtype, device=input.feats.device)

                    kmap = [imap, omap, icsr, ocsr, knnz, kpos, sum_nnz, (input_size, out_nnz)]
                    input.kmaps[(kernel_size_code, input.stride, stride_code)] = kmap
                    input.cbook[output_stride] = out_coords
                    input.init_tag = input.init_tag[1:]
            
            else:
                raise NotImplementedError

        if coded_csr_flag == 0:
            if len(kmap) == 8:
                out_feats = conv_d1.apply(
                    input.feats, kmap[4], kmap[5], kmap[0], kmap[1], kmap[2], kmap[3],
                    in_channel, out_channel, kernel, kernel_size_code, kmap[6], kmap[7], 
                    input.buffer, transposed, tensorcore
                )
            elif len(kmap) == 7:
                out_feats = conv_d2.apply(
                    input.feats, kmap[6], kmap[2], kmap[3], kmap[0], kmap[1], in_channel, out_channel, 
                    kernel, kernel_size_code, kmap[4], kmap[5], transposed, tensorcore, heuristics_flag
                )
            else:
                raise NotImplementedError
        elif coded_csr_flag == 1 or coded_csr_flag == 2 or coded_csr_flag == 3:
            out_feats = conv_coded_csr_test.apply(
                input.feats, kmap[4], kmap[5], kmap[0], kmap[1], kmap[2], kmap[3],
                in_channel, out_channel, kernel, kernel_size_code, kmap[6], kmap[7], 
                input.buffer, transposed, tensorcore, coded_csr_flag
            )
        else:
            raise NotImplementedError
        
        if bias is not None:
            out_feats += bias
        output = spTensor(feats=out_feats, coords=out_coords, buffer=input.buffer, \
            batchsize=input.batchsize, stride=output_stride, init_tag=input.init_tag, \
            coords_max=input.coords_max, coords_min=input.coords_min)
    
    output.kmaps = input.kmaps
    output.cbook = input.cbook

    # print("(%d, %d) = %s" % (in_channel, out_channel, output.init_tag))
    
    return output
