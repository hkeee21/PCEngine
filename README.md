# PCEngine(v2.0.0) Development
PCEngine(v2.0.0) contains both `Gather-MM-Scatter` and `Fetch-on-Demand` dataflows. Compared to PCEngine(v1.x), PCEngine is faster and more functional. In v2.0.0, MMs in both dataflows are implemented using `block fusion technique`. 

## Current Constraints 
1. Gather-MM-Scatter dataflow: channel size % 2 == 0.
2. Fetch-on-Demand dataflow: channel size % 2 == 0, CUDA_ARCH >= 800.

## Details on Gather-MM-Scatter
1. Feature amount for each weight position is a multiple of _M_ for tiling purpose. _M_ can be adjusted by changing the **third input** of kernel `exclusive_scan_for_kernel_quantified` in `backend/hash.cu`. Current _M_ is set to 128.
2. The specific design for channel size % 8 != 0 (e.g. channel size = 4, 6) lies in kernel `naive_gemm_{fp16, fp32}_2`, which is supposed to be replaced in the new GEMM kernel. 
3. The buffer should be reset if no padding is used in the new GEMM kernel.
4. No change in gather or scatter kernel is needed for channel size % == 2 cases.

## Install
`python3 setup.py install`

## Guide
**Correctness Validation**

`python3 results.py --dataflow={'D1', 'D2'}`

Note that channel size should be a small numnber (from 4 to 32) to accelerate CPU computing. This command can be run without uploading new data.

**Dataflow Switch**

Change `resnet_tag` or `unet_tag` in `configs/default/default.yaml`, where 'whole' denotes D1 and 'simple' denotes D2. Make sure there is an 'end' in the end.

**End-to-End Performance**

`python3 evaluate.py --fast --restart`

**Datasets should be uploaded first** and all data paths in `datasets` should be modifed respectively.




