# spconv_mod
A modified deep learning framework for 3D point clouds.


## Baseline Test Results
one convolution time (on RTX 3090, ms)

real data: (nnz = 162250)
| in channel | out channel | TorchSparse | MinkowskiEngine | HashGemm-4 | FuseConv-1 | 
| ----- | ----- | ----- | ----- | ----- | ----- |
| 3  | 32 | 1.4104 | 0.8379 | 2.2746 | 2.1151 |
| 64 | 128| 4.9774 | 5.2841 | 10.2939| 6.1627 |
| 256| 512| 24.6385| 25.5286| 119.0847| 27.944 |


