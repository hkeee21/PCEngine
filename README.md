# spconv_mod
A modified deep learning framework for 3D point clouds.


## Baseline Test Results
one convolution time (on Tesla V100, ms)

real data:
| nnz | TorchSparse | MinkowskiEngine | Baseline-4 | HashGemm-1 | 
| ----- | ----- | ----- | ----- | ----- |
| 162250 | 2.0749 | 1.2255 | 8.2806 | 4.2365 |


