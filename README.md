# spconv_mod
A modified deep learning framework for 3D point clouds.

## Dependencies (to be removed)
torchsparse

## Baseline Test Results
one convolution time (ms)

| sparsity | nnz | TorchSparse | MinkowskiEngine | Baseline-3 |
| ----- | ----- | ----- | ----- | ----- |
| 0.002 | 2000 | 0.5811 | 0.1156 | 1.9221 |
| 0.004 | 4000 | 0.6093 | 0.1231 | 2.1581 |
| 0.006 | 6000 | 0.6115 | 0.1180 | 2.4345 |
| 0.008 | 8000 | 0.6303 | 0.1266 | 3.2321 |
| 0.010 | 10000 | 0.6465 | 0.1307 | 3.9403 |
| 0.012 | 12000 | 0.9360 | 0.1742 | 4.9561 |
| 0.014 | 14000 | 0.9263 | 0.1774 | 5.4127 |
| 0.016 | 16000 | 1.0259 | 0.2123 | 6.6986 |
| 0.018 | 18000 | 0.9471 | 0.1865 | 7.8912 |
| 0.020 | 20000 | 0.9148 | 0.2415 | 8.9621 |

## Update
Baseline-4 is 3x faster than Baseline-3.

