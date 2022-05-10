# spconv_mod
A modified deep learning framework for 3D point clouds.


## Dependencies (to be removed)
torchsparse


## Baseline Test Results
one convolution time (ms)

random data:
| sparsity | nnz | TorchSparse | MinkowskiEngine | Baseline-4 |
| ----- | ----- | ----- | ----- | ----- |
| 0.002 | 2000 | 0.5811 | 0.1156 | 1.4771 |
| 0.004 | 4000 | 0.6093 | 0.1231 | 1.5279 |
| 0.006 | 6000 | 0.6115 | 0.1180 | 1.4409 |
| 0.008 | 8000 | 0.6303 | 0.1266 | 1.4753 |
| 0.010 | 10000 | 0.6465 | 0.1307 | 1.5915 |
| 0.012 | 12000 | 0.9360 | 0.1742 | 1.7092 |
| 0.014 | 14000 | 0.9263 | 0.1774 | 1.6492 |
| 0.016 | 16000 | 1.0259 | 0.2123 | 1.7697 |
| 0.018 | 18000 | 0.9471 | 0.1865 | 1.7357 |
| 0.020 | 20000 | 0.9148 | 0.2415 | 1.9200 |

real data:
| nnz | TorchSparse | MinkowskiEngine | Baseline-4 |
| ----- | ----- | ----- | ----- |
| 162250 | 2.0749 | 1.2255 | 8.2806 |


