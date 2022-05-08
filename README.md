# spconv_mod
A modified deep learning framework for 3D point clouds.

## Dependencies (to be removed)
torchsparse

## Baseline Test Results
one convolution time (ms)

| sparsity | nnz | TorchSparse | MinkowskiEngine | Baseline |
| ----- | ----- | ----- | ----- | ----- |
| 0.001 | 1000 | 0.4822 | 0.1002 | 0.9335 |
| 0.002 | 2000 | 0.5811 | 0.1156 | 2.2856 |
| 0.003 | 3000 | 0.6102 | 0.1224 | 4.9465 |
| 0.004 | 4000 | 0.6093 | 0.1231 | 7.9127 |
| 0.005 | 5000 | 0.6317 | 0.1156 | 12.2252 |
| 0.006 | 6000 | 0.6115 | 0.1180 | 17.3593 |
| 0.007 | 7000 | 0.6120 | 0.1203 | 23.3112 |
| 0.008 | 8000 | 0.6303 | 0.1266 | 30.4229 |
| 0.009 | 9000 | 0.6125 | 0.1289 | 38.2827 |
| 0.010 | 10000 | 0.6465 | 0.1307 | 47.0350 |

