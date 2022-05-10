# spconv_mod
A modified deep learning framework for 3D point clouds.

## Dependencies (to be removed)
torchsparse

## Baseline Test Results
one convolution time (ms)

| sparsity | nnz | TorchSparse | MinkowskiEngine | Baseline-3 |
| ----- | ----- | ----- | ----- | ----- |
| 0.001 | 1000 | 0.4822 | 0.1002 | 0.9840 |
| 0.002 | 2000 | 0.5811 | 0.1156 | 1.8811 |
| 0.003 | 3000 | 0.6102 | 0.1224 | 3.2595 |
| 0.004 | 4000 | 0.6093 | 0.1231 | 4.5592 |
| 0.005 | 5000 | 0.6317 | 0.1156 | 5.5349 |
| 0.006 | 6000 | 0.6115 | 0.1180 | 7.3962 |
| 0.007 | 7000 | 0.6120 | 0.1203 | 9.2045 |
| 0.008 | 8000 | 0.6303 | 0.1266 | 11.5419 |
| 0.009 | 9000 | 0.6125 | 0.1289 | 13.2387 |
| 0.010 | 10000 | 0.6465 | 0.1307 | 15.3160 |

## Update
Baseline-3 is 3x faster than Baseline-1.

