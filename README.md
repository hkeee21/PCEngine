## PCEngine
PCEngine is an efficient **Engine** for sparse convolution inference in 3D **P**oint **C**louds. Generally, PCEngine contains techniques including a novel CSR-coded mapping format, an indicator-assisted FGMS (Fused Gather-Matmul-Scatter) fusion scheme and an adaptive dataflow to improve sparse convolution inference performance.

## News
[2023.06.02] The backward kernels in Fetch-on-Demand dataflow will be merged into the framework soon.

## Install
```
    python3 setup.py install
```

## Example
1. Correctness check of forward path (compared to [SpConv](https://github.com/traveller59/spconv))
```
    python3 check_fwd.py
```

## Citation
```bibtex
@inproceedings{hong2023pcengine,
  title = {{Exploiting Hardware Utilization and Adaptive Dataflow for Efficient Sparse Convolution in 3D Point Clouds}},
  author = {Hong, Ke and Yu, Zhongming and Dai, Guohao and Yang, Xinhao and Lian, Yaoxiu and Liu, Zehao and Xu, Ningyi and Dong, Yuhan and Wang, Yu},
  booktitle = {Conference on Machine Learning and Systems (MLSys)},
  year = {2023}
}
```
