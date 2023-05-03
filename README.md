## **PCEngine-AE**
This is the repository of PCEngine code for artifacts evaluation purpose. PCEngine is an optimized engine for voxel-based point cloud neural networks. Based on a noval coded-CSR format, indicator-assisted segmented GEMM (GEneral Matrix Multiplication) fusion and heuristics adaptive dataflow, PCEngine outperforms previous works in sparse convolution inference.

**[Notice]** *This repository will not be updated in the future. Please refer to the main branch for the latest version.* 

### **Requirements**
``` 
    CUDA 11.1+
    PyTorch 1.10.0+
    TorchSparse 2.0.0
    SpConv 2.2.3+
```
Note the artifacts have been tested under following environments
``` 
    CUDA 11.1/11.6
    PyTorch 1.10.0/1.12.0
    TorchSparse 2.0.0
    SpConv 2.2.3/2.2.6
```
on an Nvidia RTX 2080 GPU or an Nvidia RTX 3090 GPU.

### **Setup**
In order to use ```ncu_report``` package, please first config your ```$CUDA_HOME``` environment. It probably locates at ```/usr/local/cuda``` or something like that.
Then config the ncu_report environment:
```export PYTHONPATH="${CUDA_HOME}/nsight-compute-xxxx.x.x/extras/python"```
If the path does not exist, try to use a higher version CUDA.

- Install PCEngine, TorchSparse and SpConv
    ```
        bash setup.sh
    ```
- (Optional) Install step by step
    - Install PCEngine
    ```
        cd lib/PCEngine
        python setup.py install
    ```
    - TorchSparse can be installed through the [official repository](https://github.com/mit-han-lab/torchsparse) as follows
    ```
        cd lib/TorchSparse
        git clone https://github.com/mit-han-lab/torchsparse.git
        cd torchsparse
        python setup.py install
    ```
    - SpConv can be installed through `pip`. Please refer to the [official repository](https://github.com/traveller59/spconv).
- Download datasets

    Please download [AE-datasets](https://drive.google.com/file/d/1137pnfO2l-EP2ZTGBGfPvGrwBl-LX331/view?usp=share_link) and unzip the `AE-datasets.zip` file into a PCEngine-AE subdirectory as follows
    ```
    |---- PCEngine-AE directory
            |---- AE-datasets
            |       |---- ModelNet40
            |       |---- S3DIS
            |       |---- KITTI
            |
            |---- evaluation
            |---- correctness-check.py
            |---- lib
            |---- setup.sh
    ```

### **Evaluation**
- To verify functionality
  
  Run the following command to compare the sparse convolution output with SpConv.
  ```
    python correctness.py
  ```
- To run artifacts evaluation
  
  Evaluate the end-to-end performance and generate the results into a `.csv` file (Fig. 9(a))
  ```
    cd evaluation
    python Fig9a-end-to-end.py
  ```
  Evaluate the sparse convolution performance and generate the results into a `.csv` file (Fig. 9(b))
  ```
    cd evaluation
    python Fig9b-kernel.py
  ```
  Compare gather and scatter performance to TorchSparse and generate the results into a `.csv` file (Fig. 10)
  - Add "--fast" to reduce the evaluation time.
  ```
    cd evaluation
    python Fig10-gather-scatter.py
  ```
  Conduct the ablation study on coded-CSR format and generate the results into a `.csv` file (Fig. 11)
  - Note that Fig11-coded-CSR.py and Fig10-gather-scatter.py can not be run simultaneously, 
    as the two scripts reuse the same file to store intermediate results.
  - Add "--fast" to reduce the evaluation time.
  ```
    cd evaluation
    python Fig11-coded-CSR.py
  ```
  Conduct the ablation study on GEMM scheme and generate the results into a `.csv` file (Fig. 12)
  ```
    cd evaluation
    python Fig12-GEMM.py
  ```
  Conduct the ablation study on heuristics adaptive dataflow and generate the results into a `.csv` file (Fig. 13)
  ```
    cd evaluation
    python Fig13-heuristics.py
  ```


