''' @ MLSys23 Artifacts Evaluation
    This code is to generate the sparse convolution kernel performance into a .csv file.
        PCEngine Dataflows: Heuristics, Gather-GEMM-Scatter, Fetch-on-Demand
        Baselines: SpConv (v2.2.6), TorchSparse (v2.0.0).
        Benchmarks: {ModelNet40, S3DIS, KITTI} X {various channel sizes}.
        Command: $ python3 sparse-convolution-kernel.py --save-file ${filename}
'''

import pandas as pd
import numpy as np
import argparse
import os
import sys
sys.path.append('../')
from lib.PCEngine.spconv_kernel import pcengine_kernel
from lib.SpConv2.spconv_kernel import spconv_kernel
from lib.TorchSparse.spconv_kernel import torchsparse_kernel

parser = argparse.ArgumentParser()
parser.add_argument('--save-file', type=str, default='Fig9b-kernel')
args = parser.parse_args()

label_list = ['input channel', 'output channel', 'dataset', 'framework', 'latency [ms]', 'normalized speedup']
channel_size_list = [(4, 16), (32, 32), (64, 96), (128, 128), (256, 384)]
dataset_list = ['modelnet40', 's3dis', 'kitti']
framework_list = ['PCEngine', 'SpConv2', 'TorchSparse']
dataflow_dict = {'PCEngine': ['gather-mm-scatter', 'fetch-on-demand'], 
                 'TorchSparse': ['gather-mm-scatter'], 
                 'SpConv2': ['implicit gemm']}
exe_dict = {'PCEngine': pcengine_kernel, 
            'SpConv2': spconv_kernel, 
            'TorchSparse': torchsparse_kernel}
latency_array = np.zeros((len(channel_size_list) * len(dataset_list), len(framework_list)))
c_in_col = []
c_out_col = []
dataset_col = []
framework_col = []

results_row = 0
for channel_size in channel_size_list:
    for dataset in dataset_list:
        results_col = 0
        for framework in framework_list:
            exe = exe_dict[framework]
            temp_latency = 10000
            for dataflow in dataflow_dict[framework]:
                temp_latency = min(temp_latency, exe(channel_size[0], channel_size[1], dataset, dataflow))
            c_in_col.append(channel_size[0])
            c_out_col.append(channel_size[1])
            dataset_col.append(dataset)
            framework_col.append(framework)
            latency_array[results_row, results_col] = temp_latency
            results_col += 1
        results_row += 1

latency_array_base = np.expand_dims(np.max(latency_array, axis=1), axis=1)
speedup_array = np.repeat(latency_array_base, len(framework_list), axis=1) / latency_array

latency_array = latency_array * 1000

latency_col = latency_array.reshape(-1).tolist()
speedup_col = speedup_array.reshape(-1).tolist()

results = list(zip(c_in_col, c_out_col, dataset_col, framework_col, latency_col, speedup_col))
results_csv = pd.DataFrame(data=results, columns=label_list)
results_csv.to_csv(os.path.join('results', args.save_file + '.csv'), index=True, float_format='%.4f')

