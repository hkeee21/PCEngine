''' @ MLSys23 Artifacts Evaluation
    This code is to generate the ablation study results of the heuristics 
    adaptive dataflow into a .csv file.
        PCEngine Dataflows: Heuristics, Gather-GEMM-Scatter, Fetch-on-Demand
        Benchmarks: {ModelNet40, S3DIS, KITTI} X {SparseResNet, MinkUNet}.
        Command: $ python3 ablation-heuristics.py --save-file ${filename}
'''

import pandas as pd
import numpy as np
import argparse
import os
import sys
sys.path.append('../')
from lib.PCEngine.inference import pcengine_exe

parser = argparse.ArgumentParser()
parser.add_argument('--save-file', type=str, default='Fig13-heuristics')
args = parser.parse_args()

label_list = ['model', 'dataset', 'framework', 'dataflow', 'latency', 'normalized speedup']
model_list = ['resnet', 'unet']
dataset_list = ['modelnet40', 's3dis', 'kitti']
framework_list = ['PCEngine']
dataflow_dict = {'PCEngine': ['heuristics', 'gather-mm-scatter', 'fetch-on-demand'], 
                 'TorchSparse': ['gather-mm-scatter'], 
                 'SpConv2': ['implicit gemm']}
exe_dict = {'PCEngine': pcengine_exe}
framework_len = 0
for fr in framework_list:
    framework_len += len(dataflow_dict[fr])
latency_array = np.zeros((len(model_list) * len(dataset_list), framework_len))
model_col = []
dataset_col = []
framework_col = []
dataflow_col = []

results_row = 0
for model in model_list:
    for dataset in dataset_list:
        results_col = 0
        for framework in framework_list:
            exe = exe_dict[framework]
            for dataflow in dataflow_dict[framework]:
                model_col.append(model)
                dataset_col.append(dataset)
                framework_col.append(framework)
                dataflow_col.append(dataflow) 
                latency_array[results_row, results_col] = exe(model, dataset, dataflow, True)
                results_col += 1
        results_row += 1

latency_array_base = np.expand_dims(np.max(latency_array, axis=1), axis=1)
speedup_array = np.repeat(latency_array_base, framework_len, axis=1) / latency_array

latency_col = latency_array.reshape(-1).tolist()
speedup_col = speedup_array.reshape(-1).tolist()

results = list(zip(model_col, dataset_col, framework_col, dataflow_col, latency_col, speedup_col))
results_csv = pd.DataFrame(data=results, columns=label_list)
results_csv.to_csv(os.path.join('results', args.save_file + '.csv'), index=True)




