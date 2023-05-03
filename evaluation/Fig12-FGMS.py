''' @ MLSys23 Artifacts Evaluation
    This code is to generate the ablation study results of the indicator-assisted 
    segmented FGMS fusion scheme into a .csv file.
        GEMM schemes: separate FGMSs, batched FGMSs, indicator-assisted segmented FGMS fusion.
        Benchmarks: {ModelNet40, S3DIS, KITTI}.
        Convolution setups: 
            input channel: 64
            output channel: 64
            kernel size: 3
        Command: $ python3 ablation-FGMS.py --save-file ${filename}
'''

import pandas as pd
import numpy as np
import argparse
import os
import ncu_report
import sys
sys.path.append('../')
from lib.PCEngine.fgms_test import fgms_test

parser = argparse.ArgumentParser()
parser.add_argument('--save-file', type=str, default='Fig12-FGMS')
args = parser.parse_args()

label_list = ['dataset', 'scheme', 'latency [ms]', 'normalized speedup', 'FLOPs', 'normalized utilization']
dataset_list = ['modelnet40', 's3dis', 'kitti']
scheme_list = ['separate', 'batched', 'fused']
dataset_col = []
scheme_col = []
latency_array = np.zeros((len(dataset_list), len(scheme_list)))

# load FLOPs of each GEMM scheme from generated .ncu-rep file first
# since the FLOPs remain the same with different gpus, 
# no new .ncu-rep file needs to be generated on GPUs beyond RTX 3090 and RTX 2080

def get_cf_metrics(action, cf_metrics):
  tmp = list()
  for m in cf_metrics:
    tmp.append(my_action.metric_by_name(m).as_double())
  return tmp

flops_array = np.zeros((len(dataset_list), len(scheme_list)))
for d, dataset in enumerate(dataset_list):
    prof_name = 'ncu-report/mlsys23-ae-fgms-test-' + dataset + '-3090.ncu-rep'
    cf_metrics = ['sm__cycles_elapsed.avg.per_second', 
              'gpu__time_duration.sum', 
              'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed',
              'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed',
              'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed']
    my_context = ncu_report.load_report(prof_name) 
    my_range = my_context.range_by_idx(0)
    for j in range(my_range.num_actions()):
        my_action = my_range.action_by_idx(j)
        kernel_name = my_action.name()
        # print(kernel_name)
        results = get_cf_metrics(my_action, cf_metrics)
        FLOPS = results[0] * results[1] * 1e-9 * (results[2] + results[3] + results[4] * 2)
        if 'fused' in kernel_name:
            flops_array[d, 0] += FLOPS
        elif 'batched' in kernel_name:
            flops_array[d, 1] += FLOPS
        elif 'separate' in kernel_name:
            flops_array[d, 2] += FLOPS
        else:
            continue

# measure the latency

for i, dataset in enumerate(dataset_list):
    for j, scheme in enumerate(scheme_list):
        dataset_col.append(dataset)
        scheme_col.append(scheme)
        latency_array[i, j] = fgms_test(scheme, dataset, False)

latency_array_base = np.expand_dims(np.max(latency_array, axis=1), axis=1)
speedup_array = np.repeat(latency_array_base, len(scheme_list), axis=1) / latency_array

utilization_array = flops_array * speedup_array
utilization_array_base = np.expand_dims(np.min(utilization_array, axis=1), axis=1)
utilization_array = utilization_array / np.repeat(utilization_array_base, len(scheme_list), axis=1)

latency_array = latency_array * 1000

flops_col = flops_array.reshape(-1).tolist()
latency_col = latency_array.reshape(-1).tolist()
speedup_col = speedup_array.reshape(-1).tolist()
utilization_col = utilization_array.reshape(-1).tolist()

results = list(zip(dataset_col, scheme_col, latency_col, speedup_col, flops_col, utilization_col))
results_csv = pd.DataFrame(data=results, columns=label_list)
results_csv.to_csv(os.path.join('results', args.save_file + '.csv'), index=True, float_format='%.4f')
