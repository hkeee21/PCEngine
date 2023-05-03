''' @ MLSys23 Artifacts Evaluation
    This code is to generate the gather and scatter speedup over torchsparse into a .csv file.
        Mapping format: coded-CSR, torchsparse v2.0.0.
        Benchmarks: {ModelNet40, S3DIS, KITTI} X {SparseResNet, MinkUNet}.
        Command: $ python3 gather-scatter.py --save-file ${filename} --fast(optional)
        "--fast" is recommanded to reduce evaluation time.
'''

import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import os
import yaml
from tqdm import tqdm
import sys
sys.path.append('../')
from lib.PCEngine.script.spconv import conv3d
from lib.PCEngine.script.utils import sparse_collate_fn
from lib.PCEngine.models.resnet import SparseResNet21D
from lib.PCEngine.models.minkunet import SparseResUNet42

def dataset_builder(dataset: str):
    builder = None
    if dataset == 'modelnet40':
        from lib.PCEngine.datasets.ModelNet40 import ModelNet40Dataset
        builder = ModelNet40Dataset(
            item=["piano", "bathtub", "airplane", "chair", "person", "sofa", "radio"],
            root="../AE-datasets/ModelNet40", 
            test_mode='kernel'
        )
    elif dataset == 's3dis':
        from lib.PCEngine.datasets.S3DIS import S3DISDataset
        builder = S3DISDataset(
            data_root='../AE-datasets/stanford_indoor3d',
            test_area=4,
            num_point=4096 * 4,
            block_size=1.0
        )
    elif dataset == 'kitti':
        from lib.PCEngine.datasets.KITTI import KITTIDataset
        builder = KITTIDataset(
            path='../AE-datasets/KITTI',
            size=1000
        )
    else:
        raise NotImplementedError

    return builder


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--save-file', type=str, default='Fig10-gather-scatter')
parser.add_argument("--fast", action="store_true")
args = parser.parse_args()

label_list = ['dataset', 'model', 'operator', 'normalized speedup']
dataset_list = ['modelnet40', 's3dis', 'kitti']
model_list = ['resnet', 'unet']
mapping_format_list = ['coded-csr', 'torchsparse']
dataset_col = []
model_col = []
operator_col = []
speedup_col = []

buffer = torch.rand((512 * 20000 * 27,), dtype=torch.float, device=device)

for d, dataset in enumerate(dataset_list):
    results_dict = {'coded-csr-gather': [],
                'coded-csr-scatter': [],
                'torchsparse-gather': [],
                'torchsparse-scatter': []}
    np.save('coded-csr-intermediate-data.npy', results_dict)
    Dataset = dataset_builder(dataset)
    DataLoader = torch.utils.data.DataLoader(
        Dataset, 
        batch_size=1, 
        collate_fn=sparse_collate_fn,
        shuffle=False)
    
    for model in model_list:
        if model == 'resnet':
            backbone = SparseResNet21D
        elif model == 'unet':
            backbone = SparseResUNet42
        else:
            raise NotImplementedError
        Model = nn.Module = backbone(in_channels=4, width_multiplier=1.0)
        Model = Model.to(device).eval()
        # print(Model)

        file_name = os.path.join('../lib/PCEngine/configs/gather-mm-scatter', model + '.yaml')
        f = open(file_name, 'r')
        init_layer_dict = yaml.load(f, Loader=yaml.FullLoader)
        init_layer_tag = init_layer_dict['dataflow_tag']

        for fm, format in enumerate(mapping_format_list):
            # switch on the coded-csr test mode
            for name, module in Model.named_modules():
                if 'conv3d' in module.__class__.__name__.lower():
                    module.coded_csr_flag = 2 * fm + 1

            i = 0
            with torch.no_grad():
                for batch in tqdm(DataLoader):
                    if i == 200 and args.fast: break

                    input = batch['input']
                    input = input.to(device)
                    input.buffer = buffer
                    input.init_tag = init_layer_tag

                    output = Model(input)

                    i += 1


        results_dict = np.load('coded-csr-intermediate-data.npy', allow_pickle=True).item()
        results_len = len(results_dict['coded-csr-gather'])
        speedup_array = np.zeros((results_len, 2))
        for r in range(results_len):
            '''print('gather: %.4f - %.4f [%.4f]' 
            % (results_dict['coded-csr-gather'][r], results_dict['torchsparse-gather'][r], 
                 (results_dict['torchsparse-gather'][r]/results_dict['coded-csr-gather'][r])))
            print('scatter: %.4f - %.4f [%.4f]' 
            % (results_dict['coded-csr-scatter'][r], results_dict['torchsparse-scatter'][r],
                 (results_dict['torchsparse-scatter'][r]/results_dict['coded-csr-scatter'][r])))'''
            speedup_array[r, 0] = results_dict['torchsparse-gather'][r] / results_dict['coded-csr-gather'][r]
            speedup_array[r, 1] = results_dict['torchsparse-scatter'][r] / results_dict['coded-csr-scatter'][r]
        mean_speedup = np.mean(speedup_array, axis=0)

        dataset_col.append(dataset)
        model_col.append(model)
        operator_col.append('gather')
        speedup_col.append(mean_speedup[0])

        dataset_col.append(dataset)
        model_col.append(model)
        operator_col.append('scatter')
        speedup_col.append(mean_speedup[1])


results = list(zip(model_col, dataset_col, operator_col, speedup_col))
results_csv = pd.DataFrame(data=results, columns=label_list)
results_csv.to_csv(os.path.join('results', args.save_file + '.csv'), index=True, float_format='%.4f')

