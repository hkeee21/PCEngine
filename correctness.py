''' @ MLSys23 Artifacts Evaluation
    This code is to check the sparse convolutional results of PCEngine.
        PCEngine Dataflow: Gather-GEMM-Scatter, Fetch-on-Demand (Heuristic
            dataflow selects between the two dataflows).
        Standards: SpConv (v2.2.6) results.
        Types: Submanifold & downsampling convolution.
        Datasets: ModelNet40, S3DIS, KITTI.
        Command: $ python3 correctness-check.py
'''

import numpy as np
import torch
from tqdm import tqdm
from lib.PCEngine.script.spconv import conv3d
from spconv.pytorch import SparseConv3d, SubMConv3d

def dataset_builder(framework: str, dataset: str):
    builder = None
    if 'PCEngine' in framework:
        if dataset == 'modelnet40':
            from lib.PCEngine.datasets.ModelNet40 import ModelNet40Dataset
            builder = ModelNet40Dataset(
                item=["piano", "bathtub", "airplane", "chair", "person", "sofa", "radio"],
                root="AE-datasets/ModelNet40"
            )
        elif dataset == 's3dis':
            from lib.PCEngine.datasets.S3DIS import S3DISDataset
            builder = S3DISDataset(
                data_root='AE-datasets/stanford_indoor3d',
                test_area=4,
                num_point=4096 * 4,
                block_size=1.0
            )
        elif dataset == 'kitti':
            from lib.PCEngine.datasets.KITTI import KITTIDataset
            builder = KITTIDataset(
                path='AE-datasets/KITTI',
                size=1000
            )
        else:
            raise NotImplementedError
    elif 'SpConv2' in framework:
        if dataset == 'modelnet40':
            from lib.SpConv2.ModelNet40 import ModelNet40Dataset as spModelNet40Dataset
            builder = spModelNet40Dataset(
                item=["piano", "bathtub", "airplane", "chair", "person", "sofa", "radio"],
                root="AE-datasets/ModelNet40"
            )
        elif dataset == 's3dis':
            from lib.SpConv2.S3DIS import S3DISDataset as spS3DISDataset
            builder = spS3DISDataset(
                data_root='AE-datasets/stanford_indoor3d',
                test_area=4,
                num_point=4096 * 4,
                block_size=1.0
            )
        elif dataset == 'kitti':
            from lib.SpConv2.KITTI import KITTIDataset as spKITTIDataset
            builder = spKITTIDataset(
                path='AE-datasets/KITTI',
                size=1000
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return builder


def collate_func_builder(framework: str):
    builder = None
    if 'PCEngine' in framework:
        from lib.PCEngine.script.utils import sparse_collate_fn
        builder = sparse_collate_fn
    elif 'SpConv2' in framework:
        from lib.SpConv2.utils import sparse_collate_fn as sp_collate_fn
        builder = sp_collate_fn
    else:
        raise NotImplementedError

    return builder


def coordinate_hash(coords: torch.Tensor):
    hash = torch.empty((coords.shape[0]), dtype=torch.int)
    hash = coords[:, 1] * 2^20 + coords[:, 2] * 2^10 + coords[:, 3]
    return hash


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


dataset_list = ['modelnet40', 's3dis', 'kitti']
framework_list = ['PCEngine(gather-mm-scatter)', 'PCEngine(fetch-on-demand)', 'SpConv2']
type_list = ['submanifold', 'downsampling']

results_array = np.zeros((len(dataset_list) * len(type_list), len(framework_list) - 1))
allclose_array = torch.ones((len(dataset_list) * len(type_list), len(framework_list) - 1))

buffer = torch.zeros((512 * 20000 * 27,), dtype=torch.float, device=device)

weight = torch.ones((3, 3, 3, 4, 32), dtype=torch.float, device=device)
co_weight = weight.reshape(-1, 4, 32)
re_weight = weight.permute(4, 0, 1, 2, 3).contiguous()

for d, dataset in enumerate(dataset_list):
    subm_output_array = {'PCEngine(gather-mm-scatter)': list(),
                        'PCEngine(fetch-on-demand)': list(), 
                        'SpConv2': list()}
    sp_output_array = {'PCEngine(gather-mm-scatter)': list(),
                        'PCEngine(fetch-on-demand)': list(), 
                        'SpConv2': list()}
    for framework in framework_list:
        Dataset = dataset_builder(framework, dataset)
        collate_func = collate_func_builder(framework)
        DataLoader = torch.utils.data.DataLoader(
            Dataset, 
            batch_size=1, 
            collate_fn=collate_func,
            shuffle=False)
        if 'PCEngine' in framework:
            subm_conv_layer = conv3d(4, 32, kernel_size=3, stride=1)
            sp_conv_layer = conv3d(4, 32, kernel_size=3, stride=2)
            subm_conv_layer.eval().to(device)
            sp_conv_layer.eval().to(device)
            # print(subm_conv_layer.state_dict().keys()) = 'kernel'
            # print(sp_conv_layer.state_dict().keys()) = 'kernel'
            subm_conv_layer.state_dict()['kernel'].copy_(co_weight)
            sp_conv_layer.state_dict()['kernel'].copy_(co_weight)
        elif 'SpConv2' in framework:
            subm_conv_layer = SubMConv3d(4, 32, kernel_size=3, padding=0, bias=False)
            sp_conv_layer = SparseConv3d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
            subm_conv_layer.eval().to(device)
            sp_conv_layer.eval().to(device)
            # print(subm_conv_layer.state_dict().keys()) = 'weight'
            # print(sp_conv_layer.state_dict().keys()) = 'weight'
            subm_conv_layer.state_dict()['weight'].copy_(re_weight)
            sp_conv_layer.state_dict()['weight'].copy_(re_weight)
        else:
            raise NotImplementedError
        if 'gather-mm-scatter' in framework:
            layer_tag = ['whole', 'whole', 'end']
        elif 'fetch-on-demand' in framework:
            layer_tag = ['simple', 'simple', 'end']
        else:
            layer_tag = None

        n = 0  
        for batch in DataLoader:
            # warning: storing too many results will cause OOM
            # set n == 50 to a larger number when a GPU with larger memory is used 
            if n == 50: break
            
            input = batch['input']

            if 'SpConv2' in framework:
                shape = (torch.max(input.indices[:, 1:], dim=0).values + 1).cpu().numpy().tolist()
                input.spatial_shape = shape

            if layer_tag is not None:
                input.buffer = buffer
                input.init_tag = layer_tag
                input = input.to(device)
            
            subm_output = subm_conv_layer(input)
            sp_output = sp_conv_layer(input)

            subm_output_array[framework].append(subm_output)
            sp_output_array[framework].append(sp_output)

            n += 1

    data_len = len(subm_output_array['SpConv2'])
    
    # check:
    #  1) if the non-zero element number matches
    #  2) if the output feature matches
    for i in range(data_len):

        ##### submanifold sparse convolution results check #####
        PCEngie_D1_coords = subm_output_array['PCEngine(gather-mm-scatter)'][i].coords
        PCEngie_D2_coords = subm_output_array['PCEngine(fetch-on-demand)'][i].coords
        SpConv2_coords = subm_output_array['SpConv2'][i].indices

        assert PCEngie_D1_coords.shape == SpConv2_coords.shape
        assert PCEngie_D2_coords.shape == SpConv2_coords.shape

        N = PCEngie_D1_coords.shape[0]

        PCEngie_D1_coords_hash = coordinate_hash(PCEngie_D1_coords) 
        PCEngie_D2_coords_hash = coordinate_hash(PCEngie_D2_coords) 
        SpConv2_coords_hash = coordinate_hash(SpConv2_coords)

        _, PCEngie_D1_i = torch.sort(PCEngie_D1_coords_hash)
        _, PCEngie_D2_i = torch.sort(PCEngie_D2_coords_hash)
        _, SpConv2_i = torch.sort(SpConv2_coords_hash)

        PCEngie_D1_feats = subm_output_array['PCEngine(gather-mm-scatter)'][i].feats
        PCEngie_D2_feats = subm_output_array['PCEngine(fetch-on-demand)'][i].feats
        SpConv2_feats = subm_output_array['SpConv2'][i].features

        PCEngie_D1_feats = PCEngie_D1_feats[PCEngie_D1_i]
        PCEngie_D2_feats = PCEngie_D2_feats[PCEngie_D2_i]
        SpConv2_feats = SpConv2_feats[SpConv2_i]

        results_array[d * 2, 0] += torch.sum(torch.abs(PCEngie_D1_feats - SpConv2_feats)) / N
        results_array[d * 2, 1] += torch.sum(torch.abs(PCEngie_D2_feats - SpConv2_feats)) / N
        allclose_array[d * 2, 0] *= torch.allclose(PCEngie_D1_feats, SpConv2_feats, rtol=1e-03, atol=1e-06)
        allclose_array[d * 2, 1] *= torch.allclose(PCEngie_D2_feats, SpConv2_feats, rtol=1e-03, atol=1e-06)

        ##### downsampling sparse convolution results check #####
        PCEngie_D1_coords = sp_output_array['PCEngine(gather-mm-scatter)'][i].coords
        PCEngie_D2_coords = sp_output_array['PCEngine(fetch-on-demand)'][i].coords
        SpConv2_coords = sp_output_array['SpConv2'][i].indices

        assert PCEngie_D1_coords.shape == SpConv2_coords.shape
        assert PCEngie_D2_coords.shape == SpConv2_coords.shape

        N = PCEngie_D1_coords.shape[0]

        PCEngie_D1_feats = sp_output_array['PCEngine(gather-mm-scatter)'][i].feats
        PCEngie_D2_feats = sp_output_array['PCEngine(fetch-on-demand)'][i].feats
        SpConv2_feats = sp_output_array['SpConv2'][i].features
        
        _, PCEngie_D1_i = torch.sort(torch.sum(PCEngie_D1_feats, dim=1))
        _, PCEngie_D2_i = torch.sort(torch.sum(PCEngie_D2_feats, dim=1))
        _, SpConv2_i = torch.sort(torch.sum(SpConv2_feats, dim=1))

        PCEngie_D1_feats = PCEngie_D1_feats[PCEngie_D1_i]
        PCEngie_D2_feats = PCEngie_D2_feats[PCEngie_D2_i]
        SpConv2_feats = SpConv2_feats[SpConv2_i]

        results_array[d * 2 + 1, 0] += torch.sum(torch.abs(PCEngie_D1_feats - SpConv2_feats)) / N
        results_array[d * 2 + 1, 1] += torch.sum(torch.abs(PCEngie_D2_feats - SpConv2_feats)) / N
        allclose_array[d * 2 + 1, 0] *= torch.allclose(PCEngie_D1_feats, SpConv2_feats, rtol=1e-03, atol=1e-06)
        allclose_array[d * 2 + 1, 1] *= torch.allclose(PCEngie_D2_feats, SpConv2_feats, rtol=1e-03, atol=1e-06)

    results_array[d * 2, 0] = results_array[d * 2, 0] / data_len
    results_array[d * 2, 1] = results_array[d * 2, 1] / data_len
    results_array[d * 2 + 1, 0] = results_array[d * 2 + 1, 0] / data_len
    results_array[d * 2 + 1, 1] = results_array[d * 2 + 1, 1] / data_len


# print the comparison results
print('**********  Sparse Convolution Mean Absolute Error Compared to SpConv  **********')
print('---------------------------------------------------------------------------------')
print('|            |        gather-mm-scatter        |         fetch-on-demand         |')
print('|   dataset  |                                 |                                 |')
print('|            |   submanifold  |  downsampling  |   submanifold  |  downsampling  |')
print('---------------------------------------------------------------------------------')
for l in range(len(dataset_list)):
    print('|%-12s|%-16.8f|%-16.8f|%-16.8f|%-16.8f|' % 
          (dataset_list[l], results_array[l*2,0], results_array[l*2+1,0], results_array[l*2,1], results_array[l*2+1,1]))
print('---------------------------------------------------------------------------------')
print('|                            torch.allclose test                                 |')
print('---------------------------------------------------------------------------------')
for l in range(len(dataset_list)):
    print('|%-12s|%-16s|%-16s|%-16s|%-16s|' % 
          (dataset_list[l], 
           bool(allclose_array[l*2,0]), 
           bool(allclose_array[l*2+1,0]), 
           bool(allclose_array[l*2,1]), 
           bool(allclose_array[l*2+1,1])))
print('---------------------------------------------------------------------------------')

    


