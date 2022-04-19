import torchsparse
import torch
import numpy as np
from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
# To use the original Conv or the modified Conv
from conv import Conv3d
# from torchsparse.nn.modules import Conv3d
import time


device = torch.device('cuda')

# Numbers of convolution operations tested
nums_profile = 200

# Here input size denotes the number of input nnz. Note that the channel of input features are 
# always small (3 for RGB, 1 for intensity), a range of input sizes are tested instead.
input_size, voxel_size, input_channel = 10000, 0.1, 3

# Consider the 3D point cloud input, the 4th dimension denotes the batch index
coords = np.random.uniform(0, 100, size=(input_size, 4))
# Consider 100/10=10 samples in the batch
# coords[:, 3] = np.floor(coords[:, 3] / 10)
feats = np.random.uniform(0, 100, size=(input_size, input_channel)) 

# Voxelization
coords[:, :3], indices = sparse_quantize(coords[:, :3],
                                voxel_size,
                                return_index=True)

coords = torch.tensor(coords, dtype=torch.int)
feats = torch.tensor(feats, dtype=torch.float)
input = SparseTensor(coords=coords, feats=feats).to(device)

conv = Conv3d(input_channel, 64, kernel_size=3, stride=1).to(device)

# Warm up
_ = conv(input)

print("Profiling starts ... ")
torch.cuda.synchronize()
start=time.time()

for _ in range(nums_profile):
    output = conv(input)

torch.cuda.synchronize()
end=time.time()
dur=(end-start)/nums_profile

print("Time for a convolution operation : {:.4f} ms".format(dur * 1000))