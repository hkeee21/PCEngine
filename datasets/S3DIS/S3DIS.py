import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from script.utils import sparse_quantize
import torch
from script.sptensor import spTensor


class S3DISDataset(Dataset):
    def __init__(self, data_root='trainval_fullarea', num_point=4096, test_area=1, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size # 1.0
        self.transform = transform
        rooms = sorted(os.listdir(data_root))  #   data_root = 'data/s3dis/stanford_indoor3d/'
        rooms = [room for room in rooms if 'Area_' in room] # 'Area_1_WC_1.npy' # 'Area_1_conferenceRoom_1.npy'
        "rooms里面存放的是之前转换好的npy数据的名字 例如 Area_1_conferenceRoom1.npy....这样的数据"

        rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
        "按照指定的test_area划分为训练集和测试集 默认是将区域5作为测试集"

        #创建一些储存数据的列表
        self.room_points, self.room_labels = [], [] # 每个房间的点云和标签
        self.room_coord_min, self.room_coord_max = [], []  # 每个房间的最大值和最小值
        num_point_all = [] # 初始化每个房间点的总数的列表
        labelweights = np.zeros(13) # 初始标签权重，后面用来统计标签的权重

        #每层初始化数据集的时候会执行以下代码
        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            #每次拿到的room_namej就是之前划分好的'Area_1_WC_1.npy'
            room_path = os.path.join(data_root, room_name) #每个小房间的绝对路径，根路径+.npy
            room_data = np.load(room_path)  # 加载数据 xyzrgbl,  (1112933, 7) N*7  room中点云的值 最后一个是标签#
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N 将训练数据与标签分开
            "前面已经将标签进行了分离，那么这里 np.histogram就是统计每个房间里所有标签的总数 例如 第一个元素就是属于类别0的点的总数"
            "将数据集所有点统计一次之后，就知道每个类别占总类别的比例，为后面加权计算损失做准备"
            tmp, _ = np.histogram(labels, range(14)) # 统计标签的分布情况 [192039 185764 488740      0      0      0  28008      0      0      0,      0      0 218382]
            #也就是有多少个点属于第i个类别
            labelweights += tmp # 将它们累计起来
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3] # 获取当前房间坐标的最值
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size) # 标签的数量  也就是点的数量
        "通过for循环后 所有的房间里类别分布情况和坐标情况都被放入了相应的变量 后面就是计算权重了"
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights) # 计算标签的权重，每个类别的点云总数/总的点云总数
        "这里应该是为了避免有的点数量比较少 计算出训练的iou占miou的比重太大 所以在这里计算一下加权 根据点标签的数量进行加权"
        # self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0) # 为什么这里还要开三次方？？？
        # print('label weight\n')
        # print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all) # 每个房间占总的房间的比例
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)  # 如果按 sample rate进行采样，那么每个区域用4096个点 计算需要采样的次数
        room_idxs = []   #[0,1,1,2,2,2,3,4,4,5...]
        # 这里求的应该就是一个划分房间的索引
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in the set.".format(len(self.room_idxs)))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]  #
        points = self.room_points[room_idx]   # N * 6 --》 debug 1112933,6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):  #  这里是不是对应的就是将一个房间的点云切分为一个区域
            center = points[np.random.choice(N_points)][:3]  #从该个房间随机选一个点作为中心点
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            "找到符合要求点的索引 min<=x,y,z<=max 坐标被限制在最小和最大值之间"
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            "如果符合要求的点至少有1024个 那么跳出循环 否则继续随机选择中心点 继续寻找"
            if point_idxs.size > self.num_point:
                break
            "这里可以尝试修改一下1024这个参数 感觉采4096个点的话 可能存在太多重复的点"
        if point_idxs.size >= self.num_point: # 如果找到符合条件的点大于给定的4096个点，那么随机采样4096个点作为被选择的点
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:# 如果符合条件的点小于4096 则随机重复采样凑够4096个点
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True) #

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6 拿到筛选后的4096个点
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = (selected_points[:, 0] - self.room_coord_min[room_idx][0]) \
            / (self.room_coord_max[room_idx][0] - self.room_coord_min[room_idx][0])
        current_points[:, 7] = (selected_points[:, 1] - self.room_coord_min[room_idx][1]) \
            / (self.room_coord_max[room_idx][1] - self.room_coord_min[room_idx][1])
        current_points[:, 8] = (selected_points[:, 2] - self.room_coord_min[room_idx][2]) \
            / (self.room_coord_max[room_idx][2] - self.room_coord_min[room_idx][2])
        # selected_points[:, 0] = selected_points[:, 0] - center[0] # 再将坐标移至随机采样的中心点
        # selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0 # 颜色信息归一化
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        
        coords, feats = current_points[:, 0:3], current_points[:, 2:6]
        coords, inds = sparse_quantize(coords, voxel_size=0.01, return_index=True)
        coords = torch.as_tensor(coords, dtype=torch.int)
        feats = torch.as_tensor(feats[inds], dtype=torch.float)
        label = torch.as_tensor(current_labels[inds], dtype=torch.int)
        input = spTensor(coords=coords, feats=feats, buffer=None)

        return {'input': input, 'label': label}

    def __len__(self):
        return len(self.room_points)
