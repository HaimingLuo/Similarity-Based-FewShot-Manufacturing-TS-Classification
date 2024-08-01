
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import arff
import pandas as pd
import os
import random

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, lengths):
        self.data = data
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.lengths[idx]

def load_arff(file_path):
    with open(file_path, encoding='utf-8') as f:
        data, meta = arff.loadarff(f)
    df = pd.DataFrame(data)
    return df

def get_datasets(file_paths):
    data_list, labels_list, lengths_list = [], [], []
    for file_path in file_paths:
        df = load_arff(file_path)
        
        # 提取特征列并转换为float32
        features = df.iloc[:, :-1].astype(np.float32).values
        # 提取标签
        labels = df.iloc[:, -1].astype(np.int64).values
        # 获取每个序列的长度
        lengths = [features.shape[1]] * features.shape[0]
        
        # 添加到列表中
        data_list.append(torch.tensor(features))
        labels_list.append(torch.tensor(labels))
        lengths_list.extend(lengths)
        
    data = torch.cat(data_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    lengths = torch.tensor(lengths_list)
    
    # 数据归一化
    data = (data - data.mean(dim=0)) / data.std(dim=0)
    
    return TimeSeriesDataset(data, labels, lengths)

def create_pairs(dataset):
    pairs = []
    labels = []
    
    # 假设dataset是一个包含(data, label, length)元组的列表
    data_dict = {}
    for data, label, length in dataset:
        if label.item() not in data_dict:
            data_dict[label.item()] = []
        data_dict[label.item()].append((data, length))
    
    for label in data_dict.keys():
        samples = data_dict[label]
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                pair = (samples[i], samples[j])
                pairs.append(pair)
                labels.append(1)  # 正样本对
                
                # 通过随机选择不同类的样本创建负样本对
                neg_label = random.choice(list(set(data_dict.keys()) - {label}))
                neg_sample = random.choice(data_dict[neg_label])
                neg_pair = (samples[i], neg_sample)
                pairs.append(neg_pair)
                labels.append(0)  # 负样本对
    
    return pairs, labels

class PairsDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (x1, len1), (x2, len2) = self.pairs[idx]
        y = self.labels[idx]
        return (x1, x2, len1), (x2, len2), y
