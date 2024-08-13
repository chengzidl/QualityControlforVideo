# 按 file 和 9:1 的比例分数据集， 存在train_data 和 test_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv('dataset.csv')

grouped = df.groupby('file')

# window sliding
def create_sequences(data, target, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = target[i+seq_length]
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)

# 获取所有文件名及其对应记录数
file_lengths = df['file'].value_counts().to_dict()
file_names = list(file_lengths.keys())


# 初始化训练集和测试集
train_files = []
test_files = []
train_size, test_size = 0, 0

# 总数据长度
total_size = sum(file_lengths.values())
train_target_size = int(0.9 * total_size)

# # 随机打乱文件顺序
np.random.seed(42)  # 可选：固定种子以确保结果可重复
np.random.shuffle(file_names)

# 分配文件
for file_name in file_names:
    file_len = file_lengths[file_name]
    if train_size + file_len <= train_target_size:
        train_files.append(file_name)
        train_size += file_len
    else:
        test_files.append(file_name)
        test_size += file_len

# 构建训练和测试数据
train_sequences = []
train_targets = []
test_names = []
test_sequences = []
test_targets = []

seq_length = 1  # window size == 1


for name, group in grouped:
    if len(group) < seq_length + 1:
        continue
    # del
    del_c = ['file', 'chunk', 'crf']
    target_column = 'crf'
    group_feature = group.drop(columns=del_c)
    scaler = StandardScaler()
    features = scaler.fit_transform(group_feature)
    # 再把 crf 拼回去 features
    group_scaled = pd.DataFrame(features, columns=group_feature.columns)
    group_scaled[target_column] = group[target_column].values

    sequences, targets = create_sequences(group_scaled, group[target_column].values, seq_length)
    if name in train_files:
        train_sequences.append(sequences)
        train_targets.append(targets)
    elif name in test_files:
        test_names.extend([name] * len(sequences))
        test_sequences.append(sequences)
        test_targets.append(targets)


# 将所有分组的数据合并
train_sequences = np.concatenate(train_sequences, axis=0)
train_targets = np.concatenate(train_targets, axis=0)
test_sequences = np.concatenate(test_sequences, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

train_file = 'train_data.npz'
test_file = 'test_data.npz'

np.savez(train_file, sequences=train_sequences, targets=train_targets)
np.savez(test_file, names=test_names, sequences=test_sequences, targets=test_targets)

