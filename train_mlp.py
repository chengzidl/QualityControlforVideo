import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

def load_dataset_from_file(cache_file):
    data = np.load(cache_file)
    return data['sequences'], data['targets']

def load_test_dataset_from_file(cache_file):
    data = np.load(cache_file)
    return data['names'], data['sequences'], data['targets']

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sequence, target

cache_file = './data/train_data_mlp.npz'
test_file = './data/test_data_mlp.npz'

all_sequences, all_targets = load_dataset_from_file(cache_file)
test_name, test_seq, test_tar = load_test_dataset_from_file(test_file)

dataset = TimeSeriesDataset(all_sequences, all_targets)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

dataset_T = TimeSeriesDataset(test_seq, test_tar)
data_loader_T = DataLoader(dataset_T, batch_size=32, shuffle=True)


# 模型超参数
input_size = 42 #
print(f"input_size :{input_size}")
hidden_size = 64
output_size = 64
torch.manual_seed(42)


# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型实例
from model import BilibiliModel
model = BilibiliModel(input_size, hidden_size, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 300
train_loss = []
valid_loss = []
best = 0.0

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for sequences, targets in data_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(sequences)
        loss = criterion(outputs.view(-1), targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(data_loader)
    train_loss.append(avg_epoch_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
    loss_total = 0.0

    correct_total = 0
    total_samples = 0

    model.eval()
    for seq, tar in data_loader_T:
        seq, tar = seq.to(device), tar.to(device)
        
        # 前向传播
        outputs = model(seq)
        loss = criterion(outputs.view(-1), tar)
        loss_total += loss

        # 计算精确度
        predicted = torch.round(outputs)  # 获取预测类别 (batch_size,)
        predicted = predicted.view(-1)
        correct = (predicted == tar).sum().item()  # 计算正确的预测数
        correct_total += correct
        total_samples += tar.size(0)  # 累加样本数
    accuracy = correct_total / total_samples
    print(f'Accuracy: {accuracy * 100:.2f}%')

    if (epoch+1) % 20 == 0: 
        torch.save(model.state_dict(), f'./checkpoints/mlp_model_epoch_{epoch}.pth')
    print(f'valid Epoch [{epoch+1}/{num_epochs}], Loss: {loss_total.item() / len(dataset_T) :.4f}')
    valid_loss.append(loss_total.item())

    if accuracy > best:
        torch.save(model.state_dict(), f'./checkpoints/mlp_best_model.pth')
        best = accuracy


with open('mlp_train_losses.json', 'w') as f:
    json.dump(train_loss, f)

with open('mlp_valid_losses.json', 'w') as f:
    json.dump(valid_loss, f)


print(f"Best Model : {best}")
