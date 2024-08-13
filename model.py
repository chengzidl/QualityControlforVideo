import torch
import  torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0, c_0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device),
                    torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        out, (h_out, c_out) = self.lstm(x, (h_0, c_0))
        out = self.fc(h_out[-1, :, :])
        return out



class BilibiliModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(BilibiliModel, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=in_features)
        self.attn = Attention_module(in_features, hidden_features, in_features)
        self.resblock = two_cascaded_res_blocks(in_features)
        self.linear = nn.Linear(in_features, 1)  

    def forward(self, x):
        x = self.bn(x)
        x = self.attn(x) #32 42
        x = self.resblock(x)
        x = self.linear(x)
        return x

class Attention_module(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(Attention_module, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.bn = nn.BatchNorm1d(num_features=hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        res = x
        y = self.linear2(self.relu(self.bn(self.linear1(x))))
        y = self.sigmoid(y)
        return y * res

class res_block(nn.Module):
    def __init__(self, features):
        super(res_block, self).__init__()
        self.layer1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(num_features=features)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(num_features=features)
        
    def forward(self, x):
        res = x
        y = self.relu(self.bn1(self.layer1(x)))
        y = self.bn2(self.layer2(y))
        return self.relu(y + res)

class two_cascaded_res_blocks(nn.Module):
    def __init__(self, features):
        super(two_cascaded_res_blocks, self).__init__()
        self.block1 = res_block(features)
        self.block2 = res_block(features)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

