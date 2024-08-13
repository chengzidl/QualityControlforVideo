import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def load_model(ckpt_dir):
    from model import LSTMPredictor
    device = 'cuda'
    input_size = 43 
    hidden_size = 64
    num_layers = 2
    output_size = 1
    model = LSTMPredictor(input_size, hidden_size, num_layers, output_size).to(device)
    checkpoint = torch.load(ckpt_dir)
    model.load_state_dict(checkpoint)

    return model


def load_dataset_from_file(cache_file):
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

def predict_windows(model, dataloader, firstItem):
    input_ = firstItem
    res = []
    model.eval()
    with torch.no_grad():  
        for i, (seq, tar) in enumerate(dataloader):
            if i == 0:  # first segment don't predict
                continue
            rf = model(input_)
            rf = rf[-1].unsqueeze(0)
            # shape: [1,1] 对第二个数字四舍五入
            rf = torch.round(rf)
            seq_trim = seq[:, :, :-1].cuda()
            current = torch.cat((seq_trim, rf.unsqueeze(0)), dim=2).cuda()
            input_ = torch.cat((input_[-1], current), dim=0).cuda()
            res.append(rf.item())
    return res

def predict_straight(model, dataloader, firsetItem):
    input_ = firsetItem
    res = []
    model.eval()
    with torch.no_grad():  
        for seq, tar in dataloader:
            seq = seq.cuda()
            rf = model(seq)
            rf = torch.round(rf)
            res.append(rf.item())
    return res

def predict(model, dataloader, firstItem):
    input_ = firstItem
    res = []
    model.eval()
    with torch.no_grad():  
        for i, (seq, tar) in enumerate(dataloader):
            if i == 0:  # first segment don't predict
                continue
            rf = model(input_)
            rf = rf[-1].unsqueeze(0)
            # shape: [1,1] 对第二个数字四舍五入
            rf = torch.round(rf)
            seq_trim = seq[:, :, :-1].cuda()
            current = torch.cat((seq_trim, rf.unsqueeze(0)), dim=2).cuda()
            input_ = torch.cat((input_, current), dim=0).cuda()
            res.append(rf.item())
    return res

def get_initial_input(dataloader):
    for seq, tar in dataloader:
        return seq.cuda()

def main():
    test_file = './data/test_data.npz'
    ckpt_dir = './checkpoints/best_model.pth'
    files, sequences, targets = load_dataset_from_file(test_file)
    # 将sequence 按 files 分类, 分别建立dataset
    unique_files = set(files)

    stat_0, stat_1, stat_2, stat_3 = 0, 0, 0, 0

    for file in unique_files:

        file_indicies = [i for i, f in enumerate(files) if f == file]
        file_sequences = [sequences[i] for i in file_indicies]
        file_targets = [targets[i] for i in file_indicies]

        dataset = TimeSeriesDataset(file_sequences, file_targets)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        firstItem = get_initial_input(dataloader) 
        print(firstItem)

        model = load_model(ckpt_dir)

        predicted_rf = predict_straight(model, dataloader, firstItem)
        # predicted_rf = predict(model, dataloader, firstItem)
        # predicted_rf = predict_windows(model, dataloader, firstItem)

        
        real_rf = [tar.item() for seq, tar in dataset]
        comparison = list(zip(real_rf, predicted_rf))


        stat_rf_0 = sum([1 for real, pred in comparison if real - pred == 0])
        stat_rf_1 = sum([1 for real, pred in comparison if -1 <= real - pred <= 1])
        stat_rf_2 = sum([1 for real, pred in comparison if -2 <= real - pred <= 2])
        stat_rf_3 = sum([1 for real, pred in comparison if abs(real - pred) > 2 ])


        for real, pred in comparison:
            print(f"Real RF: {real}, Predicted RF: {pred}")

        stat_0 += stat_rf_0
        stat_1 += stat_rf_1
        stat_2 += stat_rf_2
        stat_3 += stat_rf_3
    
    stat_sum = stat_2 + stat_3

    with open('./stats/4K.log', 'w') as f:

        f.write(f"Exact matches (±0): {stat_0}\n")
        f.write(f"Within tolerance (±1): {stat_1}\n")
        f.write(f"Within tolerance (±2): {stat_2}\n")
        f.write(f"Absolute differences > 1: {stat_3}\n")

        f.write(f"match: {stat_0 / stat_sum}\n")
        f.write(f"1: {stat_1 / stat_sum}\n")
        f.write(f"2: {stat_2 / stat_sum}\n")
        f.write(f"3: {stat_3 / stat_sum}\n")


if __name__ == "__main__":
    main()