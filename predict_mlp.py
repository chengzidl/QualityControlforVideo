import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def load_model(ckpt_dir):
    from model import BilibiliModel
    device = 'cuda'
    input_size = 42
    hidden_size = 64
    output_size = 64
    model = BilibiliModel(input_size, hidden_size, output_size).to(device)
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


def predict(model, dataloader):
    res = []
    model.eval()
    with torch.no_grad():  
        for seq, tar in dataloader:
            seq = seq.cuda()
            rf = model(seq)
            rf = torch.round(rf)
            res.append(rf.item())
    return res


def main():
    test_file = './data/test_data_mlp.npz'
    ckpt_dir = './checkpoints/mlp_best_model.pth'
    files, sequences, targets = load_dataset_from_file(test_file)

    stat_0, stat_1, stat_2, stat_3 = 0, 0, 0, 0

    dataset = TimeSeriesDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    model = load_model(ckpt_dir)

    predicted_rf = predict(model, dataloader)

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

    with open('./stats_mlp/4K.log', 'w') as f:

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