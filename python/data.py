import torch
from torch.utils.data import Dataset, DataLoader


class RockSampleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def prepare_dataloader(features, labels, batch_size=32):
    """创建数据加载器"""
    dataset = RockSampleDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
