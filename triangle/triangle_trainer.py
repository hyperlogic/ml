import numpy as np
import torch
import torch.nn.functional as F

BATCH_SIZE = 200

class Dataset(torch.utils.data.Dataset):
    def __init__(self, points_filename, categories_filename):
        self.points = np.load(points_filename, allow_pickle=True)
        self.categories = np.load(categories_filename, allow_pickle=True)

        if len(self.points) != len(self.categories):
            raise ValueError("data size mismatch")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return torch.Tensor(self.points[idx]), torch.Tensor(self.categories[idx])


# Load traning and test dataset created via triangle_data.py
train_dataset = Dataset("train-points.npy", "train-categories.npy")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print("done")
