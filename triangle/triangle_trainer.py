import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

BATCH_SIZE = 200
MAX_EPOCHS = 50

# MLP of 2-8-2
# takes a 2d point as input and outputs a one hot encoded value that indicates if the 2d point is inside or outside a triangle
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 4 * 1024)
        self.fc2 = nn.Linear(4 * 1024, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"cuda.is_available() = {torch.cuda.is_available()}")
print(f"device = {device}")

model = Net().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# print model
print("model =")
print(model)
print("parameters =")
for name, param in model.named_parameters():
    print(f"    {name}, size = {param.size()}")

# Load traning and test dataset created via triangle_data.py
full_train_dataset = Dataset("train-points.npy", "train-categories.npy")

# split the dataset into two parts, one for training and one for validation

VAL_DATASET_SIZE = 2000 #len(full_train_dataset) / 4
torch.manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(
    full_train_dataset,
    [len(full_train_dataset) - VAL_DATASET_SIZE, VAL_DATASET_SIZE],
)
torch.manual_seed(torch.initial_seed())

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

#
# Train with early stopping
#

best_val_loss = float("inf")
epochs_without_improvement = 0
max_epochs_without_improvement = 10
train_start_time = time.time()

for epoch in range(MAX_EPOCHS):
    # train the model
    train_loss = 0.0
    train_count = 0

    # for each mini-batch
    for points, categories in train_loader:
        # transfer tensors to gpu
        points, categories = points.to(device), categories.to(device)

        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, categories)

        train_loss += loss.item()
        train_count += 1

        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / train_count

    # validate the model
    val_loss = 0.0
    val_count = 0

    with torch.no_grad():
        for points, categories in val_loader:
            # transfer tensors to gpu
            points, categories = points.to(device), categories.to(device)

            outputs = model(points)
            loss = criterion(outputs, categories)
            val_loss += loss.item()
            val_count += 1

    avg_val_loss = val_loss / val_count
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch+1}: Training Loss = {avg_train_loss}, Validation Loss = {avg_val_loss}"
    )

    if epochs_without_improvement >= max_epochs_without_improvement:
        print("Early stopping triggered. Stopping training.")
        break

train_end_time = time.time()
print(f"Training took {train_end_time - train_start_time} sec")

# load the test dataset
test_dataset = Dataset("test-points.npy", "test-categories.npy")
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

model.eval()
with torch.no_grad():
    fail_count = 0
    test_count = 0
    for points, categories in test_loader:
        # transfer to gpu
        points = points.to(device)

        outputs = model(points)

        output_vecs = [x for x in outputs]
        category_vecs = [x for x in categories]

        for o, c in zip(output_vecs, category_vecs):
            expected = c.argmax()
            actual = o.argmax()

            if expected != actual:
                fail_count += 1
            test_count += 1

    print(f"Testing error rate = {100 * fail_count / test_count}%")
