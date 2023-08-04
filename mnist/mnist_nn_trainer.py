#
# Uses pytorch to train a MLP, with one hidden layer (784-512-10)
# on the MNIST dataset.
#
import mnist_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

import json


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# print model
print("model =")
print(model)
print("parameters =")
for name, param in model.named_parameters():
    print(f"    {name}, size = {param.size()}")

#
# train
#

DEBUG_COUNT = 500
NUM_EPOCHS = 30
BATCH_SIZE = 20

# load the training dataset
trainset = mnist_dataset.MNIST_Dataset(
    "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(NUM_EPOCHS):
    loss_accum = 0.0

    for i, data in enumerate(trainloader):
        image, target = data
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss_accum += loss.item()
        loss.backward()
        optimizer.step()

        if i % DEBUG_COUNT == DEBUG_COUNT - 1:
            print(f"    batch = {i}, loss = {loss_accum / DEBUG_COUNT}")
            loss_accum = 0

    print(f"Epoch {epoch} complete!")

#
# test
#

# load the test dataset
testset = mnist_dataset.MNIST_Dataset(
    "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
with torch.no_grad():
    fail_count = 0
    test_count = 0
    for i, data in enumerate(testloader):
        image, target = data
        output = model(image)

        output_vecs = [x for x in output]
        target_vecs = [x for x in target]

        for o, t in zip(output_vecs, target_vecs):
            expected = t.argmax()
            actual = o.argmax()

            if expected != actual:
                fail_count += 1
            test_count += 1

            if i % DEBUG_COUNT == 0:
                print(f"{'pass' if expected == actual else 'fail'}")
                print(f"    target = {t}")
                print(f"    output = {o}")
                print(f"    argmax(t) = {t.argmax()}")
                print(f"    argmax(o) = {o.argmax()}")

    print(f"Testing error rate = {100 * fail_count / test_count}%")

#
# dump parameters to a json file
#

params = []
for name, param in model.named_parameters():
    params.append({"name": name, "param": param.view(-1).tolist()})

with open("params.json", "w") as out:
    out.write(json.dumps(params, indent=4))

#
# dump parameters to a binary file
#

with open("params.bin", "wb") as out:
    with torch.no_grad():
        print("// structure of params.bin")
        print("struct {")
        for name, param in model.named_parameters():
            flat = torch.t(param).flatten()
            print(f"    float {name.replace('.', '_')}[{len(flat)}];")
            out.write(flat.numpy().tobytes())
        print("} Params;")

