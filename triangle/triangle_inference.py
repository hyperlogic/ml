import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"cuda.is_available() = {torch.cuda.is_available()}")
print(f"device = {device}")


# MLP of 2-16-2
# takes a 2d point as input and outputs a one hot encoded value that indicates if the 2d point is inside or outside a triangle
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def one_hot(klass, num_classes=2):
    return [1.0 if klass == i else 0.0 for i in range(num_classes)]

model = Net()
state_dict = torch.load("model_state_dict.dat")
model.load_state_dict(state_dict)
model.to(device)

criterion = nn.MSELoss()

x = torch.Tensor([0.5, 0.5]).to(device)
output = model(x)

print(f"x = {x}")
print(f"output = {output}")

expected = torch.Tensor(one_hot(1)).to(device)

print(f"expected = {expected}")

loss = criterion(output, expected)
loss.backward()

print("parameters =")
for name, param in model.named_parameters():
    print(f"{name}.data = {param.data}")
    print(f"{name}.grad = {param.grad}")

#print(model.parameters().grad)

