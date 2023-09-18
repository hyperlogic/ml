import torch
import torch.nn as nn
import torch.nn.functional as F

# start with top layer of graph

# leaves
W_2 = torch.Tensor(
    [
        [
            0.4593,
            0.3518,
            0.3042,
            0.3712,
            0.0118,
            0.1133,
            0.3646,
            0.1738,
            -0.0144,
            0.0193,
            -0.0609,
            0.1824,
            0.2625,
            0.1064,
            0.0533,
            0.3082,
        ],
        [
            -0.0658,
            -0.0626,
            -0.2896,
            -0.1136,
            -0.1361,
            -0.1233,
            -0.2244,
            0.3155,
            0.0264,
            -0.0788,
            -0.0970,
            -0.1077,
            0.0704,
            -0.0872,
            0.3110,
            0.3484,
        ],
    ]
)
W_2.requires_grad = True

u_8 = torch.Tensor(
    [
        -0.3996,
        -0.0677,
        -0.2349,
        0.2650,
        0.0062,
        0.7406,
        -0.8874,
        0.8846,
        0.0532,
        -0.3342,
        -0.4135,
        -0.3466,
        0.1355,
        0.1120,
        0.3076,
        0.2617,
    ]
)
u_8.requires_grad = True
b_2 = torch.Tensor([0.0338, -0.0688])
b_2.requires_grad = True
y_hat = torch.Tensor([0.0000, 1.0000])

u_7 = F.relu(u_8)
u_7.retain_grad()
u_5 = W_2 @ u_7
u_5.retain_grad()
y = u_5 + b_2
criterion = nn.MSELoss()
loss = criterion(y, y_hat)

loss.backward()

print(f"loss = {loss}")
print()
print(f"b_2 = {b_2}")
print(f"b_2.grad = {b_2.grad}")
print()
print(f"W_2 = {W_2}")
print(f"W_2.grad = {W_2.grad}")
print()
print(f"u_5 = {u_5}")
print(f"u_5.grad = {u_5.grad}")
print()
print(f"u_7 = {u_7}")
print(f"u_7.grad = {u_7.grad}")
print()
print(f"u_8 = {u_8}")
print(f"u_8.grad = {u_8.grad}")


u_8 = tensor([-0.3996, -0.0677, -0.2349,  0.2650,  0.0062,  0.7406, -0.8874,  0.8846, 0.0532, -0.3342, -0.4135, -0.3466,  0.1355,  0.1120,  0.3076,  0.2617],
       requires_grad=True)
u_8.grad = tensor([ 0.0000,  0.0000,  0.0000,  0.2729,  0.1046,  0.1475,  0.0000, -0.1391, -0.0265,  0.0000,  0.0000,  0.0000,  0.0839,  0.1178, -0.1978, -0.0939])
