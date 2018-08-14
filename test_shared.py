import torch
import torch.nn as nn

model_1 = nn.Sequential(nn.Linear(2,3), nn.Linear(3,1))
model_2 = nn.Sequential(nn.Linear(2,3), nn.Linear(3,1))
optim = torch.optim.Adam(model_2.parameters(), lr=1e-3)
loss = model_1(torch.Tensor([0,1])) - 3
loss.backward()

optim.zero_grad()

print("_________________________")
print("grad for model_1")

for i in model_1.parameters():
    print(i.grad)

print("_________________________")
print("grad for model_2")

for i in model_2.parameters():
    print(i.grad)

print("_________________________")
for p1, p2 in zip(model_1.parameters(), model_2.parameters() ):
    p2.grad=p1.grad

optim.step()
optim.zero_grad()

model_1.load_state_dict(model_2.state_dict())
print("grad for model_1")

for i in model_1.parameters():
    print(i.grad)
