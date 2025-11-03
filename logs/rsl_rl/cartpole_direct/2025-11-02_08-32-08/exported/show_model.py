import torch

model = torch.jit.load("policy.pt", map_location="cpu")
print(model)

x = torch.randn(1, 62)  
y = model(x)
print(y)