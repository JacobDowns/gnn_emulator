# import torch library
import torch

# create tensors with requires_grad = true
x = torch.tensor(2.0, requires_grad = True)
y = torch.tensor(1., requires_grad=True)

z = x**2+y
z.backward()

print(x.grad)
print(y.grad)

