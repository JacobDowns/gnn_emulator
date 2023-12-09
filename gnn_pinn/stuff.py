import numpy as np

A = np.random.randn(10, 10)
x = np.random.randn(10, 1)

print(A.shape)
print(x.shape)

print(np.matmul(A, x[::-1]))
print(np.matmul(x.T, A))