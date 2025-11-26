import numpy as np
import torch
print(torch.__version__)
data = [[1,2],[3,4]]
x = torch.tensor(data)


np_array = np.array(data)
x_np = torch.from_numpy(np_array)


x_ones = torch.ones_like(x_np)

x_rand = torch.rand_like(x_ones,dtype=torch.float32)

x_zeros = torch.zeros(size = (4,6))
print(x_zeros)