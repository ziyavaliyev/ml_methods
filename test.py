import torch
import numpy as np

y = torch.tensor([[[1,2],[3,4]], [[1,2],[3,4]]])

x = y.clone()

y[0][0][0] = 555

print(x)

print(y)

print(torch.cuda.is_available())

