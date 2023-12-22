import torch
width=36
iter=4
i=2
idx_r = (torch.arange(width) -width //
                     2**(iter - i)) % width
print(idx_r)