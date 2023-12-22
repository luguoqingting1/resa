import torch

height = 36
width = 100
iter = 4

for i in range(iter):
    idx_d = (torch.arange(2 * height // 3) + (height // 2 ** (iter - i))) % (2 * height // 3)
    print(idx_d)
    '''idx_u = torch.arange(height - 2 * height // 3, height)
    idx_u = torch.roll(idx_u, shifts=(height // 2 ** (iter - i)))
    print(idx_u)'''

'''import torch

# 假设特征图 a 和 b 的大小相同，通道数相同
# 这里只是示例，实际情况中可能需要适当调整
a = torch.rand(36, 100, 128)  # 36行，100列，128通道的特征图 a
b = torch.rand(36, 100, 128)  # 36行，100列，128通道的特征图 b

# 将特征图 a 的前12行添加到 b 上
b = torch.cat((a[:12], b), dim=0)

# 现在 b 的行数增加了12，前12行是来自 a 的行
print(b.shape)'''
