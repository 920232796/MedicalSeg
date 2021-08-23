import torch
import torch.nn.functional as F

x = torch.randn(1, 2, 64, 64, 64)
x = x.squeeze(dim=0)
kc, kh, kw = 16, 16, 16  # kernel size
dc, dh, dw = 16, 16, 16  # stride
new_c, new_h, new_w = 64 // 16, 64 // 16, 64 // 16
patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
patches = patches.contiguous().view(new_c, new_h, new_w, -1)
patches.unsqueeze(dim=0)
print(patches.shape)



# patches = patches.contiguous().view(-1, kc, kh, kw)
# print(patches.shape)