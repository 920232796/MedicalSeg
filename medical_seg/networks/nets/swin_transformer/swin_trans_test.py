import torch
import torch.nn.functional as F

from medical_seg.networks.nets.swin_transformer.model_2 import StageModule

## 官方代码为model.py 这个改起来比较困难，model_2 是非官方实现 model_patch_conv 是在model_2的基础上把unfold操作换成了卷积
#
# 
# 测试unfold函数。
#  x = torch.randn(1, 2, 64, 64, 64)
# x = x.squeeze(dim=0)
# kc, kh, kw = 16, 16, 16  # kernel size
# dc, dh, dw = 16, 16, 16  # stride
# new_c, new_h, new_w = 64 // 16, 64 // 16, 64 // 16
# patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
# patches = patches.contiguous().view(new_c, new_h, new_w, -1)
# patches.unsqueeze(dim=0)
# print(patches.shape)



# patches = patches.contiguous().view(-1, kc, kh, kw)
# print(patches.shape)


if __name__ == "__main__":
    model = StageModule(in_channels=3, hidden_dimension=64, layers=2, patch_size=(1, 2, 2), num_heads=8, head_dim=4, window_size=(4, 8, 8), relative_pos_embedding=True)
    t1 = torch.rand((1, 3, 32, 64, 64))

    out = model(t1)
    print(out.shape)