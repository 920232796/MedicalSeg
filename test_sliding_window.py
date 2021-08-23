
from medical_seg.inferer.inferer import SlidingWindowInferer
import torch.nn as nn 
import torch 

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

    def forward(self, x):
        return x * 2

# test infer
if __name__ == '__main__':
    infer = SlidingWindowInferer(roi_size=(9, 9, 5), sw_batch_size=2, overlap=0.5)
    t1 = torch.rand((1, 1, 100, 100, 10))
    print(t1.mean())
    net1 = net()
    res = infer(t1, net1)
    print(res.mean())
    print(res.shape)