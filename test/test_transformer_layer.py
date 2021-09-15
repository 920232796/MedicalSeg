

from medical_seg.networks.nets.transformer import TransformerLayers
import torch 


if __name__ == "__main__":

    print("hello world ")

    net = TransformerLayers(in_channels=2, out_channels=3, 
                            patch_size=(16, 16,16), img_size=(32, 32, 32),
                            mlp_size=128, num_layers=2)


    t1 = torch.rand(1, 2, 32, 32, 32)

    out = net(t1)
    print(out.shape)