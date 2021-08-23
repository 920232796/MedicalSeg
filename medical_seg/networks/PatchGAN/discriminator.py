
import torch
import torch.nn as nn
import numpy as np

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, mse=True, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, device="cpu"):
        super(NLayerDiscriminator, self).__init__()
        self.device = device
        self.to(device)
        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                # TODO: use InstanceNorm
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            # TODO: useInstanceNorm
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]


        self.model = nn.Sequential(*sequence)
        if mse:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)

    def get_loss_D(self, x, pred_res, label):
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        # x: (1, 2, 256, 256)
        # pred_res: (1, 1, 256, 256)
        fake_AB = torch.cat((x, pred_res), 1)
        pred_fake = self.forward(fake_AB.detach())# detach 是为了只更新d的参数，而不去更新分割网络的参数！
        fake_label = torch.zeros_like(pred_fake, device=self.device)
        loss_D_fake = self.loss(pred_fake, fake_label)
        # Real
        real_AB = torch.cat((x, label), 1)
        pred_real = self.forward(real_AB)
        real_label = torch.ones_like(pred_real, device=self.device)
        loss_D_real = self.loss(pred_real, real_label)
        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

if __name__ == '__main__':
    t1 = torch.rand(1, 2, 128, 128)
    label = torch.rand(1, 1, 128, 128)
    pred_res = torch.rand(1, 1, 128, 128)
    model = NLayerDiscriminator(input_nc=3)
    # out = model(t1)
    # print(out.shape)
    out_loss = model.get_loss_D(t1, pred_res, label)
    print(out_loss)
    print(out_loss.shape)