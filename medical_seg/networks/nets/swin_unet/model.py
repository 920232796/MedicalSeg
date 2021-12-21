import torch 
import numpy
import torch.nn as nn 
import copy 
from medical_seg.networks.nets.swin_unet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

class SwinUnet(nn.Module):
    def __init__(self, img_size=224, in_channel=3,  num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.loss_func = nn.CrossEntropyLoss()

        self.zero_head = zero_head
        self.in_channels = in_channel
        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                            patch_size=4,
                                            in_chans=in_channel,
                                            num_classes=self.num_classes,
                                            embed_dim=96,
                                            depths=[ 2, 2, 2, 2 ],
                                            num_heads=[ 3, 6, 12, 24 ],
                                            window_size=7,
                                            mlp_ratio=4,
                                            qkv_bias=True,
                                            qk_scale=False,
                                            drop_rate=0.2,
                                            drop_path_rate=0.2,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False)

    def forward_train(self, x):
        ## x : (batch, modality, d, w, h )
        input = x.squeeze(dim=0)

        input2d = input.permute(1, 0, 2, 3)

        d, in_channels, w, h = input2d.shape
        if w != 224 or h != 224:
            input2d = nn.functional.interpolate(input2d, size=(224, 224), mode="bilinear", align_corners=False)
        
        logits = self.swin_unet(input2d)

        logits = logits.transpose(1, 0)
        logits = logits.unsqueeze(dim=0)
        
        return logits

    def forward(self, x):
        ## x : (batch, modality, d, w, h )
        input = x.squeeze(dim=0)

        input2d = input.permute(1, 0, 2, 3)

        d, in_channels, w, h = input2d.shape
        if w != 224 or h != 224:
            input2d = nn.functional.interpolate(input2d, size=(224, 224), mode="bilinear", align_corners=False)
        
        logits = self.swin_unet(input2d)

        if w != 224 or h != 224:
            logits = nn.functional.interpolate(logits, size=(w, h), mode="bilinear", align_corners=False)
        
        
        logits = logits.transpose(1, 0)
        logits = logits.unsqueeze(dim=0)
        
        return logits
    
    def compute_loss(self, pred, label):
        b, d, w, h = label.shape
        label = label.float()

        if w != 224 or h != 224:
            label = torch.unsqueeze(label, dim=1)
            label = nn.functional.interpolate(label, size=(d, 224, 224), mode="nearest")
            label = torch.squeeze(label, dim=1).long()

        loss = self.loss_func(pred, label)

        return loss 

    def load_from(self, pretrained_path):

        device = next(self.swin_unet.parameters()).device

        print("pretrained_path:{}".format(pretrained_path))
        pretrained_dict = torch.load(pretrained_path, map_location=device)
       
        pretrained_dict = pretrained_dict['model']

        if self.in_channels != 3:
            embed_weight = pretrained_dict["patch_embed.proj.weight"]
            embed_weight = embed_weight.mean(dim=1, keepdims=True)
            embed_weight = embed_weight.repeat(1, self.in_channels, 1, 1)
            pretrained_dict["patch_embed.proj.weight"] = embed_weight

        print("---start load pretrained modle of swin encoder---")

        model_dict = self.swin_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]

        self.swin_unet.load_state_dict(full_dict, strict=False)



if __name__ == "__main__":
    net_3d = SwinUnet(img_size=224, in_channel=4, num_classes=2)
    net_3d.load_from("./medical_seg/networks/nets/swin_unet//swin_tiny_patch4_window7_224.pth")

    t1 = torch.rand(1, 4, 5, 224, 224)

    out = net_3d(t1)
    print(out.shape)
