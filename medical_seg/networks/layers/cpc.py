import torch.nn as nn 
import torch
from torch.nn.modules import transformer 

class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.
        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)
        
    def forward(self, x, y):
        """Calulate the score 
        """
        # import ipdb;ipdb.set_trace()
        x_pred = self.net(y)    # bs, emb_size

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = -(pos - neg).mean()
        return nce

class ImageCPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.
        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Conv3d(
                in_channels=y_size,
                out_channels=x_size,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Conv3d(self.y_size, self.x_size, kernel_size=1, stride=1, padding=0))
                    net.append(self.activation())
                else:
                    net.append(nn.Conv3d(self.x_size, self.x_size, kernel_size=1, stride=1, padding=0))
            self.net = nn.Sequential(*net)
        
    def forward(self, x, y):
        """Calulate the score 
        """
        # import ipdb;ipdb.set_trace()
        x_pred = self.net(y)   
        x_pred = x_pred.flatten(2)
        x_pred = x_pred.transpose(-1, -2)

        x = x.flatten(2)
        x = x.transpose(-1, -2)

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=-1, keepdim=True)
        x = x / x.norm(dim=-1, keepdim=True)
      
        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.transpose(-1, -2)), dim=-1)   # bs

        nce = -(pos - neg).mean()
        return nce


if __name__ == "__main__":
    i_cpc = ImageCPC(x_size=10, y_size=5, n_layers=1)

    x = torch.rand(1, 10, 8, 8, 8)
    y = torch.rand(1, 5, 8, 8, 8)

    loss = i_cpc(x, y)

    print(loss)