import torch

class UncertaintyLoss:
    def __init__(self):
        pass

    def __call__(self, pred, pred_uncer, label):
        pass
        pred_uncer = torch.sigmoid(pred_uncer)
        pred = pred.argmax(dim=1)
        true_pos = (pred == label).float()
        loss = pred_uncer * true_pos - pred_uncer * (1 - true_pos)

        return loss.mean()

