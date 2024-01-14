import torch.nn as nn


class {PROJECT_NAME}Metric(nn.Module):  # noqa: E999

    def __init__(self, *args, **kwargs):
        super({PROJECT_NAME}Metric, self).__init__()  # noqa: E999
        self.thresh = kwargs['thresh']

    def forward(self, cri_out, net_out, batch):
        pred = net_out['pred'] > self.thresh
        target = batch['target']
        acc = (pred == target).float().mean()
        out = {'accuracy': acc}
        return out
