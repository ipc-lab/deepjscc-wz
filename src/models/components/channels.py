import torch
from torch import nn

class ComplexAWGNChannel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x, snr = batch

        # inputs: BxCxWxH
        # snr: Bx1

        awgn = torch.randn_like(x) * torch.sqrt(10.0 ** (-snr[:, -1, None, None, None] / 10.0))

        awgn = awgn * torch.sqrt(torch.tensor(0.5, device=x.device))

        x = x + awgn

        return x

class PerfectChannel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, batch):
        x, _ = batch

        return x
