import torch
from torch import nn


class UniformCSIGenerator(nn.Module):
    def __init__(
        self,
        channel: str,
        train_min: float,
        train_max: float,
        val_min: float,
        val_max: float,
        test_min: float,
        test_max: float,
        **kwargs
    ):
        super().__init__()
        self.channel = channel
        
        self.test_min = test_min
        self.test_max = test_max
        
        self.ranges = {
            "train": (train_min, train_max),
            "val": (val_min, val_max),
            "test": (test_min, test_max),
        }

    def forward(self, num_data, dtype, device, stage):
        if self.channel == "awgn":
            csi = torch.empty(num_data, 1, dtype=dtype, device=device).uniform_(*self.ranges[stage])
        elif self.channel == "rayleigh":
            snr = torch.empty(num_data, 1, dtype=dtype, device=device).uniform_(*self.ranges[stage])
            h1 = torch.sqrt(torch.tensor(0.5)) * torch.randn(num_data, 1, dtype=dtype, device=device)
            h2 = torch.sqrt(torch.tensor(0.5)) * torch.randn(num_data, 1, dtype=dtype, device=device)
            csi = torch.cat([h1, h2, snr], dim=1)
        else:
            raise NotImplementedError

        return csi