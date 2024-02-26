import torch
from torch import nn
from torch.nn import functional as F


class ComplexAveragePowerConstraint(nn.Module):
    def __init__(self, average_power):
        super().__init__()

        self.average_power = average_power

        self.power_avg_factor = torch.sqrt(
            0.5 * torch.tensor(self.average_power)
        )  # TODO: only valid for even number of dimensions

    def forward(self, hids):
        hids_shape = hids.size()

        hids = (
            self.power_avg_factor
            * torch.sqrt(torch.prod(torch.tensor(hids_shape[1:]), 0))
            * hids
            / torch.sqrt(torch.sum(hids**2, dim=list(range(len(hids_shape)))[1:], keepdim=True))
        )

        return hids
