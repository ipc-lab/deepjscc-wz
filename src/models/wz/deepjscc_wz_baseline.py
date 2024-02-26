from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)
import torch.nn as nn
import torch
from src.models.base_deepjscc import BaseDeepJSCC
from src.models.components.afmodule import AFModule
from src.models.nets.utils import calculate_num_filters

# Standard DeepJSCC
class DeepJSCCWZBaseline(BaseDeepJSCC):
    def __init__(self, channel, power_constraint, bw_factor, N, **kwargs):
        super().__init__(channel=channel, power_constraint=power_constraint, bw_factor=bw_factor, N=N, **kwargs)

        M = calculate_num_filters(4, bw_factor)

        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=3,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=M,
                stride=2), 
            AFModule(M, 1),
            AttentionBlock(M),
        ])
        
        self.g_s = nn.ModuleList([
            AttentionBlock(M),
            ResidualBlock(
                in_ch=M,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(N, 1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=3,
                upsample=2),
            AFModule(3, 1),
            ResidualBlock(
                in_ch=3,
                out_ch=3),
        ])
    
    def forward(self, batch):
        (x, _), csi = batch
                
        for layer in self.g_a:
            if isinstance(layer, AFModule):
                x = layer((x, csi))
            else:
                x = layer(x)
        
        x = self.power_constraint(x)

        x = self.channel((x, csi))
        
        for layer in self.g_s:
            if isinstance(layer, AFModule):
                x = layer((x, csi))
            else:
                x = layer(x)

        return x

    def step(self, batch):
        x, csi = batch
        x_hat = self.forward((x, csi))
        
        x = x[0]
        
        loss = self.loss(x_hat, x)

        return loss, x_hat, x