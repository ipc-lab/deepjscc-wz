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

# no parameter sharing, multi-scale
class DeepJSCCWZJoint2(BaseDeepJSCC):
    def __init__(self, bw_factor, N, **kwargs):
        super().__init__(N=N, bw_factor=bw_factor, **kwargs)

        M = calculate_num_filters(4, bw_factor)

        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=6,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=2*N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=2*N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=2*N,
                out_ch=M,
                stride=2), 
            AFModule(M, 1),
            AttentionBlock(M),
        ])

        self.g_a3 = nn.ModuleList([
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
            None,
            None,
            None
        ])
        
        self.g_a2 = nn.ModuleList([
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
            AttentionBlock(2 * M),
            ResidualBlock(
                in_ch=2 * M,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2*N, 1),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2*N, 1),
            AttentionBlock(2 * N),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2 * N, 1),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=3,
                upsample=2),
            AFModule(3*2, 1),
            ResidualBlock(
                in_ch=3*2,
                out_ch=3),
        ])
    
    def forward(self, batch):
        (x, xs), csi = batch
        
        xs_encoder = xs
        
        csi_transmitter = csi # torch.cat([csi, torch.zeros_like(csi)], dim=1)
        csi_sideinfo = csi # torch.cat([csi, ], dim=1)
                
        for layer, layer_s in zip(self.g_a, self.g_a3):
            if isinstance(layer, ResidualBlockWithStride):
                x = torch.cat([x, xs_encoder], dim=1)
            
            if isinstance(layer, AFModule):
                x = layer((x, csi_transmitter))
                if layer_s is not None:
                    xs_encoder = layer_s((xs_encoder, csi_transmitter))
            else:
                x = layer(x)
                if layer_s is not None:
                    xs_encoder = layer_s(xs_encoder)
        
        x = self.power_constraint(x)

        x = self.channel((x, csi))
        
        xs_list = []
        for idx, layer in enumerate(self.g_a2):
            if isinstance(layer, ResidualBlockWithStride):
                xs_list.append(xs)
            
            if isinstance(layer, AFModule):
                xs = layer((xs, csi_sideinfo))
            else:
                xs = layer(xs)
        
        xs_list.append(xs)
        
        for idx, layer in enumerate(self.g_s):
            if idx in [0, 3, 6, 10, 13]:
                last_xs = xs_list.pop()
                x = torch.cat([x, last_xs], dim=1)
            
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