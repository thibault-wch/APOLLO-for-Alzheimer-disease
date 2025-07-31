import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from .routing_attention import *


class Expert(nn.Module):
    def __init__(self, input_dim, mode,drop_ratio=0):
        super(Expert, self).__init__()
        mlp_hidden_dim = 2*int(input_dim)
        approx_silu = lambda: nn.SiLU()
        self.fc = Mlp(in_features=input_dim,
                      hidden_features=mlp_hidden_dim,
                      out_features=((input_dim//2) if mode!='single' else None),
                      act_layer=approx_silu, drop=drop_ratio)

    def forward(self, x):
        return self.fc(x)


class SharedRoutingNetwork(nn.Module):
    def __init__(self, input_dim, num_heads, num_atlas, lambda_init):
        super(SharedRoutingNetwork, self).__init__()
        self.attn = AtlasMultiDiffAttn(embed_dim=input_dim,
                                       lambda_init=lambda_init,
                                       num_heads=num_heads,
                                       num_altas=num_atlas)

    def forward(self, x):
        logits = self.attn(x)
        return logits


class SparseAtlasMoE(nn.Module):
    def __init__(self, input_dim, num_atlas, num_heads, lambda_init, mode="single"):
        super(SparseAtlasMoE, self).__init__()
        approx_gelu = lambda: nn.GELU()

        self.mode = mode
        self.experts = nn.ModuleList([Expert(input_dim,mode, 0.3) for _ in range(num_atlas)])
        self.shared_experts = Expert(input_dim,mode, 0.1)
        self.recon_network = Mlp(in_features=( input_dim if mode!= 'single' else input_dim * 2),
                                 hidden_features=input_dim,
                                 out_features=input_dim,
                                 act_layer=approx_gelu, drop=0.2)
        self.routing_network = SharedRoutingNetwork(input_dim, num_heads, num_atlas, lambda_init)
        self.num_experts = num_atlas
        self.input_dim = input_dim
        self.lambda_init=lambda_init
        self.layernorm_share=nn.LayerNorm(input_dim//2 if mode!= 'single' else input_dim )
        self.layernorm_specific=nn.LayerNorm(input_dim//2 if mode!= 'single' else input_dim )
    def forward(self, x):
        # B, T, NE = ori_x.shape  # B: batch size, T: number of tokens, NE: number of experts
        # Compute routing weights, input shape: (B, T, NE), output shape: (B, T)
        routing_weights = self.routing_network(x)  # shape: (B, T)

        # Reshape routing weights for broadcasting
        routing_weights = routing_weights.unsqueeze(-1)  # shape: (B, T, 1)

        # Prepare input for experts
        expert_inputs = x.permute(1, 0, 2)  # shape: (NE, B, T)
        # Compute expert outputs in a batched manner
        expert_outputs = torch.stack([expert(expert_inputs[i]) for i, expert in enumerate(self.experts)],
                                     dim=0).permute(1, 0, 2)  # shape: (NE, B, T)

        shared_outputs = self.shared_experts(expert_inputs).permute(1, 0, 2)  # shape: (NE, B, T)
        recon_outputs = self.recon_network(torch.concat((expert_outputs, shared_outputs), dim=-1))

        # overall
        final_output = torch.concat((
           self.layernorm_share(shared_outputs.mean(1)),
            self.layernorm_specific((expert_outputs * routing_weights).sum(1))), dim=1)


        return final_output, expert_outputs, shared_outputs, expert_inputs.permute(1,0,2), recon_outputs, routing_weights
