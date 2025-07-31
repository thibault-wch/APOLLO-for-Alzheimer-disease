import torch
import torch.nn.functional as F
from torch import nn


class AtlasMultiDiffAttn(nn.Module):
    def __init__(
            self,
            embed_dim=128,
            lambda_init=0.7,
            num_altas=56,
            num_heads=4,
            qk_norm=True,
            norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_rep = self.num_heads

        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        self.q_activation= nn.SiLU()
        # self.q_proj_norm=nn.LayerNorm(self.embed_dim)
        self.q_proj_emb = nn.Conv1d(in_channels=num_altas,
                                    out_channels=num_altas,
                                    kernel_size=7, padding=3)# embed
        self.q_proj_atlas = nn.Conv1d(in_channels=embed_dim,
                                      out_channels=embed_dim,
                                      kernel_size=7, padding=3)#atlas
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.lambda_init = lambda_init
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

    def forward(self, x):
        bsz, tgt_len, embed_dim = x.size()  # B L C
        q=torch.mean(
          self.q_activation(self.q_proj_atlas(
                self.q_activation(self.q_proj_emb(x)).permute(0, 2, 1)
                )
            ),dim=2,keepdim=True).permute(0, 2, 1)  # B 1 C

        k = self.k_proj(x) # B L C

        q = q.view(bsz, 1, 2 * self.num_heads, self.head_dim)  # B 1 2H C/2H
        k = k.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)  # B L 2H C/2H
        q, k = self.q_norm(q), self.k_norm(k)

        q *= self.scaling
        attn_weights = torch.einsum("bqhe,bkhe->bhqk", q, k)
        #          = torch.matmul(q, k.transpose(-1, -2)) # k: B 2H C/2H L; attn_weights: B 2H 1 L
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, 1, tgt_len)
        # [ori_version]
        attn_weights =  (torch.softmax(attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1],-1)).mean(1).flatten(1)
        # [new_version]
        # attn_weights=(attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]).mean(1).flatten(1)
        # top_k_logits, indices = attn_weights.topk(35, dim=-1)  # shape: (B, top_k)
        # zeros = torch.zeros_like(attn_weights)
        # attn_weights = zeros.scatter(-1, indices, torch.softmax(top_k_logits,dim=-1))  # shape: (B, T)


        return attn_weights
