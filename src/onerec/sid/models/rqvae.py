import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class AttentiveResidualCombiner(nn.Module):
    """Identity-initialized attention over residual quantization levels."""

    def __init__(
        self,
        num_levels: int,
        e_dim: int,
        mode: str = "dynamic",
        use_rmsnorm: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        if mode not in {"dynamic", "static"}:
            raise ValueError(f"Unsupported attentive residual mode: {mode}")
        if temperature <= 0:
            raise ValueError("attentive residual temperature must be positive")

        self.num_levels = num_levels
        self.e_dim = e_dim
        self.mode = mode
        self.temperature = temperature
        self.norm = nn.RMSNorm(e_dim) if use_rmsnorm else nn.Identity()

        if self.mode == "dynamic":
            self.pseudo_queries = nn.Parameter(torch.zeros(num_levels, e_dim))
        else:
            self.static_logits = nn.Parameter(torch.zeros(num_levels))

    def forward(self, quantized_stack):
        # quantized_stack: [batch, num_levels, e_dim]
        if quantized_stack.shape[1] != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} residual levels, got {quantized_stack.shape[1]}"
            )

        if self.mode == "dynamic":
            keys = self.norm(quantized_stack)
            logits = (keys * self.pseudo_queries.unsqueeze(0)).sum(dim=-1)
        else:
            logits = self.static_logits.unsqueeze(0).expand(quantized_stack.shape[0], -1)

        weights = torch.softmax(logits / self.temperature, dim=1)
        gamma = self.num_levels * weights
        combined = (gamma.unsqueeze(-1) * quantized_stack).sum(dim=1)
        return combined, gamma


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 attn_residual_enable=False,
                 attn_residual_mode="dynamic",
                 attn_residual_reg_weight=0.0,
                 attn_residual_use_rmsnorm=True,
                 attn_residual_temperature=1.0,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.attn_residual_enable = attn_residual_enable
        self.attn_residual_mode = attn_residual_mode
        self.attn_residual_reg_weight = attn_residual_reg_weight
        self.attn_residual_use_rmsnorm = attn_residual_use_rmsnorm
        self.attn_residual_temperature = attn_residual_temperature

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]

        # 编码器 改变embedding维度进入量化器
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
                                 
        # 残差量化器 每层的embedding数量和维度，beta，kmeans初始化，kmeans迭代次数，soft kmeans的epsilon和迭代次数
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)
        if self.attn_residual_enable:
            self.attn_residual_combiner = AttentiveResidualCombiner(
                num_levels=len(num_emb_list),
                e_dim=e_dim,
                mode=self.attn_residual_mode,
                use_rmsnorm=self.attn_residual_use_rmsnorm,
                temperature=self.attn_residual_temperature,
            )
        else:
            self.attn_residual_combiner = None

        self._last_attn_residual_loss = None
        self._last_attn_residual_gamma = None
        # 解码器 改变embedding维度回到原始维度
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        # 输入x经过编码器得到编码表示x_e，进入残差量化器得到量化表示x_q、量化损失rq_loss和索引indices，最后经过解码器得到重构结果out
        x = self.encoder(x)
        self._last_attn_residual_loss = None
        self._last_attn_residual_gamma = None
        if self.attn_residual_combiner is not None:
            _, rq_loss, indices, quantized_stack = self.rq(x, use_sk=use_sk, return_quantized=True)
            x_q, gamma = self.attn_residual_combiner(quantized_stack)
            self._last_attn_residual_gamma = gamma.detach()
            self._last_attn_residual_loss = F.mse_loss(gamma, torch.ones_like(gamma))
        else:
            x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        if self._last_attn_residual_loss is not None and self.attn_residual_reg_weight > 0:
            loss_total = loss_total + self.attn_residual_reg_weight * self._last_attn_residual_loss

        return loss_total, loss_recon

    def get_last_attn_residual_loss(self):
        return self._last_attn_residual_loss

    def get_last_attn_residual_gamma(self):
        return self._last_attn_residual_gamma
