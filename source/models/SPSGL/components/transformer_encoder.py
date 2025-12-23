from torch.nn import TransformerEncoderLayer
from .transformer import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import torch.nn.functional as F


class brain_pathway_transformer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.5, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None,weight_matrix=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None
        self.weight_matrix = weight_matrix

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = None, weight_matrix : Optional[Tensor] =None ) -> Tensor:
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True,weight_matrix=weight_matrix)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights
