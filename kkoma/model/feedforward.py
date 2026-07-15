"""Feed-forward networks: GELU MLP (baseline) and SwiGLU (modern candidate)."""

from __future__ import annotations

import torch
import torch.nn as nn

from kkoma.config import ModelConfig


class GELUMLP(nn.Module):
    """Standard 2-layer MLP with GELU, inner dim ~ 4 * d_model."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.act(self.fc_in(x))))


class SwiGLU(nn.Module):
    """SwiGLU FFN: hidden = SiLU(gate(x)) * value(x); out = down(hidden).

    Inner dim ``d_ff`` is chosen (~8/3 * d_model) so total block params roughly
    match the GELU MLP despite the extra projection.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(hidden))


def make_ffn(config: ModelConfig) -> nn.Module:
    """Factory selecting the FFN module and tagging its output projection.

    The output projection is marked with ``_is_residual_proj`` so the
    initializer can apply the 1/sqrt(2*n_layer) residual scaling.
    """

    if config.activation == "gelu":
        ffn = GELUMLP(config.d_model, config.d_ff, bias=config.bias, dropout=config.dropout)
        ffn.fc_out._is_residual_proj = True
    elif config.activation == "swiglu":
        ffn = SwiGLU(config.d_model, config.d_ff, bias=config.bias, dropout=config.dropout)
        ffn.down_proj._is_residual_proj = True
    else:
        raise ValueError(f"unknown activation: {config.activation}")
    return ffn


__all__ = ["GELUMLP", "SwiGLU", "make_ffn"]
