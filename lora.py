import torch
from torch import nn

ATTENTION_PROJECTIONS = ("linear_q", "linear_k", "linear_v", "linear_out")


class LoRALinear(nn.Module):
    """frozen Linear + trainable low-rank update: W x + (alpha / rank) * B A x
    B starts at zero -> the wrapped layer is initially identical to the base layer"""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def add_lora(musicfm, rank, alpha):
    """wrap q / k / v / out projections of every Conformer self-attention layer"""
    musicfm.requires_grad_(False)
    n = 0
    for layer in musicfm.conformer.layers:
        attn = layer.self_attn
        for name in ATTENTION_PROJECTIONS:
            setattr(attn, name, LoRALinear(getattr(attn, name), rank, alpha))
            n += 1
    if n == 0:
        raise RuntimeError("no Conformer attention layers found for LoRA")
    return n


def is_lora_key(name):
    return ".lora_A." in name or ".lora_B." in name
