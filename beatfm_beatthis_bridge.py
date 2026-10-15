"""
Run BeatFM's model (MusicFM + MSAM + classifier) inside Beat This's own
PyTorch Lightning training harness: their dataset (augmentation), their loss
(ShiftTolerantBCELoss), their optimizer/scheduler (AdamW + weight decay + cosine warmup).

Input path: Beat This 50 fps spectrogram -> MusicFM mel -> upsampled to 100 fps -> stock MusicFM.
MusicFM is frozen; with lora_rank > 0 LoRA updates on its attention projections are trained.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / "jaehoon" / "beat_this"))

from lora import is_lora_key
from model import BeatFM                          # this repo
from beat_this.model.pl_module import PLBeatThis, CosineWarmupScheduler   # Beat This repo

EXTRACTOR = "model.beatfm.extractor."


def load_lora_init(musicfm, path, rank, alpha):
    """start the LoRA weights from a stage-1 adapter (scripts/tapt_musicfm_lora.py)"""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["rank"], saved["alpha"]) != (rank, alpha if alpha is not None else rank):
        raise ValueError(f"adapter has rank/alpha {saved['rank']}/{saved['alpha']}, model {rank}/{alpha}")
    own = {k for k in musicfm.state_dict() if is_lora_key(k)}
    if set(saved["lora"]) != own:
        raise KeyError(f"adapter keys do not match the model's {len(own)} LoRA tensors")
    musicfm.load_state_dict(saved["lora"], strict=False)
    print(f"LoRA initialized from {path} (stage-1 step {saved['step']}, fold {saved['fold']})")


class BeatFMAdapter(nn.Module):
    """BeatFM with Beat This's interface: (B, T, 128) spect at 50 fps in,
    {"beat": (B, T), "downbeat": (B, T)} logits at 50 fps out"""

    def __init__(self, layers=None, hidden_dim=512, classifier="mlp", embed_dim=16,
                 kernel_size=3, dilations=(1, 2, 4, 8), sum_head=True, lora_rank=0, lora_alpha=None,
                 transformer_layers=4):
        super().__init__()
        self.beatfm = BeatFM(layers=layers, hidden_dim=hidden_dim, classifier=classifier,
                             embed_dim=embed_dim, kernel_size=kernel_size, dilations=dilations,
                             input_type="spect", sum_head=sum_head,
                             lora_rank=lora_rank, lora_alpha=lora_alpha,
                             transformer_layers=transformer_layers)

    def forward(self, spect):
        beat, downbeat = self.beatfm(spect)          # 25 fps: feature i at i / 25 s
        t = spect.shape[1]
        return {"beat": to_50fps(beat, t), "downbeat": to_50fps(downbeat, t)}


def to_50fps(x, t):
    """(B, n) at 25 fps -> (B, t) at 50 fps with feature i placed exactly on frame 2i
    (linear in between); frames past the last feature repeat it"""
    n = x.shape[1]
    x = F.interpolate(x[:, None], size=2 * n - 1, mode="linear", align_corners=True)[:, 0]
    if x.shape[1] < t:
        x = torch.cat([x, x[:, -1:].expand(-1, t - x.shape[1])], dim=1)
    return x[:, :t]


class PLBeatFMThis(PLBeatThis):
    """PLBeatThis with self.model = BeatFMAdapter"""

    def __init__(self, layers=None, hidden_dim=512, classifier="mlp", embed_dim=16, kernel_size=3,
                 dilations=(1, 2, 4, 8), sum_head=True, lora_rank=0, lora_alpha=None, lora_lr=None,
                 lora_init=None, transformer_layers=4, **beatthis_kwargs):
        # beatthis_kwargs: lr, weight_decay, pos_weights, loss_type, warmup_steps,
        # max_epochs, use_dbn, eval_trim_beats
        beatthis_kwargs.setdefault("fps", 50)
        super().__init__(**beatthis_kwargs)
        self.lora_lr = lora_lr if lora_lr is not None else self.lr
        self.model = BeatFMAdapter(layers=layers, hidden_dim=hidden_dim, classifier=classifier,
                                   embed_dim=embed_dim, kernel_size=kernel_size, dilations=dilations,
                                   sum_head=sum_head, lora_rank=lora_rank, lora_alpha=lora_alpha,
                                   transformer_layers=transformer_layers)
        if lora_init is not None:
            load_lora_init(self.model.beatfm.extractor.musicfm, lora_init, lora_rank, lora_alpha)

    def configure_optimizers(self):
        # Beat This's groups (decay only >= 2-dim tensors) plus a separate group for LoRA
        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        lora = [p for n, p in named if is_lora_key(n)]
        rest = [p for n, p in named if not is_lora_key(n)]
        groups = [
            {"params": [p for p in rest if p.ndim >= 2], "weight_decay": self.weight_decay},
            {"params": [p for p in rest if p.ndim <= 1], "weight_decay": 0},
        ]
        if lora:
            groups.append({"params": lora, "weight_decay": self.weight_decay, "lr": self.lora_lr})
        optimizer = torch.optim.AdamW(groups, lr=self.lr)
        self.lr_scheduler = CosineWarmupScheduler(optimizer, self.warmup_steps,
                                                  self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": self.lr_scheduler, "interval": "step"}}

    # frozen MusicFM is reloaded from its own checkpoint -> store only the trainable LoRA part of it
    def on_save_checkpoint(self, ckpt):
        ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items()
                              if not k.startswith(EXTRACTOR) or is_lora_key(k)}

    def on_load_checkpoint(self, ckpt):
        own = self.state_dict()
        missing = [k for k in own if k.startswith(EXTRACTOR) and is_lora_key(k) and k not in ckpt["state_dict"]]
        if missing:
            raise KeyError(f"checkpoint lacks {len(missing)} LoRA tensors, e.g. {missing[0]}")
        ckpt["state_dict"].update({k: v for k, v in own.items()
                                   if k.startswith(EXTRACTOR) and not is_lora_key(k)})
