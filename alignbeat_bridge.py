"""
AlignBeat's alignment head (Ahn et al., arXiv:2510.14391) on the BeatFM encoder.

Encoder: Beat This spectrogram -> MusicFM (frozen, optional LoRA) -> MSAM -> the first
layer of BeatFM's classifier (N*F -> hidden, ReLU) as a projection, moved from 25 to 50 fps.
Head, loss, decoding and Lightning steps: AlignBeat's PLAlignBeat, unchanged.

Warm start mirrors AlignBeat's protocol (encoder from a trained checkpoint, fresh head):
init_from loads a train_beatfm_via_beatthis.py checkpoint into the BeatFM part.
"""
import sys
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / "jaehoon" / "beat_this"))
sys.path.insert(0, str(Path.home() / "jaehoon" / "beatFCOS_new"))

from alignbeat.integration.config import split_subset_kwargs
from alignbeat.integration.lightning import PLAlignBeat
from alignbeat.model.head import SubsetHead, librosa_logmel_onset_strength
from beat_this.model.pl_module import CosineWarmupScheduler

from beatfm_beatthis_bridge import EXTRACTOR, to_50fps
from lora import is_lora_key
from model import BeatFM


class BeatFMSubsetTracker(nn.Module):
    """(B, T, 128) Beat This spect at 50 fps -> AlignBeat head outputs"""

    def __init__(self, head_kwargs, hidden_dim=512, lora_rank=0, lora_alpha=None):
        super().__init__()
        # the whole BeatFM model, so trained checkpoints load by name; its last
        # classifier layer (hidden -> 2 logits) is not used here
        self.beatfm = BeatFM(classifier="mlp", hidden_dim=hidden_dim, input_type="spect",
                             lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.beatfm.classifier[2].requires_grad_(False)
        self.task_heads = SubsetHead(hidden_dim, **head_kwargs)

    def encode(self, spect):
        h = self.beatfm.msam(self.beatfm.features(spect))             # (B, N, F, T25)
        h = rearrange(h, "b n f t -> b t (n f)")
        h = self.beatfm.classifier[1](self.beatfm.classifier[0](h))   # (B, T25, hidden)
        b, n, d = h.shape
        h = to_50fps(rearrange(h, "b t d -> (b d) t"), spect.shape[1])  # feature i on frame 2i
        return rearrange(h, "(b d) t -> b t d", b=b, d=d)

    def forward(self, spect):
        onset = librosa_logmel_onset_strength(spect) if self.task_heads.onset_adjustment else None
        return self.task_heads(self.encode(spect), onset_strength=onset)


class PLAlignBeatFM(PLAlignBeat):
    def __init__(self, hidden_dim=512, lora_rank=0, lora_alpha=None, lora_lr=None,
                 init_from=None, **alignbeat_kwargs):
        alignbeat_kwargs["subset_head"] = True
        super().__init__(**alignbeat_kwargs)                           # builds criterion + decode settings
        head_kwargs, _, _ = split_subset_kwargs(alignbeat_kwargs["subset_kwargs"])
        self.lora_lr = lora_lr if lora_lr is not None else self.lr
        # replace AlignBeat's Beat This encoder with BeatFM, keeping the same head settings
        self.model = BeatFMSubsetTracker(head_kwargs, hidden_dim, lora_rank, lora_alpha)
        if init_from is not None:
            self._init_from(init_from)

    def _init_from(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        own = self.state_dict()
        # train_beatfm_via_beatthis.py names the module model.beatfm.*, as here
        load = {k: v for k, v in sd.items() if k.startswith("model.beatfm.") and k in own}
        lora_needed = [k for k in own if k.startswith(EXTRACTOR) and is_lora_key(k)]
        if any(k not in load for k in lora_needed):
            raise KeyError("init checkpoint lacks this model's LoRA weights (rank mismatch?)")
        trainable = [k for k in own if k.startswith("model.beatfm.") and not k.startswith(EXTRACTOR)]
        missing = [k for k in trainable if k not in load]
        if missing:
            raise KeyError(f"init checkpoint lacks {len(missing)} encoder tensors, e.g. {missing[0]}")
        self.load_state_dict(load, strict=False)
        print(f"[init] loaded {len(load)} BeatFM tensors from {path}; AlignBeat head is fresh")

    def configure_optimizers(self):
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

    # frozen MusicFM is reloaded from its own checkpoint -> store only its LoRA part
    def on_save_checkpoint(self, ckpt):
        ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items()
                              if not k.startswith(EXTRACTOR) or is_lora_key(k)}

    def on_load_checkpoint(self, ckpt):
        own = self.state_dict()
        ckpt["state_dict"].update({k: v for k, v in own.items()
                                   if k.startswith(EXTRACTOR) and not is_lora_key(k)})
