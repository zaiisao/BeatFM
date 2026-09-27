"""
Run BeatFM's model (frozen MusicFM + MSAM + classifier) inside Beat This's own
PyTorch Lightning training harness: their dataset (18 datasets, augmentation),
their loss (ShiftTolerantBCELoss), their optimizer/scheduler (AdamW + weight
decay + cosine warmup).

Only the model is swapped; everything else in PLBeatThis is inherited unchanged.
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / "jaehoon" / "beat_this"))

from model import BeatFM                          # this repo
from beat_this.model.pl_module import PLBeatThis   # Beat This repo


class BeatFMAdapter(nn.Module):
    """Wraps BeatFM to match BeatThis's forward interface: (B, T, 128) spect in,
    {"beat": (B,T), "downbeat": (B,T)} logits out, at Beat This's own fps (50)."""

    def __init__(self, layers=None, hidden_dim=512, classifier="mlp", embed_dim=16,
                 kernel_size=3, dilations=(1, 2, 4, 8), student_adapter=None,
                 sum_head=True):
        super().__init__()
        # spect_fps=50 + student_adapter: MusicFM's frontend is adapted to emit
        # natively 50 fps features (no interpolation) -> matches Beat This's fps=50
        # dataset/target resolution exactly. Requires a trained student_adapter.
        self.beatfm = BeatFM(layers=layers, hidden_dim=hidden_dim, classifier=classifier,
                             embed_dim=embed_dim, kernel_size=kernel_size, dilations=dilations,
                             input_type="spect", spect_fps=50, student_adapter=student_adapter,
                             sum_head=sum_head)

    def forward(self, spect):                      # spect: (B, T, 128) Beat This mel, 50 fps in
        beat, downbeat = self.beatfm(spect)         # BeatFM always outputs 25 fps (data.py: FPS=25)
        target_t = spect.shape[1]                   # upsample 25fps -> 50fps to match BeatThis's targets
        beat = torch.nn.functional.interpolate(
            beat.unsqueeze(1), size=target_t, mode="linear", align_corners=True).squeeze(1)
        downbeat = torch.nn.functional.interpolate(
            downbeat.unsqueeze(1), size=target_t, mode="linear", align_corners=True).squeeze(1)
        return {"beat": beat, "downbeat": downbeat}


class PLBeatFMThis(PLBeatThis):
    """Same as PLBeatThis, but self.model is BeatFMAdapter instead of BeatThis."""

    def __init__(self, student_adapter=None, layers=None, hidden_dim=512,
                 classifier="mlp", embed_dim=16, kernel_size=3, dilations=(1, 2, 4, 8),
                 sum_head=True, **beatthis_kwargs):
        # beatthis_kwargs: lr, weight_decay, pos_weights, loss_type, warmup_steps,
        # max_epochs, use_dbn, eval_trim_beats - all inherited.
        beatthis_kwargs.setdefault("fps", 50)   # always 50 for this bridge; avoid dup kwarg on reload
        super().__init__(**beatthis_kwargs)
        # override the model PLBeatThis's __init__ just built
        self.model = BeatFMAdapter(layers=layers, hidden_dim=hidden_dim, classifier=classifier,
                                   embed_dim=embed_dim, kernel_size=kernel_size, dilations=dilations,
                                   student_adapter=student_adapter, sum_head=sum_head)
