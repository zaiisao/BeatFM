import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from MusicFMExtractor import MusicFMExtractor
from MSAM import MSAM

from spect_convert import bt_to_musicfm, build_freq_map


class BeatFM(nn.Module):
    """BeatFM = frozen MusicFM -> MSAM -> FC classifier
    in : wav (B, samples), 24 kHz mono
    out: beat logits (B, T), downbeat logits (B, T), T at 25 fps
    """

    def __init__(self, layers=None, feat_dim=1024, hidden_dim=512, classifier="mlp",
                 dilations=(1, 2, 4, 8), kernel_size=3, embed_dim=16, input_type="wav",
                 spect_fps=100, student_adapter=None, sum_head=False):
        super().__init__()
        if spect_fps not in (50, 100):
            raise ValueError(f"unsupported spectrogram frame rate: {spect_fps}")
        if input_type == "wav" and spect_fps != 100:
            raise ValueError("spect_fps=50 applies only to spectrogram input")
        if student_adapter is not None and (input_type != "spect" or spect_fps != 50):
            raise ValueError("student_adapter requires 50 fps spectrogram input")
        self.extractor = MusicFMExtractor(layers=layers, student_adapter=student_adapter)
        self.input_type = input_type
        self.spect_fps = spect_fps
        self.student_adapter = student_adapter
        self.sum_head = sum_head          # Beat This's SumHead: downbeat logit added into beat logit
        if input_type == "spect":
            width_bt, freq_map = build_freq_map()                 # fixed; rebuilt on load, not saved
            self.register_buffer("width_bt", width_bt, persistent=False)
            self.register_buffer("freq_map", freq_map, persistent=False)
        elif input_type != "wav":
            raise ValueError(f"unknown input_type: {input_type}")
        n_layers = self.extractor.n_layers
        self.msam = MSAM(n_layers, dilations, kernel_size, embed_dim)

        # paper: "fully connected layers" (depth / activation not specified)
        self.classifier_type = classifier
        in_dim = n_layers * feat_dim
        if classifier == "weighted":
            # learnable softmax weights over the N layers -> (B, T, F), then the same MLP on F = 1024 dims
            self.layer_logits = nn.Parameter(torch.zeros(n_layers))           # uniform at init
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        elif classifier == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        elif classifier == "linear":
            self.classifier = nn.Linear(in_dim, 2)
        else:
            raise ValueError(f"unknown classifier: {classifier}")


    def features(self, x):
        if self.input_type == "wav":
            return self.extractor(x)
        mel = bt_to_musicfm(x, self.extractor.musicfm.stat, self.width_bt, self.freq_map,
                            output_fps=self.spect_fps)
        h = self.extractor.forward_mel(mel)
        if self.spect_fps == 50 and self.student_adapter is None:
            # Unchanged MusicFM downsamples 50 fps input to 12.5 fps. Align its
            # features to the existing 25 fps targets after the frozen encoder.
            b, n, f, t = h.shape
            target_frames = (x.shape[1] + 1) // 2
            h = F.interpolate(h.reshape(b, n * f, t), size=2 * t - 1,
                              mode="linear", align_corners=True)
            if h.shape[-1] < target_frames:
                h = torch.cat((h, h[..., -1:]), dim=-1)
            h = h.reshape(b, n, f, target_frames)
        return h
    
    def forward(self, x):                                     # wav (B, samples) or spect (B, T_bt, 128)
        h = self.features(x)                                  # (B, N, F, T)   Eq.(1)(2)
        h = self.msam(h)                                      # (B, N, F, T)   Eq.(3)-(10)
        if self.classifier_type == "weighted":
            w = self.layer_logits.softmax(0)                  # (N,)
            h = torch.einsum("bnft,n->btf", h, w)             # (B, T, F)
        else:
            h = rearrange(h, "b n f t -> b t (n f)")          # (B, T, N*F)
        logits = self.classifier(h)                           # (B, T, 2)
        beat, downbeat = logits[..., 0], logits[..., 1]
        if self.sum_head:                                     # Beat This Eq.: beat = beat + downbeat (pre-sigmoid)
            beat = beat + downbeat
        return beat, downbeat
