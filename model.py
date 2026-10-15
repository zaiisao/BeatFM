import torch
from torch import nn
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
                 sum_head=False, lora_rank=0, lora_alpha=None):
        super().__init__()
        self.extractor = MusicFMExtractor(layers=layers, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.sum_head = sum_head          # Beat This SumHead: downbeat logit added into beat logit
        self.input_type = input_type
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
        mel = bt_to_musicfm(x, self.extractor.musicfm.stat, self.width_bt, self.freq_map)
        return self.extractor.forward_mel(mel)
    
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
        if self.sum_head:
            beat = beat + downbeat
        return beat, downbeat

