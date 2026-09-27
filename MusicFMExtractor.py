import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))
from musicfm.model.musicfm_25hz import MusicFM25Hz

MUSICFM_DIR = Path(__file__).resolve().parent / "third_party" / "musicfm"
DEFAULT_STAT_PATH = MUSICFM_DIR / "data" / "msd_stats.json"
DEFAULT_MODEL_PATH = MUSICFM_DIR / "data" / "pretrained_msd.pt"

class MusicFMExtractor(nn.Module):
    def __init__(self, stat_path=DEFAULT_STAT_PATH, model_path=DEFAULT_MODEL_PATH, layers=None,
                 freeze=True, student_adapter=None):
        super().__init__()
        self.musicfm = MusicFM25Hz(is_flash=False, stat_path=stat_path, model_path=model_path)
        if student_adapter is not None:
            saved = torch.load(student_adapter, map_location="cpu", weights_only=False)
            if (saved.get("format_version"), saved.get("student_mel_hop"),
                    tuple(saved.get("student_time_strides", ()))) != (1, 480, (1, 2)):
                raise ValueError("Expected a 50 fps student adapter with (1, 2) time strides")
            for stage, time_stride in zip(self.musicfm.conv.conv, (1, 2)):
                stage.conv1.stride = (2, time_stride)
                stage.conv3.stride = (2, time_stride)
            self.musicfm.conv.load_state_dict(saved["frontend"])
            n = saved["config"]["train_last_n"]
            if len(saved["last_layers"]) != n:
                raise ValueError("Adapter Conformer layer count is inconsistent")
            for layer, state in zip(self.musicfm.conformer.layers[-n:] if n else [],
                                    saved["last_layers"]):
                layer.load_state_dict(state)
        self.layers = list(layers) if layers is not None else list(range(13))
        self.freeze = freeze
        if freeze:
            for p in self.musicfm.parameters():
                p.requires_grad = False
            self.musicfm.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.musicfm.eval()
        return self

    @property
    def n_layers(self):
        return len(self.layers)

    def forward(self, wav):
        with torch.set_grad_enabled(not self.freeze):
            _, hidden_emb = self.musicfm.get_predictions(wav)
        h = torch.stack([hidden_emb[i] for i in self.layers], dim = 1)
        return h.transpose(2, 3)

    def forward_mel(self, mel):
        """Normalized mel (B, 128, T); returns one feature frame per four input frames."""
        with torch.set_grad_enabled(not self.freeze):
            _, hidden_emb = self.musicfm.encoder(mel)
        h = torch.stack([hidden_emb[i] for i in self.layers], dim=1)
        return h.transpose(2, 3)

if __name__ == "__main__":
    import argparse
    import librosa

    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="path to an audio file")
    args = parser.parse_args()

    y, sr = librosa.load(args.audio, sr=24000, mono=True)   # whole track, 24 kHz mono
    wav = torch.from_numpy(y).unsqueeze(0)                  # (1, samples)
    sec = wav.shape[1] / sr

    ext = MusicFMExtractor()                                # default weights in third_party/musicfm/data
    h = ext(wav)
    print(f"{sec:.2f} s -> expected T ~ {sec * 25:.0f}")
    print(h.shape)
