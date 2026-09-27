"""Compare frozen MusicFM embeddings with the same GTZAN genre classifier."""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from compare_musicfm_50 import select_gtzan
from train_musicfm_student import (
    MelSTFT, MusicFM25Hz, ROOT, hidden_states, load_adapter, pair_mels,
    set_50fps_stride,
)


class GTZANCrops(Dataset):
    def __init__(self, paths, seconds=6):
        self.paths = paths
        self.seconds = seconds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            wav, sr = torchaudio.load(str(path))
            wav = wav.mean(dim=0)
            native_samples = self.seconds * sr
            if wav.numel() > native_samples:
                start = (wav.numel() - native_samples) // 2
                wav = wav[start:start + native_samples]
            else:
                wav = F.pad(wav, (0, native_samples - wav.numel()))
            if sr != 24000:
                wav = torchaudio.functional.resample(wav, sr, 24000)
            target = self.seconds * 24000
            wav = F.pad(wav[:target], (0, max(0, target - wav.numel())))
            return wav, path.name.split(".")[0], index % 100 < 80
        except (OSError, RuntimeError) as exc:
            print(f"Skipping {path}: {str(exc).splitlines()[0]}", flush=True)
            return None


def collate(items):
    valid = [item for item in items if item is not None]
    if not valid:
        return None
    wav, label, is_train = zip(*valid)
    return torch.stack(wav), list(label), list(is_train)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtzan-dir", type=Path,
                        default=Path("/disk1/jaehoon/dataset_store/meter2800/GTZAN"))
    parser.add_argument("--distilled", type=Path,
                        default=ROOT / "runs/musicfm_student_50_first/best.pt")
    parser.add_argument("--selfsup", type=Path,
                        default=ROOT / "runs/musicfm_50_selfsup_first/best.pt")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    paths = select_gtzan(args.gtzan_dir, 100)
    loader = DataLoader(GTZANCrops(paths), batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=args.device.startswith("cuda"))
    teacher = MusicFM25Hz(is_flash=False,
                          stat_path=ROOT / "third_party/musicfm/data/msd_stats.json",
                          model_path=ROOT / "third_party/musicfm/data/pretrained_msd.pt")
    base, distilled, selfsup = (copy.deepcopy(teacher) for _ in range(3))
    set_50fps_stride(base, (1, 2))
    load_adapter(distilled, args.distilled)
    load_adapter(selfsup, args.selfsup)
    models = {"teacher100": teacher, "unadapted50": base,
              "distilled50": distilled, "selfsup50": selfsup}
    for model in models.values():
        model.to(args.device).eval().requires_grad_(False)
    mel50 = MelSTFT(sample_rate=24000, n_fft=2048, hop_length=480,
                    n_mels=128, is_db=True).to(args.device).eval()
    vectors = {name: [] for name in models}
    labels, train_flags = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            wav, genre, is_train = batch
            wav = wav.to(args.device, non_blocking=True)
            mel100, mel_50 = pair_mels(wav, teacher, mel50)
            for name, model in models.items():
                mel = mel100 if name == "teacher100" else mel_50
                emb = hidden_states(model, mel)[12].mean(dim=1)
                vectors[name].append(emb.cpu().numpy())
            labels.extend(genre)
            train_flags.extend(is_train)
    y = np.asarray(labels)
    train = np.asarray(train_flags, dtype=bool)
    print(f"GTZAN tracks: train={train.sum()} test={(~train).sum()}", flush=True)
    for name, pieces in vectors.items():
        x = np.concatenate(pieces, axis=0)
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))
        probe.fit(x[train], y[train])
        pred = probe.predict(x[~train])
        print(f"{name} accuracy={accuracy_score(y[~train], pred):.4f} "
              f"macro_f1={f1_score(y[~train], pred, average='macro'):.4f}", flush=True)


if __name__ == "__main__":
    main()
