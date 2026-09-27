"""SMC analogue of probe_musicfm_50_gtzan.py: instead of a genre linear probe,
train a frame-level beat vs. non-beat logistic regression directly on frozen
embeddings, and compare across student adapters -- including the FMA-oversampled
one -- to test whether oversampling changed the RAW representation's linear
separability for SMC specifically, isolated from the full BeatFM+MSAM pipeline
and its trained classifier/decoding.

IMPORTANT: the embeddings extracted here (hidden_states(model, mel)[layer]) are at
MusicFM's native 25 fps, NOT 50 fps, even for the 50fps-strided student adapters --
the 50fps mel INPUT still gets downsampled 2x internally (this is the same reason
beatfm_beatthis_bridge.py has to interpolate 25fps model output back up to 50fps to
match Beat This's targets). Labels here are built directly at 25 fps to match, rather
than interpolating, since this script measures raw representational quality, not the
downstream bridge's behavior. An earlier version of this script built labels at 50fps
and truncated both arrays to the same length instead of resampling -- that silently
paired embedding frame i (time i/25s) with label frame i (time i/50s), a systematic
2x misalignment, not just a truncation. Caught by an independent Codex review.
"""
import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from train_musicfm_student import MelSTFT, MusicFM25Hz, ROOT, hidden_states, load_adapter, pair_mels, set_50fps_stride

SR = 24000
FPS_EMB = 25   # native embedding rate -- see module docstring


class SMCTracks(Dataset):
    """Whole SMC tracks (cropped to a max length) with frame-level beat labels at the
    embeddings' own native 25 fps.

    single.split marks every SMC piece "train" (no held-out portion), so this does a
    hash-based 80/20 track-level split itself rather than relying on the (here,
    degenerate) official split file. Hashing (not index % 100 on sorted filenames)
    avoids grouping neighboring/similar filenames into the same split.
    """
    def __init__(self, audio_dir, annotation_dir, max_seconds=180):
        self.items = []
        for wav_path in sorted(Path(audio_dir).glob("*.wav")):
            stem = wav_path.stem
            ann_path = Path(annotation_dir) / f"{stem}.beats"
            if ann_path.exists():
                self.items.append((wav_path, ann_path, stem))
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        wav_path, ann_path, stem = self.items[index]
        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.mean(dim=0)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        max_samples = self.max_seconds * SR
        wav = wav[:max_samples]
        beats = np.loadtxt(ann_path, ndmin=1)
        n_frames = wav.numel() // (SR // FPS_EMB)
        target = np.zeros(n_frames, dtype=bool)
        frames = np.round(beats * FPS_EMB).astype(int)
        frames = frames[(frames >= 0) & (frames < n_frames)]
        target[frames] = True
        is_train = int(hashlib.sha1(stem.encode()).hexdigest(), 16) % 100 < 80
        return wav, torch.from_numpy(target), is_train


def collate(items):
    return items  # keep variable length, process one at a time downstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=Path("/disk1/jaehoon/dataset_store/audio_by_stem/smc"))
    parser.add_argument("--annotation-dir", type=Path,
                        default=ROOT / "data_beatthis/annotations/smc/annotations/beats")
    parser.add_argument("--distilled", type=Path, default=ROOT / "runs/musicfm_student_50_first_3000/best.pt",
                        help="baseline (non-oversampled) distilled adapter")
    parser.add_argument("--oversampled", type=Path, default=ROOT / "runs/musicfm_student_50_oversampled/best.pt")
    parser.add_argument("--selfsup", type=Path, default=None,
                        help="optional: masked-token self-supervised adapter (baseline or oversampled)")
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ds = SMCTracks(args.audio_dir, args.annotation_dir, args.max_seconds)
    print(f"SMC tracks: {len(ds)}", flush=True)

    teacher = MusicFM25Hz(is_flash=False,
                          stat_path=ROOT / "third_party/musicfm/data/msd_stats.json",
                          model_path=ROOT / "third_party/musicfm/data/pretrained_msd.pt")
    base = copy.deepcopy(teacher)
    set_50fps_stride(base, (1, 2))

    models = {"teacher100": teacher, "unadapted50": base}
    distilled = copy.deepcopy(teacher)
    load_adapter(distilled, args.distilled)
    models["distilled50_baseline"] = distilled
    oversampled = copy.deepcopy(teacher)
    load_adapter(oversampled, args.oversampled)
    models["distilled50_oversampled"] = oversampled
    if args.selfsup is not None:
        selfsup = copy.deepcopy(teacher)
        load_adapter(selfsup, args.selfsup)
        models["selfsup50"] = selfsup

    for model in models.values():
        model.to(args.device).eval().requires_grad_(False)
    mel50 = MelSTFT(sample_rate=24000, n_fft=2048, hop_length=480,
                    n_mels=128, is_db=True).to(args.device).eval()

    vectors = {name: [] for name in models}
    labels, train_flags = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            wav, target, is_train = ds[i]
            wav = wav.unsqueeze(0).to(args.device)
            mel100, mel_50 = pair_mels(wav, teacher, mel50)   # properly normalized, matches training
            for name, model in models.items():
                mel = mel100 if name == "teacher100" else mel_50
                emb = hidden_states(model, mel)[12][0]   # (T, F) layer-12 hidden states
                t = min(emb.shape[0], target.shape[0])
                vectors[name].append(emb[:t].cpu().numpy())
            t = min(min(v[-1].shape[0] for v in vectors.values()), target.shape[0])
            for name in models:
                vectors[name][-1] = vectors[name][-1][:t]
            labels.append(target[:t].numpy())
            train_flags.append(np.full(t, is_train))
            if i % 20 == 0:
                print(f"progress: {i}/{len(ds)}", flush=True)

    y = np.concatenate(labels)
    train = np.concatenate(train_flags)
    print(f"Frames: train={train.sum()} test={(~train).sum()}  "
          f"beat frames: train={y[train].sum()} test={y[~train].sum()}", flush=True)

    for name, pieces in vectors.items():
        x = np.concatenate(pieces, axis=0)
        probe = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1, max_iter=1000, class_weight="balanced"))
        probe.fit(x[train], y[train])
        pred = probe.predict(x[~train])
        print(f"{name:24s} beat_F1={f1_score(y[~train], pred):.4f}", flush=True)


if __name__ == "__main__":
    main()
