"""Select unlabeled FMA clips that resemble SMC, per SMC song, for stage-1 adaptation.

Every clip goes through the beat-training input path: Beat This log-mel (22.05 kHz, 50 fps)
-> MusicFM mel at 100 fps -> stock (frozen) MusicFM. A song's embedding is the mean and the
standard deviation over time of each layer, averaged over layers 4-12 (so it carries how the
features vary in time, not only their average), z-scored with FMA statistics, L2-normalised.

For each SMC song of the fold's TRAINING split, its k nearest FMA clips (cosine) are kept;
the union is written with its similarity statistics, and the selected clips' Beat This
spectrograms are saved for scripts/tapt_musicfm_lora.py --spect-dir.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "jaehoon" / "beat_this"))
from beat_this.dataset import BeatDataModule
from beat_this.preprocessing import LogMelSpect, load_audio

from MusicFMExtractor import MusicFMExtractor
from spect_convert import bt_to_musicfm, build_freq_map

FRAMES = 1500                                    # 30 s at 50 fps
LAYERS = list(range(4, 13))


class FMASpect(Dataset):
    """FMA mp3 -> Beat This spectrogram (T, 128) float16, centre 30 s; None if unreadable"""

    def __init__(self, paths):
        self.paths = paths
        self.spect = LogMelSpect()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            wav, sr = load_audio(str(self.paths[i]), dtype="float32")
            wav = torch.as_tensor(wav)
            if wav.ndim > 1:
                wav = wav.mean(dim=-1)
            if sr != 22050:
                import torchaudio
                wav = torchaudio.functional.resample(wav, sr, 22050)
            s = self.spect(wav)
            if len(s) < FRAMES // 2:
                return None
            start = max(0, (len(s) - FRAMES) // 2)
            return i, s[start:start + FRAMES].half()
        except Exception:
            return None


def collate(items):
    items = [x for x in items if x is not None]
    if not items:
        return None
    t = min(len(s) for _, s in items)
    return torch.tensor([i for i, _ in items]), torch.stack([s[:t] for _, s in items])


@torch.no_grad()
def embed(spect, musicfm, width_bt, freq_map):
    """(B, T, 128) Beat This spect -> (B, 2 * 1024) mean/std over time, averaged over LAYERS"""
    mel = bt_to_musicfm(spect.float(), musicfm.stat, width_bt, freq_map)
    _, hidden = musicfm.encoder(mel)
    h = torch.stack([hidden[l] for l in LAYERS])                      # (L, B, T, 1024)
    return torch.cat([h.mean(2), h.std(2)], dim=-1).mean(0)          # (B, 2048)


def main(args):
    dev = torch.device(f"cuda:{args.gpu}")
    musicfm = MusicFMExtractor().musicfm.to(dev).eval()
    width_bt, freq_map = (x.to(dev) for x in build_freq_map())

    # SMC songs of this fold's training split (never the held-out fold)
    dm = BeatDataModule(ROOT / "data_beatthis", batch_size=1, fold=args.fold, spect_fps=50,
                        test_dataset="gtzan", num_workers=4)
    dm.setup("fit")
    smc_items = [i for i in dm.train_items if i.startswith("smc/")]
    ds = dm.train_dataset
    smc_emb = []
    for item in smc_items:
        spect = torch.as_tensor(np.asarray(ds._get_spect({"spect_path": Path(item) / "track.npy"})))
        chunks = [spect[s:s + FRAMES] for s in range(0, max(1, len(spect) - FRAMES // 2), FRAMES)]
        e = torch.cat([embed(c[None].to(dev), musicfm, width_bt, freq_map) for c in chunks if len(c) > 200])
        smc_emb.append(e.mean(0))
    smc_emb = torch.stack(smc_emb).cpu()
    print(f"embedded {len(smc_items)} SMC training songs (fold {args.fold})", flush=True)

    paths = sorted(Path(args.fma_dir).rglob("*.mp3"))
    rng = np.random.default_rng(args.seed)
    paths = [paths[i] for i in rng.permutation(len(paths))[:args.fma_sample]]
    loader = DataLoader(FMASpect(paths), batch_size=32, num_workers=args.num_workers, collate_fn=collate)
    idx_all, emb_all = [], []
    for n, batch in enumerate(loader):
        if batch is None:
            continue
        idx, spect = batch
        emb_all.append(embed(spect.to(dev), musicfm, width_bt, freq_map).cpu())
        idx_all.append(idx)
        if n % 100 == 0:
            print(f"FMA {n * 32}/{len(paths)}", flush=True)
    fma_emb, fma_idx = torch.cat(emb_all), torch.cat(idx_all)
    print(f"embedded {len(fma_emb)} readable FMA clips", flush=True)

    mu, sd = fma_emb.mean(0), fma_emb.std(0).clamp_min(1e-6)
    norm = lambda e: torch.nn.functional.normalize((e - mu) / sd, dim=-1)
    S, F_ = norm(smc_emb), norm(fma_emb)
    sim = S @ F_.T                                                    # (n_smc, n_fma)
    top = sim.topk(args.k, dim=1)
    chosen = {}
    for s in range(len(S)):
        for v, j in zip(top.values[s].tolist(), top.indices[s].tolist()):
            if j not in chosen or v > chosen[j][0]:
                chosen[j] = (v, smc_items[s])
    ss = S @ S.T
    ss.fill_diagonal_(float("nan"))
    print(f"similarity  SMC-SMC mean {np.nanmean(ss.numpy()):.3f}  "
          f"SMC-random FMA mean {sim.mean():.3f}  "
          f"SMC-selected (k={args.k}) mean {top.values.mean():.3f}; {len(chosen)} unique clips")

    out = Path(args.out_dir)
    (out / "spect").mkdir(parents=True, exist_ok=True)
    np.save(out / "smc_emb.npy", smc_emb.numpy())
    np.save(out / "fma_emb.npy", fma_emb.numpy())
    rows = []
    for j, (v, ref) in sorted(chosen.items(), key=lambda x: -x[1][0]):
        p = paths[int(fma_idx[j])]
        _, spect = FMASpect([p])[0]
        f = out / "spect" / f"{p.stem}.npy"
        np.save(f, spect.numpy())
        rows.append({"fma": str(p), "spect": str(f), "similarity": round(v, 4), "nearest_smc": ref})
    with open(out / "selected.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} clips to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--fma-dir", default="/disk1/jaehoon/dataset_store/fma/fma_large")
    p.add_argument("--fma-sample", type=int, default=40000)
    p.add_argument("--k", type=int, default=50, help="FMA neighbours per SMC song")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/fma_like_smc_f0")
    main(p.parse_args())
