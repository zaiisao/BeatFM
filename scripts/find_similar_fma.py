"""Step 2 of the FMA-oversampling pipeline: given the hardest labeled songs (from
scripts/score_song_hardness.py), find similar unlabeled FMA tracks via nearest-
neighbor search in MusicFM's own (frozen, student-adapted) feature space.

Both the labeled "hard reference" songs and the FMA candidates are embedded by
mean-pooling MusicFM's hidden states (all 13 layers, averaged over time) -- the
same feature space the beat classifier actually consumes, so similarity here
should reflect "does the frontend represent this audio in a similarly
ambiguous/weak way", not just generic genre/timbre similarity.

Usage:
    python scripts/find_similar_fma.py --ckpt PATH/TO/checkpoint.ckpt \
        --hardness-csv runs/hardness_scores.csv --top-n-hard 60 \
        --fma-dir /disk1/jaehoon/dataset_store/fma/fma_large --fma-sample 5000 \
        --k 20 --out runs/fma_oversample_candidates.csv
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path.home() / "jaehoon" / "beat_this"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beat_this.dataset.dataset import BeatTrackingDataset
from beat_this.inference import split_predict_aggregate
from beatfm_beatthis_bridge import PLBeatFMThis


def embed_from_features(feat):
    """(B, N, F, T) frozen hidden states -> (B, N*F) embedding, mean-pooled over time."""
    return feat.mean(dim=-1).flatten(1)


class FMASingle(Dataset):
    """Loads one 30s clip per FMA path at 24kHz, matching train_musicfm_student.py's convention."""
    def __init__(self, paths, seconds=30):
        self.paths = paths
        self.seconds = seconds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for offset in range(8):
            path = self.paths[(index + offset * 7919) % len(self.paths)]
            try:
                wav, sr = torchaudio.load(str(path))
                wav = wav.mean(dim=0)
                if sr != 24000:
                    wav = torchaudio.functional.resample(wav, sr, 24000)
                native_samples = round(self.seconds * 24000)
                if wav.numel() > native_samples:
                    start = (wav.numel() - native_samples) // 2
                    wav = wav[start:start + native_samples]
                else:
                    wav = F.pad(wav, (0, native_samples - wav.numel()))
                return wav, str(path)
            except Exception:
                continue
        return torch.zeros(round(self.seconds * 24000)), "FAILED"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data_beatthis")
    p.add_argument("--beat-this-dir", default=str(Path.home() / "jaehoon" / "beat_this"))
    p.add_argument("--hardness-csv", required=True)
    p.add_argument("--top-n-hard", type=int, default=60, help="use the N hardest labeled songs as reference")
    p.add_argument("--fma-dir", default="/disk1/jaehoon/dataset_store/fma/fma_large")
    p.add_argument("--fma-sample", type=int, default=5000, help="random FMA tracks to embed and search over")
    p.add_argument("--k", type=int, default=20, help="nearest FMA neighbors per hard reference song")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/fma_oversample_candidates.csv")
    args = p.parse_args()

    sys.path.insert(0, args.beat_this_dir)

    model = PLBeatFMThis.load_from_checkpoint(args.ckpt, map_location="cpu", weights_only=False)
    model = model.cuda().eval()

    # --- embed hard reference songs (Beat This spect -> bt_to_musicfm -> student-adapted encoder) ---
    with open(args.hardness_csv) as f:
        rows = list(csv.DictReader(f))
    hard_items = [r["item"] for r in rows[:args.top_n_hard]]
    print(f"Embedding {len(hard_items)} hard reference songs...")

    # train_length=1500 (30s @ 50fps) to match FMA's clip length and avoid OOM on long
    # pieces (candombe/asap full recordings) -- fine for embedding similarity, which
    # doesn't need the whole piece.
    ref_ds = BeatTrackingDataset(hard_items, deterministic=True, augmentations={},
        train_length=1500, data_folder=Path(args.data_dir), spect_fps=50)
    ref_dl = DataLoader(ref_ds, batch_size=1, num_workers=4)
    ref_embeds = []
    with torch.no_grad():
        for i, batch in enumerate(ref_dl):
            spect = batch["spect"].cuda()
            # reuse the BeatFM feature extractor (bt_to_musicfm + student-adapted encoder), pre-MSAM
            feat = model.model.beatfm.features(spect)   # (1, N, F, T)
            ref_embeds.append(embed_from_features(feat).cpu())
            if i % 20 == 0:
                print(f"  ref progress: {i}/{len(hard_items)}", flush=True)
    ref_embeds = torch.cat(ref_embeds, dim=0)  # (n_ref, N*F)
    ref_embeds = F.normalize(ref_embeds, dim=-1)

    # --- embed a random FMA sample (raw wav -> native extractor path) ---
    all_fma = list(Path(args.fma_dir).rglob("*.mp3"))
    random.Random(args.seed).shuffle(all_fma)
    fma_paths = all_fma[:args.fma_sample]
    print(f"\nEmbedding {len(fma_paths)} FMA candidates (of {len(all_fma)} total)...")

    fma_dl = DataLoader(FMASingle(fma_paths), batch_size=8, num_workers=8)
    fma_embeds, fma_names = [], []
    with torch.no_grad():
        for i, (wav, paths) in enumerate(fma_dl):
            wav = wav.cuda()
            feat = model.model.beatfm.extractor(wav)   # (B, N, F, T), native wav path, student-adapted
            fma_embeds.append(embed_from_features(feat).cpu())
            fma_names.extend(paths)
            if i % 20 == 0:
                print(f"  fma progress: {i * fma_dl.batch_size}/{len(fma_paths)}", flush=True)
    fma_embeds = torch.cat(fma_embeds, dim=0)
    fma_embeds = F.normalize(fma_embeds, dim=-1)

    # cache embeddings so re-analysis doesn't require re-embedding 5000 FMA tracks
    torch.save({"ref_embeds": ref_embeds, "hard_items": hard_items,
               "fma_embeds": fma_embeds, "fma_names": fma_names},
               Path(args.out).with_suffix(".embeds.pt"))

    # --- nearest-neighbor search: for each hard reference, top-k FMA matches ---
    sims = ref_embeds @ fma_embeds.T   # (n_ref, n_fma) cosine similarity
    candidates = {}
    per_reference_log = []
    for r, item in enumerate(hard_items):
        topk = sims[r].topk(args.k)
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            path = fma_names[idx]
            if path == "FAILED":
                continue
            candidates[path] = max(candidates.get(path, -1), score)
            per_reference_log.append((item, path, score))

    with open(Path(args.out).with_suffix(".per_reference.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hard_reference_item", "fma_path", "cosine_sim"])
        w.writerows(per_reference_log)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fma_path", "max_cosine_sim_to_hard_reference"])
        for path, score in sorted(candidates.items(), key=lambda x: -x[1]):
            w.writerow([path, f"{score:.4f}"])
    print(f"\nWrote {len(candidates)} unique FMA oversampling candidates to {args.out}")


if __name__ == "__main__":
    main()
