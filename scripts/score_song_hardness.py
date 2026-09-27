"""Step 1 of the FMA-oversampling pipeline: score every labeled training song by
raw separability gap (see scripts/probe_separability.py for the metric and the
reasoning behind it). Produces a per-song ranking -- the hardest (lowest-gap)
songs become the "hard reference" set for scripts/find_similar_fma.py's
nearest-neighbor search into FMA.

Runs over train+val items for the given fold (all labeled non-GTZAN datasets),
with an optional per-dataset cap for tractability.

Usage:
    python scripts/score_song_hardness.py --ckpt PATH/TO/checkpoint.ckpt \
        --data-dir data_beatthis --fold 0 --per-dataset-cap 40 \
        --out runs/hardness_scores.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path.home() / "jaehoon" / "beat_this"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beat_this.dataset import BeatDataModule
from beat_this.dataset.dataset import BeatTrackingDataset
from beat_this.inference import split_predict_aggregate
from beatfm_beatthis_bridge import PLBeatFMThis


def score_song(model, spect, truth_beat):
    pred = split_predict_aggregate(spect[0], chunk_size=1500, border_size=6,
                                   overlap_mode="keep_first", model=model.model)
    prob = pred["beat"].sigmoid().cpu().numpy()
    near_beat = np.zeros_like(truth_beat)
    for b in np.nonzero(truth_beat)[0]:
        near_beat[max(0, b - 5):b + 6] = True
    background = ~near_beat
    if truth_beat.sum() == 0 or background.sum() == 0:
        return float("nan")
    return prob[truth_beat].mean() - prob[background].mean()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data_beatthis")
    p.add_argument("--beat-this-dir", default=str(Path.home() / "jaehoon" / "beat_this"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--per-dataset-cap", type=int, default=40,
                   help="max songs sampled per dataset, for tractability")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/hardness_scores.csv")
    args = p.parse_args()

    sys.path.insert(0, args.beat_this_dir)

    dm = BeatDataModule(args.data_dir, batch_size=8, train_length=1500,
                        spect_fps=50, num_workers=4, test_dataset="gtzan", fold=args.fold)
    dm.setup(stage="fit")

    model = PLBeatFMThis.load_from_checkpoint(args.ckpt, map_location="cpu", weights_only=False)
    model = model.cuda().eval()

    all_items = dm.train_items + dm.val_items
    by_dataset = {}
    for item in all_items:
        ds = item.split("/", 1)[0]
        by_dataset.setdefault(ds, []).append(item)

    rng = np.random.default_rng(args.seed)
    sampled = []
    for ds, items in by_dataset.items():
        items = list(items)
        rng.shuffle(items)
        sampled.extend(items[:args.per_dataset_cap])
    print(f"Scoring {len(sampled)} songs across {len(by_dataset)} datasets "
          f"(cap={args.per_dataset_cap}/dataset)")

    full_ds = BeatTrackingDataset(sampled, deterministic=True, augmentations={},
        train_length=None, data_folder=Path(args.data_dir), spect_fps=50)
    dl = DataLoader(full_ds, batch_size=1, num_workers=4)

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            spect = batch["spect"].cuda()
            truth_beat = batch["truth_beat"][0].numpy().astype(bool)
            item_name = sampled[i]
            gap = score_song(model, spect, truth_beat)
            rows.append((item_name, gap))
            if i % 50 == 0:
                print(f"progress: {i}/{len(sampled)}", flush=True)

    rows.sort(key=lambda r: (r[1] if not np.isnan(r[1]) else 1e9))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "separability_gap"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} scores to {args.out}")
    print("\nHardest 20 (lowest gap):")
    for item, gap in rows[:20]:
        print(f"  {gap:.3f}  {item}")


if __name__ == "__main__":
    main()
