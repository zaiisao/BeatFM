"""Diagnose whether MusicFM's frozen features carry a strong beat signal for a given
dataset, without running full decoding/postprocessing.

For each dataset in the fold's held-out validation split, computes:
  - separability gap: mean raw sigmoid(beat_logit) at true-beat frames minus at
    background frames (>5 frames / 100ms from any true beat). Higher = the frozen
    features more clearly discriminate beat vs. non-beat at that dataset's audio.
  - decoded beat_F: the usual predict_step metric, for comparison.

Then reports the Pearson correlation between the two across datasets. A strong
correlation means dataset-level performance gaps are explained by the frozen
representation itself (upstream of decoding), not by postprocessing choices --
useful as a cheap, decode-free diagnostic before investing in more training or a
different postprocessing strategy.

Usage:
    python scripts/probe_separability.py --ckpt PATH/TO/checkpoint.ckpt \
        --data-dir data_beatthis --fold 0 [--datasets smc ballroom ...] [--limit N]

Background: this was built to test why the student-adapter BeatFM-via-BeatThis
bridge (beatfm_beatthis_bridge.py) underperforms specifically on SMC even though
training and DBN postprocessing help every other dataset. r=0.687 across 15
datasets on the epoch-19 checkpoint confirmed the gap; SMC's low separability
(0.426, near the bottom) showed the frozen MusicFM features themselves don't
discriminate SMC's ambiguous/weak-onset rhythm well -- not a decoding problem.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "beat_this"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beat_this.dataset import BeatDataModule
from beat_this.dataset.dataset import BeatTrackingDataset
from beat_this.inference import split_predict_aggregate
from beatfm_beatthis_bridge import PLBeatFMThis


def analyze_dataset(model, ds_name, val_items, data_dir, limit=None):
    items = [i for i in val_items if i.startswith(f"{ds_name}/")]
    if limit:
        items = items[:limit]
    ds = BeatTrackingDataset(items, deterministic=True, augmentations={},
                             train_length=None, data_folder=data_dir, spect_fps=50)
    dl = DataLoader(ds, batch_size=1, num_workers=2)

    beat_probs, nonbeat_probs, beat_F_list = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            batch_gpu = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            spect = batch_gpu["spect"]
            truth_beat = batch["truth_beat"][0].numpy().astype(bool)

            # chunked forward (avoids OOM on long pieces, e.g. asap/harmonix)
            pred = split_predict_aggregate(spect[0], chunk_size=1500, border_size=6,
                                           overlap_mode="keep_first", model=model.model)
            prob = pred["beat"].sigmoid().cpu().numpy()

            near_beat = np.zeros_like(truth_beat)
            for b in np.nonzero(truth_beat)[0]:
                near_beat[max(0, b - 5):b + 6] = True
            background = ~near_beat

            if truth_beat.sum() > 0:
                beat_probs.append(prob[truth_beat])
            if background.sum() > 0:
                nonbeat_probs.append(prob[background])

            metrics, _, _, _ = model.predict_step(batch_gpu, i)
            beat_F_list.append(metrics["F-measure_beat"])

    if not beat_probs or not nonbeat_probs:
        return float("nan"), float("nan"), len(items)
    gap = np.concatenate(beat_probs).mean() - np.concatenate(nonbeat_probs).mean()
    beat_F = np.mean(beat_F_list) * 100
    return gap, beat_F, len(items)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="PLBeatFMThis checkpoint (.ckpt)")
    p.add_argument("--data-dir", default="data_beatthis", help="Beat This data dir (annotations/ + audio/spectrograms/)")
    p.add_argument("--beat-this-dir", default=str(Path.home() / "jaehoon" / "beat_this"),
                   help="path to the beat_this repo, for imports")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="only these datasets (default: all in the fold's val split)")
    p.add_argument("--limit", type=int, default=None, help="max songs per dataset (default: all)")
    p.add_argument("--dbn", action="store_true", help="use DBN instead of minimal postprocessing for beat_F")
    args = p.parse_args()

    sys.path.insert(0, args.beat_this_dir)

    dm = BeatDataModule(args.data_dir, batch_size=8, train_length=1500,
                        spect_fps=50, num_workers=4, test_dataset="gtzan", fold=args.fold)
    dm.setup(stage="fit")

    model = PLBeatFMThis.load_from_checkpoint(args.ckpt, map_location="cpu",
                                              weights_only=False, use_dbn=args.dbn)
    model = model.cuda().eval()

    datasets = args.datasets or sorted(set(i.split("/", 1)[0] for i in dm.val_items))
    results = {}
    for ds_name in datasets:
        gap, beat_F, n = analyze_dataset(model, ds_name, dm.val_items, Path(args.data_dir), args.limit)
        results[ds_name] = (gap, beat_F, n)
        print(f"{ds_name:16s} n={n:3d}  gap={gap:.3f}  beat_F={beat_F:.1f}")

    gaps = np.array([v[0] for v in results.values()])
    beat_Fs = np.array([v[1] for v in results.values()])
    valid = ~np.isnan(gaps)
    if valid.sum() >= 2:
        r = np.corrcoef(gaps[valid], beat_Fs[valid])[0, 1]
        print(f"\nPearson correlation (separability gap vs beat_F) across {valid.sum()} datasets: r = {r:.3f}")


if __name__ == "__main__":
    main()
