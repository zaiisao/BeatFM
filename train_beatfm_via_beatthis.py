"""
Train BeatFM's model (MusicFM, frozen or + LoRA, + MSAM + classifier) using Beat This's own
dataset (18 datasets incl. augmentation), loss (ShiftTolerantBCELoss), and
optimizer/scheduler (AdamW + weight decay + cosine warmup).

Mirrors beat_this/launch_scripts/train.py almost exactly; only the model
construction (PLBeatFMThis instead of PLBeatThis) differs.
"""
import argparse
import csv
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path.home() / "jaehoon" / "beat_this"))
from beat_this.dataset import BeatDataModule

from beatfm_beatthis_bridge import PLBeatFMThis


def hardness_weights(items, scores_csv, mix, max_weight):
    """per-excerpt sampling weights from out-of-fold beat F of a reference model:
    w = (1 - mix) + mix * d / mean(d),  d = 1 - F,  clipped to max_weight.
    Only training items are looked up, so the held-out fold never affects the weights."""
    with open(scores_csv) as f:
        f1 = {r["item"]: float(r["beat_f_measure"]) for r in csv.DictReader(f)}
    names = [str(item["spect_path"].parent) for item in items]
    d = np.array([1.0 - f1.get(n, np.nan) for n in names])
    n_missing = int(np.isnan(d).sum())
    d[np.isnan(d)] = np.nanmean(d)                       # unscored -> average difficulty
    w = np.minimum((1 - mix) + mix * d / d.mean(), max_weight)
    print(f"hard sampling: {len(w)} excerpts, {n_missing} without a score, weight range "
          f"{w.min():.2f}-{w.max():.2f}")
    share = defaultdict(lambda: [0.0, 0.0])
    for n, wi in zip(names, w):
        share[n.split("/")[0]][0] += 1
        share[n.split("/")[0]][1] += wi
    for ds, (cnt, tot) in sorted(share.items()):
        print(f"  {ds:16s} share {100 * cnt / len(w):5.1f}% -> {100 * tot / w.sum():5.1f}%")
    return torch.as_tensor(w, dtype=torch.double)


class HardSamplingDataModule(BeatDataModule):
    """BeatDataModule whose training loader draws excerpts in proportion to given weights"""

    # plain values only: Lightning pickles datamodule init args into every checkpoint
    def __init__(self, *args, hard_scores=None, hard_mix=0.5, hard_max_weight=4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.hard_scores, self.hard_mix, self.hard_max_weight = hard_scores, hard_mix, hard_max_weight
        self.sample_weights = None

    def train_dataloader(self):
        if self.sample_weights is None:
            self.sample_weights = hardness_weights(self.train_dataset.items, self.hard_scores,
                                                   self.hard_mix, self.hard_max_weight)
        sampler = WeightedRandomSampler(self.sample_weights, num_samples=len(self.sample_weights),
                                        replacement=True)
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
                          sampler=sampler, drop_last=True, pin_memory=True)


def main(args):
    seed_everything(args.seed, workers=True)

    data_dir = Path(__file__).resolve().parent / "data_beatthis"
    checkpoint_dir = Path(args.out_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    augmentations = {}
    if args.tempo_augmentation:
        augmentations["tempo"] = {"min": -20, "max": 20, "stride": 4}
    if args.pitch_augmentation:
        augmentations["pitch"] = {"min": -5, "max": 6}
    if args.mask_augmentation:
        augmentations["mask"] = {
            "kind": "permute", "min_count": 1, "max_count": 6,
            "min_len": 0.1, "max_len": 2, "min_parts": 5, "max_parts": 9,
        }

    if args.hard_scores:
        datamodule_cls = partial(HardSamplingDataModule, hard_scores=args.hard_scores,
                                 hard_mix=args.hard_mix, hard_max_weight=args.hard_max_weight)
    else:
        datamodule_cls = BeatDataModule
    datamodule = datamodule_cls(
        data_dir, batch_size=args.batch_size, train_length=args.train_length,
        spect_fps=50, num_workers=args.num_workers, test_dataset="gtzan",
        augmentations=augmentations, fold=args.fold,
    )
    datamodule.setup(stage="fit")
    pos_weights = datamodule.get_train_positive_weights(widen_target_mask=3)
    print("Using positive weights:", pos_weights)

    pl_model = PLBeatFMThis(
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_lr=args.lora_lr, lora_init=args.lora_init,
        classifier=args.classifier, hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim, kernel_size=args.kernel_size,
        sum_head=args.sum_head,
        lr=args.lr, weight_decay=args.weight_decay, pos_weights=pos_weights,
        loss_type=args.loss, warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs, use_dbn=args.dbn,
        eval_trim_beats=args.eval_trim_beats,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(every_n_epochs=1, dirpath=str(checkpoint_dir),
                        filename=f"{args.name} S{args.seed}"),
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs, accelerator="auto", devices=[args.gpu],
        num_sanity_val_steps=1, logger=CSVLogger(args.out_dir, name="logs"), callbacks=callbacks,
        log_every_n_steps=1, precision="32-true",   # 16-mixed + high pos_weight (88) overflowed to NaN in epoch 0
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.val_frequency,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(pl_model, datamodule)
    trainer.test(pl_model, datamodule)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="beatfm_via_beatthis")
    p.add_argument("--lora-rank", type=int, default=0, help="0 = frozen MusicFM")
    p.add_argument("--lora-alpha", type=float, default=None, help="default: = rank (scale 1)")
    p.add_argument("--lora-lr", type=float, default=None, help="default: = --lr")
    p.add_argument("--lora-init", default=None, help="stage-1 adapter from scripts/tapt_musicfm_lora.py")
    p.add_argument("--classifier", choices=["mlp", "linear", "weighted"], default="mlp")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--sum-head", action="store_true", default=True)
    p.add_argument("--no-sum-head", dest="sum_head", action="store_false")
    p.add_argument("--tempo-augmentation", action="store_true", default=True)
    p.add_argument("--no-tempo-augmentation", dest="tempo_augmentation", action="store_false")
    p.add_argument("--pitch-augmentation", action="store_true", default=True)
    p.add_argument("--no-pitch-augmentation", dest="pitch_augmentation", action="store_false")
    p.add_argument("--mask-augmentation", action="store_true", default=True)
    p.add_argument("--no-mask-augmentation", dest="mask_augmentation", action="store_false")
    p.add_argument("--hard-scores", default=None,
                   help="CSV (item, fold, beat_f_measure) of out-of-fold scores; enables hardness-biased sampling")
    p.add_argument("--hard-mix", type=float, default=0.5, help="0 = uniform, 1 = proportional to 1 - F")
    p.add_argument("--hard-max-weight", type=float, default=4.0)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulate-grad-batches", type=int, default=8)
    p.add_argument("--train-length", type=int, default=1500)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--loss", default="shift_tolerant_weighted_bce")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--eval-trim-beats", type=int, default=5)
    p.add_argument("--dbn", action="store_true", default=False,
                   help="use DBN post-processing instead of Beat This's minimal peak-picking; "
                        "default False since ShiftTolerantBCELoss already tolerates small shifts")
    p.add_argument("--val-frequency", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/beatfm_via_beatthis")
    main(p.parse_args())
