"""
Train AlignBeat's alignment head on the BeatFM encoder (alignbeat_bridge.py).

Mirrors AlignBeat's launch_scripts/train.py for the subset head with the paper's settings
(KEEP: beat-only datasets kept): priors from this fold's training split, N from the tempo
ceiling, 20 epochs at lr 3e-4 from a trained encoder with a fresh head (--init-from).
"""
import argparse
import sys
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "jaehoon" / "beat_this"))
sys.path.insert(0, str(Path.home() / "jaehoon" / "beatFCOS_new"))

from alignbeat.constants import BPM_MAX
from alignbeat.integration.data import AlignBeatDataModule
from alignbeat.model.downsample import stages_from_tempo
from alignbeat.training.priors import (get_train_class_prior, get_train_meter_prior,
                                       get_train_positive_weights)

from alignbeat_bridge import PLAlignBeatFM


def main(args):
    seed_everything(args.seed, workers=True)
    augmentations = {} if args.no_augmentation else {
        "tempo": {"min": -20, "max": 20, "stride": 4},
        "pitch": {"min": -5, "max": 6},
        "mask": {"kind": "permute", "min_count": 1, "max_count": 6, "min_len": 0.1,
                 "max_len": 2, "min_parts": 5, "max_parts": 9},
    }
    datamodule = AlignBeatDataModule(ROOT / "data_beatthis", batch_size=args.batch_size,
                                     train_length=args.train_length, spect_fps=50,
                                     num_workers=args.num_workers, test_dataset="gtzan",
                                     augmentations=augmentations, fold=args.fold)
    datamodule.setup(stage="fit")
    pos_weights = get_train_positive_weights(datamodule, widen_target_mask=3)
    meter_prior = get_train_meter_prior(datamodule.train_dataset, candidates=(2, 3, 4, 5, 6, 8))
    data_prior = get_train_class_prior(datamodule.train_dataset)
    stages, n_cand, floor = stages_from_tempo(args.train_length, 50, BPM_MAX)
    print("meter prior", {k: round(v, 4) for k, v in meter_prior.items()}, "class prior", data_prior)
    print(f"N={n_cand} candidates from {stages} halvings (tempo ceiling {floor})")

    pl_model = PLAlignBeatFM(
        hidden_dim=512, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_lr=args.lora_lr,
        init_from=args.init_from,
        fps=50, lr=args.lr, weight_decay=0.01, pos_weights=pos_weights,
        warmup_steps=args.warmup_steps, max_epochs=args.max_epochs, eval_trim_beats=5,
        subset_kwargs={
            "num_candidates": n_cand, "train_length": args.train_length,
            "downsample_stages": stages, "detect_tau": 0.5, "attention_layers": 2,
            "gamma": 1.0, "meter_prior": meter_prior, "data_prior": data_prior,
            "match_event_cost": True, "latent_bar": True,
        },
    )
    out = Path(args.out_dir)
    trainer = Trainer(
        max_epochs=args.max_epochs, accelerator="auto", devices=[args.gpu],
        num_sanity_val_steps=1, logger=CSVLogger(out, name="logs"), log_every_n_steps=1,
        callbacks=[LearningRateMonitor(logging_interval="step"),
                   ModelCheckpoint(every_n_epochs=1, dirpath=str(out / "checkpoints"),
                                   filename=f"{args.name} S{args.seed}")],
        precision="32-true", gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
    )
    trainer.fit(pl_model, datamodule)
    if not args.skip_test:
        trainer.test(pl_model, datamodule)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="alignbeat_fm")
    p.add_argument("--init-from", default=None, help="train_beatfm_via_beatthis.py checkpoint (encoder)")
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--lora-alpha", type=float, default=None)
    p.add_argument("--lora-lr", type=float, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulate-grad-batches", type=int, default=8)
    p.add_argument("--train-length", type=int, default=1500)
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-train-batches", type=float, default=1.0)
    p.add_argument("--limit-val-batches", type=float, default=1.0)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--out-dir", default="runs/alignbeat_fm")
    main(p.parse_args())
