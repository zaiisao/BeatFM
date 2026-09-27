"""Evaluate a saved checkpoint on the test split without training (F / CMLt / AMLt per dataset)."""
import argparse
import sys
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import PieceDataset, read_manifest, split_rows
from train import PLBeatFM, first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--fold", type=int, default=0, help="must match the training fold")
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="val: pick epochs on validation pieces, never on the test set")
    p.add_argument("--val-ratio", type=float, default=0.1, help="must match training")
    p.add_argument("--seed", type=int, default=0, help="must match training")
    p.add_argument("--datasets", nargs="*", default=None, help="only these datasets, e.g. gtzan")
    p.add_argument("--limit-tracks", type=int, default=0, help="debug: first N tracks")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--no-dbn", action="store_true")
    p.add_argument("--cache-dir", type=str, default=None, help="audio cache directory for wav input")
    p.add_argument("--spect-fps", type=int, choices=[50, 100], default=None,
                   help="override checkpoint spectrogram rate to compare the same weights without retraining")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    overrides = {"dbn": not args.no_dbn}
    if args.spect_fps is not None:
        overrides["spect_fps"] = args.spect_fps
    model = PLBeatFM.load_from_checkpoint(args.ckpt, map_location="cpu", **overrides)
    input_type = model.hparams.input_type

    _, val_rows, test_rows = split_rows(read_manifest(args.manifest), args.fold, args.val_ratio, args.seed)
    rows = val_rows if args.split == "val" else test_rows
    if args.datasets:
        rows = [r for r in rows if r["dataset"] in args.datasets]
    if args.limit_tracks:
        rows = rows[:args.limit_tracks]
    print(f"ckpt {args.ckpt} | input {input_type} | {args.split} tracks {len(rows)}")

    test_dl = DataLoader(PieceDataset(rows, cache_dir=args.cache_dir, input_type=input_type), batch_size=1,
                         collate_fn=first, num_workers=args.num_workers)
    cuda = torch.cuda.is_available()
    trainer = Trainer(accelerator="gpu" if cuda else "cpu", devices=[args.gpu] if cuda else 1, logger=False)
    trainer.test(model, test_dl)



if __name__ == "__main__":
    main()
