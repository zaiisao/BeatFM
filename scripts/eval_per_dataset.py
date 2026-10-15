"""Evaluate a train_beatfm_via_beatthis.py checkpoint on full pieces, per dataset.

--datasplit val: the held-out CV fold of every dataset (Beat This's k-fold protocol)
--datasplit test: GTZAN
Pieces are predicted whole (chunked by Beat This's split_predict_aggregate).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "jaehoon" / "beat_this"))
from beat_this.dataset import BeatDataModule

from beatfm_beatthis_bridge import PLBeatFMThis

KEYS = ("F-measure_beat", "CMLt_beat", "AMLt_beat", "F-measure_downbeat", "CMLt_downbeat", "AMLt_downbeat")


def main(args):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    hp["use_dbn"] = args.dbn
    model = PLBeatFMThis(**hp)
    model.on_load_checkpoint(ckpt)                    # restore the frozen MusicFM weights
    model.load_state_dict(ckpt["state_dict"])

    fold = ckpt["datamodule_hyper_parameters"]["fold"]
    datamodule = BeatDataModule(ROOT / "data_beatthis", batch_size=1, num_workers=args.num_workers,
                                spect_fps=50, test_dataset="gtzan", fold=fold,
                                predict_datasplit=args.datasplit)
    datamodule.setup(stage="predict")

    trainer = Trainer(accelerator="gpu", devices=[args.gpu], logger=False, precision="32-true")
    out = trainer.predict(model, datamodule.predict_dataloader())
    metrics = {k: np.asarray([o[0][k] for o in out]) for k in out[0][0]}
    dataset = np.asarray([o[2][0] for o in out])

    print(f"\n{args.ckpt}  fold {fold}  split {args.datasplit}  dbn {args.dbn}")
    print(f"{'dataset':14s} {'n':>4s} " + " ".join(f"{k:>19s}" for k in KEYS))
    table = {}
    for d in np.unique(dataset):
        sel = dataset == d
        row = {k: float(np.nanmean(metrics[k][sel])) for k in KEYS if k in metrics}
        table[d] = {"n": int(sel.sum()), **row}
        print(f"{d:14s} {sel.sum():4d} " + " ".join(f"{100 * row.get(k, np.nan):19.1f}" for k in KEYS))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt")
    p.add_argument("--datasplit", choices=("val", "test"), default="val")
    p.add_argument("--dbn", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out", default=None, help="write per-dataset metrics as JSON")
    main(p.parse_args())
