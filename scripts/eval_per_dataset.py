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

class CASMPost:
    """Beat This postprocessor interface around CASM's frozen 7F decoder (He et al., 2026)"""

    def __init__(self):
        from casm_beat_tracking import CASMProcessor
        self.casm = CASMProcessor(input_type="logits")       # 50 fps, 30-300 BPM

    def __call__(self, beat, downbeat, padding_mask=None):
        beats, downbeats = [], []
        for b, d in zip(beat.double().cpu().numpy(), downbeat.double().cpu().numpy()):
            ev = self.casm(np.stack([b, d], axis=1))
            beats.append(ev[:, 0])
            downbeats.append(ev[ev[:, 1] == 1, 0])
        return tuple(beats), tuple(downbeats)                 # Beat This metrics expect tuples


def wide_dbn(fps=50):
    """madmom DBN with the 30-300 BPM range used by CASM's paper (Beat This defaults to 55-215)"""
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    return DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=30.0, max_bpm=300.0,
                                        fps=fps, transition_lambda=100)


KEYS = ("F-measure_beat", "CMLt_beat", "AMLt_beat", "F-measure_downbeat", "CMLt_downbeat", "AMLt_downbeat")


def main(args):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    hp["use_dbn"] = args.post in ("dbn", "dbn_wide")
    model = PLBeatFMThis(**hp)
    if args.post == "dbn_wide":
        model.postprocessor.dbn = wide_dbn()
    elif args.post == "casm":
        model.postprocessor = CASMPost()
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

    print(f"\n{args.ckpt}  fold {fold}  split {args.datasplit}  post {args.post}")
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
    p.add_argument("--post", choices=("minimal", "dbn", "dbn_wide", "casm"), default="minimal",
                   help="minimal: Beat This peak picking; dbn: madmom DBN 55-215 BPM; dbn_wide: 30-300 BPM; casm")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out", default=None, help="write per-dataset metrics as JSON")
    main(p.parse_args())
