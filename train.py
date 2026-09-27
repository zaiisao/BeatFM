import argparse
from collections import defaultdict
from pathlib import Path

import mir_eval
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from data import FPS, RATE, ClipDataset, PieceDataset, build_cache, read_manifest, split_rows
from model import BeatFM


def masked_bce(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def beat_scores(ref, est, trim_sec):
    ref = mir_eval.beat.trim_beats(np.asarray(ref, float), min_beat_time=trim_sec)
    est = mir_eval.beat.trim_beats(np.asarray(est, float), min_beat_time=trim_sec)
    _, cmlt, _, amlt = mir_eval.beat.continuity(ref, est)
    return {"F": mir_eval.beat.f_measure(ref, est), "CMLt": cmlt, "AMLt": amlt}   # 70 ms window


def first(batch):
    return batch[0]


class PLBeatFM(LightningModule):
    def __init__(self, lr=3e-4, layers=None, hidden_dim=512, classifier="mlp", embed_dim=16,
                 kernel_size=3, dilations=(1, 2, 4, 8),
                 dbn=True, chunk_sec=0.0, trim_sec=5.0, input_type="wav", dbn_fps=50,
                 spect_fps=100, student_adapter=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = BeatFM(layers=layers, hidden_dim=hidden_dim, classifier=classifier, embed_dim=embed_dim,
                            kernel_size=kernel_size, dilations=tuple(dilations),
                            input_type=input_type, spect_fps=spect_fps,
                            student_adapter=student_adapter)
        self.per_frame = RATE[input_type] // FPS                             # input steps per output frame
        self.dbn = None
        if dbn:
            from madmom.features.downbeats import DBNDownBeatTrackingProcessor
            # the DBN models beat intervals in whole frames: at 25 fps, 120 BPM (12.5 frames) can only be
            # 12 or 13 frames -> tempo drifts. Activations are upsampled to dbn_fps before decoding.
            self.dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0,
                                                    fps=dbn_fps, transition_lambda=100)
        self.test_results = []



    # ---- training ---------------------------------------------------------------
    def _losses(self, batch):
        beat, down = self.model(batch["x"])                                  # (B, T) each
        mask = batch["mask"].float()
        db_mask = mask * batch["has_downbeats"].float()[:, None]             # e.g. SMC has no downbeats
        return masked_bce(beat, batch["beat"], mask), masked_bce(down, batch["downbeat"], db_mask)

    def training_step(self, batch, _):
        lb, ld = self._losses(batch)
        self.log_dict({"train_loss": lb + ld, "train_beat": lb, "train_downbeat": ld},
                      on_step=False, on_epoch=True, batch_size=len(batch["x"]))
        return lb + ld

    def validation_step(self, batch, _):
        lb, ld = self._losses(batch)
        self.log_dict({"val_loss": lb + ld, "val_beat": lb, "val_downbeat": ld},
                      on_epoch=True, prog_bar=True, batch_size=len(batch["x"]))

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr)

    # frozen MusicFM is reloaded from its own checkpoint -> don't store 1.3 GB per save
    def on_save_checkpoint(self, ckpt):
        ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith("model.extractor.")}

    def on_load_checkpoint(self, ckpt):
        ckpt["state_dict"].update({k: v for k, v in self.state_dict().items() if k.startswith("model.extractor.")})

    # ---- testing on whole pieces --------------------------------------------------
    @torch.no_grad()
    def predict_piece(self, x):
        """whole piece (wav (samples,) or spect (T_bt, 128)) -> framewise probabilities (T,)
        chunk_sec > 0 splits long pieces"""
        p = self.per_frame                                                   # 960 samples or 2 spect frames
        n = -(-len(x) // p) * p
        x = torch.cat([x, x.new_zeros((n - len(x),) + x.shape[1:])])         # multiple of p -> T = n / p
        T, win = n // p, int(self.hparams.chunk_sec * FPS)
        if win <= 0 or T <= win:
            beat, down = self.model(x[None])
            return beat[0].sigmoid(), down[0].sigmoid()
        border = win // 8
        beat, down = torch.zeros(T, device=x.device), torch.zeros(T, device=x.device)
        for s in list(range(0, T - win, win - 2 * border)) + [T - win]:
            b, d = self.model(x[s * p:(s + win) * p][None])
            lo = 0 if s == 0 else border                                     # later windows overwrite borders
            beat[s + lo:s + win], down[s + lo:s + win] = b[0, lo:].sigmoid(), d[0, lo:].sigmoid()
        return beat, down


    def decode(self, beat, down):
        beat, down = beat.float().cpu().numpy(), down.float().cpu().numpy()
        if self.dbn is not None:
            fps = self.hparams.dbn_fps
            t_in = np.arange(len(beat)) / FPS                               # model output: 25 fps
            t_out = np.arange(int(len(beat) * fps / FPS)) / fps             # DBN input: dbn_fps
            beat, down = np.interp(t_out, t_in, beat), np.interp(t_out, t_in, down)
            eps = 1e-5                                                       # madmom takes log -> avoid log(0)
            act = np.stack([np.clip(beat - down, eps, 1), np.clip(down, eps, 1)], axis=1)   # (beat-only, downbeat)
            out = self.dbn(act)
            if len(out) == 0:
                return np.zeros(0), np.zeros(0)
            return out[:, 0], out[out[:, 1] == 1, 0]

        def peaks(p):                                                        # fallback: local maxima > 0.5
            return np.flatnonzero((p > 0.5) & (p >= np.roll(p, 1)) & (p >= np.roll(p, -1))) / FPS
        return peaks(beat), peaks(down)


        def peaks(p):                                                        # fallback: local maxima > 0.5
            return np.flatnonzero((p > 0.5) & (p >= np.roll(p, 1)) & (p >= np.roll(p, -1))) / FPS
        return peaks(beat), peaks(down)
    
    def test_step(self, item, _):
        est_beats, est_downbeats = self.decode(*self.predict_piece(item["x"]))
        res = {"dataset": item["dataset"]}
        res.update({f"beat_{k}": v for k, v in beat_scores(item["beats"], est_beats, self.hparams.trim_sec).items()})
        if item["has_downbeats"]:
            res.update({f"downbeat_{k}": v for k, v in
                        beat_scores(item["downbeats"], est_downbeats, self.hparams.trim_sec).items()})
        self.test_results.append(res)


    def on_test_epoch_end(self):
        by_ds = defaultdict(list)
        for r in self.test_results:
            by_ds[r["dataset"]].append(r)
        keys = ["beat_F", "beat_CMLt", "beat_AMLt", "downbeat_F", "downbeat_CMLt", "downbeat_AMLt"]
        print(f"\n{'dataset':12s} {'n':>4s} " + " ".join(f"{k:>13s}" for k in keys))
        for ds, rs in sorted(by_ds.items()):
            vals = []
            for k in keys:
                xs = [r[k] for r in rs if k in r]
                vals.append(f"{100 * np.mean(xs):13.1f}" if xs else f"{'-':>13s}")
                if xs:
                    self.log(f"test/{ds}/{k}", float(np.mean(xs)))
            print(f"{ds:12s} {len(rs):4d} " + " ".join(vals))
        self.test_results.clear()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--input", choices=["wav", "spect"], default="wav",
                   help="wav: audio -> MusicFM mel / spect: Beat This spectrogram converted to MusicFM mel")
    p.add_argument("--spect-fps", type=int, choices=[50, 100], default=100,
                   help="MusicFM mel frame rate for spect input; 50 interpolates encoder features to 25 fps")
    p.add_argument("--student-adapter", default=None,
                   help="distilled 50 fps MusicFM adapter; requires --input spect --spect-fps 50")
    p.add_argument("--cache-dir", default=None, help="wav only: where 24 kHz mono .npy copies are stored")
    p.add_argument("--fold", type=int, default=0, help="8-fold CV fold held out for testing")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--val-ratio", type=float, default=0.1, help="fraction of training pieces for validation")
    p.add_argument("--classifier", choices=["mlp", "linear", "weighted"], default="mlp")
    p.add_argument("--hidden-dim", type=int, default=512, help="classifier MLP hidden size")
    p.add_argument("--embed-dim", type=int, default=16, help="channel-attention Q/K/V dim (MSAM)")
    p.add_argument("--kernel-size", type=int, default=3, help="MS-Conv kernel size (MSAM)")
    p.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8], help="MS-Conv dilation rates (MSAM)")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="MusicFM hidden states (default: all 13)")
    p.add_argument("--no-dbn", action="store_true", help="peak picking instead of the DBN")
    p.add_argument("--chunk-sec", type=float, default=0.0, help="test-time window; 0 = whole piece")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--precision", default="32-true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--save-all", action="store_true", help="keep a checkpoint for every epoch (default: best only)")
    p.add_argument("--no-progress-bar", action="store_true", help="suppress batch progress output")
    p.add_argument("--resume-ckpt", default=None, help="resume Lightning training state from a checkpoint")
    p.add_argument("--limit-tracks", type=int, default=0, help="debug: keep only N tracks per split")
    args = p.parse_args()
    if args.student_adapter is not None and (args.input != "spect" or args.spect_fps != 50):
        p.error("--student-adapter requires --input spect --spect-fps 50")

    seed_everything(args.seed, workers=True)
    rows = read_manifest(args.manifest)
    train_rows, val_rows, test_rows = split_rows(rows, args.fold, args.val_ratio, args.seed)
    if args.limit_tracks:
        train_rows, val_rows, test_rows = (r[:args.limit_tracks] for r in (train_rows, val_rows, test_rows))
    if args.input == "wav":
        if args.cache_dir is None:
            p.error("--cache-dir is required with --input wav")
        build_cache(train_rows + val_rows + test_rows, args.cache_dir)
    print(f"tracks  train {len(train_rows)}  val {len(val_rows)}  test {len(test_rows)}  (input: {args.input})")

    train_dl = DataLoader(ClipDataset(train_rows, args.cache_dir, input_type=args.input), batch_size=args.batch_size,
                          shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_dl = DataLoader(ClipDataset(val_rows, args.cache_dir, input_type=args.input), batch_size=args.batch_size,
                        num_workers=args.num_workers)
    test_dl = DataLoader(PieceDataset(test_rows, args.cache_dir, input_type=args.input), batch_size=1,
                         collate_fn=first, num_workers=args.num_workers)

    model = PLBeatFM(lr=args.lr, layers=args.layers, classifier=args.classifier,
                     hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
                     kernel_size=args.kernel_size, dilations=tuple(args.dilations),
                     dbn=not args.no_dbn, chunk_sec=args.chunk_sec, input_type=args.input,
                     spect_fps=args.spect_fps, student_adapter=args.student_adapter)
    run_dir = Path(args.out_dir) / f"fold{args.fold}"
    if args.input == "spect" and args.spect_fps == 50:
        run_dir /= "spect50_student" if args.student_adapter else "spect50"
    ckpt = ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor="val_loss", mode="min",
                           save_top_k=-1 if args.save_all else 1)
    cuda = torch.cuda.is_available()
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if cuda else "cpu",
        devices=[args.gpu] if cuda else 1,
        precision=args.precision,
        enable_progress_bar=not args.no_progress_bar,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=args.patience), ckpt],
        logger=CSVLogger(run_dir, name="logs"),
    )
    trainer.fit(model, train_dl, val_dl, ckpt_path=args.resume_ckpt)
    trainer.test(model, test_dl, ckpt_path="best")




if __name__ == "__main__":
    main()
