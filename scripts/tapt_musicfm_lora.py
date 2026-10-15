"""Stage 1: task-adaptive pretraining of MusicFM with LoRA (Gururangan et al., 2020).

Continues MusicFM's own masked-token objective on the unlabeled audio of the hard
beat-tracking datasets (training folds only), through the same input path used for
beat training: Beat This 50 fps spectrogram -> MusicFM mel -> 100 fps.

MusicFM masks 400 ms spans of raw audio (p = 0.6) with N(0, 0.1^2) noise. Without raw
audio, the masked mel frames are replaced by the mel of that same noise. Targets are
MusicFM's quantized tokens of the clean mel. Only LoRA weights are trained.

The adapter is loaded for beat training with train_beatfm_via_beatthis.py --lora-init.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "jaehoon" / "beat_this"))
from beat_this.dataset import BeatDataModule
from beat_this.dataset.dataset import BeatTrackingDataset

from lora import add_lora, is_lora_key
from MusicFMExtractor import MusicFMExtractor
from spect_convert import bt_to_musicfm, build_freq_map

HARD = ("smc", "simac", "asap", "rwc_classical")
SPAN = 40                                   # 400 ms at 100 fps = 10 tokens


@torch.no_grad()
def subset(item):
    """dataset name of an item; RWC is split by collection (rwc/rwc_classical_CD... -> rwc_classical)"""
    dataset, stem = item.split("/", 1)
    return stem.split("_CD")[0] if dataset == "rwc" else dataset


def noise_mel(musicfm, seconds=60):
    """normalized mel of MusicFM's masking noise, N(0, 0.1^2) audio at 24 kHz: (128, T)"""
    wav = torch.randn(1, 24000 * seconds) * 0.1
    mel = musicfm.preprocessing(wav, features=["melspec_2048"])
    return musicfm.normalize(mel)["melspec_2048"][0]


def mask_mel(mel, noise, prob):
    """mel (B, 128, T), T % 40 == 0 -> masked mel, token-domain masked indices (like MusicFM.masking)"""
    b, _, t = mel.shape
    spans = torch.rand(b, t // SPAN, device=mel.device) < prob
    frame_mask = spans.repeat_interleave(SPAN, dim=1)                     # (B, T)
    starts = torch.randint(0, noise.shape[1] - t, (b,))
    fill = torch.stack([noise[:, s:s + t] for s in starts]).to(mel)       # (B, 128, T)
    masked = torch.where(frame_mask[:, None], fill, mel)
    token_idx = torch.nonzero(spans.repeat_interleave(SPAN // 4, dim=1))
    return masked, token_idx


def to_mel(spect, musicfm, width_bt, freq_map):
    """Beat This spect (B, T, 128) -> normalized 100 fps MusicFM mel, length cut to a multiple of 40"""
    mel = bt_to_musicfm(spect, musicfm.stat, width_bt, freq_map)
    t = mel.shape[-1] // SPAN * SPAN
    return mel[..., :t]


def step_loss(musicfm, mel, noise, prob):
    # MusicFM.tokenize looks up "quantizer_melspec_2048", but the module is registered with a codebook suffix
    flat = musicfm.rearrange({"melspec_2048": mel.clone()})["melspec_2048"]
    with torch.no_grad():
        targets = {"melspec_2048": musicfm.quantizer_melspec_2048_0(flat)}
    masked, idx = mask_mel(mel, noise, prob)
    logits, _ = musicfm.encoder(masked)
    losses, accs = musicfm.get_loss(logits, targets, idx)
    return sum(losses.values()), sum(accs.values()) / len(accs)


@torch.no_grad()
def evaluate(musicfm, loader, noise, prob, convert, device, seed=0):
    musicfm.eval()
    g = torch.random.get_rng_state()
    torch.manual_seed(seed)                                               # same masks every evaluation
    tot_l = tot_a = n = 0
    for batch in loader:
        mel = convert(batch["spect"].to(device))
        loss, acc = step_loss(musicfm, mel, noise, prob)
        tot_l += loss.item() * len(mel)
        tot_a += acc.item() * len(mel)
        n += len(mel)
    torch.random.set_rng_state(g)
    return tot_l / n, tot_a / n


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    dm = BeatDataModule(ROOT / "data_beatthis", batch_size=args.batch_size, fold=args.fold,
                        spect_fps=50, test_dataset="gtzan", num_workers=args.num_workers)
    dm.setup("fit")                                    # train_items exclude the held-out fold
    items = [i for i in dm.train_items if subset(i) in args.datasets]
    rng = np.random.default_rng(args.seed)
    items = list(rng.permutation(items))
    n_val = max(16, len(items) // 20)                  # monitoring split, taken from training folds
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"fold {args.fold}: {len(train_items)} train / {len(val_items)} monitor tracks from {args.datasets}")

    aug = {} if args.no_augmentation else {"tempo": {"min": -20, "max": 20, "stride": 4},
                                            "pitch": {"min": -5, "max": 6}}
    common = dict(data_folder=ROOT / "data_beatthis", spect_fps=50, train_length=args.train_length)
    train_ds = BeatTrackingDataset(train_items, deterministic=False, augmentations=aug, **common)
    val_ds = BeatTrackingDataset(val_items, deterministic=True, augmentations={}, **common)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    extractor = MusicFMExtractor(lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    musicfm = extractor.musicfm.to(device)
    width_bt, freq_map = (x.to(device) for x in build_freq_map())
    convert = lambda spect: to_mel(spect, musicfm, width_bt, freq_map)
    noise = noise_mel(musicfm.cpu()).to(device)
    musicfm.to(device)

    params = [p for n, p in musicfm.named_parameters() if p.requires_grad]
    assert params and all(is_lora_key(n) for n, p in musicfm.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    vl, va = evaluate(musicfm, val_dl, noise, args.mask_prob, convert, device)
    print(f"step 0  monitor loss {vl:.4f} acc {va:.4f}", flush=True)
    best = vl
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    step, it = 0, iter(train_dl)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl)
            batch = next(it)
        musicfm.eval()                                  # base model has no dropout to enable; LoRA has none
        mel = convert(batch["spect"].to(device, non_blocking=True))
        loss, acc = step_loss(musicfm, mel, noise, args.mask_prob)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            print(f"step {step}  train loss {loss.item():.4f} acc {acc.item():.4f}  lr {sched.get_last_lr()[0]:.2e}",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            vl, va = evaluate(musicfm, val_dl, noise, args.mask_prob, convert, device)
            print(f"step {step}  monitor loss {vl:.4f} acc {va:.4f}", flush=True)
            if vl < best:
                best = vl
                torch.save({"rank": args.lora_rank, "alpha": args.lora_alpha or args.lora_rank,
                            "fold": args.fold, "datasets": list(args.datasets), "step": step,
                            "monitor_loss": vl, "monitor_acc": va,
                            "lora": {k: v.detach().cpu() for k, v in musicfm.state_dict().items()
                                     if is_lora_key(k)}}, out)
                print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--datasets", nargs="+", default=list(HARD))
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=None)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-length", type=int, default=1500, help="50 fps frames (30 s)")
    p.add_argument("--mask-prob", type=float, default=0.6)
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/tapt_lora8_f0/adapter.pt")
    main(p.parse_args())
