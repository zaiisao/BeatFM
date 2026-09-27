"""Fine-tune the MusicFM frontend with its original masked-token objective.

The 50 fps student sees masked audio as 50 fps mel, while token targets are
created from unmasked audio with MusicFM's original 100 fps quantizer. Both
paths have 25 fps token positions, preserving the pretrained token vocabulary.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_musicfm_student import (
    FMAClips, MelSTFT, MusicFM25Hz, ROOT, adapt_student, load_adapter,
    save_adapter, split_audio,
)


def masked_token_loss(model, wav, mel50):
    with torch.no_grad():
        # Upstream get_targets() calls tokenize(), whose lookup omits the
        # "_0" suffix on the registered quantizer. Reuse its preprocessing and
        # grouping, then address the pretrained quantizer by its actual name.
        clean = model.preprocessing(wav, features=["melspec_2048"])
        clean = model.normalize(clean)
        clean = model.rearrange(clean)
        targets = {"melspec_2048": model.quantizer_melspec_2048_0(clean["melspec_2048"])}
        masked_wav, masked_indices = model.masking(wav)
        if masked_indices.numel() == 0:
            raise RuntimeError("No masked token positions in this batch")
        mel = mel50(masked_wav)[..., :-1]
        mean = model.stat["melspec_2048_mean"]
        std = model.stat["melspec_2048_std"]
        mel = (mel - mean) / std
    logits, _ = model.encoder(mel)
    if logits["melspec_2048"].shape[:2] != targets["melspec_2048"].shape:
        raise ValueError("50 fps logits and 100 fps token targets are misaligned")
    losses, accuracies = model.get_loss(logits, targets, masked_indices)
    return losses["melspec_2048"], accuracies["melspec_2048"]


@torch.no_grad()
def validate(model, mel50, loader, device, seed):
    model.eval()
    total_loss, total_accuracy, total_count = 0.0, 0.0, 0
    for index, wav in enumerate(loader):
        wav = wav.to(device, non_blocking=True)
        # Use the same masked positions at every validation checkpoint.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 100_000 + index)
            loss, accuracy = masked_token_loss(model, wav, mel50)
        total_loss += loss.item() * wav.shape[0]
        total_accuracy += accuracy.item() * wav.shape[0]
        total_count += wav.shape[0]
    return total_loss / total_count, total_accuracy / total_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path,
                        default=Path("/disk1/jaehoon/dataset_store/fma/fma_large"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/musicfm_50_selfsup_first"))
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume weights and optimizer from an adapter checkpoint; data order restarts")
    parser.add_argument("--model-path", type=Path,
                        default=ROOT / "third_party/musicfm/data/pretrained_msd.pt")
    parser.add_argument("--stat-path", type=Path,
                        default=ROOT / "third_party/musicfm/data/msd_stats.json")
    parser.add_argument("--clip-seconds", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-percent", type=int, default=5)
    parser.add_argument("--val-tracks", type=int, default=64)
    parser.add_argument("--train-last-n", type=int, default=0)
    parser.add_argument("--time-strides", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--oversample-csv", type=Path, default=None,
                        help="CSV with an 'fma_path' column listing tracks to oversample "
                             "(see scripts/find_similar_fma.py). Requires --oversample-weight.")
    parser.add_argument("--oversample-weight", type=float, default=10.0,
                        help="relative sampling weight for tracks in --oversample-csv "
                             "vs. weight 1.0 for the rest of the training set")
    args = parser.parse_args()
    if tuple(args.time_strides) not in ((1, 2), (2, 1)):
        parser.error("time-strides must be 1 2 or 2 1")
    if not 0 <= args.train_last_n <= 12:
        parser.error("train-last-n must be 0..12")
    if min(args.clip_seconds, args.batch_size, args.max_steps, args.val_every,
           args.val_tracks, args.grad_clip) <= 0 or not 0 < args.val_percent < 100:
        parser.error("clip, batch, steps, validation and grad-clip settings must be positive")
    args.objective = "masked_token"

    torch.manual_seed(args.seed)
    train_paths, val_paths = split_audio(args.audio_dir, args.val_percent, args.val_tracks)
    print(f"FMA: {len(train_paths)} training tracks, {len(val_paths)} validation tracks", flush=True)
    sampler, shuffle = None, True
    if args.oversample_csv is not None:
        import csv as _csv
        with open(args.oversample_csv) as f:
            oversample_set = {Path(row["fma_path"]) for row in _csv.DictReader(f)}
        weights = [args.oversample_weight if p in oversample_set else 1.0 for p in train_paths]
        hit = sum(w > 1.0 for w in weights)
        print(f"Oversampling: {hit}/{len(train_paths)} training tracks matched "
              f"--oversample-csv (weight={args.oversample_weight})", flush=True)
        if hit == 0:
            parser.error("--oversample-csv matched zero training tracks -- check paths line up "
                         "with --audio-dir")
        sampler, shuffle = WeightedRandomSampler(weights, num_samples=len(train_paths),
                                                 replacement=True), False

    train_loader = DataLoader(FMAClips(train_paths, args.clip_seconds, True),
                              batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))
    val_loader = DataLoader(FMAClips(val_paths, args.clip_seconds, False),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=args.device.startswith("cuda"))

    model = MusicFM25Hz(is_flash=False, stat_path=args.stat_path,
                        model_path=args.model_path).to(args.device)
    adapt_student(model, args.train_last_n, args.time_strides)
    mel50 = MelSTFT(sample_rate=24000, n_fft=2048, hop_length=480,
                    n_mels=128, is_db=True).to(args.device).eval()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters", flush=True)

    step = 0
    best = float("inf")
    if args.resume is not None:
        saved = load_adapter(model, args.resume)
        if saved.get("objective") != "masked_token":
            parser.error("--resume must point to a masked-token checkpoint")
        for key in ("audio_dir", "train_last_n", "time_strides", "clip_seconds",
                    "model_path", "stat_path", "val_percent", "val_tracks", "seed",
                    "batch_size", "lr", "grad_clip"):
            saved_value = (list(saved["student_time_strides"]) if key == "time_strides"
                           else saved["config"][key])
            if saved_value != getattr(args, key):
                parser.error(f"--{key.replace('_', '-')} must match the resumed checkpoint")
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["step"]
        best = saved.get("best_loss", saved["val_loss"])
        print(f"Resumed step {step} from {args.resume}", flush=True)

    initial_loss, initial_accuracy = validate(model, mel50, val_loader, args.device, args.seed)
    print(f"step {step} validation loss {initial_loss:.5f} accuracy {initial_accuracy:.5f}", flush=True)
    if args.resume is None:
        best = initial_loss
        save_adapter(args.out_dir / "best.pt", model, optimizer, args, step, initial_loss, best)
        save_adapter(args.out_dir / "last.pt", model, optimizer, args, step, initial_loss, best)
    while step < args.max_steps:
        for wav in train_loader:
            model.train()
            model.conformer.eval()  # frozen layers remain deterministic
            wav = wav.to(args.device, non_blocking=True)
            loss, accuracy = masked_token_loss(model, wav, mel50)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                           args.grad_clip)
            optimizer.step()
            step += 1
            if step == 1 or step % 10 == 0:
                print(f"step {step} training loss {loss.item():.5f} accuracy {accuracy.item():.5f}",
                      flush=True)
            if step % args.val_every == 0 or step == args.max_steps:
                val_loss, val_accuracy = validate(model, mel50, val_loader, args.device, args.seed)
                print(f"step {step} validation loss {val_loss:.5f} accuracy {val_accuracy:.5f}",
                      flush=True)
                improved = val_loss < best
                best = min(best, val_loss)
                save_adapter(args.out_dir / "last.pt", model, optimizer, args, step, val_loss, best)
                if improved:
                    save_adapter(args.out_dir / "best.pt", model, optimizer, args, step, val_loss, best)
            if step >= args.max_steps:
                return


if __name__ == "__main__":
    main()
