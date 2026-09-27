"""Distill the released MusicFM checkpoint into a 50 fps mel-input student.

The teacher and student receive the same FMA audio crop. The teacher uses the
original 100 fps mel and 4x frontend; the student uses a 50 fps mel and a 2x
frontend. Both produce 25 fps hidden states for the distillation loss.
"""

import argparse
import copy
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))
from musicfm.model.musicfm_25hz import MusicFM25Hz
from musicfm.modules.features import MelSTFT


class FMAClips(Dataset):
    def __init__(self, paths, seconds, random_crop):
        self.paths = paths
        self.seconds = seconds
        self.random_crop = random_crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # A small number of FMA files are known to be damaged. Keep the batch
        # usable by trying distant tracks if decoding fails. Some damaged files
        # occur in consecutive ID runs, so adjacent retries are ineffective.
        for offset in range(8):
            path = self.paths[(index + offset * 7919) % len(self.paths)]
            try:
                wav, sr = torchaudio.load(str(path))
                wav = wav.mean(dim=0)
                native_samples = round(self.seconds * sr)
                if wav.numel() > native_samples:
                    limit = wav.numel() - native_samples
                    start = int(torch.randint(limit + 1, ()).item()) if self.random_crop else limit // 2
                    wav = wav[start:start + native_samples]
                else:
                    wav = F.pad(wav, (0, native_samples - wav.numel()))
                if sr != 24000:
                    wav = torchaudio.functional.resample(wav, sr, 24000)
                target = self.seconds * 24000
                clip = F.pad(wav[:target], (0, max(0, target - wav.numel())))
                return clip
            except (OSError, RuntimeError) as exc:
                print(f"Skipping unreadable FMA file {path}: {exc}", file=sys.stderr)
        raise RuntimeError(f"Could not decode eight tracks near {self.paths[index]}")


def split_audio(audio_dir, val_percent, val_tracks):
    all_paths = sorted(audio_dir.rglob("*.mp3"))
    # FMA-large includes tiny MP3 placeholders that FFmpeg cannot decode.
    # A genuine 30-second excerpt is much larger than 16 KiB.
    paths = [path for path in all_paths if path.stat().st_size >= 16_384]
    if not paths:
        raise FileNotFoundError(f"No MP3 files found under {audio_dir}")
    print(f"Skipped {len(all_paths) - len(paths)} tiny MP3 files", flush=True)
    train, val = [], []
    for path in paths:
        digest = int(hashlib.sha1(path.stem.encode()).hexdigest(), 16)
        (val if digest % 100 < val_percent else train).append(path)
    if not train or not val:
        raise ValueError("Train/validation split is empty; adjust --val-percent")
    val.sort(key=lambda path: hashlib.sha1(path.stem.encode()).digest())
    return train, val[:val_tracks]


def set_50fps_stride(student, time_strides=(1, 2)):
    """Set frontend time strides; both supported layouts reduce 50 to 25 fps."""
    if tuple(time_strides) not in ((1, 2), (2, 1)):
        raise ValueError(f"Unsupported 50 fps time strides: {time_strides}")
    for stage, time_stride in zip(student.conv.conv, time_strides):
        stage.conv1.stride = (2, time_stride)
        stage.conv3.stride = (2, time_stride)


def adapt_student(student, train_last_n, time_strides=(1, 2)):
    student.requires_grad_(False)
    set_50fps_stride(student, time_strides)
    student.conv.requires_grad_(True)
    if train_last_n:
        for layer in student.conformer.layers[-train_last_n:]:
            layer.requires_grad_(True)


def hidden_states(model, mel):
    # Skip the pretrained codebook prediction head: this experiment matches
    # representations, not the original BEST-RQ token targets.
    return model.conformer(model.conv(mel), output_hidden_states=True).hidden_states


def pair_mels(wav, teacher, student_mel):
    with torch.no_grad():
        a = teacher.preprocessor_melspec_2048(wav)[..., :-1]
        b = student_mel(wav)[..., :-1]
        mean = teacher.stat["melspec_2048_mean"]
        std = teacher.stat["melspec_2048_std"]
        return (a - mean) / std, (b - mean) / std


def distill_loss(student_hidden, teacher_hidden, layers):
    terms = []
    for layer in layers:
        s, t = student_hidden[layer].float(), teacher_hidden[layer].float()
        if s.shape != t.shape:
            raise ValueError(f"Layer {layer}: student {s.shape} != teacher {t.shape}")
        cosine = 1 - F.cosine_similarity(s, t, dim=-1).mean()
        terms.append(cosine + 0.05 * F.smooth_l1_loss(s, t))
    return torch.stack(terms).mean()


@torch.no_grad()
def validate(teacher, student, student_mel, loader, layers, device):
    student.eval()
    total, count = 0.0, 0
    for wav in loader:
        wav = wav.to(device, non_blocking=True)
        a, b = pair_mels(wav, teacher, student_mel)
        loss = distill_loss(hidden_states(student, b), hidden_states(teacher, a), layers)
        total += loss.item() * wav.shape[0]
        count += wav.shape[0]
    return total / count


def save_adapter(path, student, optimizer, args, step, val_loss, best_loss):
    state = {
        "format_version": 1,
        "student_mel_hop": 480,
        "student_time_strides": tuple(args.time_strides),
        "step": step,
        "val_loss": val_loss,
        "best_loss": best_loss,
        "config": vars(args),
        "frontend": student.conv.state_dict(),
        "last_layers": [layer.state_dict() for layer in student.conformer.layers[-args.train_last_n:]]
        if args.train_last_n else [],
        "optimizer": optimizer.state_dict(),
    }
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def load_adapter(student, path):
    """Apply a saved adapter to MusicFM initialized from its base checkpoint."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("format_version"), saved.get("student_mel_hop")) != (1, 480) or \
            saved.get("student_time_strides") not in ((1, 2), (2, 1)):
        raise ValueError(f"Unsupported student adapter format: {path}")
    set_50fps_stride(student, saved["student_time_strides"])
    student.conv.load_state_dict(saved["frontend"])
    n = saved["config"]["train_last_n"]
    if len(saved["last_layers"]) != n:
        raise ValueError("Adapter's final Conformer layer count is inconsistent")
    for layer, state in zip(student.conformer.layers[-n:] if n else [], saved["last_layers"]):
        layer.load_state_dict(state)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path,
                        default=Path("/disk1/jaehoon/dataset_store/fma/fma_large"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/musicfm_student_50_first"))
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
    parser.add_argument("--train-last-n", type=int, default=0,
                        help="also train this many final Conformer layers; default trains frontend only")
    parser.add_argument("--time-strides", type=int, nargs=2, default=[1, 2], metavar=("FIRST", "SECOND"),
                        help="50 fps frontend time strides: 1 2 (default) or 2 1")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 6, 12])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.clip_seconds <= 0 or args.batch_size <= 0 or args.max_steps <= 0 or args.grad_clip <= 0:
        parser.error("clip-seconds, batch-size, max-steps and grad-clip must be positive")
    if not 0 <= args.train_last_n <= 12 or not all(0 <= i <= 12 for i in args.layers):
        parser.error("train-last-n must be 0..12 and layers must be 0..12")
    if tuple(args.time_strides) not in ((1, 2), (2, 1)):
        parser.error("time-strides must be 1 2 or 2 1")
    if not 0 < args.val_percent < 100 or args.val_tracks <= 0 or args.val_every <= 0:
        parser.error("validation settings must be positive and val-percent below 100")

    torch.manual_seed(args.seed)
    train_paths, val_paths = split_audio(args.audio_dir, args.val_percent, args.val_tracks)
    print(f"FMA: {len(train_paths)} training tracks, {len(val_paths)} validation tracks")

    train_loader = DataLoader(FMAClips(train_paths, args.clip_seconds, True),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))
    val_loader = DataLoader(FMAClips(val_paths, args.clip_seconds, False),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=args.device.startswith("cuda"))

    teacher = MusicFM25Hz(is_flash=False, stat_path=args.stat_path,
                          model_path=args.model_path).to(args.device).eval()
    teacher.requires_grad_(False)
    student = copy.deepcopy(teacher)
    adapt_student(student, args.train_last_n, args.time_strides)
    student_mel = MelSTFT(sample_rate=24000, n_fft=2048, hop_length=480,
                          n_mels=128, is_db=True).to(args.device).eval()
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training {sum(p.numel() for p in student.parameters() if p.requires_grad):,} student parameters")

    step = 0
    best = float("inf")
    if args.resume is not None:
        saved = load_adapter(student, args.resume)
        for key in ("audio_dir", "train_last_n", "time_strides", "layers", "clip_seconds", "model_path",
                    "stat_path", "val_percent", "val_tracks", "seed"):
            saved_value = (list(saved["student_time_strides"]) if key == "time_strides"
                           else saved["config"][key])
            if saved_value != getattr(args, key):
                parser.error(f"--{key.replace('_', '-')} must match the resumed checkpoint")
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["step"]
        best = saved.get("best_loss", saved["val_loss"])
        print(f"Resumed step {step} from {args.resume}", flush=True)

    baseline = validate(teacher, student, student_mel, val_loader, args.layers, args.device)
    print(f"step {step} validation loss {baseline:.5f}", flush=True)
    if args.resume is None:
        best = baseline
        save_adapter(args.out_dir / "best.pt", student, optimizer, args, step, baseline, best)
        save_adapter(args.out_dir / "last.pt", student, optimizer, args, step, baseline, best)
    while step < args.max_steps:
        for batch in train_loader:
            student.train()
            student.conformer.eval()  # deterministic frozen layers and targets
            wav = batch
            wav = wav.to(args.device, non_blocking=True)
            a, b = pair_mels(wav, teacher, student_mel)
            with torch.no_grad():
                target = hidden_states(teacher, a)
            prediction = hidden_states(student, b)
            loss = distill_loss(prediction, target, args.layers)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad),
                                           args.grad_clip)
            optimizer.step()
            step += 1
            if step == 1 or step % 10 == 0:
                print(f"step {step} training loss {loss.item():.5f}", flush=True)
            if step % args.val_every == 0 or step == args.max_steps:
                score = validate(teacher, student, student_mel, val_loader,
                                 args.layers, args.device)
                print(f"step {step} validation loss {score:.5f}", flush=True)
                improved = score < best
                best = min(best, score)
                save_adapter(args.out_dir / "last.pt", student, optimizer, args, step, score, best)
                if improved:
                    save_adapter(args.out_dir / "best.pt", student, optimizer, args, step, score, best)
            if step >= args.max_steps:
                return


if __name__ == "__main__":
    main()
