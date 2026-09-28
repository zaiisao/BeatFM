"""Train a 50 FPS MusicFM student to match the frozen 100 FPS teacher."""

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

# Fixed experiment recipe. Change these together only when defining a new run.
CLIP_SECONDS = 6
BATCH_SIZE = 2
NUM_WORKERS = 4
MAX_DECODE_ATTEMPTS = 8
VALIDATION_EVERY = 100
VALIDATION_PERCENT = 5
VALIDATION_TRACKS = 64
TIME_STRIDES = (1, 2)
DISTILL_LAYERS = (0, 6, 12)
LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
SEED = 0

DEFAULT_AUDIO_DIR = Path("/disk1/jaehoon/dataset_store/fma/fma_large")
DEFAULT_MODEL_PATH = ROOT / "third_party/musicfm/data/pretrained_msd.pt"
DEFAULT_STAT_PATH = ROOT / "third_party/musicfm/data/msd_stats.json"


class FMAClips(Dataset):
    """Load mono, 24 kHz crops; substitute another track if decoding fails."""

    def __init__(self, paths, random_crop):
        self.paths = paths
        self.random_crop = random_crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        target_samples = CLIP_SECONDS * 24000
        last_error = None
        # Try distant tracks so a cluster of corrupt FMA IDs does not exhaust retries.
        stride = max(1, len(self.paths) // MAX_DECODE_ATTEMPTS)
        for attempt in range(min(MAX_DECODE_ATTEMPTS, len(self.paths))):
            path = self.paths[(index + attempt * stride) % len(self.paths)]
            try:
                wav, sample_rate = torchaudio.load(str(path))
                wav = wav.mean(dim=0)
                if sample_rate != 24000:
                    wav = torchaudio.functional.resample(wav, sample_rate, 24000)
                if wav.numel() > target_samples:
                    if self.random_crop:
                        start = int(torch.randint(wav.numel() - target_samples + 1, ()).item())
                    else:
                        start = (wav.numel() - target_samples) // 2
                    wav = wav[start:start + target_samples]
                return F.pad(wav, (0, target_samples - wav.numel()))
            except (OSError, RuntimeError) as exc:
                last_error = exc
                print(f"Skipping unreadable FMA file {path}", file=sys.stderr)
        raise RuntimeError(f"No decodable FMA audio near {self.paths[index]}") from last_error


def split_audio(audio_dir):
    """Use a stable 5% track split and the same 64 validation tracks each run."""
    all_paths = sorted(audio_dir.rglob("*.mp3"))
    # FMA-large contains tiny placeholder files that the audio decoder cannot read.
    paths = [path for path in all_paths if path.stat().st_size >= 16_384]
    if not paths:
        raise FileNotFoundError(f"No usable MP3 files found under {audio_dir}")
    print(f"Skipped {len(all_paths) - len(paths)} tiny MP3 files", flush=True)

    train, validation = [], []
    for path in paths:
        digest = int(hashlib.sha1(path.stem.encode()).hexdigest(), 16)
        (validation if digest % 100 < VALIDATION_PERCENT else train).append(path)
    validation.sort(key=lambda path: hashlib.sha1(path.stem.encode()).digest())
    validation = validation[:VALIDATION_TRACKS]
    if not train or not validation:
        raise ValueError("FMA train/validation split is empty")
    return train, validation


def set_time_strides(model, strides):
    for stage, time_stride in zip(model.conv.conv, strides):
        stage.conv1.stride = (2, time_stride)
        stage.conv3.stride = (2, time_stride)


def prepare_student(student):
    student.requires_grad_(False)
    set_time_strides(student, TIME_STRIDES)
    student.conv.requires_grad_(True)


def hidden_states(model, mel):
    return model.conformer(model.conv(mel), output_hidden_states=True).hidden_states


def paired_mels(wav, teacher, student_mel):
    with torch.no_grad():
        teacher_mel = teacher.preprocessor_melspec_2048(wav)[..., :-1]
        student_mel = student_mel(wav)[..., :-1]
        mean = teacher.stat["melspec_2048_mean"]
        std = teacher.stat["melspec_2048_std"]
        return (teacher_mel - mean) / std, (student_mel - mean) / std


def distillation_loss(student_states, teacher_states):
    losses = []
    for layer in DISTILL_LAYERS:
        student_state = student_states[layer].float()
        teacher_state = teacher_states[layer].float()
        if student_state.shape != teacher_state.shape:
            raise ValueError(f"Layer {layer} shapes differ: {student_state.shape}, {teacher_state.shape}")
        cosine_loss = 1 - F.cosine_similarity(student_state, teacher_state, dim=-1).mean()
        losses.append(cosine_loss + 0.05 * F.smooth_l1_loss(student_state, teacher_state))
    return torch.stack(losses).mean()


@torch.no_grad()
def validation_loss(teacher, student, student_mel, loader, device):
    student.eval()
    total_loss, total_clips = 0.0, 0
    for wav in loader:
        wav = wav.to(device, non_blocking=True)
        teacher_mel, student_input = paired_mels(wav, teacher, student_mel)
        loss = distillation_loss(hidden_states(student, student_input),
                                 hidden_states(teacher, teacher_mel))
        total_loss += loss.item() * wav.shape[0]
        total_clips += wav.shape[0]
    return total_loss / total_clips


def save_adapter(path, student, optimizer, step, loss, best_loss, config):
    # Keep this format compatible with the BeatFM adapter loader.
    checkpoint = {
        "format_version": 1,
        "student_mel_hop": 480,
        "student_time_strides": TIME_STRIDES,
        "step": step,
        "val_loss": loss,
        "best_loss": best_loss,
        "config": {**config, "train_last_n": 0},
        "frontend": student.conv.state_dict(),
        "last_layers": [],
        "optimizer": optimizer.state_dict(),
    }
    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_adapter(student, path):
    """Restore the student frontend from a saved run."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1 or checkpoint.get("student_mel_hop") != 480:
        raise ValueError(f"Unsupported student checkpoint: {path}")
    strides = tuple(checkpoint["student_time_strides"])
    if strides not in ((1, 2), (2, 1)):
        raise ValueError(f"Unsupported student strides: {strides}")
    if checkpoint.get("last_layers"):
        raise ValueError("This script supports frontend-only student checkpoints")
    set_time_strides(student, strides)
    student.conv.load_state_dict(checkpoint["frontend"])
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stat-path", type=Path, default=DEFAULT_STAT_PATH)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--resume", type=Path, help="adapter checkpoint to continue")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = (args.resume.parent if args.resume
                        else Path("runs/musicfm_student_50_audit"))
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.resume and args.resume.resolve().parent != args.out_dir.resolve():
        parser.error("--resume checkpoint must be inside --out-dir")
    existing = [name for name in ("best.pt", "last.pt") if (args.out_dir / name).exists()]
    if not args.resume and existing:
        parser.error(f"{args.out_dir} already contains {', '.join(existing)}; choose a new --out-dir or resume")

    torch.manual_seed(SEED)
    train_paths, validation_paths = split_audio(args.audio_dir)
    print(f"FMA: {len(train_paths)} training tracks, {len(validation_paths)} validation tracks")

    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(
        FMAClips(train_paths, random_crop=True), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        FMAClips(validation_paths, random_crop=False), batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )

    teacher = MusicFM25Hz(is_flash=False, stat_path=args.stat_path,
                          model_path=args.model_path).to(args.device).eval()
    teacher.requires_grad_(False)
    student = copy.deepcopy(teacher)
    prepare_student(student)
    student_mel = MelSTFT(sample_rate=24000, n_fft=2048, hop_length=480,
                          n_mels=128, is_db=True).to(args.device).eval()
    optimizer = torch.optim.AdamW(student.conv.parameters(), lr=LEARNING_RATE)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "audio_dir": str(args.audio_dir),
        "model_path": str(args.model_path),
        "stat_path": str(args.stat_path),
        "clip_seconds": CLIP_SECONDS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "val_every": VALIDATION_EVERY,
        "layers": DISTILL_LAYERS,
        "time_strides": TIME_STRIDES,
        "lr": LEARNING_RATE,
        "grad_clip": GRAD_CLIP,
        "val_percent": VALIDATION_PERCENT,
        "val_tracks": VALIDATION_TRACKS,
        "seed": SEED,
        "max_steps": args.max_steps,
        "device": args.device,
    }
    step, best_loss = 0, float("inf")
    if args.resume:
        checkpoint = load_adapter(student, args.resume)
        saved_config = checkpoint["config"]
        if saved_config.get("train_last_n", 0) != 0:
            raise ValueError("Cannot resume a checkpoint that trained Conformer layers")
        expected = {
            "audio_dir": args.audio_dir,
            "model_path": args.model_path,
            "stat_path": args.stat_path,
            "clip_seconds": CLIP_SECONDS,
            "batch_size": BATCH_SIZE,
            "time_strides": TIME_STRIDES,
            "layers": DISTILL_LAYERS,
            "lr": LEARNING_RATE,
            "val_percent": VALIDATION_PERCENT,
            "val_tracks": VALIDATION_TRACKS,
            "seed": SEED,
        }
        for name, current in expected.items():
            previous = saved_config.get(name, current)
            if name.endswith("_dir") or name.endswith("_path"):
                previous, current = Path(previous).resolve(), Path(current).resolve()
            elif name in ("time_strides", "layers"):
                previous, current = tuple(previous), tuple(current)
            if previous != current:
                raise ValueError(f"Cannot resume: {name} differs from the saved run")
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        best_loss = checkpoint["best_loss"]
        print(f"Resuming at step {step} from {args.resume}")

    current_loss = validation_loss(teacher, student, student_mel, validation_loader, args.device)
    print(f"step {step} validation loss {current_loss:.5f}")
    if step == 0:
        best_loss = current_loss
        save_adapter(args.out_dir / "best.pt", student, optimizer, step,
                     current_loss, best_loss, config)
        save_adapter(args.out_dir / "last.pt", student, optimizer, step,
                     current_loss, best_loss, config)

    while step < args.max_steps:
        for wav in train_loader:
            student.train()
            student.conformer.eval()
            wav = wav.to(args.device, non_blocking=True)
            teacher_mel, student_input = paired_mels(wav, teacher, student_mel)
            with torch.no_grad():
                teacher_states = hidden_states(teacher, teacher_mel)
            student_states = hidden_states(student, student_input)
            loss = distillation_loss(student_states, teacher_states)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.conv.parameters(), GRAD_CLIP)
            optimizer.step()
            step += 1

            if step == 1 or step % 10 == 0:
                print(f"step {step} training loss {loss.item():.5f}", flush=True)
            if step % VALIDATION_EVERY == 0 or step == args.max_steps:
                current_loss = validation_loss(
                    teacher, student, student_mel, validation_loader, args.device
                )
                print(f"step {step} validation loss {current_loss:.5f}", flush=True)
                improved = current_loss < best_loss
                best_loss = min(best_loss, current_loss)
                save_adapter(args.out_dir / "last.pt", student, optimizer, step,
                             current_loss, best_loss, config)
                if improved:
                    save_adapter(args.out_dir / "best.pt", student, optimizer, step,
                                 current_loss, best_loss, config)
            if step == args.max_steps:
                return


if __name__ == "__main__":
    main()
