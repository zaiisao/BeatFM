"""Compare 50 fps MusicFM adapters on teacher agreement and token prediction."""

import argparse
import copy
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_musicfm_50_selfsup import masked_token_loss
from train_musicfm_student import (
    FMAClips, MelSTFT, MusicFM25Hz, ROOT, hidden_states, load_adapter,
    pair_mels, set_50fps_stride, split_audio,
)


def select_gtzan(root, per_genre):
    groups = defaultdict(list)
    for path in root.glob("*.wav"):
        if re.fullmatch(r"[a-z]+\.\d{5}\.wav", path.name):
            groups[path.name.split(".")[0]].append(path)
    if len(groups) != 10:
        raise ValueError(f"Expected 10 GTZAN genres, found {len(groups)}")
    paths = []
    for genre in sorted(groups):
        candidates = sorted(groups[genre], key=lambda p: hashlib.sha1(p.name.encode()).digest())
        if len(candidates) < per_genre:
            raise ValueError(f"Only {len(candidates)} tracks available for {genre}")
        paths.extend(candidates[:per_genre])
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["fma", "gtzan"], required=True)
    parser.add_argument("--fma-dir", type=Path,
                        default=Path("/disk1/jaehoon/dataset_store/fma/fma_large"))
    parser.add_argument("--gtzan-dir", type=Path,
                        default=Path("/disk1/jaehoon/dataset_store/meter2800/GTZAN"))
    parser.add_argument("--gtzan-per-genre", type=int, default=10)
    parser.add_argument("--distilled", type=Path,
                        default=ROOT / "runs/musicfm_student_50_first/best.pt")
    parser.add_argument("--selfsup", type=Path,
                        default=ROOT / "runs/musicfm_50_selfsup_first/best.pt")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.source == "fma":
        _, paths = split_audio(args.fma_dir, 5, 64)
    else:
        paths = select_gtzan(args.gtzan_dir, args.gtzan_per_genre)
    loader = DataLoader(FMAClips(paths, 6, False), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))
    print(f"source={args.source} tracks={len(paths)}", flush=True)

    teacher = MusicFM25Hz(is_flash=False,
                          stat_path=ROOT / "third_party/musicfm/data/msd_stats.json",
                          model_path=ROOT / "third_party/musicfm/data/pretrained_msd.pt")
    base = copy.deepcopy(teacher)
    distilled = copy.deepcopy(teacher)
    selfsup = copy.deepcopy(teacher)
    set_50fps_stride(base, (1, 2))
    load_adapter(distilled, args.distilled)
    load_adapter(selfsup, args.selfsup)
    if any(m.conv.conv[0].conv1.stride != (2, 1) for m in (base, distilled, selfsup)):
        raise ValueError("All comparison models must use the [1, 2] time-stride layout")
    teacher = teacher.to(args.device).eval()
    students = {"unadapted": base.to(args.device).eval(),
                "distilled": distilled.to(args.device).eval(),
                "selfsup": selfsup.to(args.device).eval()}
    teacher.requires_grad_(False)
    for model in students.values():
        model.requires_grad_(False)
    mel50 = MelSTFT(sample_rate=24000, n_fft=2048, hop_length=480,
                    n_mels=128, is_db=True).to(args.device).eval()

    sums = {name: {"token_loss": 0.0, "token_accuracy": 0.0,
                   **{f"cosine_{layer}": 0.0 for layer in (0, 6, 12)}}
            for name in students}
    count = 0
    with torch.no_grad():
        for batch_index, wav in enumerate(loader):
            wav = wav.to(args.device, non_blocking=True)
            mel100, mel_50 = pair_mels(wav, teacher, mel50)
            reference = hidden_states(teacher, mel100)
            for name, model in students.items():
                hidden = hidden_states(model, mel_50)
                for layer in (0, 6, 12):
                    value = F.cosine_similarity(hidden[layer], reference[layer], dim=-1).mean()
                    sums[name][f"cosine_{layer}"] += value.item() * wav.shape[0]
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(args.seed + 100_000 + batch_index)
                    loss, accuracy = masked_token_loss(model, wav, mel50)
                sums[name]["token_loss"] += loss.item() * wav.shape[0]
                sums[name]["token_accuracy"] += accuracy.item() * wav.shape[0]
            count += wav.shape[0]
    for name, metrics in sums.items():
        print(name, " ".join(f"{key}={value / count:.5f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
