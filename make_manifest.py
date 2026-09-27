"""Build the manifest CSV (dataset, name, audio, spect, annotation, fold, has_downbeats).

Annotations and 8-fold splits come from the Beat This annotation set.
  --input wav   : audio paths resolved per dataset (edit AUDIO below); tracks without audio are skipped
  --input spect : Beat This spectrograms (SPECT_ROOT/<dataset>.npz, key <name>/track); no audio needed
"""

import argparse
import csv
import json
from pathlib import Path

ANN_ROOT = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
LABELED = Path("/home/taegum/mnt/labeled_data")
SPECT_ROOT = Path("/disk4/shared/beat_this/data/audio/spectrograms")


AUDIO = {
    "ballroom": LABELED / "ballroom/data",
    "beatles": LABELED / "beatles/data",
    "hainsworth": LABELED / "hains/data",
    "gtzan": LABELED / "gtzan/data",
    "rwc_popular": LABELED / "rwc_popular/data",
    "smc": Path("/disk1/taegum/mnt/SMC_MIREX/SMC_MIREX/SMC_MIREX_Audio"),
    "harmonix": None,                     # TODO: aligned Harmonix audio (<name>.wav)
}
RWC_CD_SIZES = (16, 16, 16, 16, 16, 10, 10)   # RWC-P CD1..CD7 -> RM-P001..RM-P100


def audio_path(dataset, name):
    root = AUDIO[dataset]
    if dataset in ("ballroom", "beatles"):
        return root / f"{name[len(dataset) + 1:]}.wav"           # ballroom_X -> X.wav
    if dataset == "hainsworth":
        return root / f"{name.split('_')[1]}.wav"                 # hainsworth_001 -> 001.wav
    if dataset == "smc":
        return root / f"SMC_{name.split('_')[1]}.wav"             # smc_001 -> SMC_001.wav
    if dataset == "gtzan":
        _, genre, idx = name.split("_")                           # gtzan_blues_00000
        hits = list(root.glob(f"*_{genre}.{idx}.wav"))            # 0001_blues.00000.wav
        return hits[0] if hits else root / f"missing_{name}.wav"
    if dataset == "rwc_popular":
        cd, track = name.split("_")[2:]                           # rwc_popular_CD1_01
        idx = sum(RWC_CD_SIZES[:int(cd[2:]) - 1]) + int(track)
        hits = list(root.rglob(f"RM-P{idx:03d}.wav"))
        return hits[0] if hits else root / f"RM-P{idx:03d}.wav"
    if dataset == "harmonix":
        return root / f"{name}.wav"
    raise ValueError(dataset)


def read_folds(ann_dir):
    split = ann_dir / "8-folds.split"
    if not split.exists():
        return {}
    return {name: int(fold) for name, fold in (line.split() for line in split.read_text().splitlines() if line.strip())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="manifest.csv")
    p.add_argument("--datasets", nargs="*", default=list(AUDIO))
    p.add_argument("--input", choices=["wav", "spect"], default="wav")
    args = p.parse_args()

    from data import CV_DATASETS
    from mmnpz import MemmappedNpzFile

    rows, missing = [], {}
    for dataset in args.datasets:
        if args.input == "wav" and AUDIO[dataset] is None:
            print(f"[skip] {dataset}: audio path not set")
            continue
        ann_dir = ANN_ROOT / ("rwc" if dataset == "rwc_popular" else dataset)
        spect = SPECT_ROOT / f"{ann_dir.name}.npz"
        spect_keys = set(MemmappedNpzFile(spect).files) if args.input == "spect" else set()
        has_db = json.loads((ann_dir / "info.json").read_text()).get("has_downbeats", True)
        folds = read_folds(ann_dir)
        beat_files = sorted((ann_dir / "annotations/beats").glob("*.beats"))
        if dataset == "rwc_popular":
            beat_files = [f for f in beat_files if f.stem.startswith("rwc_popular_")]
        n_ok = 0
        for ann in beat_files:
            name = ann.stem
            if ann.stat().st_size == 0:                               # no beats annotated (beatles Revolution 9)
                continue
            audio = audio_path(dataset, name) if AUDIO[dataset] is not None else None
            found = f"{name}/track" in spect_keys if args.input == "spect" else audio.exists()
            if not found:
                missing.setdefault(dataset, []).append(name)
                continue
            fold = folds.get(name, -1) if dataset in CV_DATASETS else -1
            rows.append({"dataset": dataset, "name": name, "audio": str(audio or ""), "spect": str(spect),
                         "annotation": str(ann), "fold": fold, "has_downbeats": int(has_db)})
            n_ok += 1
        print(f"{dataset:12s} {n_ok:4d} tracks  (missing {args.input}: {len(missing.get(dataset, []))})")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "name", "audio", "spect", "annotation", "fold", "has_downbeats"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    for dataset, names in missing.items():
        print(f"  missing {dataset}: {names[:5]}{' ...' if len(names) > 5 else ''}")


if __name__ == "__main__":
    main()
