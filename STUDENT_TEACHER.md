# MusicFM 50 FPS student–teacher experiment

This experiment adapts the released MusicFM-MSD model to accept 50 FPS mel
input. For each FMA audio crop, the frozen teacher receives MusicFM's original
100 FPS mel, while the student receives 50 FPS mel. The student's default
frontend time strides are `(1, 2)`, so both models produce 25 FPS hidden
states. By default, only the student's convolutional frontend is trained.

The loss matches hidden layers 0, 6, and 12 using cosine distance plus
`0.05 * SmoothL1`. Training and validation use deterministic track-level FMA
splits. The focused implementation is `scripts/train_musicfm_student.py`.

## Requirements and local inputs

Use the project's `musicfm` Conda environment with PyTorch, torchaudio,
Transformers 4.x, and einops. MusicFM code is provided by the
`third_party/musicfm` submodule. The script expects the released MSD files at
`third_party/musicfm/data/msd_stats.json` and
`third_party/musicfm/data/pretrained_msd.pt`; obtain them from the MusicFM
release if they are not present locally.

FMA-large must already be extracted. The default path is
`/disk1/jaehoon/dataset_store/fma/fma_large`; use `--audio-dir` to override it.
The data and model weights are not committed to this repository.

## Train

```bash
python scripts/train_musicfm_student.py \
  --audio-dir /disk1/jaehoon/dataset_store/fma/fma_large \
  --out-dir runs/musicfm_student_50 \
  --max-steps 1000
```

The run writes `best.pt` and `last.pt` adapter checkpoints under `--out-dir`.
To resume, pass `--resume PATH_TO_LAST_PT` and retain the same model, data,
and split settings.

This branch contains the representation-distillation experiment. Direct
masked-token fine-tuning, BeatFM beat-classifier training, Beat This conversion,
and later hard-track sampling experiments are separate work.
