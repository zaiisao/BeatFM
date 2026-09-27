# BeatFM 재현

**BeatFM: Improving Beat Tracking with Pre-trained Music Foundation Model** (ICME 2025) 재현 코드입니다.
목표는 논문 Table I의 GTZAN 점수를 다시 얻는 것입니다.

| GTZAN (BeatFM, MusicFM) | F-measure | CMLt | AMLt |
|---|---|---|---|
| Beat | 89.1 | 80.6 | 93.5 |
| Downbeat | 79.6 | 74.4 | 88.7 |

**현재 상태:** 입력으로 **Beat This가 제공하는 spectrogram**(`<dataset>.npz`)을 받을 수 있게 했습니다(`--input spect`). 오디오 입력(`--input wav`)도 그대로 쓸 수 있습니다. fold 0 학습 결과 GTZAN beat F 88.0 / downbeat F 77.5로, 논문과 F는 1–2점, CMLt는 2–3점 차이입니다(아래 "결과").

---

## 구조

```
wav (B, samples) @ 24 kHz                  --input wav   : MusicFM이 직접 mel 계산
Beat This spect (B, T_bt, 128) @ 50 fps    --input spect : spect_convert.py로 MusicFM mel(100 fps)로 변환
  → MusicFMExtractor   frozen MusicFM, 층별 hidden state를 쌓음      → h  (B, N=13, F=1024, T)   Eq.(1)(2)
  → MSAM               temporal / frequency / channel attention      → h̃ (B, N, F, T)           Eq.(3)-(10)
  → classifier         (B, T, N·F) → Linear → ReLU → Linear(→2)      → beat, downbeat logits (B, T)
  → DBN (madmom)       beat / downbeat 시각
```
T는 25 fps (MusicFM 출력 해상도 그대로, 1 frame = 40 ms).

| 파일 | 내용 |
|---|---|
| `MusicFMExtractor.py` | frozen MusicFM. `forward(wav)` / `forward_mel(mel)`. `.train()`을 불러도 eval 모드 유지 |
| `spect_convert.py` | Beat This spectrogram → MusicFM 입력 mel 변환 (아래 "Beat This spectrogram 입력" 참고) |
| `mmnpz.py` | `.npz` memory-map 로더 (Beat This 코드, MIT) |
| `MSAM.py` | `MSConv`, `TemporalAggregation`, `FrequencyAggregation`, `channelAggregation`, `MSAM` |
| `model.py` | `BeatFM` = extractor → MSAM → classifier (`input_type="wav"` / `"spect"`) |
| `data.py` | 오디오 캐시 또는 Beat This `.npz` 읽기, 15 s clip / 5 s overlap, ±2 frame soft label, 분할 |
| `make_manifest.py` | 데이터셋별 라벨·fold와 오디오(`--input wav`) 또는 spectrogram(`--input spect`) 경로를 모아 manifest 생성 |
| `train.py` | Lightning 학습 + 곡 전체 test (DBN, mir_eval) |
| `scripts/` | 확인용 스크립트 (서버 절대경로 포함). `check_convert.py` / `check_align.py` 등 변환 검증 |
| `third_party/musicfm` | MusicFM 원본 (git submodule) |

---

## 설치

```bash
git clone --recursive https://github.com/Hwang-Tae-Gum/beatFM.git
cd beatFM

conda create -n musicfm python=3.10 -y
conda activate musicfm
# S3 서버 드라이버는 CUDA 12.4까지 지원 → cu124 빌드 필요
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers<5" einops librosa pytorch_lightning mir_eval tqdm setuptools cython
pip install git+https://github.com/CPJKU/madmom      # PyPI 0.16은 최신 python에서 import 실패

# MusicFM 가중치 (MSD)
wget -P third_party/musicfm/data https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json
wget -P third_party/musicfm/data https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt
```

> **transformers 5.x에서는 MusicFM이 깨집니다** (`KeyError: 'hidden_states'` — 5.x의 `Wav2Vec2ConformerEncoder`가 hidden states를 반환하지 않음). 반드시 4.x.

---

## 실행

### Beat This spectrogram 입력 (현재 사용)

```bash
# 1. manifest: Beat This .npz 기준 (Harmonix 포함 3,313곡)
python make_manifest.py --input spect --out manifest_spect.csv

# 2. 학습 + test (최종 설정 B: fold 0, lr 1e-4, 10 epoch, 모든 epoch 체크포인트 저장)
nohup python -u train.py --manifest manifest_spect.csv --input spect --fold 0 --gpu 0 \
    --lr 1e-4 --max-epochs 10 --patience 10 --save-all --out-dir runs_B > logs/B.log 2>&1 &

# 3. 저장된 체크포인트를 따로 평가 (--split val: epoch 선택용, test: 최종 점수)
python scripts/eval_ckpt.py --ckpt runs_B/fold0/checkpoints/epoch=2-step=4995.ckpt \
    --manifest manifest_spect.csv --split test --gpu 0
```
- spectrogram 위치: `make_manifest.py`의 `SPECT_ROOT` (`<dataset>.npz`, key `<name>/track`)
- 오디오 캐시가 필요 없습니다 (`--cache-dir` 불필요).

### 오디오 입력

```bash
# 1. manifest (dataset, name, audio, spect, annotation, fold, has_downbeats)
python make_manifest.py --out manifest.csv

# 2. smoke test
python train.py --manifest manifest.csv --cache-dir /path/to/cache \
    --limit-tracks 2 --batch-size 2 --max-epochs 1 --num-workers 0 --gpu 0

# 3. 본 학습 (fold k = 8-fold CV에서 test로 뺄 fold)
python train.py --manifest manifest.csv --cache-dir /path/to/cache --fold 0 --gpu 0
```
- 첫 실행 시 모든 오디오를 24 kHz mono float16 `.npy`로 캐시합니다 (약 15 GB).
- 끝나면 best 체크포인트(val_loss 기준)로 test를 돌리고 데이터셋별 F / CMLt / AMLt 표를 출력합니다.
- 체크포인트에는 frozen MusicFM(1.3 GB)을 빼고 저장합니다 (~80 MB). 로드 시 MusicFM 가중치 파일에서 다시 채웁니다.
- 로그: `runs/fold{k}/logs/` (CSV)

---

## 데이터

라벨과 8-fold 분할은 Beat This annotation set(`/disk1/jaehoon/dataset_store/beat_this_annotations`)을 사용합니다. 데이터셋 구성과 분할은 BeatFM 논문(Sec. IV-A) 방식입니다: Ballroom / Hainsworth / SMC는 8-fold 중 fold k를 test, GTZAN은 test only, 나머지 곡의 10%를 val.

| 데이터셋 | 📄 용도 | 곡 수: spect | 곡 수: 오디오 | 비고 |
|---|---|---|---|---|
| Beatles | train | 179 | 179 | 빈 annotation 1곡(Revolution 9) 제외 |
| RWC Popular | train | 100 | 100 | |
| **Harmonix** | train | **911** | 0 | 정렬된 오디오가 없어 오디오 입력으로는 사용 불가 |
| Ballroom | 8-fold | 685 | 672 | |
| Hainsworth | 8-fold | 222 | 222 | |
| SMC | 8-fold | 217 | 217 | downbeat 라벨 없음 → downbeat loss에서 제외 |
| GTZAN | test only | 999 | 993 | |
| **합계** | | **3,313** | 2,383 | |

fold 0 기준 (spect): train 1,956 / val 217 / test 1,140곡 (GTZAN 999 + Ballroom 86 + Hainsworth 28 + SMC 27).

---

## Beat This spectrogram 입력

두 모델의 입력 spectrogram은 설정이 다릅니다.

| | Beat This (저장된 값) | MusicFM (원래 입력) |
|---|---|---|
| 샘플레이트 / n_fft / hop | 22.05 kHz / 1024 / 441 (**50 fps**) | 24 kHz / 2048 / 240 (**100 fps**) |
| mel | slaney, 30–11000 Hz, **magnitude**를 합산 | htk, 0–12000 Hz, **power**를 합산 |
| 크기 | `log1p(1000 · x)` | `10·log10(x)` → MSD 통계로 정규화 |

`spect_convert.py`는 Beat This 단계를 거꾸로 풀고 MusicFM 단계를 다시 적용합니다.
```
bt_to_mag        log1p 역변환 (정확)
mel_bt_to_fm     band 폭으로 나눔 → 제곱 → MusicFM band 중심으로 보간 → × MusicFM band 폭 (band 안 평탄 가정)
power_to_musicfm 10·log10 + DB_OFFSET(34.49) → (x − 6.77) / 18.42
upsample_time    50 → 100 fps 선형 보간
```

**검증** (GTZAN, 같은 곡을 오디오로 만든 MusicFM mel과 비교, `scripts/check_convert.py`, `scripts/check_align.py`)
- mel 오차 2.77 dB. band별로 중간 대역 1.2–1.4 dB, 최저역 4.8 dB, 최고역 8.3 dB. Beat This에 30 Hz 미만, 11 kHz 이상 정보가 없어서입니다.
- MusicFM hidden state cosine: layer 0 **0.734**, layer 6 **0.787**, layer 12 **0.873**. 참고로 같은 오디오를 10 ms 밀었을 때 약 0.95입니다.
- 시간 정렬: 오디오 입력과 spect 입력의 feature를 ±3 frame 밀어 비교했을 때 lag 0에서 최대(대칭). frame 수도 일치(30 s → 750).
- 앞뒤 ±2 frame을 보는 선형 매핑을 데이터로 학습하면 cosine 0.93–0.96까지 올라갑니다(GTZAN 10곡 학습, 10곡 평가). 아직 적용하지 않았습니다.

### Experimental 50 fps MusicFM input

`--spect-fps 50` skips temporal interpolation of Beat This frames. The second
MusicFM convolution stage keeps its original time stride of 2. The unchanged
encoder therefore produces 12.5 fps features from 50 fps input; those features
are interpolated to 25 fps before MSAM and the classifier. The default
`--spect-fps 100` retains the original input interpolation path.

Run `python scripts/check_model_spect.py --spect-fps 50` or use the VS Code
"BeatFM: spectrogram forward (50 fps input)" configuration. A saved
spectrogram checkpoint can be evaluated with
`python scripts/eval_ckpt.py --ckpt PATH --manifest manifest_spect.csv --split val --spect-fps 50`
to compare the same learned weights without retraining.

Beat tracking was also evaluated with the same trained 100 fps checkpoint
(`runs_B/fold0/checkpoints/epoch=2-step=4995.ckpt`) on its original 217-track
validation split. The checkpoint and manifest are stored under
`/disk1/taegum/mnt/musicfm_bridge/`. Scores below are track-weighted means;
downbeat metrics exclude the 29 SMC tracks without downbeat labels.

| Inference input | Beat F | Beat CMLt | Downbeat F | Downbeat CMLt |
|---|---:|---:|---:|---:|
| Interpolated 100 fps (trained setting) | 88.1 | 81.6 | 86.3 | 80.4 |
| Direct 50 fps, interpolated encoder features (same weights) | 60.3 | 35.0 | 51.2 | 34.9 |

This measures an inference-time input-rate change with the original MusicFM
network. A model trained from the start with 50 fps input could perform
differently.

### Experimental 50 fps MusicFM student

`scripts/train_musicfm_student.py` trains a student from the released MusicFM-MSD
checkpoint using paired crops of FMA audio. The frozen teacher receives its
original 100 fps mel. The student receives a separately computed 50 fps mel;
the default frontend time strides are `(1, 2)`, so both networks emit 25 fps
hidden states. The initial training scope is the student frontend only, and the loss
matches hidden layers 0, 6, and 12. This is representation distillation, not
BeatFM training or the original BEST-RQ pretraining objective.

The downloaded `fma_large.zip` contains 30-second excerpts. It is smaller in
audio duration than the full-length FMA collection used in the MusicFM paper.
After extraction, a first run is:

```bash
conda activate musicfm
python scripts/train_musicfm_student.py \
  --audio-dir /disk1/jaehoon/dataset_store/fma/fma_large \
  --out-dir runs/musicfm_student_50_first \
  --max-steps 1000
```

The script reserves a deterministic track-level validation subset and writes
`best.pt` and `last.pt` containing the adapted frontend, any selected final
Conformer layers, and optimizer state. Pass `--resume runs/musicfm_student_50_first/last.pt`
to continue an interrupted run; the data shuffle starts a new order on resume.
It skips the tiny, undecodable MP3 placeholders present in FMA-large.
`load_adapter(student, path)` in the script
restores both the learned weights and the required 50 fps frontend stride.
It does not alter the original
MusicFM-MSD checkpoint. This first version does not use LoRA. Agreement on FMA
must still be checked on other music, and the Beat This spectrogram conversion
remains a separate input-domain mismatch.

Two FMA-large runs used 1,000 steps, 6-second crops, batch size 2, and the
same 64 held-out FMA tracks. Changing the first stride preserves the original
frontend's intermediate 50 fps rate; changing the second gives an intermediate
25 fps rate.

| Student time strides | Validation loss before training | Best validation loss | Best step | Checkpoints |
|---|---:|---:|---:|---|
| `(2, 1)` | 0.2948 | 0.0320 | 900 | `runs/musicfm_student_50/` |
| `(1, 2)` | 0.1272 | 0.0257 | 600 | `runs/musicfm_student_50_first/` |

These losses measure teacher–student representation agreement on FMA, not beat
tracking or generalization to Beat This.

### Direct fine-tuning versus teacher–student adaptation

`scripts/train_musicfm_50_selfsup.py` provides a direct fine-tuning baseline.
It starts from the same MusicFM-MSD checkpoint, changes the frontend time
strides to `(1, 2)`, and trains the same 26.2 million frontend parameters as
the distilled student. It uses 50 fps masked mel input and the original
MusicFM masked-token loss. Token targets come from the pretrained quantizer
applied to unmasked 100 fps mel, so both paths have 25 token positions per
second. This retains the original token vocabulary while changing the audio
input rate. Both runs used 1,000 steps, batch size 2, six-second FMA crops,
the same 64 held-out FMA tracks, and learning rate 1e-4.

```bash
python scripts/train_musicfm_50_selfsup.py --device cuda
python scripts/compare_musicfm_50.py --source fma --device cuda
python scripts/compare_musicfm_50.py --source gtzan --device cuda
```

| Evaluation data | 50 fps frontend | Masked-token loss ↓ | Token accuracy ↑ | Teacher layer-12 cosine ↑ |
|---|---|---:|---:|---:|
| FMA, 64 held-out tracks | Unadapted | 3.1414 | 0.3001 | 0.9313 |
| FMA, 64 held-out tracks | Distilled | 3.1615 | 0.3061 | 0.9858 |
| FMA, 64 held-out tracks | Direct fine-tuning | **2.7173** | **0.3594** | 0.8751 |
| GTZAN, 100 tracks | Unadapted | 3.1038 | 0.3127 | 0.9438 |
| GTZAN, 100 tracks | Distilled | 3.1898 | 0.2982 | **0.9892** |
| GTZAN, 100 tracks | Direct fine-tuning | **2.8466** | **0.3310** | 0.8909 |

Direct fine-tuning improves the task it trains on, while distillation better
preserves the original teacher features. These metrics do not establish which
features work better for downstream tasks. The GTZAN comparison uses separate
audio from the FMA training run; one unreadable file was replaced by the
dataset loader's fallback crop. Checkpoints are in
`runs/musicfm_50_selfsup_first/` and `runs/musicfm_student_50_first/`.

For a downstream check, `scripts/probe_musicfm_50_gtzan.py` extracts one
centered six-second crop per GTZAN track, averages the final hidden layer over
time, and fits the same standardized logistic regression for each frozen
model. Each genre uses 80 hash-selected tracks for training and 20 for test;
one unreadable training WAV is skipped (799 train, 200 test). No MusicFM
weights are updated by this probe.

| Frozen MusicFM features | Genre accuracy ↑ | Macro F1 ↑ |
|---|---:|---:|
| Original 100 fps teacher | 0.805 | 0.799 |
| Unadapted 50 fps | 0.805 | 0.801 |
| Distilled 50 fps | **0.815** | **0.813** |
| Direct fine-tuned 50 fps | 0.795 | 0.790 |

On this small probe, distillation has a modest advantage over direct
fine-tuning (2 percentage points accuracy). This is one crop and one split,
so it does not establish a general ranking or beat-tracking performance.

#### Continued training to 3,000 steps

Both runs were resumed from their 1,000-step `last.pt` checkpoints, including
optimizer state, and trained for 2,000 more steps with the same settings.
The data order restarts on resume. The original runs were preserved; extended
checkpoints and logs are in `runs/musicfm_student_50_first_3000/` and
`runs/musicfm_50_selfsup_first_3000/`. Each row below uses the checkpoint
with the lowest validation loss for its own training objective.

| Objective | Best step | Best FMA validation loss | GTZAN token accuracy | GTZAN teacher cosine, layer 12 | GTZAN genre accuracy |
|---|---:|---:|---:|---:|---:|
| Distillation, 1,000-step budget | 600 | 0.02570 | 0.2982 | 0.9892 | 0.815 |
| Distillation, 3,000-step budget | 2900 | **0.02010** | 0.2971 | **0.9914** | 0.805 |
| Direct masked-token, 1,000-step budget | 1000 | 2.71732 | 0.3310 | 0.8909 | 0.795 |
| Direct masked-token, 3,000-step budget | 1900 | **2.71681** | **0.3325** | **0.8928** | 0.790 |

Longer training improved each method's own validation objective slightly,
especially teacher agreement for distillation. Genre accuracy did not improve
for either method on this one split. The two validation losses use different
objectives and cannot be compared numerically. At batch size 2, 3,000 steps
sample only 6,000 crops from the 101,171-track FMA-large training pool.

---

## 논문 명시 사항 vs 직접 정한 값

**📄 논문에 명시 (Sec. III, IV) → 그대로 구현**
- frozen FM, 층별 출력 concat → `h ∈ R^{b×n×f×t}`
- MS-Conv: M = 4, dilation [1, 2, 4, 8], 병렬 → concat → MLP → sigmoid; frequency는 같은 구조, 파라미터 비공유
- channel: C-AvgPool → Conv Q/K/V → softmax(QKᵀ/√C)V → Conv → sigmoid
- `Attn = Attn_t · Attn_f · Attn_c`, `h̃ = h + Attn * h`
- BCE, beat·downbeat 동시 학습, 라벨 ±2 frame 확장 (weight 0.5, 0.25)
- 15 s clip, 5 s overlap, 같은 곡은 train/val 중 한쪽에만
- Adam, lr 3e-4, batch 16, val loss 20 epoch patience early stopping
- DBN 후처리, 곡 전체 test, mir_eval, 70 ms

**논문에 없어서 직접 정한 값 (전부 인자로 변경 가능)** — 검토 부탁드립니다

| 항목 | 현재 값 | 인자 / 위치 |
|---|---|---|
| MusicFM 체크포인트 | MSD | `MusicFMExtractor.py` |
| 사용 층 N | 13 (conv 출력 + conformer 12) — 논문은 "Encoder 1…n" | `--layers` |
| 출력 fps | 25 (업샘플 없음) → ±2 frame = ±80 ms | `data.py` |
| MS-Conv kernel / padding | 3 / 길이 유지 | `MSAM.py` |
| MS-Conv 채널 해석 | N(층)을 채널로, T 또는 F 방향 conv | `MSAM.py` |
| MS-Conv 뒤 MLP | 1×1 conv 2층, hidden = N, ReLU | `MSAM.py` |
| channel Q/K/V Conv | 층당 스칼라 → `Conv1d(1, C=16, 1)`, head 1개, 출력 `Conv1d(C, 1, 1)` | `MSAM.py` |
| classifier | N·F flatten → Linear(512) → ReLU → Linear(2) ("fully connected layers"). `weighted`: 층 softmax 가중합(1024) → 같은 MLP | `--classifier mlp/linear/weighted` |
| beat/downbeat 출력 | 독립 | `model.py` |
| val 비율 | 학습 곡의 10% (곡 단위) | `--val-ratio` |
| weight decay / scheduler / augmentation | 없음 | — |
| DBN | madmom `DBNDownBeatTrackingProcessor`, beats_per_bar [3,4], 55–215 bpm, transition_lambda 100 (Beat This 설정, madmom 기본값) | `train.py` |
| DBN 입력 frame rate | 모델 출력(25 fps)을 **50 fps로 선형 보간** 후 DBN (Beat This와 동일). 25 fps 그대로면 템포가 정수 frame으로 양자화돼 CMLt가 크게 떨어짐 | `dbn_fps` |
| test 입력 | 곡 전체 한 번에 | `--chunk-sec` |
| 평가 시 앞부분 제외 | 5 s (mir_eval 관례) | `train.py` |
| precision | fp32 | `--precision` |
| spect 변환 dB offset | 34.49 (GTZAN 10곡으로 맞춤, 곡별 편차 0.14 dB) | `spect_convert.py` |
| DBN 입력 하한 | 1e-5 (madmom log(0) 방지) | `train.py` |

---

## 결과 (fold 0, GTZAN)

모든 실험은 Beat This spectrogram 입력, 10 epoch, 모든 epoch 체크포인트 저장, val loss 최저 epoch으로 test, DBN 50 fps.

| 실행 | 설정 | best ep | beat F | CMLt | AMLt | downbeat F | CMLt | AMLt |
|---|---|---|---|---|---|---|---|---|
| A | 기본 (MLP, lr 3e-4) | 1 | 87.9 | 78.5 | 92.6 | 76.8 | 70.7 | 89.0 |
| **B** | **lr 1e-4 (최종)** | 2 | **88.0** | **78.9** | **92.4** | **77.5** | **71.8** | **89.4** |
| C | linear classifier (2.7만 파라미터) | 4 | 87.8 | 78.1 | 92.7 | 77.3 | 71.5 | 89.3 |
| D | 층 가중합 classifier (0.53M) | 4 | 87.4 | 77.5 | 92.2 | 76.7 | 70.5 | 89.2 |
| E | **오디오 입력** (Harmonix 없음, GTZAN 993곡) | 1 | 86.9 | 76.0 | 93.6 | 80.1 | 72.2 | 91.3 |
| 📄 논문 | BeatFM (MusicFM) | – | 89.1 | 80.6 | 93.5 | 79.6 | 74.4 | 88.7 |

- **DBN frame rate가 가장 큰 요인이었습니다.** 같은 모델(A, epoch 1)에서 DBN을 25 fps → 50 fps로 바꾸면 beat F 78.5 → 87.8, beat CMLt 56.5 → 78.5, downbeat CMLt 51.5 → 70.6.
- classifier 구조, lr, epoch은 점수를 1점 안쪽으로만 바꿉니다. val loss는 epoch 1–2 이후 오르지만 val 점수(F, CMLt)는 epoch 내내 거의 일정합니다.
- 층 가중합(D)에서 학습된 층 가중치는 9번째 conformer 층이 0.27로 가장 크고 8, 10번이 그다음입니다(초기값 1/13 = 0.077).
- DBN 설정(λ 50/100/200, 50/100 fps, BPM 범위)과 시간 보정(0/+10/+20 ms)을 val로 비교했지만 GTZAN CMLt는 최대 +0.6이었습니다. 남은 CMLt 차이는 후처리가 아니라 입력 feature나 평가 조건(GTZAN annotation 버전 등) 쪽으로 보입니다.
- E(오디오 입력)는 downbeat가 3점 높지만 학습 데이터에 Harmonix가 없어 변환 손실만의 효과로 볼 수는 없습니다.

---

## 확인된 것 / 남은 것

- ✅ 실제 오디오(GTZAN 30 s) → `(1, 13, 1024, 751)`, MusicFM 가중치 보존, 학습 파라미터 6.8 M / frozen 329 M
- ✅ Beat This spectrogram 입력: 변환 검증, 시간 정렬, 15 s clip → 375 frame
- ✅ GPU(RTX A6000) 학습 → val → 체크포인트 → test(DBN) 점수표까지 동작
- ✅ 긴 곡 chunk 추론 로직 (`--chunk-sec`)
- ✅ fold 0 학습, 설정 비교 (A–E), DBN frame rate 문제 해결
- ⏳ 나머지 fold (1–7)
- ⏳ Harmonix를 뺀 spectrogram 학습으로 변환 손실만 따로 측정
