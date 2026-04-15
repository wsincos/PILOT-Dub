# PILOT-Dub

PILOT-Dub is the cleaned paper-version project for the **v19 Real-Time Acoustic Interface** visual TTS / video dubbing framework.

This repository exposes:

```text
pilot-dub/final   # released default v19 generator
pilot-dub/strong  # scaled-up v19 strong config (decoder16 + planner4)
```

The final system thesis is:

```text
text + target video + reference audio
    -> planner hidden
    -> real-time acoustic interface prediction
    -> autoregressive dubbing generation
```

## Project Layout

```text
configs/
  evaluate/                    # LRS3 evaluation configs
  inference/                   # example inference config
  model/pilot-dub/final.yaml   # released default model config
  model/pilot-dub/strong.yaml  # scaled-up strong model config
  model/glo-var/               # minimal legacy inheritance chain required by v19
src/
  data/
  modeling/
  lightning/
scripts/
  evaluate_npy.py
  inference.py
  run_lrs3_metrics.sh
  run_lrs3_metrics_mini.sh
artifacts/
  PILOT-Dub/                   # physical final v19 checkpoint
  pretrained_models/           # physical EnCodec / AV-HuBERT files
  vocab/phn2num.txt
data/dataset -> shared LRS3 dataset symlink
reports/                       # method notes and historical migration / analysis docs
```

## Public Model Configs

Released default generator config:

```text
pilot-dub/final
```

Compatible released checkpoint:

```text
artifacts/PILOT-Dub/generator_epoch00.ckpt
```

Scaled-up strong config:

```text
pilot-dub/strong
```

Current strong recipe:

```text
v19 method unchanged
decoder: 8 -> 16
planner: 2 -> 4
total params: 1.048B
```

The strong config is intended for training from the released `final` checkpoint. A released strong checkpoint is not bundled yet.

## Environment

Use the existing runtime environment:

```bash
conda activate vcdub
pip install -e .
```

Metrics are evaluated through the shared metrics project:

```text
/data1/jinyu_wang/projects/metrics
```

## Run Full LRS3 Evaluation

```bash
bash scripts/run_lrs3_metrics.sh \
  pilot-dub/final \
  pilotdub-v19-epoch00-full \
  /data1/jinyu_wang/projects/PILOT-Dub/artifacts/PILOT-Dub/generator_epoch00.ckpt \
  0
```

## Run Mini50 Evaluation

```bash
bash scripts/run_lrs3_metrics_mini.sh \
  mini50 \
  pilot-dub/final \
  pilotdub-v19-epoch00-mini50 \
  /data1/jinyu_wang/projects/PILOT-Dub/artifacts/PILOT-Dub/generator_epoch00.ckpt \
  0
```

## Run Example Inference

```bash
conda activate vcdub
python scripts/inference.py --config-name inference/inference
```

## Notes

- `pilot-dub/final` remains the default released runtime path.
- `pilot-dub/strong` is the scaled-up strong training path for the same `v19` method.
- The legacy `configs/model/glo-var/` files are retained only as a checkpoint-compatible inheritance chain required by `v19`.
