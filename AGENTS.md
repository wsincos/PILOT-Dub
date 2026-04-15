# Repository Guidelines

## Scope

This is the cleaned paper-version PILOT-Dub project. Keep it focused on the released **v19 real-time acoustic interface** generator and its direct scale-up strong variant. Do not reintroduce unrelated exploratory branches unless explicitly needed for a paper ablation.

## Environment

Use local GPU access and the existing environment:

```bash
conda activate vcdub
pip install -e .
```

Available GPUs are local. Evaluation scripts accept a GPU id argument.

## Data and Artifacts

- Dataset files are symlinked under `data/dataset`.
- Model artifacts are physical copies under `artifacts/PILOT-Dub`.
- Required pretrained files are physical copies under `artifacts/pretrained_models`.

Do not modify the original exploratory workspace from this project.

## Final Method

The paper path is now:

```text
text + target video + reference audio
    -> planner hidden
    -> real-time acoustic interface prediction
    -> autoregressive dubbing generation
```

Public generator configs:

```text
released default:
  model config: pilot-dub/final
  checkpoint: artifacts/PILOT-Dub/generator_epoch00.ckpt

scaled-up strong:
  model config: pilot-dub/strong
  checkpoint: not bundled yet
  scale change: decoder16 + planner4
```

## Code Style

Follow the existing Python style. Prefer small, explicit scripts over reintroducing a large experiment framework. Keep paths project-root based where possible.

## Validation

Minimum useful validation:

```bash
bash scripts/run_lrs3_metrics_mini.sh mini50 pilot-dub/final ...
```

Strong config can be evaluated the same way once a strong checkpoint exists:

```bash
bash scripts/run_lrs3_metrics_mini.sh mini50 pilot-dub/strong ...
```

Full paper validation:

```bash
bash scripts/run_lrs3_metrics.sh pilot-dub/final ...
```

Report system1/system2 separately and include WER, LSE-D, LSE-C, UTMOS, speaker similarity, zero-rate, `>25`, and `>50`.
