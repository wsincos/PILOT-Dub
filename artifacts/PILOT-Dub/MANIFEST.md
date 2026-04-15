# PILOT-Dub Artifacts Manifest

This directory tracks the released **v19 final generator** used by the cleaned PILOT-Dub project.

## Artifact Layout

| File | Type | Source | Purpose |
|---|---|---|---|
| `generator_epoch00.ckpt` | physical copy | `Glo-Var v19 epoch00` | final public v19 generator checkpoint |
| `generator_epoch01.ckpt` | physical copy | `Glo-Var v19 epoch01` | optional later v19 checkpoint for analysis |
| `avsync_scorer_best.pt` | historical physical copy | previous reranking phase | kept only as archived historical artifact |
| `avsync_scorer_last.pt` | historical physical copy | previous reranking phase | archived historical artifact |
| `avsync_scorer_resume_step=6000.pt` | historical physical copy | previous reranking phase | archived historical artifact |
| `avsync_scorer_wandb_id.txt` | historical physical copy | previous reranking phase | archived historical artifact |

## Final Project Semantics

The public project exposes:

```text
pilot-dub/final   # released default
pilot-dub/strong  # scaled-up strong config, checkpoint not bundled yet
```

Default runtime:

```text
generator config: pilot-dub/final
generator ckpt:   artifacts/PILOT-Dub/generator_epoch00.ckpt
```

## Notes

- `generator_epoch00.ckpt` is the current default final checkpoint.
- `generator_epoch01.ckpt` is retained as a nearby follow-up checkpoint for analysis, not as a second public variant.
- `pilot-dub/strong` is config-only for now; a released strong checkpoint has not been added to this artifact directory yet.
- The AVSync scorer files remain only for historical reference and are not part of the current public final method narrative.
