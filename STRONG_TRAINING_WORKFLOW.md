# PILOT-Dub Strong Training Workflow

This file records the exact operational steps for launching the `pilot-dub/strong` model.

## Strong Model Definition

Current strong recipe:

```text
v19 method unchanged
decoder: 8 -> 16
planner: 2 -> 4
```

Config entry:

```text
model: pilot-dub/strong
train config: v19_strong_real_time_acoustic_interface_formal
```

## Scripts

Bootstrap script:

```text
scripts/run_pilot_dub_v19_strong_bootstrap.sh
```

Resume script:

```text
scripts/run_pilot_dub_v19_strong_resume.sh
```

## Before Training

The strong scripts default to:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
trainer.devices=6
```

So before launching, make sure GPUs `0-5` are free.

## Step 1: Bootstrap

Purpose:

```text
start from the released final checkpoint
produce the first resume checkpoint at step=5
generate the wandb id
```

Command:

```bash
cd /data1/jinyu_wang/projects/PILOT-Dub
tmux new -s pilotdub_strong_bootstrap
bash scripts/run_pilot_dub_v19_strong_bootstrap.sh
```

What this script does:

- Initializes from:

```text
artifacts/PILOT-Dub/generator_epoch00.ckpt
```

- Writes step checkpoints to:

```text
/data1/jinyu_wang/projects/PILOT-Dub/resume_steps
```

- Writes wandb id to:

```text
/data1/jinyu_wang/projects/PILOT-Dub/outputs/formal/pilot_dub_v19_strong_run6/wandb_id.txt
```

## Step 2: Wait For The First Resume Checkpoint

Check:

```bash
ls -lh /data1/jinyu_wang/projects/PILOT-Dub/resume_steps
```

Wait until a checkpoint for step 5 appears, e.g.:

```text
resume_step=5.ckpt
```

Once that file exists, bootstrap is complete.

## Step 3: Stop Bootstrap

If you are inside the tmux session:

```bash
Ctrl+C
```

Or from outside:

```bash
tmux kill-session -t pilotdub_strong_bootstrap
```

## Step 4: Resume Formal Training

Command:

```bash
cd /data1/jinyu_wang/projects/PILOT-Dub
tmux new -s pilotdub_strong_train
bash scripts/run_pilot_dub_v19_strong_resume.sh
```

What this script does:

- Reads the wandb id from:

```text
outputs/formal/pilot_dub_v19_strong_run6/wandb_id.txt
```

- Uses the latest checkpoint from:

```text
resume_steps/
```

- Continues training with wandb resume enabled

## Important Paths

Active resume checkpoints:

```text
/data1/jinyu_wang/projects/PILOT-Dub/resume_steps
```

Bootstrap output:

```text
/data1/jinyu_wang/projects/PILOT-Dub/outputs/formal/pilot_dub_v19_strong_run6/bootstrap
```

Resume output:

```text
/data1/jinyu_wang/projects/PILOT-Dub/outputs/formal/pilot_dub_v19_strong_run6/resume
```

Wandb id file:

```text
/data1/jinyu_wang/projects/PILOT-Dub/outputs/formal/pilot_dub_v19_strong_run6/wandb_id.txt
```

## Minimal Checklist

1. Free GPUs `0-5`.
2. Run:

```bash
bash scripts/run_pilot_dub_v19_strong_bootstrap.sh
```

3. Wait until `resume_steps/` contains `resume_step=5.ckpt`.
4. Stop bootstrap.
5. Run:

```bash
bash scripts/run_pilot_dub_v19_strong_resume.sh
```

That is the full training workflow.
