# CausualMemoryQA / PoST

Code for support-strategy prediction with posterior training targets,
strategy histories, and supervision diagnostics.

## Code map

- `esc/train_cls.py`: baseline strategy classifiers, soft-target training,
  strategy-history input and teacher-sampled histories.
- `esc/oof.py`: assembly of out-of-fold teacher logits.
- `esc/recognize.py`, `esc/diagnose.py`, `esc/panel.py`: supervision diagnostics.
- `esc/rollout.py`: evaluation with generated strategy histories.
- Other `esc/` and `src/` modules: alternative learning methods, controls,
  data preparation and metrics.
- Root `run_*.py` and `analyze_*.py`: additional experimental entry points.

## Environment and data

Use Python 3.10 or newer, a CUDA-capable PyTorch environment, and:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

The `esc/data.py` loader expects the
[EmoDynamiX-v2](https://github.com/cw-wan/EmoDynamiX-v2) data layout.
Obtain the upstream project and datasets separately, then set `EMODX` to its
root directory. Required files include:

- `data/esconv/ESConv.json` and `data/esconv/strategies.json`.
- `data/annomi/AnnoMI-clean.json` and `data/annomi/strategies.json`.

Preserve upstream dataset permissions. This repository does not redistribute
the dialogues.

## Baseline and posterior-target workflow

Run commands from the repository root. A baseline with strategy history is:

```bash
python -m esc.train_cls --dataset esconv --with_plan --seed 0 --out runs/esconv/baseline_s0
```

The following Bash example constructs a five-fold, same-input teacher and
trains a student from its posterior targets. It illustrates the workflow;
select the seed, model and hyperparameters required for your experiment.

```bash
for fold in 0 1 2 3 4; do
  python -m esc.train_cls --dataset esconv --with_plan --fold "$fold" --nfold 5 --seed 0 --out "runs/esconv/self_f$fold"
done
python -m esc.oof runs/esconv/self_f 5
python -m esc.train_cls --dataset esconv --with_plan --distill runs/esconv/self_foof.npz --seed 0 --out runs/esconv/post_s0
```

See `python -m esc.train_cls --help` for `--alpha`, `--T`,
`--hist_teacher`, model and baseline-control options. Teacher inputs should
match the intended supervision condition; response-aware diagnostic teachers
are distinct from same-input teachers.

## Release contents

This repository contains the project source code, training and evaluation
entry points, baseline implementations, and available tests. Experiment and
metric-analysis code is retained. Manuscripts, downloaded reference papers,
editorial notes, publication figures and LaTeX-table exporters are not included.

Datasets, model weights, local prediction caches, logs, and machine credentials
are not distributed with this release. Obtain datasets and model weights from
their original providers and follow their respective access and usage terms.
Configuration defaults in some experimental scripts refer to the original
Windows or Linux workspace; use the documented environment variables and
command-line options, and review path constants before a full run.

The release was checked for Python syntax and accidental inclusion of manuscript
files and embedded credentials. This publication check does not rerun GPU
training or certify that every experimental entry point is turnkey.

## Repository status

Public code repository: https://github.com/xkk9866/CausualMemoryQA

The public release is a code snapshot. Dataset and checkpoint availability is
separate from source-code availability.
