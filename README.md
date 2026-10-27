# CausualMemoryQA / PoST

PoST trains support-strategy predictors with posterior targets and dialogue strategy histories. This repository contains the classifiers, teacher-target construction and evaluations on ESConv and AnnoMI. It also includes controls for teacher inputs, historical labels and sequential prediction.

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

## Working with teacher targets

Each teacher fold writes a `logits.npz` file in its output directory.
`esc.oof` combines the held-out predictions into a training-set target array;
the example above produces `runs/esconv/self_foof.npz`.
Pass that file to the student's `--distill` option.

Keep the dataset split and example ordering fixed between teacher and student.
Use a separate output directory for each seed and supervision condition.
The cross-fitted example above specifies one target-generation recipe; full-data
teachers and response-aware diagnostic teachers are separate experimental conditions.

## Data and configuration

Download benchmark datasets and pretrained models from their original providers
and follow their usage terms. Data, checkpoints and generated predictions are
stored locally and are not included in this repository.

Set the paths described above before running experiments. Some auxiliary scripts
use fixed local path defaults; adjust those paths for your installation.
The command examples assume execution from the repository root.
