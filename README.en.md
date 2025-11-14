# Rank Bandit Lab

[日本語はこちら](README.md)

Rank Bandit Lab is a lightweight playground for studying ranking bandits. You can define attraction probabilities per document, plug in different ranking policies, and observe click behavior under a cascade-style user model. The project ships with command-line tools, tutorials, scenarios, and notebooks so you can move quickly between simulation and analysis.

## Components

- `CascadeEnvironment` / `PositionBasedEnvironment` / `DependentClickEnvironment`: user-behavior models (cascade, PBM, DCM).
- `EpsilonGreedyRanking` / `ThompsonSamplingRanking` / `UCB1Ranking` / `SoftmaxRanking`: baseline policies that only rely on click feedback.
- `BanditSimulator`: runs policies inside an environment and records CTR, seen/click counts, regret, etc.
- CLI (`rank-bandit-lab`): run simulations, log full interaction traces, and compare algorithms.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Example runs:

```bash
# Cascade model with epsilon-greedy
rank-bandit-lab --algo epsilon --epsilon 0.15 --steps 3000 --slate-size 3

# UCB1
rank-bandit-lab --algo ucb --ucb-confidence 1.5 --steps 3000 --slate-size 3

# Softmax exploration
rank-bandit-lab --algo softmax --softmax-temp 0.2 --steps 3000 --slate-size 3
```

## Scenario Presets

Use `--scenario <name>` to load predefined document sets, position biases, and satisfaction probabilities.

| Scenario | Use case | Characteristics |
| --- | --- | --- |
| `news_headlines` | General news portal | Strong headline + medium tail |
| `ecommerce_longtail` | Online store | Best-sellers plus clearance items |
| `video_streaming` | Streaming service | Emphasizes originals and genre variety |
| `education_catalog` | Online courses | Mix of beginner and advanced tracks |

Scenarios live under `src/rank_bandit_lab/scenarios/`. Add your own JSON and it automatically becomes available in the CLI.

### Notebooks / Docs

- `docs/TUTORIAL.md`: CLI + logging + sweep workflow
- `notebooks/bandit_walkthrough.ipynb`: basic API example
- `notebooks/scenario_gallery.ipynb`: compare presets
- `notebooks/sweep_comparison.ipynb`: analyze sweep outputs

## Visualization

Install matplotlib to save plots:

```bash
rank-bandit-lab --steps 5000 --slate-size 3 \
  --plot-learning learning.png \
  --plot-docs docs.png \
  --plot-regret regret.png
```

`--show-plot` opens figures interactively. APIs `plot_learning_curve` / `plot_doc_distribution` / `plot_regret_curve` are available for custom scripts.

## Logging & Replay

```bash
rank-bandit-lab --scenario news_headlines --algo thompson \
  --steps 3000 --log-json runs/thompson.json

rank-bandit-lab --load-json runs/thompson.json \
  --plot-learning replay.png
```

Compare logs:

```bash
rank-bandit-lab-compare runs/*.json --sort-by regret --plot-regret compare.png
```

Parameter sweeps:

```bash
rank-bandit-lab-sweep \
  --scenario video_streaming \
  --run eps05:algo=epsilon,epsilon=0.05 \
  --run ts:algo=thompson,alpha_prior=1.0,beta_prior=1.0 \
  --run ucb07:algo=ucb,ucb_confidence=0.7 \
  --steps 4000 --slate-size 4 \
  --output-dir sweep_logs --summary-json sweep_logs/summary.json \
  --plot-regret sweep_logs/regret.png
```

## API Quick Example

```python
from random import Random
from rank_bandit_lab import (
    BanditSimulator,
    CascadeEnvironment,
    Document,
    EpsilonGreedyRanking,
    plot_learning_curve,
    plot_regret_curve,
    write_log,
)

documents = [
    Document("x", 0.35),
    Document("y", 0.25),
    Document("z", 0.10),
]
env = CascadeEnvironment(documents, slate_size=2, rng=Random(0))
policy = EpsilonGreedyRanking([doc.doc_id for doc in documents], slate_size=2, epsilon=0.1)
log = BanditSimulator(env, policy).run(rounds=1000)
print(log.summary())
plot_learning_curve(log, output_path="learning.png")
plot_regret_curve(log, output_path="regret.png")
write_log("latest.json", log, metadata={"doc_ids": [doc.doc_id for doc in documents]})
```

## Development

```bash
ruff check .
PYTHONPATH=src python -m unittest discover -s tests -v
```

CI (GitHub Actions) runs Ruff, Mypy, and the unit suite on every push.
