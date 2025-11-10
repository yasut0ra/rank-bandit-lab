# rank-bandit-lab

ランキングバンディットの学習・研究を目的にした、軽量なシミュレーションラボです。ドキュメントごとの誘引確率を設定し、ランキングポリシーを差し替えながら cascade クリックモデルでの挙動を観測できます。

## 主なコンポーネント
- `CascadeEnvironment` / `PositionBasedEnvironment` / `DependentClickEnvironment`: 異なるクリックモデルを切り替え可能なシミュレーション環境。
- `EpsilonGreedyRanking` / `ThompsonSamplingRanking`: クリックフィードバックのみを使う基本的なランキングバンディット方策。
- `BanditSimulator`: 方策を環境で反復実行し、CTR や各ドキュメントの露出/クリック回数を収集。
- CLI (`rank-bandit-lab`): サンプルシナリオや任意の `doc_id=確率` 指定でシミュレーションを一発実行。

## 使い方
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### サンプルシミュレーション
```bash
# カスケード (既定)
rank-bandit-lab --algo epsilon --epsilon 0.15 --steps 3000 --slate-size 3

# 位置バイアスを持つ Position-Based Model
rank-bandit-lab --model position --slate-size 3 \
  --position-bias 0.95 --position-bias 0.8 --position-bias 0.6

# Dependent Click Model (クリック後に満足確率で離脱)
rank-bandit-lab --model dependent --slate-size 3 \
  --doc-satisfaction doc-A=0.8 --doc-satisfaction doc-B=0.2
```

ドキュメント構成を変えたい場合は `--doc docA=0.45 --doc docB=0.2 ...` のように複数指定します。DCM の満足確率 (`--doc-satisfaction`) を指定しない場合は `--default-satisfaction` (既定 0.5) が使われます。

### API で扱う場合
```python
from random import Random
from rank_bandit_lab import (
    BanditSimulator,
    CascadeEnvironment,
    Document,
    EpsilonGreedyRanking,
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
```

`SimulationLog` では `total_reward` がクリック数の合計となり、PBM/DCM のように複数クリックが発生した場合にも集計されます。

## テスト
```
python -m unittest discover -s tests -v
```
