# Rank Bandit Lab チュートリアル

このガイドでは CLI・ログ・可視化・スイープ機能を一通り触りつつ、Jupyter からの API 利用方法も紹介します。

## 1. 事前準備

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]  # dev extras がなければ pip install -e . でOK
```

可視化やノートブックを使う場合は追加で `uv pip install matplotlib jupyter` を実行してください。

## 2. CLI での実験

### シナリオ指定

```bash
rank-bandit-lab --scenario news_headlines --algo softmax \
  --softmax-temp 0.2 --steps 4000 --slate-size 4 \
  --plot-learning runs/news-learning.png \
  --plot-regret runs/news-regret.png \
  --log-json runs/news-softmax.json
```

### ログ再生と比較

```bash
rank-bandit-lab --scenario news_headlines --algo ucb \
  --ucb-confidence 0.7 --steps 4000 \
  --log-json runs/news-ucb.json

rank-bandit-lab-compare runs/news-*.json --sort-by regret \
  --plot-regret runs/news-compare.png
```

### パラメータスイープ

```bash
rank-bandit-lab-sweep \
  --scenario ecommerce_longtail \
  --run eps05:algo=epsilon,epsilon=0.05 \
  --run eps15:algo=epsilon,epsilon=0.15 \
  --run ts:algo=thompson,alpha_prior=1.0,beta_prior=1.0 \
  --run ucb07:algo=ucb,ucb_confidence=0.7 \
  --steps 5000 --slate-size 4 \
  --output-dir sweep_logs --summary-json sweep_logs/summary.json \
  --plot-regret sweep_logs/regret.png
```

## 3. Jupyter / API 例

`notebooks/bandit_walkthrough.ipynb` にセルを用意しているので `jupyter notebook` で開いてください。内容は以下のとおりです。

1. `scenario_loader` からプリセットを読み込む
2. `CascadeEnvironment` と任意のポリシーを構築して `BanditSimulator` を実行
3. `plot_learning_curve` / `plot_regret_curve` で結果を可視化
4. `write_log` で JSON 保存 → `compare` / `sweep` と組み合わせ

## 4. シナリオギャラリー / スイープ比較

現在読み込めるシナリオは `rank_bandit_lab.scenario_loader.list_scenarios()` で確認できます。

```bash
python - <<'PY'
from rank_bandit_lab import scenario_loader
print(scenario_loader.list_scenarios())
PY
```

`notebooks/scenario_gallery.ipynb` では複数シナリオのクリック分布・リグレットを比較する例を掲載しています。ご自身の JSON を `src/rank_bandit_lab/scenarios/` に追加すると自動的に CLI からも選択できるようになります。

`notebooks/sweep_comparison.ipynb` では `rank-bandit-lab-sweep` で生成したログを Notebook 上で読み込み、学習曲線・リグレット・CTR を比較する例を載せています。CLI との併用で解析がしやすくなるので、応用編として活用してください。

## 5. Tips

- ログはすべて JSON なので、お好みのツールで解析したり、他チームと共有する際に便利です。
- シナリオ JSON を追加するだけで、資料に合わせた実験条件を簡単に配布できます。
- `rank-bandit-lab-sweep` の結果はそのまま `rank-bandit-lab-compare` に渡せるようになっています。
