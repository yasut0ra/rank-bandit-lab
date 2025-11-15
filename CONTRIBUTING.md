# Contributing to rank-bandit-lab

Thanks for your interest in improving **rank-bandit-lab**!  
このプロジェクトへ貢献していただきありがとうございます。以下のガイドラインに沿っていただけると、レビューとマージがスムーズになります。

## 1. Ground rules / 基本方針
- 重大な仕様変更や大きな実装追加は、まず Issue で提案し方向性をすり合わせてください。
- すべてのコードは型ヒント付きで記述し、既存のテスト＋新規テストを添えてください。
- Python 3.10+ かつ `src/` 配下のモジュールに合わせた [PEP 484](https://peps.python.org/pep-0484/) スタイルと [Ruff](https://docs.astral.sh/ruff/) に準拠したフォーマットを維持します。
- 新しい CLI フラグや API を追加した場合は README / ドキュメント / チュートリアルノートブックを更新してください。

## 2. Development workflow / 開発フロー
1. Fork & Clone the repository.
2. Create a topic branch from `main` (例: `feature/better-pbm`, `docs/guides`).
3. Set up the environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
   > `.[dev]` には `pytest`, `ruff`, `mypy` など開発向け依存を定義している想定です。`pip install -e .` のあとに必要パッケージを追加しても構いません。
4. Implement the change with clear commits.
5. Run quality checks locally (described below) before opening a PR.

## 3. Testing & quality / テストと静的解析
- **Unit tests**: `python -m pytest tests/`
- **Lint**: `ruff check src tests`
- **Type check**: `mypy src`
- 変更内容に合わせて既存テストの修正や追加を行ってください。再現手順のあるバグ修正 PR では、必ずテストで再現ケースをカバーしてください。

## 4. Pull Request checklist
- [ ] Issue / motivation is linked in the PR description.
- [ ] Tests & linters pass locally.
- [ ] Public API or CLI changes are documented (README / docs / notebooks).
- [ ] 新規ファイルにはライセンス表記が必要な場合、ヘッダーを追加済み。
- [ ] スクリーンショットやログなど、レビューに必要な補足情報を添付。

## 5. Coding conventions / コーディング規約
- 1 ファイル 100 行超の大改造は可能であれば複数コミットに分割してください。
- 変数名は `doc_ids`, `slate_size` のように既存コードに合わせて snake_case を使用します。
- コメントは必要最小限にとどめ、複雑なアルゴリズムや根拠がある場合のみ追加してください。
- CLI / Notebook で生成される成果物は `.gitignore` に追加し、リポジトリをクリーンな状態に保ってください。

## 6. Communication / コミュニケーション
- 提案やレビュー内容に疑問がある場合は遠慮なく質問してください。
- コミュニティ・コントリビューターは [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) に従って行動する必要があります。
- Issue, Discussion, PR すべてにおいて丁寧な言葉遣いと建設的なフィードバックを心がけてください。

Open-source contributions are welcome regardless of your experience level. 🙌  
皆さんの挑戦をお待ちしています！
