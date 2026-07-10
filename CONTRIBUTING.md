# Contributing

Thanks for your interest in improving this project. Contributions should keep the core framing intact: **inferential A/B testing** (cleaning, two-proportion tests, effect-size / logistic adjustment, power), not predictive ML or unrelated stacks.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- Bug reports (broken install, notebook execution errors, incorrect stats usage)
- Documentation fixes (README clarity, wrong metrics, broken commands)
- Analysis improvements (clearer power checks, better plots, sensitivity analyses) that stay within the statistical design
- Reproducibility hardening (`uv` / kernel / nbconvert workflow)

**Out of scope (please discuss in an issue first):**

- Turning the project into AutoML / LazyPredict / accuracy competitions
- Adding Ollama, fine-tuning, or RAG for its own sake
- Changing α, dropping rows, or re-specifying models after the fact to force significance

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python **3.13.13** (fallback 3.12.10 only if 3.13 is unavailable).

```bash
git clone https://github.com/pypi-ahmad/ab-test-analysis-marketing-campaign.git
cd ab-test-analysis-marketing-campaign

uv python install 3.13.13
uv venv --python 3.13.13
uv python pin 3.13.13
uv sync

uv run python -m ipykernel install --user --name ab-test-marketing-project
```

## Run the analysis

```bash
# Preferred: execute the notebook end-to-end
uv run jupyter nbconvert --to notebook --execute \
  notebooks/ab_test_marketing_analysis.ipynb \
  --output ab_test_marketing_analysis.ipynb \
  --ExecutePreprocessor.kernel_name=ab-test-marketing-project \
  --ExecutePreprocessor.timeout=600
```

Or edit the jupytext source, then convert:

```bash
uv run jupytext --to ipynb notebooks/ab_test_marketing_analysis.py
```

Data is downloaded from the public Udacity DAND mirror on first run and cached under `data/`.

## Smoke-check notebook outputs

```bash
uv run python - <<'PY'
import nbformat
from pathlib import Path
nb = nbformat.read(Path("notebooks/ab_test_marketing_analysis.ipynb"), as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
empty = [i for i, c in enumerate(code) if not c.get("outputs")]
errs = [
    (i, o.get("ename"), o.get("evalue"))
    for i, c in enumerate(code)
    for o in c.get("outputs", [])
    if o.get("output_type") == "error"
]
print(f"code_cells={len(code)} empty={len(empty)} errors={len(errs)}")
assert not empty and not errs, (empty, errs)
print("OK")
PY
```

## Pull request process

1. **Open an issue first** for non-trivial changes so scope stays clear.
2. Fork the repo and create a branch from `main`.
3. Keep diffs focused; do not commit `.venv`, secrets, or unrelated reformatting.
4. If you change analysis code, **re-execute the notebook** and update README numbers when results change.
5. Prefer reporting **actual metrics** over placeholder claims.
6. Open a PR with:
   - What changed and why
   - How you verified it (commands + outcome)
   - Any residual risks or limitations

## Coding / analysis guidelines

- Use `uv add` for new dependencies; keep `uv.lock` updated.
- Preserve the teaching structure of the notebook (why → code → interpretation).
- Do not invent predictive “best model” sections.
- State cleaning rules, α, and hypotheses explicitly.
- Leave null or borderline results honest; do not α-shop.

## Reporting security issues

Do not open a public issue for security-sensitive reports. Contact the repository owner ([@pypi-ahmad](https://github.com/pypi-ahmad)) privately via GitHub.

## License

By contributing, you agree that your contributions will be licensed under the project’s [MIT License](LICENSE).
