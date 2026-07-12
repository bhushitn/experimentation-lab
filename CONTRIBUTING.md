# Contributing

Corrections beat features here. If a statistical claim, a simulation, or an
explainer number is wrong, that is the most valuable issue you can open.

## Ground rules

- Every statistical claim must either follow from a simulation in this
  repository or carry a citation to a verifiable source. If you cannot
  verify it, do not add it.
- Every figure and number must regenerate from committed code:
  `PYTHONPATH=src python scripts/regenerate_figures.py`. CI compares the
  regenerated headline numbers to the committed ones and fails on drift, so
  if you change a simulation, regenerate and commit its outputs together.
- Notebooks are generated, not edited:
  `PYTHONPATH=src python scripts/build_notebooks.py`.
- Before opening a PR: `ruff check src tests scripts eval`, `mypy src`, and
  `pytest tests/ -q` must pass.

## Adding a pathology

New pathologies are welcome if they follow the house pattern: a module in
`src/lab/pathologies/` exposing a `simulate_*` function, a test in
`tests/integration/test_pathologies.py` that verifies the naive analysis's
error and the corrected method's recovery against constructed ground truth,
and (optionally) a figure and explainer. Sample ratio mismatch is the most
wanted addition; see DECISIONS.md, ADR-1.
