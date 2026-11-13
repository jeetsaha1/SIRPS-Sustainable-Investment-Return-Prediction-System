# SIRPS Methodology

This document records the implementation decisions that connect the research
protocol to the software. The final implementation blueprint is the governing
scope; this file explains the controls that make its outputs reproducible.

## Environment

SIRPS uses Python 3.12, pinned dependencies from `requirements.txt`, and a
`src` package layout. The project is tested with pytest, Ruff, and Black in CI.

## Data flow

Raw source snapshots are read-only and versioned. Cleaning writes to `data/interim`,
point-in-time joins and targets write to `data/processed`, and dashboard-ready
files write to `data/serving`. Raw vendor data must not be committed to GitHub.

Every shared table must conform to the contracts in `docs/data_dictionary.md`
and the validators in `src/sirps/common/schemas.py`.

## Reproducibility

Synthetic pilot data uses the fixed seed in `config/settings.yaml` and is labelled
synthetic. Every model run records its run ID, Git commit, configuration hash,
data version, seed, experiment, and model in `results/runs.csv`.

## Point-in-time discipline

Financial and ESG facts are usable only after their recorded availability date
plus one trading day. First-reported facts are retained for historical modelling;
restatements are stored separately. No feature may use information unavailable at
the prediction date.

## Validation gates

Before a module is merged, it must have schema tests, provenance, leakage tests
where applicable, and a two-company sample. The final holdout is run once only
after protocol freeze and final model selection.
