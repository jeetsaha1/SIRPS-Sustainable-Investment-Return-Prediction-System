## Change summary

Describe the change and its owning module.

## Contract and protocol review

- [ ] No schema or protocol change
- [ ] Schema/protocol change documented and versioned
- [ ] Data provenance and licence notes are included
- [ ] Synthetic data is labelled, if used

## Validation

- [ ] `python -m ruff check src/ tests/`
- [ ] `python -m black --check src/ tests/`
- [ ] `python -m pytest tests/ -v`
- [ ] Relevant two-company sample included
- [ ] Reviewer approval obtained

## Leakage and reproducibility

- [ ] Point-in-time availability is preserved
- [ ] Random seeds and configuration are recorded
- [ ] No final holdout was used for development decisions