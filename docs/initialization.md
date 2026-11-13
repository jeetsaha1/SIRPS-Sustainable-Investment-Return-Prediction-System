# M1 Initialization Record

## Completed foundation

- Python 3.12 environment and pinned dependencies.
- `src` package layout, schema validator, feature registry, and CI checks.
- Synthetic two-company pilot company master and deterministic price generator.
- Trading-calendar primitive with explicit-date semantics.
- Universe eligibility rules, source-register template, and pilot configuration.
- Run registry header and pull-request review checklist.
- Pre-registration baseline and methodology documentation.

## Handoff rules

Members may build against the schemas in `docs/data_dictionary.md` and
`src/sirps/common/schemas.py`. Every producer must first submit a two-company
sample, include provenance, and add tests before full data production.

The synthetic pilot is not the approved NSE universe. The production universe
remains blocked until evidence is collected and two reviewers approve it.

## Remaining non-initialization work

Market acquisition, ESG extraction, point-in-time fundamentals, feature
engineering, regime classification, modelling, statistical evaluation,
backtesting, and dashboard pages are implementation work owned by M2-M7.