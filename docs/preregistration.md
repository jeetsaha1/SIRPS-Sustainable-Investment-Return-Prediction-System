# SIRPS Pre-registration Baseline

**Status:** initialization baseline; protocol-v1 requires team approval before
any ESG result is examined.

## Locked research choices

- Python 3.12 and the pinned dependencies in `requirements.txt`.
- Primary target: 63-trading-day forward excess log return versus NIFTY 50 TRI.
- Entry at close `t+1`; exit at close `t+64`.
- One company-date observation every five trading days.
- Expanding walk-forward validation with a 63-trading-day purge and a
  5-trading-day embargo.
- One final holdout run after model selection and protocol freeze.
- Primary ESG comparison: A' versus B on identical rows, folds, models, and
  tuning, with Ridge as the pre-specified primary model.
- Primary metric: date-aggregated MSFE and one-sided Clark-West test.

## Freeze procedure

M1 records the protocol version, schema version, configuration hash, and Git
commit before the first ESG model result. Any later change requires a pull
request, a version bump, a reason, and reviewer approval.

## Pilot boundary

The initialization pilot uses synthetic data only. It demonstrates contracts and
software behavior and must never be reported as historical NSE evidence.