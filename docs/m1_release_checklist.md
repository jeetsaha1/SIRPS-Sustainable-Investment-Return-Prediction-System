# M1 Release Checklist

This checklist is the gate for handing the initialized repository to M2-M7.

## Repository checks

- [x] Python 3.12 is declared and dependencies are pinned.
- [x] Package installation works from a clean checkout.
- [x] CI runs Ruff, Black, and pytest.
- [x] Shared schemas, I/O, UI, calendar, and mock generation contracts exist.
- [x] `run_all.py` generates all eight mock tables and serving output.
- [x] Mock data is labelled synthetic and is not presented as NSE history.
- [x] All current tests pass locally.

## Human approval gates

- [ ] M1 reviews and signs the protocol/schema freeze.
- [ ] M2/M5 review the universe eligibility rules.
- [ ] M2 verifies market-source fields and licence terms.
- [ ] M3/M4 verify the ESG source register and extraction assumptions.
- [ ] Real-company evidence is collected and independently reviewed.
- [ ] The production `company_master.csv` is frozen and signed.
- [ ] A reviewer approves the pull request.
- [ ] `main` branch protection is enabled on GitHub.
- [ ] Release tags are created from the approved merge commit.

The unchecked items require evidence, approval, or GitHub administration. They
must not be replaced with synthetic values or marked complete by code alone.