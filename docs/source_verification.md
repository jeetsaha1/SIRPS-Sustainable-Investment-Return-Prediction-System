# Source Verification Register

Source verification is an M1 governance gate, not a synthetic-data task. The
register in `config/source_register.csv` remains `pending` until a member has
checked access, fields, coverage, licence/terms, and provenance.

For each source, the reviewer must record:

1. Official URL or filing location.
2. Date and method of access.
3. Available fields and date coverage.
4. Licence or terms-of-use decision.
5. Cross-check performed and result.
6. Reviewer identity and approval date.

Raw vendor files and PDFs must remain outside the public repository unless the
licence permits redistribution. Synthetic files under `data/mock/` are exempt
from source verification because they are generated locally and labelled as
synthetic.