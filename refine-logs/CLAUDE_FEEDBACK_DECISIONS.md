# Decisions on Claude's C1 Feedback

Claude Opus 5 independently scored C1 at 4/10 for novelty and recommended proceeding with caution. The contribution remains clear, but the literature position is narrower than the internal design review alone suggested.

## Accepted

- **A1**: Call each horizon a dataset snapshot cutoff. Use nested cutoffs inside one immutable raw release for the controlled pilot; real cross-release snapshots may add curation and schema confounds.
- **A2**: Call the matched comparison a historical-balance null, not a causal null or p-value.
- **A3**: Report entrant share and donor-pool balance before interpreting the null.
- **A4**: Add a pre-run power calculation for the binary per-user Recall@20 difference and justify the practical floor before scoring.
- **A5**: Report the exact entrant-share decomposition, sign, and any ranking reversal.
- **A6**: Require a stronger model pair and second dataset for a paper, after the deterministic pilot passes.

## Not accepted

- **N1**: A second all-target evaluation endpoint would reopen target-definition and multiple-testing problems. Keep one frozen target in the pilot.
- **N2**: A second random-future-record null changes the intervention and adds scope before the primary balance-null result is known.
- **N3**: Real Amazon release-to-release comparisons are not the primary pilot because release construction may change alongside time coverage.
- **N4**: A one-category negative result is not automatically a paper contribution. It remains a no-go for expansion unless replicated with tight uncertainty.
- **N5**: Claude's suggested 2% entrant cutoff is not adopted without a power calculation.

## Resulting position

C1 is ready for a two-day diagnostic pilot, but the publication decision is conditional. Its method contribution is a fixed-history, fixed-prediction snapshot audit with a historical-balance null. Its paper-level contribution exists only if the measured effect is nontrivial and survives stronger models, a second dataset, and filtered-pipeline attribution.

