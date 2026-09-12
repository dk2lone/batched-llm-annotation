# Round 2 Review

<details>
<summary>Full reviewer response</summary>

# Round 2 Re-evaluation

**CALIBRATION: none**

## Scores

| Dimension | Score | Weighted |
|---|---:|---:|
| Problem Fidelity | 9/10 | 1.35 |
| Method Specificity | 8/10 | 2.00 |
| Contribution Quality | 6/10 | 1.50 |
| Frontier Leverage | 7/10 | 1.05 |
| Feasibility | 8/10 | 0.80 |
| Validation Focus | 7/10 | 0.35 |
| Venue Readiness | 6/10 | 0.30 |
| **Weighted composite** | **7.35/10** | |

## Anchor and method assessment

The original Problem Anchor is preserved. The revision still asks whether post-test records change historical recommender conclusions solely through future-conditioned eligibility. Training on unfiltered historical rows is a diagnostic choice that isolates selection; it does not introduce future information or change the anchored problem.

The fixed universe and scoring interface are now mostly coherent. \(U^*\) is constructed before k-core filtering, every user has historical input and an in-catalog test target, models are trained once on suffix-independent rows, and \(d_u\) is defined before horizon-specific eligibility is applied. This resolves the earlier contradiction in which a baseline-eligible universe could not gain members and later entrants could be unscoreable.

One ambiguity remains: “the user’s frozen test target” assumes one target, while window-based evaluation may yield several. The proposal must define whether there is one prespecified target per user, whether all targets must survive the horizon mask, or whether the unit of analysis is a user-target pair. Changing which labels count would violate the fixed-label claim.

The dominant contribution is much sharper. It is now one estimand with an entrant/incumbent decomposition and matched null. The invariance audit and 2×2 analysis have correctly been demoted to controls. The method is simpler and no longer reads as several parallel contributions.

## GAP

The remaining gap is between a clean diagnostic and a publishable contribution. The estimand, decomposition, and matched resampling use standard statistical tools. Their application appears distinctive, but the intellectually important result remains unverified: future-conditioned entrants must exhibit model-relative difficulty beyond what historical covariates already explain. The unfiltered reference models also form a deliberate hybrid rather than the exact k-core benchmark pipeline. A paper-level claim therefore needs the gated 2×2 analysis to connect the diagnostic selection effect back to the benchmark construction used in practice.

## Scores below 7

### Contribution Quality: 6/10

**Weakness:** The obviousness objection is reduced but not eliminated. Any selected subgroup can have a different pairwise mean, and the entrant decomposition is an algebraic identity. The substantive novelty depends on exceeding a credible matched null.

**Concrete fix:** Define the null risk set and matching procedure exactly. Sample pseudo-entrants from \(U^*\setminus E_{t_0}\), using only information available by \(t_0\), with prespecified degree/activity/popularity strata or a fixed propensity-score procedure. State that the method contribution is the controlled audit; systematic excess selection is a separate empirical claim.

**Priority: CRITICAL**

### Venue Readiness: 6/10

**Weakness:** The design is ready for a pilot, but not yet for a paper claim. A positive result from two simple models on one dataset could still be dataset- or model-pair-specific. A one-dataset null is correctly treated as a failed positive pilot, not a publishable boundary.

**Concrete fix:** Make the paper-stage bridge mandatory after a positive pilot: run the 2×2 construction on a second dataset and at least one stronger model pair, while preserving the same frozen estimand and thresholds. This connects the controlled diagnostic to practical benchmark conclusions without expanding the method.

**Priority: IMPORTANT**

## Simplification Opportunities

1. Treat the entrant/incumbent decomposition as an interpretation of the primary estimand, not a second contribution.
2. Choose one explicit evaluation unit: preferably one deterministically selected frozen target per user for the pilot.
3. Keep the 2×2 analysis outside the two-day pilot unless both continuation thresholds pass.

## Modernization Opportunities

**NONE for the method.** Foundation-model components would be forced and irrelevant. If the pilot passes, one maintained contemporary recommender can replace a larger collection of legacy baselines for external validity.

## Remaining action items

1. Define the multi-target eligibility rule without allowing horizon-dependent label changes.
2. Specify the matched-null risk set, matching algorithm, balance checks, and number of repetitions.
3. Replace “a fraction of the baseline gap” with a threshold that remains meaningful when the baseline gap is near zero.
4. State explicitly that the unfiltered frozen models estimate selection in isolation, while the gated 2×2 analysis establishes relevance to the actual filtered benchmark.

## Drift Warning

**NONE.** The revisions clarify the original problem and reduce contribution sprawl.

## Verdict

**REVISE — 7.35/10**

The revision fixes the primary design contradiction and produces a focused, feasible pilot. It is not READY because the null construction and target-level eligibility remain underspecified, while paper-level novelty still depends on an unverified empirical effect.

</details>

