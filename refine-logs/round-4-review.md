# Round 4 Review

<details>
<summary>Full reviewer response</summary>

# Round 4 Re-evaluation

**CALIBRATION: none**

## Scores

| Dimension | Score | Weighted |
|---|---:|---:|
| Problem Fidelity | 10/10 | 1.50 |
| Method Specificity | 10/10 | 2.50 |
| Contribution Quality | 8/10 | 2.00 |
| Frontier Leverage | 8/10 | 1.20 |
| Feasibility | 9/10 | 0.90 |
| Validation Focus | 10/10 | 0.50 |
| Venue Readiness | 9/10 | 0.45 |
| **Weighted composite** | **9.05/10** | |

## Assessment

The Problem Anchor remains fully preserved. The design isolates future-conditioned eligibility while keeping the historical task, target, model inputs, predictions, and primary comparison fixed.

The primary endpoint is now unambiguous: the popularity-versus-item-kNN Recall@20 pairwise-gap shift at the final observation horizon. Requiring both the matched-null threshold and a 0.002 absolute shift prevents statistical extremity without practical relevance. Intermediate horizons and NDCG@10 cannot be promoted opportunistically because they are explicitly descriptive.

The null is sufficiently reproducible. Its strata are frozen at \(t_0\), and sparse-stratum merging follows a deterministic hierarchy with a declared failure condition. Reporting the null as unsupported when donor quantity or balance remains inadequate is preferable to silently relaxing the comparison.

The proposal retains one dominant contribution. The estimator, entrant interpretation, and matched null form one audit rather than separate claims. The 2×2 analysis remains a gated attribution control. No foundation-model component is needed.

## GAP

The remaining gap is empirical, not a proposal-design defect. The audit may find that realistic future-conditioned admission does not exceed historically matched cohort variation or the 0.002 practical threshold. If so, the positive pilot fails. Conversely, a passing result establishes a candidate empirical contribution but still requires the planned replication and filtered-pipeline attribution before broader claims.

Method novelty remains moderate rather than foundational: the statistical pieces are standard, while their fixed-history application to future-dependent recommender eligibility is the contribution. The potentially stronger claim, that this mechanism materially changes model comparisons, remains unverified until execution.

## Drift Warning

**NONE.** Every revision sharpens measurement of the original future-conditioned selection problem.

## Remaining blockers

**NONE at the proposal-design level.**

An unsupported matched null is a prespecified pilot failure state, not an unresolved design decision. An unrun or negative empirical result is also not a design blocker.

## Verdict

**READY — 9.05/10**

The proposal is specific, focused, feasible, and ready for the two-day pilot. READY applies to the research design only; it does not imply that the future-conditioned model-gap effect exists.

</details>

