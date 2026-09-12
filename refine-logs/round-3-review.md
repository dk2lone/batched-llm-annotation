# Round 3 Review

<details>
<summary>Full reviewer response</summary>

# Round 3 Re-evaluation

**CALIBRATION: none**

## Scores

| Dimension | Score | Weighted |
|---|---:|---:|
| Problem Fidelity | 10/10 | 1.50 |
| Method Specificity | 9/10 | 2.25 |
| Contribution Quality | 8/10 | 2.00 |
| Frontier Leverage | 8/10 | 1.20 |
| Feasibility | 9/10 | 0.90 |
| Validation Focus | 8/10 | 0.40 |
| Venue Readiness | 9/10 | 0.45 |
| **Weighted composite** | **8.70/10** | |

## Anchor, focus, and simplicity

The Problem Anchor is preserved. Future records affect only \(E_h\); the historical task, catalog, targets, training rows, models, and per-user predictions remain fixed.

The previous scoreability problem is resolved. The deterministic single-target rule removes multi-target ambiguity, while \(U^*\) guarantees historical training input and an in-catalog target for every evaluated user. The unfiltered models are correctly presented as an isolation instrument rather than the exact practical benchmark pipeline.

The contribution is focused: one horizon-sensitivity audit, interpreted through an entrant decomposition and tested against a matched null. The decomposition is no longer presented as independent novelty. Hash checks and the gated 2×2 study remain controls. No component should be removed.

Foundation-model machinery would not improve this design. The planned contemporary recommender pair is sufficient for paper-stage relevance.

## GAP

The proposal-design gap is now small but important: six possible claim tests remain available across three horizons and two metrics. “Continue if the absolute shift exceeds” does not specify whether success on any combination counts. Using separate 95th-percentile null thresholds would allow opportunistic endpoint selection and inflate false positives.

The larger uncertainty is empirical, not methodological. The audit method is ready to test, but no claim can yet be made that future-conditioned admission causes a practically meaningful excess model-gap shift. A null remains possible and would end the one-dataset positive pilot as specified.

Method novelty and empirical novelty are appropriately separated:

- **Method contribution:** the fixed-history, fixed-prediction audit and prespecified matched comparison.
- **Unverified empirical contribution:** future-conditioned entrants exhibit model-relative difficulty beyond historically comparable cohorts.

## Remaining method-level action items

### A1. Freeze one primary endpoint

Choose one model pair, metric, and horizon as the primary continuation test. Recommended: Recall@20 at the final observation horizon, with NDCG@10 and intermediate horizons treated as diagnostic trajectories.

Alternatively, construct each null replicate’s maximum absolute shift across all six metric-horizon combinations and compare the observed maximum against that familywise null.

**Priority: CRITICAL**

### A2. Complete the null specification

State that stratum boundaries are computed once from \(U^*\) using information available by \(t_0\). Prespecify what happens when a stratum has fewer eligible donors than required: merge adjacent strata using a fixed order or mark that horizon’s null unsupported. Do not adapt boundaries after seeing \(d_u\) or entrant outcomes.

**Priority: IMPORTANT**

## Drift Warning

**NONE.** The unfiltered frozen-model analysis isolates the anchored selection question, and the gated 2×2 study reconnects it to practical filtered pipelines without changing the research problem.

## Verdict

**REVISE — 8.70/10**

The proposal is ready to implement as a pilot, but it does not meet the formal READY rule because the primary endpoint and multiple-comparison handling remain unresolved. This is the only blocking design issue; the absence of empirical results is expected at the proposal stage and is not itself a design defect.

</details>

