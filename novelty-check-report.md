# Novelty Check Report

Date: 2026-09-08. Scope: bounded literature check for the two surviving IEEE BigData 2026 proposals. This is not a systematic review. The fresh reviewer is same-family, so its judgment is provisional.

## Search coverage

Each of eight core claims was searched with at least three formulations spanning direct phrasing, adjacent vocabulary, and the domain-free mechanism. Coverage included OpenAlex, arXiv, authenticated Semantic Scholar, general web scholarly discovery, ICLR 2025/2026, NeurIPS 2025, ICML 2025/2026, and March–September 2026 arXiv results.

Semantic Scholar remained intermittent despite requests spaced below the approved limit: this run received one HTTP 200 response plus HTTP 429 and 500 responses. Failed requests were recorded as unknown and were not treated as empty results. Earlier authenticated three-angle Semantic Scholar results in this workspace were included.

## C1: The Future Chooses the Test Set

### Proposed method

Hold all historical training, validation, test, and candidate data fixed, then append future records solely when computing k-core eligibility. Separate effects caused by changing historical training rows from effects caused only by changing the scored user/item cohort.

### Core claims

| Claim | Novelty | Closest work | Judgment |
|---|---|---|---|
| C1.1 Future-only records alter historical eligibility | Medium | Zhao et al. | Exact intervention not located, but k-core sensitivity and selection effects are established. |
| C1.2 A 2×2 training-mask/cohort design isolates the source | High | Zhao et al.; Ji et al. | Strongest method contribution. Prior work does not freeze predictions while independently manipulating training membership and evaluation membership. |
| C1.3 Realistic suffixes materially alter scores or rankings | Undetermined; potentially high | Parajuli et al.; Time to Split | This is an empirical hypothesis. Its novelty depends on effect size, realism, and replication. |
| C1.4 An invariance audit detects post-test dependence | Medium | SplitLight; RecNextEval | Useful packaging of standard provenance and reproducibility checks, not a new auditing primitive. |

### Closest prior work

| Paper | Year | Venue | Overlap | Remaining difference |
|---|---:|---|---|---|
| [A Revisiting Study of Appropriate Offline Evaluation for Top-N Recommendation Algorithms](https://doi.org/10.1145/3545796) | 2023 | TOIS | k-core filtering changes composition and model rankings | Does not isolate future-only eligibility with fixed historical records. |
| [A Critical Study on Data Leakage in Recommender System Offline Evaluation](https://arxiv.org/abs/2010.11060) | 2023 | TOIS | Future records affect historical evaluation and rankings | Adds future interactions to training; C1 forbids this. |
| Time to Split | 2025 | RecSys | Split choice changes recommender evaluation | Does not test future-only filtering provenance. |
| SplitLight | 2026 | arXiv | Audits data and split properties | No located fixed-history future-suffix intervention. |
| [On the Convergent Validity of Offline Evaluation Designs for Recommender Systems](https://arxiv.org/abs/2607.25097) | 2026 | RecSys | Evaluation designs alter model-ranking validity and k-core effects are discussed | Its experiments vary candidate sets, relevance, and list length; it leaves k-core effects for future analysis. |

### Overall novelty assessment

- Score: **7.5/10**
- Recommendation: **PROCEED**
- Key differentiator: future-dependent eligibility is isolated while historical labels, candidates, fitted predictions, and optionally training rows remain fixed.
- Main risk: the pilot may recover only the established fact that filtering changes dataset composition, or effects may require unrealistic suffixes.

### Suggested positioning

Frame C1 as a controlled measurement of future-dependent eligibility or benchmark non-invariance. Do not claim to discover temporal leakage, selection bias, or k-core sensitivity. The publishable evidence must show a meaningful consequence beyond changed row counts, or establish credible conditions under which the effect is negligible.

## C2: Silent Cache Hits

### Proposed method

Instrument online log parsers to identify wrong accepted cache hits, measure their persistence and delay before a distinct labeled event family is correctly recognized, and compare natural failures with injected stress tests and budget-matched audits.

### Core claims

| Claim | Novelty | Closest work | Judgment |
|---|---|---|---|
| C2.1 Wrong accepted cache matches silently persist | Low | Analemma manuscript; EFParser | Persistent silent trie absorption is already public prior art. |
| C2.2 Wrong hits delay discovery of a distinct event family | Medium–high | Analemma manuscript | Exact labeled family-discovery delay was not located, but it is a close consequence of prior silent-absorption results. |
| C2.3 Path-conditioned persistence and discovery-delay metrics add information | Medium | Analemma manuscript; error-taxonomy work | Discovery delay appears distinct; persistence duration substantially overlaps prior work. |
| C2.4 Budget-matched selective auditing mitigates the failure | Low | VET-Parser; SCOPE | Verification and selective arbitration are established. A new heuristic is insufficient without clear gains over strong risk baselines. |

### Closest prior work

| Paper | Year | Venue/status | Overlap | Remaining difference |
|---|---:|---|---|---|
| [Persistent Demo-Pool Poisoning Attacks on Online LLM Log Parsers](https://analemma.ai/papers/cb266f60-e436-42c1-b85b-19fb9f051637) | 2026 | Public automated-research manuscript; no located standard venue | Demonstrates 900+ log persistence and silent trie absorption under crafted injection | Does not study natural accepted-hit errors or labeled family-discovery delay. |
| [Small is Beautiful: A Practical and Efficient Log Parsing Framework](https://arxiv.org/abs/2601.22590) | 2026 | arXiv | Accepted EFParser cache hits bypass generation/correction; cache errors and updates | Does not report persistence duration or family-discovery delay. |
| [SCOPE](https://arxiv.org/abs/2603.27247) | 2026 | arXiv | Matching-time correction and evolving trees | Does not provide the proposed natural-failure characterization. |
| [VET-Parser](https://doi.org/10.1109/WCCIS70285.2026.11650850) | 2026 | WCCIS | Risk detection, selective LLM arbitration, verified cache updates | Indexed abstract does not report accepted-hit persistence or discovery delay; full text was unavailable. |
| [LILAC](https://arxiv.org/abs/2310.01796) | 2024 | FSE | Adaptive LLM parsing cache | Does not report the proposed temporal failure metrics. |

The Analemma paper explicitly states that it was generated by an automated research system. Its status weakens confidence in its evidence, but the public method and result still constrain novelty claims.

### Overall novelty assessment

- Score: **5.5/10**
- Recommendation: **PROCEED WITH CAUTION**
- Key differentiator: natural, path-confirmed cache-hit errors and first-correct-recognition delay for labeled event families.
- Main risk: the headline persistence mechanism is already occupied, while natural errors may be too rare and verification baselines may remove the effect.

### Suggested positioning

Frame C2 as a non-adversarial characterization study. State that prior work demonstrates persistent cache/trie corruption, then ask whether ordinary online parsing produces the same failure and whether discovery delay exposes harm hidden by aggregate accuracy. Do not claim persistent cache errors, cache verification, or selective arbitration as new.

## Final decision

Pilot C1 first. It retains the cleaner unresolved comparison and can support an informative null result if the intervention is realistic and tested across datasets. Keep C2 only as a gated backup: before building the full study, confirm that natural wrong accepted hits and measurable family-discovery delays occur.

The raw reviewer trace is stored at `.aris/traces/novelty-check/2026-09-08_run01/`.
