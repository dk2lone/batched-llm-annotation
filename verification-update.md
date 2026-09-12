# BigData idea verification update

Date: 2026-09-08. Research planning only. No experiments implemented or run.

## Decision

C1, the temporal cohort-selection audit, remains the preferred project for a two-day feasibility pilot. C2, accepted-cache-hit error persistence in log parsing, survives as the backup. C3, projection heatmap locality, is rejected as an insufficiently novel contribution. Targeted three-angle checks have now completed across OpenAlex, arXiv, and Semantic Scholar for all three candidates. This is a bounded literature verdict, not evidence of experimental results or an exhaustive systematic review.

The previous pause was premature. A failed source prohibits a novelty claim from that search, but the idea-generation skill provides authenticated retries followed by an explicitly qualified OpenAlex/arXiv fallback. It does not require abandoning useful primary-paper verification.

## Search recovery and limits

Authenticated minimal Semantic Scholar search requests and a bulk search returned HTTP 200. Other requests still returned HTTP 500 or 429. The same previously failing recommendation query succeeded with fewer metadata fields and without a year filter. Because multiple parameters changed and the service is intermittent, the cause is not isolated.

The successful direct recommendation diagnostic returned RecNextEval and Don't Get Ahead of Yourself. C1 subsequently received successful adjacent-vocabulary and mechanism responses. C2 received successful direct, adjacent-vocabulary, and mechanism responses. Both mechanism responses were noisy and were supplemented with primary-paper checks. HTTP success is not sufficient evidence of useful coverage.

C3's direct, adjacent-vocabulary, and mechanism requests were rerun sequentially below the authenticated one-request-per-second limit and returned HTTP 200. The direct round returned the 2026 Gap Index paper and its 2025 predecessor. The adjacent round returned projection-quality work but no study of remote perturbations changing geometrically untouched triangles. The mechanism round returned established work on compositional closure artifacts and subcompositional coherence. Some supplemental requests still returned HTTP 429 or 500; those failures were not used as negative evidence.

The accompanying s2-recovery-results.jsonl contains the earlier recovery session, including failures. The final successful calls are summarized here rather than appended with full abstracts. Credentials are excluded and were not saved in any project file.

## C1: exact contribution boundary

[Zhao et al., A Revisiting Study of Appropriate Offline Evaluation for Top-N Recommendation Algorithms](https://doi.org/10.1145/3545796) already compares raw data with 5-core and 10-core filtering and measures model-ranking changes. Sections 5.1.2 and 5.1.3 explain effects through the composition and sparsity of the retained data. Section 5.2.1 leaves deeper global-time-split analysis to future work because of cold-start complications. Online publication was December 2022, though the volume is commonly indexed as 2023.

[A Critical Study on Data Leakage in Recommender System Offline Evaluation](https://arxiv.org/html/2010.11060v4) fixes a historical test period and progressively adds later interactions to training. Our proposed experiment must not reproduce that intervention: future interactions affect only filtering eligibility, never training, hyperparameter selection, test labels, or candidate construction.

[SplitLight](https://arxiv.org/html/2602.19339v1) already audits dataset and split properties, including temporal overlap, cold start, and distribution changes. Its reported experiments use pre-split core filtering and vary other preprocessing choices. The inspected paper does not report the proposed fixed-window, future-suffix intervention with a training-mask/evaluation-cohort decomposition. This is a bounded differentiation statement, not proof that no other work does.

[Towards Robust Offline Evaluation](https://arxiv.org/html/2504.03997v1) studies selection and exposure bias using a causal and information-theoretic resampling framework. [Overview of leakage scenarios in supervised machine learning](https://doi.org/10.1186/s40537-025-01193-8) covers preprocessing leakage generally. Therefore neither selection bias nor preprocessing leakage is a new general mechanism here.

Claim worth testing: historical recommender conclusions can depend on later records solely because benchmark construction uses those records to select historical users and items. The contribution would be a controlled attribution study, its measured consequences, and a reproducible invariance test. Train-only core filtering itself is established practice and is not the contribution.

## C2: new competitors and risks

[EFParser](https://arxiv.org/html/2601.22590v1), section 3.1, returns accepted cache matches without invoking the miss-path generation and correction steps. Its limitations discuss retrieval failures and propagated template mistakes. This supports inspecting accepted-hit paths separately. However, section 5.2 reports that over-general errors are relatively rare. Natural error persistence may therefore be too small to sustain a paper.

[SCOPE](https://arxiv.org/html/2603.27247v1) already performs syntax and semantic checks during matching, and [KELP](https://arxiv.org/html/2601.00633v1) already evaluates evolving log structures. Neither generic cache verification nor generic drift evaluation is a defensible new contribution.

The recovered S2 direct search added [Towards a Classification of Log Parsing Errors](https://users.encs.concordia.ca/~abdelw/papers/ICPC23_ParsingErrors.pdf), which categorizes parsing failures, and [TraceDoctor](https://netsec.ccert.edu.cn/publications/ase25-logparsing), which analyzes reasoning errors and fine-tunes models on targeted variants. Our proposed path-specific persistence and event-discovery analysis differs from those contributions. Merely making another error taxonomy would not suffice.

[VET-Parser](https://doi.org/10.1109/WCCIS70285.2026.11650850) was resolved through Semantic Scholar paper ID `d5f9f248332d525e38335bebcf9a12c37dc15fce`. It is a 2026 WCCIS conference paper by Mei-Chen Li, Kuntharrgyal Khysru, Di Jin, and Jianguo Wei. Its indexed abstract describes variable-evidence risk detection, selective LLM arbitration for uncertain clusters, and verification before cache updates across 14 LogHub-2k datasets. This is medium overlap and a mandatory competitor for mitigation experiments, but the abstract does not report accepted-hit error persistence or event-family discovery delay. The full text remains closed-access, so this positioning is limited to verified metadata and abstract.

## C3: corrections to the secondary review

The primary [Gap Index paper](https://arxiv.org/html/2607.28324v1), section 2.2, does explicitly say that a point's position affects only quality values of incident triangles. The earlier review's claim that this statement was absent is incorrect.

The [2025 predecessor](https://arxiv.org/html/2511.14544v1), immediately after equation 1, defines triangle area after division by the largest triangle's area. Its normalization is therefore globally dependent. Looking only at the local max in equation 1 misses this definition.

The completed search rejects C3 in its current form. The July 2026 paper explicitly intends relative-area semantics, while shared-denominator dependence, closure artifacts, and subcompositional coherence are established mathematics. A remote perturbation changing an untouched triangle's relative score follows directly from those definitions. Without an already-demonstrated material interpretation failure, the proposed audit is too small for a paper.

## Next research step

Use c1-pilot-plan.md to decide whether C1 has an empirical contribution before committing to a full paper. C1's literature verdict is novel with medium confidence because no located work performs the fixed-history, future-suffix selection-only intervention and cohort/training decomposition. C2 is also novel with medium confidence at the question level, but VET-Parser, SCOPE, and EFParser make its mitigation space crowded. The plan does not authorize implementation beyond the user's current no-code boundary.
