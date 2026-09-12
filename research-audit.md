# IEEE BigData 2026 High School Symposium: research ideation audit

Search date: 2026-09-08, America/Toronto. Research planning only. No experimental results or implementation code were produced.

## Coverage and limitations

The requested idea-generation workflow was followed in order: survey, proposal generation, then adversarial verification and refinement. OpenAlex, arXiv, and Semantic Scholar all returned successful direct, adjacent-vocabulary, and domain-free mechanism searches for the three finalists after authenticated requests were throttled below one request per second. The targeted three-source novelty check is complete. It is still a bounded search rather than a systematic review, and empirical contribution remains subject to each candidate's stop rule.

The search script was `/Users/dz/.claude/skills/lit-research/scripts/search.py`. Initial multi-source failure was followed by the skill's OpenAlex/arXiv fallback. Completed novelty rounds using those two sources exited 0. Raw JSONL files in this directory retain irrelevant results as well as relevant ones. Counts are retrieval counts, not numbers of relevant papers or evidence of novelty. Multiword queries were often noisy, so title-specific searches, primary-paper reading, and additional refined rounds supplemented them. This is a bounded targeted search, not a systematic review or exhaustive literature clearance.

Survey files cover recommendation evaluation, entity matching, log parsing, and projection metrics from 2023 onward. Separate recent files cover 2025 onward, including recent-recommendation.jsonl, recent-logs.jsonl, and recent-visualization.jsonl. Adjacent tabular-shift papers were checked through primary conference proceedings. Older mechanism-level work was deliberately included in novelty verification.

## Authenticated Semantic Scholar retry, September 8, 2026

The user supplied an API key for a transient retry. It was passed through hidden terminal input and held in process memory, not saved in these notes or shell configuration. The initial authenticated search returned HTTP 429. A subsequent search through the skill's existing script exited 0 and returned three 2025-onward log-parsing papers. A separate lookup of arXiv:2504.04877 returned HTTP 200 and the indexed title "System Log Parsing with Large Language Models: A Review."

The follow-on survey queries `recommendation evaluation` and `entity matching large language models`, both filtered to 2023 onward, each exited 0 with 15 papers. Results are saved in `s2-auth-survey-rec.jsonl` and `s2-auth-survey-entity.jsonl`. Their titles and available abstracts were inspected. The next query, `log parsing` from 2023 onward, exhausted the script's retries with HTTP 429 and exited 1. Its empty output file is a failed lookup, not evidence of an empty literature.

The later recovery used one authenticated request at a time. C1 and C2 each obtained successful direct, adjacent-vocabulary, and mechanism responses; C3 then obtained the same three successful angles. A paper-ID lookup also resolved VET-Parser's venue, DOI, authors, abstract, and method boundary. Supplemental searches still intermittently returned HTTP 429 or 500, but failed calls were not treated as empty results. The resulting verdicts are recorded below and in finalists.json.

## Venue constraints

[Official symposium page](https://bigdataieee.org/BigData2026/high-school-symposium/): submission deadline September 20, 2026; maximum five pages including references; first author must be enrolled in high school at submission and be the primary research contributor. At least one author must register and present in person. An undergraduate-only submission does not meet that first-author requirement.

## Landscape

- F1: Recommendation evaluation. Temporal leakage, filtering, test-target selection, and population selection already affect model comparisons. Generic split comparisons are saturated.
- F2: LLM entity matching. Prompting, example selection, multi-candidate matching, and consistency constraints have substantial prior art. Merely adding transitivity is insufficient.
- F3: LLM log parsing. Caching and self-correction are crowded. Cache-path error persistence and event-family discovery are narrower evaluation questions.
- F4: Projection reliability. Metric fooling, metric correlation, and empty-space distortion have recent papers. The July 2026 Gap Index supersedes an audit aimed solely at its 2025 precursor.
- F5: Tabular shift. TabReD and TabFSBench already compare models under temporal and feature shifts. A generic model comparison would not suffice.

## Candidate ledger

| ID | Proposed question and hypothesis | Documented starting point | Decision |
|---|---|---|---|
| C1 | Can future-only interactions change historical benchmark cohorts and conclusions? Hypothesis: full-timeline core filtering changes retention and some model comparisons while train-only filtering is invariant. | Zhao et al. document inconsistent filtering and changes to rankings; Time to Split documents protocol-dependent evaluation. The future-suffix decomposition is our inferred extension, not an author-stated open problem. | Finalist, rank 1. Must distinguish from known train-only filtering. |
| C2 | Do incorrect accepted cache hits delay discovery of new log-event families? Hypothesis: cache reuse amplifies initial errors, and budgeted checks reduce persistence. | SoK: LLM-based Log Parsing explicitly discusses initially incorrect stored templates. | Finalist, rank 2. Contribution is path-specific causal characterization, not inventing cache verification. |
| C3 | Can remote point movements change distortion colors on geometrically unchanged triangles? Hypothesis: global normalization produces measurable nonlocal heatmap changes. | Gap Index discusses relative/raw-area interpretability and makes a local-influence claim in section 2.2. | Reject after three-source verification. The effect follows from established shared-denominator mathematics and the metric's intended relative-area semantics; no sufficient paper contribution remains without prior empirical evidence of material misinterpretation. |
| C4 | Does chronological splitting change recommender rankings? Hypothesis: rankings change after temporal leakage is removed. | Time to Split's evaluation-protocol limitations. | Reject. Don't Get Ahead of Yourself, Time to Split, and RecNextEval already cover the broad direction. |
| C5 | Can transitivity constraints improve LLM entity matches? Hypothesis: graph consistency reduces inconsistent pair decisions. | Match, Compare, or Select? identifies limitations of independent matching. | Reject. Multi-candidate matching and Exploiting Transitivity Constraints already establish the ingredients and mechanism. |
| C6 | Does checking a template before caching reduce log-parsing errors? Hypothesis: fewer erroneous templates enter the cache. | SoK's initial-template-error discussion. | Reject. EFParser and SCOPE already perform correction/semantic checks. vCache and GroundedCache also defeat generic cache-verification novelty. |
| C7 | Can we construct visually bad embeddings with good quality scores? Hypothesis: misleading layouts retain high metric values. | Necessary but not Sufficient documents inadequate metric sensitivity. | Reject. That paper already fools 17 metrics. Metric Design != Metric Behavior also defeats generic metric aggregation. |

## Novelty-search rounds

Every listed raw round used OpenAlex and arXiv. All completed original, refined, and final rounds exited 0. Semantic Scholar was unavailable, so the requested all-source verdict remains unknown.

| Finalist | Direct wording | Adjacent vocabulary | Domain-free mechanism |
|---|---|---|---|
| C1 | recommendation k-core filtering temporal leakage | survivorship bias offline evaluation cohort selection | future information sample selection preprocessing leakage |
| C2 | log parsing cache hit errors auditing | semantic cache false hits drift detection | memoization error amplification selective verification |
| C3, initial | projection quality warping index outlier sensitivity | embedding distortion metric robustness contamination | maximum normalization nonlocal influence robust statistics |

Additional refinements are recorded in rec-r1/r2/r3-refined.jsonl, log-r1/r2/r3-refined.jsonl, and vis-r1/r2/r3-refined.jsonl. They searched k-core recommendation, selection bias, preprocessing leakage, log-parsing caches, semantic caching, approximate memoization, projection metrics, dimensionality-reduction distortion, and normalization-induced rank reversal.

After discovering the July 2026 Gap Index, C3 received three further rounds: `"Gap Index" AND distortion`; `"local distortion" AND normalization`; `"compositional data" AND "subcomposition"`. Results are vis-final-r1/r2/r3.jsonl. The second query was particularly noisy; complementary projection-specific searches and the primary paper were necessary. Compositional-data literature establishes that shared denominators create dependencies, so this general mechanism is not a novelty claim.

## Closest prior work and positioning

### C1: The Future Chooses the Test Set

- [A Revisiting Study of Appropriate Offline Evaluation for Top-N Recommendation Algorithms](https://doi.org/10.1145/3545796), TOIS 41(2), indexed 2023, online 2022: already evaluates 5/10-core filtering versus unfiltered data and ranking changes. Proposed difference: fixed historical windows with appended unused future data and a separation of cohort selection from training/candidate changes.
- [Time to Split](https://arxiv.org/html/2507.16289v1), 2025: compares splitting and target-selection protocols. Proposed difference: hold the split and targets fixed while intervening on filtering provenance.
- [Don't Get Ahead of Yourself](https://researchportal.helsinki.fi/en/publications/dont-get-ahead-of-yourself-a-critical-study-on-data-leakage-in-of/), 2025: temporal leakage comparison with training-size controls. Proposed difference: selection-only effects when no future interactions enter model training.
- [RecNextEval](https://arxiv.org/html/2604.13665v1), SIGIR 2026 demo: provides temporal next-batch evaluation and configurable cold-start handling. Do not claim to invent temporal evaluation infrastructure.
- [tgr README](https://github.com/kiriennn/tgr/blob/main/README.md): explicitly implements train-window k-core filtering and warns about future information. This rules out claiming the filtering correction itself as new.
- [On Survivorship Bias in MS MARCO](https://arxiv.org/abs/2204.12852): adjacent selection-bias prior. Future-dependent recommendation cohorts are a distinct experiment, not a new general theory of selection bias.

### C2: Silent Cache Hits

- [SoK: LLM-based Log Parsing](https://arxiv.org/html/2504.04877v1), 2025: initial erroneous templates and revision are known problems.
- [EFParser / Small Is Beautiful](https://arxiv.org/html/2601.22590v1), 2026: corrects generated templates and uses adaptive caching. Proposed difference: accepted-hit error persistence and delayed family discovery.
- [SCOPE](https://arxiv.org/html/2603.27247v1), 2026: matching-time semantic checks already exist. Mandatory strong comparison, not an ignorable competitor.
- [KELP](https://arxiv.org/html/2601.00633v1), 2026: structural evolution and adaptive grouping are already benchmarked. Proposed experiment isolates false acceptance of arrivals by existing templates.
- [vCache](https://arxiv.org/html/2502.03771v5), 2025/2026: verification and adaptive correctness thresholds are prior art; assumptions include i.i.d. data. Do not claim a new generic verified cache or error guarantee.
- [GroundedCache](https://arxiv.org/html/2605.27494v1), 2026: wrong-hit metrics and cache safety stress tests for RAG already exist. Proposed difference is log-family discovery and initial-error-to-reuse attribution, not simply a new domain for its verifier.

### C3: Local Colors, Global Dependencies

- [Gap Index](https://arxiv.org/html/2607.28324v1), July 2026: current method uses sum-normalized areas, already evaluates jitter across 18 datasets, discusses raw/relative-area tradeoffs, and compares perimeter deformation. None of those alone is a new proposal.
- [Mind the Gaps](https://arxiv.org/html/2511.14544v1), 2025: predecessor uses a different normalization. Retain as a historical baseline, not the sole target.
- [Necessary but not Sufficient](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70101), 2025: broad metric-fooling experiments are already done. Proposed difference is untouched-region influence under controlled remote perturbations.
- [Metric Design != Metric Behavior](https://arxiv.org/abs/2507.02225), 2025: metric selection and correlation-based aggregation are prior art.
- [Aitchison's Compositional Data Analysis 40 Years On](https://arxiv.org/abs/2201.05197), revised 2023: normalization-induced dependencies are established. The potential paper is a specific metric-semantics audit with empirical consequences, not a new robust-statistics method.

## Verified data inventory

- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/): approximately one million ratings, 6,000 users, and 4,000 movies. This release has pre-existing user selection, which must be acknowledged.
- [Amazon 2014 Beauty](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html): 2,023,070 raw ratings. Use ratings-only, not the prefiltered 198,502-row 5-core release. Direct ratings download returned HTTP 200 and Content-Length 82,432,164 bytes.
- [Loghub-2.0](https://github.com/logpai/loghub-2.0): Linux 23,921 logs / 338 templates; Mac 100,314 / 626; Hadoop 179,993 / 236. Combined 304,228 logs. Existing labels suffice. Zenodo hosting was reached earlier but a later web open failed and curl timed out. Data-page availability is verified; complete archive download/checksums are not verified.
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): 70,000 images, 28 by 28 pixels, ten classes. Use a fixed 5,000-image subset for pairwise metrics.
- [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html): 1,797 images with 64 features and ten classes.

No new participants, manual labels, private data, specialized hardware, or model training are needed for C2/C3. C1 fits small conventional recommenders on a CPU. Public-data use must still follow dataset terms and institutional requirements; public availability is not an automatic ethics exemption.

## Ranking and stop rules

Scores are planning judgments, not acceptance probabilities. Novelty scores indicate potential differentiation under the bounded search, not verified novelty.

| Rank | Candidate | Novelty potential | Feasibility | Reviewer appeal | BigData fit | Completion | Interestingness |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | C1 | 6 | 9 | 7 | 9 | 9 | 7 |
| 2 | C2 | 6 | 7 | 8 | 10 | 7 | 8 |
| 3 | C3 | 3 | 8 | 4 | 8 | 7 | 5 |

C1 stop rule: no meaningful historical cohort or model-comparison effect, or an exact prior controlled decomposition is located. Do not write a paper that only repeats 'filtering matters.'

C2 stop rule: effects occur only with deliberately corrupted caches, stronger SCOPE checks already eliminate them, labels make family discovery ambiguous, or baseline reproduction exceeds the available budget.

C3 stop rule: only the known shared-denominator algebra is recovered, or realistic perturbations show no material local-interpretation consequences. Do not relabel intended relative-area changes as metric errors.

Recommendation: C1, subject to the two-day pilot. C2 is the backup. C3 is rejected. No implementation has begun.

## Recovery update, 2026-09-08

See [verification-update.md](verification-update.md) for the completed authenticated Semantic Scholar checks, newly inspected competitors, and primary-paper corrections. All three candidates obtained successful responses for direct, adjacent-vocabulary, and mechanism searches across Semantic Scholar, complementing the completed OpenAlex and arXiv rounds. VET-Parser is resolved at the metadata-and-abstract level and remains a mandatory C2 competitor; its full text is closed-access.

The concrete next phase is specified in [c1-pilot-plan.md](c1-pilot-plan.md). It isolates post-test information used only for historical eligibility filtering, separates training-mask and evaluation-cohort effects, and specifies invariant controls and stop criteria. The plan is written; no experiment implementation or results are claimed.
