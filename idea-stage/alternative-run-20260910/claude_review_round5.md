# Final independent audit of three corrected candidates

Use WebSearch and WebFetch independently. Verify every paper and mathematical claim. This is for the IEEE BigData 2026 High School Symposium, interests 2/5/7. Hard-pass only if novelty >= 6 and depth >= 6. Kill routine applications of known tools. If you kill a candidate, search and repair or replace it in the same response until exactly three survivors remain.

## Why round-4 candidates were revised

Round 4 proposed S1-S3. A new OpenAlex/arXiv/Semantic Scholar audit found:

- Original S2 collides with Li et al., *Towards Reliable Item Sampling for Recommendation Evaluation*, which already derives an error-optimized estimator and adaptive sampling: https://arxiv.org/abs/2211.15743. Treat original S2 as killed.
- Original S1's response permutation may sever the memorized model from the newly permuted validation labels and therefore restore the independence it intended to test. It cannot honestly claim to measure pretraining-induced Type-I inflation without controlled exposure.
- Original S3 remains provisional. Audit single-run interference and whether content/co-occurrence separation is identifiable.

## C1 — controlled validation-data exposure and Type-I inflation in agentic science

Question: When an LLM agent has seen the validation rows and outcomes during pretraining, how badly can that break a sequential hypothesis-validation pipeline's nominal Type-I error, and can re-keying restore validity?

Design:

1. Before training, generate many independent tabular datasets under a known global null. Include a discovery split and a hidden validation split. Randomize entire dataset instances to `full exposure` (model sees discovery + validation rows/outcomes in continued pretraining), `discovery-only exposure`, and `no exposure`.
2. Continue-pretrain small open twin models, or use a rigorously justified single-run controlled-pretraining design. Afterwards the agent receives only the discovery split through the normal pipeline. It proposes hypotheses and the Popper-style validation step uses the held-out validation split.
3. Since the exposed model weights contain validation outcomes, sample splitting's measurability condition can fail. Across independently generated null datasets, estimate actual rejection probability at nominal alpha and the exposure-arm causal contrast.
4. Mitigation: random token re-keying of columns, categories, and row identifiers at validation time. Measure Type-I error and power versus re-key strength.
5. Add a planted-alternative arm for power. Use a fixed non-LLM hypothesis generator as a negative control. Use a deliberately leaked in-context agent as a positive control.

Claim only the causal effect of controlled exposure under this training/text distribution. Do not generalize directly to proprietary pretraining or public datasets.

Audit: whether Popper's Type-I theorem really depends on agent/validation measurability in the claimed way; whether the validation test itself remains valid even if the agent memorized outcomes; how to define hypotheses so the exposed model can exploit validation noise; whether controlled-pretraining or adaptive-data-analysis papers already run this experiment; and whether re-keying is more than a heuristic.

## C2 — anytime-valid full-catalog rank certification under adaptive item sampling

Question: Can two expensive LLM recommenders be compared against the full catalog while adaptively querying only enough item scores to certify the winner with finite-sample coverage?

Design:

1. For user u and model a, the target rank is `R_ua = 1 + sum_j 1{s_uaj > s_uat}` over the finite catalog. The comparison target is the paired difference in full-catalog Recall@k (and separately DCG@k) across a fixed user test set.
2. Items are sampled without replacement using known adaptive inclusion probabilities based on cheap collaborative scores and all expensive scores observed so far. Construct finite-population martingale/e-process confidence sequences for each rank total, valid under optional stopping and adaptive sampling.
3. Map rank confidence sets to deterministic lower/upper bounds on aggregate Recall@k or DCG@k and the paired system difference. Stop when the difference interval excludes zero or the budget ends.
4. Prove a worst-case lower bound: exact deterministic certification of top-k membership needs linear catalog queries without assumptions. The statistical certificate therefore states its coverage probability explicitly.
5. Compare with uniform sampling, Li et al. 2022 adaptive sampling, Krichene-Rendle corrections, and full-catalog ground truth. Report coverage, calls saved, and abstention rate.

Do not claim a new optimal sampling distribution. The proposed novelty is an anytime-valid *paired winner certificate* for full-catalog metrics under adaptive item queries, not another point estimator.

Audit: search confidence sequences/e-processes in recommender evaluation, active ranking, finite-population adaptive sampling, and best-arm identification. Determine whether the construction is technically possible with known sampling probabilities and whether per-user multiple confidence sequences make the interval useless. Kill if it is just standard confidence sequences pasted onto Li et al.

## C3 — randomized pretraining exposure split into item content and co-occurrence

Question: How much of an LLM recommender's performance comes from pretraining on item semantics versus item-item co-occurrence?

Design:

1. Use synthetic item identifiers absent from the base model. Create item content attributes and a latent user-preference/co-occurrence graph. Randomize items in a factorial design: content documents absent/present x co-occurrence documents absent/present, with randomized exposure dose.
2. Continue pretrain one small open model using the Bordt-Pawelczyk `Train Once, Answer All` design only if its assumptions apply; otherwise train twin models per factorial contrast. Run CPDT interaction checks when using a single run.
3. Fine-tune on identical user-item sequences whose held-out targets are balanced across arms. Estimate item-level intention-to-treat effects on target log-rank/hit@10 and the content-by-co-occurrence interaction using design-based randomization inference.
4. Pre-register a closed-book probe confirming that each injected document family was learned. Falsification: probe succeeds but ranking effects remain zero.
5. Replicate the direction on real post-cutoff Amazon items only as an external-validity check, with no causal claim there.

Audit: interference from shared model parameters, whether item-level factorial randomization identifies the stated direct effects in one model, whether Bordt-Pawelczyk already covers this exact contribution, whether content/co-occurrence corpora can be separated without semantic leakage, and closest recommendation/pretraining-exposure papers.

## Required output

For each C1-C3:

1. exact estimand and strongest defensible claim;
2. closest 3-5 primary papers with exact URLs and a same-estimand/design comparison;
3. mathematical/identification audit, including any corrected theorem language;
4. smallest decisive experiment, compute, and strongest falsification;
5. fatal assumption/risk;
6. novelty/depth/feasibility/venue scores 1-10 and hard pass/kill.

Return exactly three survivors. If a candidate fails, independently find and fully audit a replacement before finishing. Do not ask the user a question.
