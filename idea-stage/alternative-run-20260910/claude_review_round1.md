# Adversarial research-idea review

You are the hostile second reviewer for three proposed IEEE BigData 2026 High School Symposium papers. The venue allows five pages including references and scores technical quality, novelty, relevance, and clarity. Read `idea.md` first so you do not duplicate the existing provenance-replay idea. Inspect the JSONL evidence in `literature/stage3_*` and use WebSearch/WebFetch to check current literature through 2026.

The student wants master/PhD-level ideas spanning LLM data mining, recommendation/personalization, and statistical ML. Public data and open-weight models are preferred. GPU access exists, but the design must be executable by one strong high-school student before 2026-09-20.

Review these candidates:

1. **Simulator-error rank bounds.** Given an LLM user simulator and a small human validation sample, estimate a high-confidence bound on the simulator-human utility gap. Propagate that bound into pairwise intervals over conversational recommender performance. Certify a winner only when its worst-case lower bound beats every competitor; otherwise return a set of plausible winners. Validate empirical coverage on held-out human conversations from ConvApparel and compare against raw simulator rankings.

2. **Item-name placebo intervention.** For a fixed user history and fixed candidate item attributes, randomize which canonical, shuffled, or opaque title is attached to each item. Use randomization inference to estimate how much an LLM recommender's ranking is causally driven by recognizable item names rather than the supplied preference evidence. Produce an alias-averaged title-free score and validate on injected-memorization oracles plus MovieLens/Amazon data.

3. **Serialization-invariance certificate.** Repeatedly permute feature order in serialized tabular rows. Use an anytime-valid confidence sequence or exact binomial procedure to stop once a black-box LLM prediction is certified stable or unstable under the permutation distribution. Report per-row certificates and a population coverage statement; validate the claimed error rate in simulation and test on public UCI/OpenML data.

Already rejected: **factorial variance decomposition of prompt instability**, because Kweon et al., *Uncertainty Quantification and Decomposition for LLM-based Recommendation* (WWW 2025) already decomposes predictive and prompt uncertainty, while a 2026 paper uses crossed variance components for LLM brand answers.

For each surviving candidate, answer:

- What is the closest prior work through September 2026? Give exact titles and links.
- Is the central contribution actually new, or a routine application/renaming?
- What precise theorem or validated property can honestly be claimed?
- What hidden assumption would make the result invalid?
- What is the smallest defensible experiment and strongest falsification test?
- Score novelty, technical depth, feasibility, and venue fit from 1 to 10.

Kill any candidate that does not deserve a paper. If fewer than three survive, propose a narrowly scoped replacement grounded in a limitation from a specific cited paper. End with a ranked verdict and concrete revisions. Do not edit files.
