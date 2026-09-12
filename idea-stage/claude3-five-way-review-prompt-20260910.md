# Independent five-way research-idea audit

You are conducting an independent review for the IEEE BigData 2026 High School Symposium
project in `/Users/dz/bigdata`.

## Objective

Independently evaluate and rank these five ideas:

- I1: The Reviews from Tomorrow
- A1: Repository self-disagreement via provenance replay
- A2: Factorial pretraining exposure, item content versus co-occurrence
- A3: Controlled validation-data exposure and agentic Type-I error
- A4: Contamination-immune agentic tests and the price of open-endedness

The project has access to adequate GPUs and time. Do not penalize A2 or A3 merely because
they require model training. Still evaluate whether each design is technically executable and
whether its result would support the intended claim.

## Independence protocol

1. Read `AGENTS.md` for the project boundaries.
2. Read only the idea descriptions needed to understand the designs:
   - I1 in `ideas.md` and `idea-stage/IDEA_REPORT.md`
   - A1 in `idea-stage/alternative-run-20260910/idea.md`
   - A2, A3, and A4 in
     `idea-stage/alternative-run-20260910/idea-generation-report.md`
3. Before forming your ranking, do not read:
   - `idea-stage/FIVE_WAY_COMPARISON_20260910.md`
   - any `claude_review_round*` file
   - `ideas.md` beyond the material required to understand I1
4. Conduct your own live literature search. Search each idea through at least three distinct
   formulations: direct design match, mechanism/estimand match, and adjacent terminology.
   Prefer primary papers and official venue pages. Verify that cited papers exist and support
   the claim attributed to them.
5. After fixing your independent scores and ranking, read
   `idea-stage/FIVE_WAY_COMPARISON_20260910.md` and briefly explain agreements or disagreements.

## Required audit for every idea

For each candidate, provide:

- the exact research question and strongest defensible contribution;
- the closest published work, with direct links and same-design comparison;
- novelty from 1 to 10;
- contribution clarity from 1 to 10;
- IEEE BigData venue fit from 1 to 10;
- feasibility from 1 to 10 under the stated resource assumption;
- the main fatal risk or assumption;
- one smallest decisive feasibility gate;
- a verdict: hard pass, conditional pass, or reject.

Use a score of 7 or higher for novelty only when you found no same-design work after the
required searches and can state precisely what remains different. If any search source fails,
record the failure and keep the novelty judgment provisional.

Audit the mathematical validity of A3 and A4, including selection-induced Type-I error,
training-data independence, the reachable hypothesis family, max-statistic correction, and
whether A4's claimed necessity result is more than a direct `argmax` corollary.

For A1, distinguish the archived report's claimed pilot from reproducible evidence. Check
whether the scripts and outputs needed to reproduce it are actually present.

For I1 and A2, distinguish general leakage or memorization papers from papers that match the
proposed intervention and estimand.

## Final decision

Produce one ranked table for all five ideas. State which single idea should be pursued, which
idea is the best recommender-specific alternative, and what evidence would reverse the top
choice. Do not preserve the current ranking merely for consistency.

Write the complete report to:

`idea-stage/CLAUDE3_FIVE_WAY_REVIEW_20260910.md`

Do not modify any other project file. Do not write experiment code or run a pilot. Do not
claim experimental results. End your response with a short statement confirming the report
path and your top-ranked idea.
