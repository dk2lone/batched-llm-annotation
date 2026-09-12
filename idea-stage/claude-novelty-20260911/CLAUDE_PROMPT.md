/novelty-check

Perform a fresh, independent literature review and novelty audit of all seven proposals in
`/Users/dz/bigdata/idea-stage/claude-novelty-20260911/BLIND_IDEA_DOSSIER.md`.

Independence rules:

1. Read only that dossier for the proposal descriptions. Do not open `ideas.md`, any file whose
   name contains `NOVELTY`, `REVIEW`, or `COMPARISON`, or any existing literature/search artifact
   in this project.
2. Ignore any novelty score, ranking, or prior-work claim that appears outside the dossier.
3. Conduct your own live literature searches. Do not infer novelty from the dossier's wording.
4. Do not edit any files. Return the complete report as Markdown in your final response.

Search requirements:

- Extract three to five potentially novel technical claims for each proposal.
- For each claim, try at least three distinct query formulations.
- Search arXiv, OpenAlex, and Semantic Scholar, plus ICLR 2025/2026, NeurIPS 2025, and ICML
  2025/2026. Include the most recent six months of arXiv through 2026-09-11.
- Open and inspect the abstract, method, or related-work text for plausible collisions. A search
  snippet alone is not enough for a collision judgment.
- Record unavailable, blocked, or rate-limited sources. Never treat a failed query as zero hits.
- Prefer primary paper pages and provide a direct URL for every paper used in a verdict. Verify
  title, year, and venue before citing it.
- Search adjacent terminology, not only the proposal's chosen labels.

Evaluation requirements:

- Be strict. Applying an established method to a new dataset is not a new method.
- Separate method novelty, experimental-design novelty, and potentially novel empirical findings.
- Identify the closest prior work, exact overlap, defensible delta, and strongest reviewer
  objection for every proposal.
- Give each proposal a novelty score from 0 to 10 and contribution-strength score from 0 to 10.
  Do not estimate acceptance probability.
- Rank all seven primarily by defensible novel contribution. Feasibility and venue fit may break
  close ties but must not override a novelty collision.
- State PROCEED, PROCEED WITH CAUTION, REPAIR THEN PROCEED, or ABANDON for each.
- End with a comparison to no pre-existing ranking: clearly state which proposal you independently
  rank first and the exact result it would need to establish to justify a paper.

Use this structure:

# Independent Claude novelty audit

## Search coverage and limitations
## Ranking
## K
## A
## T
## A2
## A3
## A1
## I1
## Cross-idea conclusion

For each idea section include core claim verdicts, a closest-prior-work table, score reasoning,
recommendation, defensible positioning, and kill criterion. Keep the final report under 8,000
words while preserving concrete citations and search limitations.
