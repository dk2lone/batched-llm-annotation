# Batched LLM Annotation

Code and evidence for **When LLM Labels Fail Together**, accepted to the IEEE BigData 2026
High School Symposium. The study asks whether errors in labels generated together in one
LLM call are dependent, even after controlling for the neighboring texts.

[Read the camera-ready paper](manuscript/main.pdf).

## Contents

| Directory | Contents |
|---|---|
| [`manuscript/`](manuscript/) | Five-page camera-ready PDF, LaTeX source, references, figures, and table source data |
| [`analysis/`](analysis/) | CPU reproduction package, analysis-ready error arrays, dataset indices, exact prompts and revisions, and tests |
| [`study-120h/`](study-120h/) | Original implementation, sealed protocols, schedules, compact results, recovery records, and historical reports |
| [`gate-experiments/a/`](gate-experiments/a/) | Earlier feasibility pilot, kept separate from the main study |
| [`refine-logs/EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md) | Original extension design |

Internal handoffs, review emails, editing notes, credentials, private dataset text, and local
cloud-operation logs are not part of the published working tree. Historical scientific records
are retained without rewriting their results. Large earlier run archives remain under
[Releases](https://github.com/dk2lone/batched-llm-annotation/releases).

## Reproduce the analyses

The analysis package includes stored error arrays. It needs Python 3.11--3.13 and
[uv](https://docs.astral.sh/uv/), but no GPU, model weights, API key, or new model calls.

```sh
cd analysis
uv sync --locked --extra dev
uv run reproduce all
uv run pytest
uv run ruff check .
```

The reproduction command writes covariance, batch-size, repeated-sampling, reference-label
sensitivity, and figure results to `analysis/outputs/`. See the
[analysis instructions](analysis/README.md) for individual commands and evidence limits.

The primary grid contains four models by four tasks, with 8,192 items and three assignments
per cell. The batch-size comparison uses four selected cells and sizes 2--32. Coverage results
are a secondary sensitivity analysis: 1,095 malformed or truncated calls were recovered under
a changed output contract. Error means disagreement with a dataset reference.

The camera-ready paper separates tests against zero from the 0.002 descriptive threshold,
treats batch-size trends as descriptive, and distinguishes total joint-call design effects
from joint-minus-control contrasts. Older sealed reports preserve their original wording;
use the paper and analysis instructions for the current interpretation.

## Build the paper

```sh
cd manuscript
python3 build_tables.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Table generation uses saved summaries included in this repository and makes no model calls.
The PDF includes references within the five-page limit. A source archive is available at
[manuscript/source.zip](manuscript/source.zip).

Dataset text and model weights must be obtained from their original providers under their
respective licenses. The CPU analysis code is covered by [analysis/LICENSE](analysis/LICENSE).
