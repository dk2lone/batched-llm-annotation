# CPU analysis reproduction

This package reproduces the statistical analyses of **When LLM Labels Fail Together** from
stored error arrays. It compares joint annotation with target-only calls showing the same
documents and order. No new model inference is needed.

## Evidence

| Path | Contents |
|---|---|
| `evidence/covariance_grid/` | Error arrays for the 16-cell grid, assignment replays, and recovered-output sensitivity |
| `evidence/batch_size/` | Error arrays and execution counts for four cells at batch sizes 2, 4, 8, 16, and 32 |
| `evidence/repeated_sampling/` | Original and recovered error arrays, recovery masks, and fixed bootstrap seeds for four cells |
| `evidence/dataset_indices/` | Public source indices, document hashes, reference labels, lengths, strata, and Civil Comments toxicity scores |
| `protocol.json` | Prompts, model and dataset revisions, decoding settings, conditions, and recovery constraints |
| `src/batched_annotation/` | Statistical analyses, prompts, parsing, dataset verification, and figure generation |
| `tests/` | Numerical reproduction and protocol checks |

Arrays use `0` for agreement with the dataset reference, `1` for disagreement, and `NaN` for
an unresolved output. Dataset text, model weights, private cloud configuration, and credentials
are not included. These arrays reproduce the analyses but do not independently validate the
original model execution or parsing. Historical execution code and compact provenance remain
in the parent repository's `study-120h/` directory.

## Install and run

Use Python 3.11--3.13 and [uv](https://docs.astral.sh/uv/). From this directory:

```sh
uv sync --locked --extra dev
uv run reproduce all
uv run pytest
uv run ruff check .
```

The complete command writes results and figures under ignored `outputs/`. Dependencies,
stored replays, and bootstrap seeds are pinned. It uses CPU only and does not download
datasets or launch cloud resources.

Individual commands are:

```sh
uv run reproduce covariance-grid
uv run reproduce batch-size
uv run reproduce repeated-sampling
uv run reproduce reference-sensitivity
uv run reproduce followups
uv run reproduce figures
uv run reproduce check-gpu-protocol
```

Run `figures` after the result commands. `check-gpu-protocol` validates configuration and
evidence paths without downloading models or starting a GPU.

## Results and interpretation

The grid's equal-cell covariance contrast is `0.00961043`. All 16 cells are Holm-significant
against zero, while 13 point estimates separately meet the prespecified `0.002` reporting
threshold. The `0.0002` raw p-value is the minimum attainable with 4,999 replays. Replay
ranges hold outputs fixed and do not measure uncertainty over new items, prompts, or models.

The mean batch-size slope is `-0.00909521` per doubling. This is descriptive. Its three
complete-assignment slopes are `-0.00960244`, `-0.00879470`, and `-0.00888848`. Stored
block-bootstrap intervals remain labeled sensitivities because their resampling does not
preserve recurring item identity across assignments.

The legacy output field `excess_design_effect` equals
`1 + (b - 1) * (rho_joint - rho_exact)`. It is not the total joint-call design effect and
is not a joint/control variance ratio. The total design effect is
`1 + (b - 1) * rho_joint`; the camera-ready table exporter calculates this separately.
Repeated-sampling variance ratios are measured across fresh annotation runs and are a
third quantity.

Recovered item-level Wald / call-mean t / whole-call bootstrap coverage is:

| Cell | Item Wald | Call t | Call bootstrap |
|---|---:|---:|---:|
| Granite / IMDb | 0.7775 | 0.9300 | 0.9175 |
| Llama / Civil Comments | 0.9675 | 0.9650 | 0.9400 |
| Mistral / MultiNLI | 0.9075 | 0.9525 | 0.9475 |
| Qwen / Civil Comments | 0.8800 | 0.9600 | 0.9525 |

Coverage is secondary, post-execution sensitivity evidence. Recovery completed 1,095 calls
using modified output-format and response-length settings while retaining successful original
outputs. The outputs include exact binomial intervals, reference-uncertainty checks, and
repair-free subsets. Neither call-aware method is universally calibrated.

`reference-sensitivity` recomputes Civil Comments analyses at thresholds 0.4, 0.5, and 0.6,
and after excluding scores strictly between 0.4 and 0.6. These are post-hoc checks.

## Optional dataset checks

To download the original datasets and verify the indexed rows, hashes, references, and lengths:

```sh
uv sync --locked --extra data
uv run python -m batched_annotation.datasets civil_comments
uv run python -m batched_annotation.datasets ag_news
uv run python -m batched_annotation.datasets multi_nli
uv run python -m batched_annotation.datasets imdb
```

These checks need network access and use the revisions in `protocol.json`. They do not need
a GPU and are not required for reproducing the saved analyses.

Code is under the MIT License in `LICENSE`. Dataset and model use remains subject to the
original providers' licenses.
