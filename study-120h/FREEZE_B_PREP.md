# Freeze B preparation: throughput-informed expansion

Updated: 2026-09-12. Status: **superseded preparation record**. The current CPU scientific rules are in [FREEZE_B_PROTOCOL.md](FREEZE_B_PROTOCOL.md); its seal status is recorded separately. No Freeze B model output or paid benchmark has been generated. The sealed 20260912T041633Z Step 0 and completed Step 1 are unchanged. The historical Qwen3-8B/Civil Comments Step 2 remains the first joint-result anchor, but must wait until the independent Freeze B scientific choices are sealed.

## Why a second freeze is needed

The Step 1 solo screen used 0.190 metered A100-hours for 10,240 calls, but it did not measure batch-16 or batch-32 prefill and decode costs. It also failed the planned Mistral model axis. Therefore neither the old 120-hour allocations nor the new 16-cell/5,000-replicate aspirations are a credible runtime forecast. The new study targets 120 cumulative metered A100-hours and stops before 150, using a fixed scientific expansion order. It cannot promise to consume 120 hours until long-context timings are measured; no calls should be added solely to spend GPU time.

## Ordered candidate and task registry

These are public Hugging Face API revision hashes checked on 2026-09-12. They are candidate pins, not admission decisions. Before sealing, verify tokenizer chat templates, vLLM compatibility, licenses, A100 memory fit, context limits, and access under the exact serving image. Gemma access is manual and its failure to be available is a technical exclusion, not a capability failure.

| order | model policy | revision | access |
|---:|---|---|---|
| 1 | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | public |
| 2 | `allenai/OLMo-2-1124-7B-Instruct` | `470b1fba1ae01581f270116362ee4aa1b97f4c84` | public; model length must not exceed declared 4,096 |
| 3 | `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | public |
| 4 | `microsoft/Phi-3.5-mini-instruct` | `2fe192450127e6a83f7441aef6e3ca586c338b77` | public |
| 5 | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | public |
| 6 | `allenai/OLMo-2-1124-7B` | `7df9a82518afdecae4e8c026b27adccc8c1f0032` | public; matched base/instruct policy only if its solo screen passes |
| 7 | `ibm-granite/granite-3.3-2b-instruct` | `707f574c62054322f6b5b04b6d075f0a8f05e0f0` | public |
| 8 | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` | license acceptance required; never bypass access |

The first four policies span four families without depending on gated Gemma. A failed Mistral cell from Step 1 is historical evidence and will not be silently reclassified. An OLMo retest requires a corrected serving policy in this new freeze; it does not alter the Step 1 load-failure record. BF16 is the common primary precision. No single-family quantization substitution is allowed.

| order | task policy | dataset/revision | frame and label rule to freeze |
|---:|---|---|---|
| 1 | toxicity | `google/civil_comments` / `f2970eb3a55777454c94069077cc8d9b5866312d` | validation; toxicity >= 0.5 |
| 2 | topic | `fancyzhx/ag_news` / `eb185aade064a813bc0b7f42de02595523103ca4` | train; original four classes |
| 3 | inference | `nyu-mll/multi_nli` / `da70db2af9d09693783c3320c4249840212ee221` | validation_matched; original three classes |
| 4 | sentiment | `stanfordnlp/imdb` / `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` | test; original binary class; cap item length before sampling |
| 5 | multilabel emotion | `google-research-datasets/go_emotions` / `add492243ff905527e67aeb8b80c082af02207c3` | simplified train; exact label list, multiple-label grammar, and reference matching must be frozen |
| 6 | entity extraction | `Babelscape/multinerd` / `2814b78e7af4b5a1f1886fe7ad49632de4d9dd25` | English train; exact entity-span canonicalization and scoring must be frozen; noncommercial-sharealike license |

Pinned-revision streaming checks found IMDb fields `text,label`, GoEmotions simplified fields `text,labels,id` with 28 named labels, and MultiNERD fields `tokens,ner_tags,lang`. MultiNERD's training stream begins with German rows, so English must be selected explicitly; its integer tags need a pinned mapping before any prompt is written. The last two policies cannot yet be called ready: a single-label covariance test does not automatically apply to a multilabel set or an entity-span list. Freeze a binary item-level *disagreement* indicator for each and retain task-specific accuracy separately. Confirm split sizes, data access, use rights, and row reconstruction before sealing. Publish only IDs and hashes when text redistribution rights are uncertain. Civil Comments rows seen in Gate A or the original Step 0 anchor must be excluded from Freeze B confirmation. The original Qwen/Civil anchor uses its sealed rows unchanged.

## Scientific rules that must be sealed before any joint result

1. Materialize all six candidate row frames and the filler pools required by a declared intervention on CPU. Pin row IDs, content hashes, exclusion lists, train/test roles, prompt grammar, output parser, model/tokenizer revisions, and exact seed streams. Screening rows and all confirmation rows must be disjoint; no pilot or historical anchor row may enter Freeze B Qwen/Civil confirmation.
2. Freeze a solo-only screen: exact tokenizer context fit for every planned batch size and condition, first-response parse rate >= 0.98, task-specific reference-accuracy floors, and an error-information rule set by CPU simulation. Civil 0.70, AG News 0.70, and MultiNLI 0.55 are existing Step 0 floors; IMDb, GoEmotions, and MultiNERD floors still require explicit pre-screen calibration. A zero-fit or failed candidate is reported, not hidden.
3. Freeze one deterministic balanced matching algorithm before solo outputs. It must choose up to 16 primary and 12 held-out admitted cells, count each family and task at most once per round before repeat, and maximize distinct family/task coverage under fixed tie breaks. If four families or four task families are unavailable, narrow the recurrence claim before joint calls. Never use a joint result to select cells.
4. Freeze the item-count rule (2,048, 4,096, or 8,192 distinct items) using solo-only error rates and CPU power simulations. Preserve the same item IDs across conditions, three balanced assignments, and a condition-interleaved execution schedule. A cell whose eligible frame cannot support its assigned size is a technical exclusion.
5. Fix the E2 primary signed joint-minus-exact covariance estimator, call-level randomization/bootstrap, multiplicity family, 0.002 absolute effect floor, malformed-output bounds, and full recurrence gate. Fix the six E3 cells, four discovery and four held-out E4 cells, six primary and six held-out E5 cells, and all fallback ordering using the balanced algorithm before any E2 output. The original Step 0 test code can be reused only after it is tested against these new data types and designs.
6. Specify actual E4 prompt interventions, especially whether earlier decisions are visible in the same decoding context, and ensure exact-context controls hold neighbors and position fixed. Freeze the batch-2/4/8/16/32 dose arms and two prompt variants. If an intervention cannot be implemented without changing another factor, label its inference as limited before results.
7. Fix E5's independent 250-reference and 400/1,000/2,500/5,000-evaluation prefixes per cell, 512 item appearances and 32 new joint calls per replicate, no output reuse, empirical coverage target, item-level and call-aware intervals, reference-target uncertainty sensitivity, and exact numerical coverage gate. The first six cells at full precision require 1,008,000 joint calls, which could exceed 150 hours. Secondary Type I/power claims need separate truth and null generators and an explicit call ledger.
8. Freeze a complete expansion queue, seed schedule, and a 150-hour meter that sums **all** A100 worker seconds, including loads and failed calls. Block admission may use throughput and technical status only. At 120 hours stop after the current complete affordable unit; never start a unit whose 1.6x forecast plus reserve exceeds 150. Preserve partial calls and mark unrun blocks.

## Paid-work boundary and benchmark handoff

Before paid work, the remaining CPU tasks are the exact frames, parsers, simulations, selection algorithm, execution ledger, and tests above. Seal those scientific rules as **Freeze B protocol part 1** before new model output. Then run a separately authorized **synthetic** short-, batch-16-, and batch-32-context timing benchmark on the pinned models and engine. Record prefill, decode, wall time, metered A100 time, and failure rates; synthetic outputs never enter science. Use measured tokens and all-in seconds per call to seal base/1.6x/3x cost projections in a separate immutable runtime ledger. Next run solo-only screens, apply the frozen selection algorithm, and seal **Freeze B admission lock part 2** with the exact panel, item counts, and block order. Only after that lock may the original Step 2 joint anchor or any Freeze B joint condition open. Reforecast at every block boundary. No Step 2 or paid benchmark has been launched by this preparation record.

The Modal token previously exposed in chat must be rotated. Configure authentication locally when paid work is ready; do not paste a token into chat or place it in a research artifact.
