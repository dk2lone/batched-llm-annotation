# Blind seven-idea dossier

This dossier contains technical proposals only. It intentionally omits prior novelty scores,
rankings, literature conclusions, and acceptance estimates.

## K: Distance concentration and generic LLM tabular failure

Test whether the drop in generic decoder-only LLM accuracy on in-context tabular prediction as
dimension rises is caused by concentration of distances between the query and demonstration-row
representations, rather than by dimensionality itself. The decisive intervention compares
high-dimensional datasets engineered to have low representation-space concentration against
matched datasets where distances concentrate. Measure prediction accuracy and relative contrast
in the model's own row representations. Use row-level attention only as descriptive evidence;
include causal removal of nearest rows, prompt-row permutation, and nearest-row ordering controls.

## A: Batched LLM annotation dependence

Test whether labeling multiple documents in one prompt induces within-batch covariance among
annotation errors even when model, prompt, decoding, and item population are fixed. Randomize the
same items into multiple batch assignments while sweeping batch size. Estimate error-indicator
intra-batch correlation and its effect on confidence-interval coverage and Type-I error for
downstream estimators that consume LLM labels. Controls include length-matched unlabelled filler,
slot position, greedy decoding, fake-batch permutation, and evaluation of every item at every
batch size. Any cluster-robust correction is a baseline, not the proposed novelty.

## T: Within-row redundancy and tabular likelihood pathology

Test whether the rarity of likelihood-based out-of-distribution failures in native tabular data,
relative to image or text-derived embeddings, is explained by a lack of measurable within-sample
redundancy rather than dimensionality or ordinary feature correlation. Reproduce likelihood
pathology rates on native tabular and embedding subsets, develop or justify a redundancy measure
with enough resolution for short mixed-type rows, and construct controls that separate redundancy
from correlation. Byte compression and a marginal-model likelihood ratio are excluded as novel
methods because they are respectively unsuitable at this scale and established prior techniques.

## A2: Factorial content versus co-occurrence pretraining exposure

Build a synthetic recommendation catalogue with unfamiliar item identifiers, content attributes,
and a separately generated latent preference graph that creates item co-occurrences. Randomize
items into a 2 by 2 continued-pretraining exposure design: neither source, content documents only,
co-occurrence documents only, or both, with multiple doses. Verify acquisition using closed-book
probes, then fine-tune on interactions that exclude held-out target transitions. Estimate the two
main effects and their interaction using the randomized assignment to decompose what pretraining
contributes to recommendation.

## A3: Validation exposure and agentic Type-I error

Measure whether training exposure to hidden validation outcomes inflates the false-positive rate
of an LLM agent that adaptively chooses statistical tests. Generate many dataset instances under a
known global null, each with discovery and hidden validation splits. Use direct in-context
validation access as a positive control. For weight-level exposure, avoid a one-model-per-arm
confound by randomizing instance-level exposure within a shared model or using several independent
model seeds per arm. Compare validation exposure, discovery-only exposure, and no exposure while
holding the interface and reachable test family fixed; measure recall of validation noise,
realized Type-I error, fixed-selector controls, and power on planted alternatives.

## A1: Repository provenance replay

Determine whether active public-repository entries derived from the same canonical dataset contain
undocumented preprocessing differences that change model comparisons. Starting from checksummed
canonical files, reconstruct each repository entry with an explicit transformation graph and
require exact row- and column-level agreement. Then intervene on one named transformation at a time
under fixed folds, seeds, population, and models to attribute changes in performance or winner.
The proposed contribution is exact provenance reconstruction plus causal per-transformation
attribution, rather than merely observing that preprocessing affects rankings.

## I1: Non-target future-review leakage

Measure how much review-aware recommender accuracy changes when user and item text profiles contain
reviews written after the historical prediction cutoff. Freeze the interactions, test targets,
candidate set, split, and model settings. Remove every held-out target review in all conditions,
then compare past-only text with later non-target text and separately add later reviews by the same
user and later reviews about the same item. The proposed contribution is the residual non-target
future-text effect and a timestamp-safe evaluation protocol, not the already-known use of the
unavailable target review.
