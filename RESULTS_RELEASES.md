# Results release policy

Source code, tests, plans, lockfiles, manifests, and readable analysis reports belong in Git. Immutable frozen inputs and raw result payloads belong in milestone GitHub Releases with SHA-256 checksums.

## Current prereleases

The `v0.1.0-step1` prerelease records feasibility and screening evidence only. It contains:

- the complete Step 0 freeze history, including the sealed active protocol;
- all Step 1 preflights, host records, raw solo ledgers, screening reports, billing records, and the execution amendment;
- the complete Gate A pilot artifacts;
- the complete Gate K stop-decision artifacts;
- a release manifest and checksums for every archive.

These files do not contain Step 2 joint-condition outputs or a paper-scale confirmation.

The `v0.2.0-interim` prerelease records the later Freeze B and extension checkpoint. It contains:

- the Freeze B part-1 protocol and part-2 admission lock;
- the sealed runtime lock and its source runtime evidence;
- the complete 36-cell solo-screen payload;
- the completed historical Qwen3-8B/Civil Comments bridge;
- the completed first fresh Qwen3-8B/MultiNLI E2 cell;
- all X1 through X1R4 extension protocol seals, the passing five-model synthetic
  batch-invariance preflight, and checkpoint-staging provenance;
- a release manifest and SHA-256 checksums for every archive.

This is an interim prerelease. It does not establish recurrence, confidence-interval coverage, or
a final paper result.

The `v0.3.0-extension-m1` private prerelease records the completed X1R4 A6 solo screen. It
contains the signed runtime/access lock and its evidence, all 40,960 first-response records,
the 20 cell summaries, the first-1,000-attempt release decision, checkpoint inventories,
meter receipts, admission inputs, and checksums. Seven cells in four families passed the
frozen solo gates. It contains no extension joint-condition output and establishes no
dependence or coverage claim.

## Release boundary

`study-120h/freeze/` is gitignored, so seals are distributed only through release archives. The
active Freeze B admission lock is `20260913T022442Z-B2`. The active extension protocol is X1R4,
manifest SHA256 `3838c6e201388480eb08aa48e04a1ef3a5af6a86fac5d920e9837459382677d2`.
X1, X1R1, X1R2, and X1R3 are preserved as explicitly superseded audit history.

The private extension frame text in `study-120h/ext-work/` is also gitignored and must not be
released until the redistribution terms of each dataset are verified. Publish the row identifiers,
content hashes, and the reconstruction recipe instead.

## Final paper release

Create a separate final release only after the confirmatory study and paper are frozen. It must include:

- the exact protocol, sampled data identifiers, assignments, prompts, and model revisions;
- all raw outputs and failure ledgers needed to reconstruct exclusions;
- analysis-ready tables and the scripts and lockfiles that produce every reported statistic;
- manuscript figures and tables in editable and publication formats;
- the submitted manuscript or an exact version identifier;
- a machine-readable manifest and SHA-256 checksum file.

Do not include credentials, local environments, caches, agent traces, or model weights.

Before making a results release public, verify the redistribution terms for every dataset and model output. If source text cannot be redistributed, publish stable row identifiers, content hashes, sampling metadata, and reconstruction code instead.

The current private `v0.1.0-step1` Gate A archive contains copied Civil Comments text. Replace it
with an identifier-and-hash-only archive before changing the repository or release to public,
even though the current Hugging Face dataset record labels Civil Comments as CC0-1.0.
