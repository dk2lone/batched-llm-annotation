# Results release policy

Source code, tests, plans, lockfiles, manifests, and readable analysis reports belong in Git. Immutable frozen inputs and raw result payloads belong in milestone GitHub Releases with SHA-256 checksums.

## Current prerelease

The `v0.1.0-step1` prerelease records feasibility and screening evidence only. It contains:

- the complete Step 0 freeze history, including the sealed active protocol;
- all Step 1 preflights, host records, raw solo ledgers, screening reports, billing records, and the execution amendment;
- the complete Gate A pilot artifacts;
- the complete Gate K stop-decision artifacts;
- a release manifest and checksums for every archive.

These files do not contain Step 2 joint-condition outputs or a paper-scale confirmation.

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
