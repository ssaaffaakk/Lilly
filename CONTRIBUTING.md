# Contributing to Lilly

Lilly is an evidence-first project. A contribution is welcome when it makes
the product more useful, the measurement more honest, or the artifact easier
to reproduce.

## Before you change something

- Read the root [`README.md`](README.md), [`AGENTS.md`](AGENTS.md), and the
  relevant document under [`docs/`](docs/README.md).
- Keep `origin/main` as the source of truth. Do not commit secrets, tokens,
  `kaggle.json`, `.env` files, or model/data scratch space.
- Put GPU work and multi-hour benchmarks on Kaggle. A local change should be
  a smoke test, audit, or a small reproducible measurement.
- For a new experiment, write the test set, pass bar, and failure condition
  before the run. A `COMPLETE` job is not success if a gate failed.

## Make the change easy to trust

1. Keep the serving path and the evaluation path explicit.
2. Report counts beside percentages and identify the exact artifact measured.
3. Add or update a focused test for behavior changes.
4. Run `.venv/bin/python -m pytest tests -q` and `git diff --check`.
5. Update the relevant README, result file, or roadmap in the same change.

For model or dataset publication, use the repository’s preflight and publisher
scripts. They verify file lists, fingerprints, attribution, and credentials
before anything is uploaded.

## Pull requests

Please explain the user-visible change, the evidence behind it, and any known
limit. Small, focused pull requests are easier to review than a mixed code,
data, and narrative rewrite.
