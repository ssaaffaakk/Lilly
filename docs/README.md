# Lilly docs

![Lilly — Bosnian first](images/lilly-hero.jpg)

The repository has two doors: the root [`README.md`](../README.md) is the
product landing page; this index is the evidence, operating rules, and research
behind it.

## Reading order

| Start here | What you get |
| :--- | :--- |
| [`README.md`](../README.md) | Product overview, live demo, local setup, measured results |
| [`WHITE-PAPER.md`](WHITE-PAPER.md) | IMRaD write-up: system, data, methods, results, limitations |
| [`STORY.md`](STORY.md) | Why Lilly exists and how the first builds became a measured artifact |
| [`V4-PLAN.md`](V4-PLAN.md) | Current queue: what gets measured next and what is closed |

The project is intentionally evidence-first: failures remain next to wins,
training runs stop at known bad states, and a number does not count unless its
test set and serving path are named.

## Live status

| Doc | Use when |
| :--- | :--- |
| [`V4-PLAN.md`](V4-PLAN.md) | **Where we are now** — the lane order for what gets trained next, and what does not |
| [`V3-PLAN.md`](V3-PLAN.md) | Older v3 plan (points to V4 for current status) |
| [`V2-PLAN.md`](V2-PLAN.md) | Older v2 architecture sketch (points to V3 for current status) |
| [`V2-BOUNDARIES.md`](V2-BOUNDARIES.md) | Scope contract — Latin Bosnian, Kaggle not Mac, no secrets |

## Kaggle / training law

| Doc | Use when |
| :--- | :--- |
| [`kaggle-notebooks.md`](kaggle-notebooks.md) | How to write speech/OCR notebooks (wrong vs right) |
| [`kaggle-fail-stop.md`](kaggle-fail-stop.md) | Short list of COMPLETE-past-failure mistakes |
| [`../training/README.md`](../training/README.md) | How to train each ability |

Cursor agents also load `.cursor/rules/kaggle-fail-stop.mdc` (always apply).

## Product / research

| Doc | Use when |
| :--- | :--- |
| [`STORY.md`](STORY.md) | **Why Lilly exists** — founder story |
| [`WHITE-PAPER.md`](WHITE-PAPER.md) | **The paper** — IMRaD write-up: system, data, methods, results, limits. Failures stay in. |
| [`BOSNIAN_METRIC.md`](BOSNIAN_METRIC.md) | Why Bosnian metrics are hard |
| [`OCR-ROADMAP.md`](OCR-ROADMAP.md) | **The reader's queue** — closed lines, numbers to trust, do-not-repeat list, status board |
| [`ROADMAP.md`](ROADMAP.md) | Early phase roadmap (historical; prefer V4-PLAN for now) |
| [`REPORT-reverse-direction-2026-09-12.md`](REPORT-reverse-direction-2026-09-12.md) | The reply direction, scored through the path a user actually meets |
| [`REPORT-what-would-raise-the-numbers-2026-09-13.md`](REPORT-what-would-raise-the-numbers-2026-09-13.md) | Ranked levers that would raise a measured number, and which need no GPU |

## Agent teams

| Doc | Use when |
| :--- | :--- |
| [`agent-teams-lilly.md`](agent-teams-lilly.md) | Lilly-specific team rules (rule 9 = no band-aids) |
| [`agent-teams.md`](agent-teams.md) | Generic team patterns |

## Images

README assets live in [`images/`](images/). Prefer under ~500 KB each.
