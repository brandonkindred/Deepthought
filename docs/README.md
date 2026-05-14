# Deepthought Documentation

This folder is the canonical entry point for Deepthought engineering
documentation. Each file targets a specific audience — start with the
one that matches your goal.

## Architectural overviews

| File | Audience | What it covers |
|---|---|---|
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Anyone new to the system | C4 context / container / component diagrams, all four subsystems (`/rl`, `/llm`, `/lr`, `/images`), networking & deployment topology, cross-cutting use cases, build/test/run, observability and failure modes |
| [`DATA_MODEL.md`](./DATA_MODEL.md) | Engineers touching Neo4j | Every node label, every relationship type, every property, every verbatim Cypher query the app issues, per-endpoint read/write effects, recommended constraints, dormant schema |
| [`BRAIN_SUBSYSTEM.md`](./BRAIN_SUBSYSTEM.md) | Engineers touching ML logic | Algorithms, constants, and engineering caveats for Brain/QLearn, LanguageModelService, LogisticRegressionService, and ImageProcessingService — plus a cross-subsystem comparison |

## Endpoint deep-dives

| File | Endpoint |
|---|---|
| [`PREDICT_ENDPOINT.md`](./PREDICT_ENDPOINT.md) | `POST /rl/predict` — full request lifecycle, sequence diagram, response shape, error/edge cases |
| [`LEARN_ENDPOINT.md`](./LEARN_ENDPOINT.md) | `POST /rl/learn` — Q-learning math, reward table, Cypher executed, before/after graph state, worked examples |
| [`TRAIN_ENDPOINT.md`](./TRAIN_ENDPOINT.md) | `POST /rl/train` — current behavior (a no-op today) and gap analysis |

## Reference

| File | Purpose |
|---|---|
| [`API_SPEC.md`](./API_SPEC.md) | High-level technical overview targeted at API integrators |
| [`CODE_REVIEW_PLAN.md`](./CODE_REVIEW_PLAN.md) | Known findings and remediation plan |
| [`../openapi.yaml`](../openapi.yaml) | Machine-readable OpenAPI 3 spec |
| [`../Deepthought API.postman_collection.json`](../Deepthought%20API.postman_collection.json) | Postman collection for manual exploration |

## Quick reading paths

- **"I'm onboarding."** Start with the root [`../README.md`](../README.md),
  then [`ARCHITECTURE.md`](./ARCHITECTURE.md) end-to-end.
- **"I need to extend the graph schema."**
  [`DATA_MODEL.md`](./DATA_MODEL.md) first, then the matching
  endpoint deep-dive for any behavior change.
- **"I'm changing the prediction or learning algorithm."**
  [`BRAIN_SUBSYSTEM.md`](./BRAIN_SUBSYSTEM.md), then
  [`PREDICT_ENDPOINT.md`](./PREDICT_ENDPOINT.md) and
  [`LEARN_ENDPOINT.md`](./LEARN_ENDPOINT.md).
- **"I'm deploying this somewhere new."**
  [`ARCHITECTURE.md`](./ARCHITECTURE.md) §8 (networking / topology) and
  [`DATA_MODEL.md`](./DATA_MODEL.md) §7 (recommended constraints).
