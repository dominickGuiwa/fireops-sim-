# Benchmark results

This directory contains one checked-in, script-generated baseline smoke run:

- [`benchmark.json`](benchmark.json) contains configuration and software
  provenance, all 80 raw episodes, and recomputable grouped summaries;
- [`benchmark.csv`](benchmark.csv) contains the eight flat policy-reliability
  summary rows.

It used 10 episodes per cell, base seed `2026`, paired episode seeds across
reliabilities, no injected failures, and the random and rule policies only. The
mean snapshot is:

| Reliability | Random reward / affected area | Rule reward / affected area |
| ---: | ---: | ---: |
| 100% | -7.24 / 5.40 | -3.08 / 3.70 |
| 75% | -7.24 / 5.40 | -3.59 / 4.10 |
| 50% | -7.24 / 5.40 | -3.31 / 3.90 |
| 25% | -7.24 / 5.40 | -4.33 / 4.80 |

These are actual outputs, not placeholder scores. They are also deliberately
limited: `n=10` is small, no PPO model was evaluated, no chaos event was
injected, and the recorded runner is Python 3.11.4 while the supported release
floor is Python 3.12. They prove that the benchmark path executes; they do not
establish policy superiority, degradation sensitivity, or robust
generalization. Rerun under Python 3.12 with a preregistered seed budget and
appropriate uncertainty analysis before citing results.

From the repository root, run:

```bash
python experiments/benchmark.py --output-dir results
```

The default run evaluates the random and rule-based policies at sensor
reliabilities `1.00`, `0.75`, `0.50`, and `0.25`. Add trained policies only when
their model artifacts exist:

```bash
python experiments/benchmark.py \
  --nominal-model models/ppo_nominal.zip \
  --robust-model models/ppo_robust.zip \
  --output-dir results
```

Each run writes a JSON document containing metadata, every episode, and grouped
summary statistics, plus a flat CSV containing the same grouped summaries. The
reported metrics are:

- cumulative episode reward;
- critical infrastructure preserved;
- fire-affected grid cells (`burned_area` includes burning and burned cells); and
- response resources consumed.

The same episode seeds, reset-time failure schedule, and policy-side seeds are
reused at every reliability level. This paired design changes sensor quality
without also changing the initial synthetic worlds or random-policy action
streams. When chaos is enabled, every episode records the actual seeded failure
type, and summaries include per-type counts. Seeds, episode counts, privacy-safe
model labels and hashes, reliability levels, and runner settings are recorded
with the result. A model is never silently substituted with a fallback policy.
Generated JSON and CSV files are research artifacts; review their provenance
before publishing them. Never hand-edit a score. Regenerate the complete pair so
the raw episodes and summaries stay consistent.
