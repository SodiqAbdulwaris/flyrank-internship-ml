# Week 2 — Readable model vs hand rule (notebook 02, "your turn")

## Experiment
Swept `max_depth` in {2, 3, 4} for a decision tree, scored with a **client-holdout**
split (GroupShuffleSplit, 25% of clients held out entirely) so no client's pages appear
in both train and test. Compared against the Section 1 `stale x visible` hand rule on
Precision@20 and Precision@50.

Environment: 30,000 pages; label `is_declining_label = (trend_direction == "down")`;
`client_id` used only for grouping, never as a feature.

## Results (test pages: 7,115 from 8 held-out clients)

| max_depth | hand@20 | tree@20 | hand@50 | tree@50 |
|---|---|---|---|---|
| 2 | 0.500 | 0.550 | 0.620 | 0.580 |
| 3 | 0.500 | 0.500 | 0.620 | 0.560 |
| 4 | 0.500 | 0.600 | 0.620 | 0.600 |

## Observation (observed / directional)
On held-out clients, a depth-2 to depth-4 decision tree and the `stale x visible` hand
rule perform similarly. The hand rule holds at Precision@50 (0.620) and only loses at
Precision@20 when the tree depth is 4 (0.600 vs 0.500). Deeper trees (depth 3) actually
score *worse* at @50 than the depth-2 tree — they overfit to training clients. So the
in-sample "model beats rule" gap seen earlier largely disappears once clients are held
out. Directional only; no causal claim about any ranking algorithm.

## Lesson reinforced
Prefer a model you can read (`export_text`) and validate honestly (client-holdout). The
label is derived from `trend_direction`/`trend_pct`, so those columns are never features
(leakage — shown in Section 3, where a leaky tree hits Precision@50 = 1.000 by splitting
on `trend_pct`).
