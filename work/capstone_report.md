# Capstone Report — Content Refresh Prioritization

- **Author:** Sodiq Abdulwaris
- **Lane:** Content Refresh Prioritization (freestyle, rooted in the shipped pipeline)
- **Repo:** https://github.com/SodiqAbdulwaris/flyrank-internship-ml
- **Date:** 2026-08-16

## 0. Abstract

Which of a client's existing pages should an SEO editor review first, when 54.2% of pages in
this sample are declining and the obvious signal — keyword search volume — barely correlates
with actual traffic (r = 0.001)? Using FlyRank's 30,000-page anonymized starter export (32
clients, trailing 90-day search and analytics metrics), a Logistic Regression model trained on
pre-decision signals only (visibility, position, age, content depth — never the trend fields
the label is built from) ranks declining pages at Precision@50 = 0.70 on a held-out set of 8
clients the model never trained on, against a base rate of 0.517 and a transparent staleness
rule that manages 0.545. The output is a three-tier, reason-coded review queue an editor can act
on directly — decision-support for where to spend a limited review hour, not a verdict on any
one page.

## 1. Problem framing

**Decision supported:** which page an SEO/content editor should open first in a weekly review
backlog. **Unit of analysis:** one row = one pseudonymized content page. **Output:** a
priority-ranked queue with a confidence tier, a probability, and a human-readable reason code
per page. **Action a human takes:** refresh, expand thin content, fix CTR, or simply monitor —
the editor decides; the queue only orders the backlog. **Cost of a wrong call:** a false flag
wastes an editor's hour on a healthy page; a missed decline lets a traffic-losing page sit
untouched, and because misses compound (ongoing lost traffic) the method needs to favor recall
at the top of the queue without drowning editors in false alarms.

**Why ML helps here:** the pattern is too messy for one if-statement. A depth-2 decision tree
matches a hand rule at the very top of the list, but a transparent staleness-and-visibility rule
only ever fires on 17 of 30,000 pages in this sample — everything past that is an arbitrary tie.
A model that weighs many weak, individually-unremarkable signals together does better where a
single rule runs out of signal.

## 2. Data safety

**Data used:** `data/raw/content_refresh_anonymized.csv` — 30,000 rows, one row per
pseudonymized content item, 32 pseudonymized clients, trailing 90-day metrics as of one export
date. The full FlyRank warehouse (519,606 items, 104 clients) exists but is out of scope for
this capstone.

**Deliberately excluded, and why:**
- `trend_direction`, `trend_pct` — the label's source (`is_declining_label = trend_direction ==
  "down"`).
- `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, `impressions_prev_30d`,
  `clicks_prev_30d`, `sessions_prev_30d` — proven to be `trend_pct`'s literal inputs by
  recomputing it from them (100% match rate), not just correlated with it.
- `provider_used`, `model_used` — content-generation metadata, not a performance signal.
- Raw `impressions_90d`/`clicks_90d`/`sessions_90d`/`ai_sessions_90d` — replaced by their
  `log1p` versions (heavy-tailed).
- `content_id`, `client_id` — context only: grouping and the client-holdout split, never
  features.

**Leakage risk confirmed by attack test:** adding `trend_pct` back into the otherwise-honest
feature set pushed Precision@50 from 0.70 to 1.00 and ROC AUC to 0.999 on the identical split —
confirming both that the excluded column really is that leaky, and that the evaluation harness
actually detects a real leak rather than never finding one.

Nothing client-identifying — no names, domains, URLs, or raw queries — appears anywhere in this
repository or this paper.

## 3. Baseline

**Rule (frozen once model work started):** a page is worth reviewing if it's stale
(`days_since_last_update >= 180`) **and** still visible (`impressions_90d >= 500`), ranked by
how much exposure it has. Simple enough to state in one sentence, and 94% correct on the pages
it actually flags.

**The catch, found by hand-reviewing the top 20:** only 17 of 30,000 pages ever pass both
conditions — 99.4% of this slice was updated within 180 days (median 20 days), so the threshold
barely fires. Everything past rank 17 is a zero-score tie ordered by row position, not by the
rule. Only 3 held-out pages receive a positive score, so 47 of the top-50 slots land inside the
zero-score tie. The evaluation therefore reports expected Precision@K across that tied group,
instead of letting CSV order or a pandas sorting implementation choose the result. On the
client-holdout test set this gives Precision@20 = 0.589 and Precision@50 = 0.545 (base rate
0.517): deterministic, slightly above random selection, and honestly limited by the rule's low
coverage.

## 4. Model / analysis

**Method:** Logistic Regression, then Random Forest, evaluated by `predict_proba` at
Precision@K — the toolkit's guidance for an observed-label ranking task is to start readable and
add complexity only if it earns its keep.

**Features (18 numeric + 8 categorical, one-hot encoded):** `search_volume`, `competition`,
`cpc`, `word_count`, `char_count`, `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`,
`log_ai_sessions_90d`, `days_with_impressions`, `days_with_sessions`, `content_age_days`,
`days_since_last_update`, `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`,
`ai_traffic_pct`, plus `has_keyword_data`/`has_word_count`/`has_scroll_data` missingness flags
(added because `search_volume`/`word_count`/`scroll_rate` go blank by `content_type` — a blind
`fillna(0)` would silently encode content type into the model); `competition_level`,
`content_type`, `main_intent`, `age_tier`, `freshness_tier`, `word_count_tier`,
`impression_tier`, `position_tier`. Left out on purpose: everything in Section 2's exclusion
list.

**Target/proxy, in one sentence:** `is_declining_label = 1` when `trend_direction == "down"` —
an observed proxy computed from one 90-day snapshot's 30-vs-30-day impression comparison, not a
verified future outcome.

## 5. Evaluation

**Split:** `GroupShuffleSplit(test_size=0.25, random_state=42)` grouped on `client_id` — 24
clients train / 8 clients test (22,885 / 7,115 pages), so no client's pages appear on both
sides.

**Why grouped, quantified:** the identical model scored on a random row-level split instead
reads Precision@50 = 0.86 — 16 points higher — because 31 of 32 clients leak across train and
test under a random split. The grouped number (0.70) is the one this paper reports; the gap
itself is a finding about how much the model leans on client-specific pattern versus
generalizable signal.

| Model | ROC AUC | Avg. precision | Precision@20 | Precision@50 |
|---|---:|---:|---:|---:|
| Baseline rule | — | — | 0.589 | 0.545 |
| Logistic Regression | 0.610 | 0.605 | **0.80** | **0.70** |
| Random Forest | 0.603 | 0.587 | 0.55 | 0.56 |

Base rate (test set, 7,115 pages / 8 clients): **0.517**.

**Errors:** false positives cluster on moderate-traffic pages actually trending up or stable —
they match every pre-decision decliner signal available, but trend history itself is excluded
as leakage, so the model has no way to see which way they're actually moving. False negatives
cluster on very-low-traffic old pages that genuinely are declining but carry almost no signal to
separate from noise.

## 6. Interpretation

Logistic Regression wins outright over both the baseline and Random Forest at Precision@20 and
Precision@50 — reported as the finding, not smoothed into "the model helps." Top Random Forest
feature importances — `days_with_impressions` (0.149), `log_impressions_90d` (0.130),
`avg_position` (0.103), `content_age_days` (0.087), `char_count` (0.046) — are all plausible
correlates of a page's trajectory, and none dominates suspiciously (the top feature is 0.149 of
1.0, not 0.9+), which is the sanity check that matters: a single feature owning nearly all the
importance is usually leakage, and that isn't what this shows.

**Negative result worth stating plainly:** Random Forest, the more complex model, did *not* beat
Logistic Regression here (0.56 vs 0.70 at Precision@50). More complexity was not automatically
better on this feature set and this sample size.

## 7. Recommendation

The Logistic Regression winner, refit on all 30,000 pages, tiered by its own probability:
**high** (≥ 0.70, 5,119 pages / 17.1%), **medium** (0.50–0.70, 11,242 pages / 37.5%), **low**
(< 0.50, 13,639 pages / 45.5%). Reason codes reuse the baseline's human-checkable flags where
they fire (`stale_but_visible`, `thin_content`, `page_one_low_ctr`), plus `model_flagged` when a
high-tier page passes none of them — **75.8% of high-tier pages** carry no simple rule
explanation, an honest label for "the model's combined read of many weak signals says review
this, but no single rule explains why." That's exactly the model's edge over the rule baseline,
and exactly why a human, not a machine, closes the loop.

**Before acting:** confirm the page isn't already scheduled for retirement or consolidation;
for `model_flagged` rows specifically, read the page since no simple rule explains the flag;
confirm the export is recent. **Never automate:** auto-publishing, auto-redirecting, or
auto-deleting content from this queue; exposing the model's probability to a client as a
verdict; judging a content creator's performance from this label.

**Confidence and limits, stated explicitly:** Precision@50 = 0.70 is a client-holdout number on
one 30,000-page snapshot, not a promise for a brand-new client or a different time period — see
Section 8's limitations.

## 8. Reproducibility

**Environment:** Python 3.11, `pip install -r requirements.txt` (pandas, numpy, scikit-learn,
matplotlib, reportlab, duckdb, huggingface_hub). **Random seeds:** `random_state=42` everywhere
a seed applies (split, Logistic Regression, Random Forest).

**To re-run everything from a fresh clone:**

```bash
pip install -r requirements.txt
jupyter execute work/notebooks/capstone.ipynb --inplace
```

The notebook rebuilds the feature set from `data/raw/content_refresh_anonymized.csv`, refits
the baseline and both models on the exact grouped split above, runs the leakage attack test, and
regenerates every chart and number in this report. Receipts are committed at
`work/outputs/capstone_metrics.json` (the metrics this report cites) and
`docs/assets/*.svg` (the charts this report and the deployed page embed) — "evaluated once,
honest" is checkable from this repo, not taken on faith.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai).

---

> **Claims checklist:** every number above is observed / measured / directional /
> decision-support. No causal claims, no "predicted Google's algorithm," no client-identifying
> details. Precision@K is always reported next to its base rate. Numbers in this report match a
> fresh re-run of `work/notebooks/capstone.ipynb`.
