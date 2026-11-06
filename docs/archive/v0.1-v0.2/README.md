# Archive: results and documents of v0.1 and v0.2

These files are kept for traceability only. **They are superseded by v0.3** and should not be
cited; the current results are in [`../../reference_results/v03_apple_m1/`](../../reference_results/v03_apple_m1/)
and are described in [`REPORT.md`](../../../REPORT.md).

Why their numbers differ from v0.3:

| Item | v0.1 / v0.2 | v0.3 |
|---|---|---|
| Aggregation of metrics | v0.1 (`comparison_v0.1.md`, `*_line_pac/`): mean of per-fold metrics; v0.2 (`battery_*`): metrics on pooled out-of-fold predictions per seed | pooled per seed, as in v0.2, plus paired per-seed comparisons |
| Block B | 168 features (one missingness flag per covariate column, 69 near-identical flags) | 101 features (one flag per source) |
| Calibrated threshold metrics | threshold chosen on raw probabilities and applied to calibrated ones (bug: calibrated F1 ≈ 0.02) | threshold chosen on calibrated validation probabilities |
| Baselines | logistic regression only in v0.1, outside the main table | lookup table and logistic regressions under the battery's folds and seeds |
| Probability quality | ECE only | Brier skill score against the base rate |

`README_technical_v0.1.md` was the front page of v0.1; its claims are replaced by `REPORT.md`.
