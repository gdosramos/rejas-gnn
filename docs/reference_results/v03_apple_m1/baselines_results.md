# Non-neural baselines (same spatial folds, seeds and pooled OOF metrics as the battery)

| Baseline       | PR-AUC        | ROC-AUC       |   Brier cal |   BSS cal |
|:---------------|:--------------|:--------------|------------:|----------:|
| prevalence     | 0.071 ± 0.009 | 0.440 ± 0.063 |      0.0725 |    -0.011 |
| table_road_cds | 0.202 ± 0.025 | 0.805 ± 0.018 |      0.071  |     0.011 |
| logreg_all     | 0.286 ± 0.024 | 0.871 ± 0.012 |      0.0677 |     0.056 |
| logreg_no_road | 0.197 ± 0.018 | 0.772 ± 0.013 |      0.072  |    -0.003 |
| logreg_no_loc  | 0.218 ± 0.018 | 0.822 ± 0.016 |      0.0719 |    -0.002 |
