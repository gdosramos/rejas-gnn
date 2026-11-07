# Closures by registration cohort (logistic regression, block A, spatial folds)

## Profile

| cohort    |    n |   dead_end_share |   living_street_share |   median_length_m |
|:----------|-----:|-----------------:|----------------------:|------------------:|
| 2012-2015 |  109 |            0.312 |                 0.835 |            90.58  |
| 2022-2025 |  108 |            0.185 |                 0.694 |            82.065 |
| open      | 2573 |            0.038 |                 0.248 |            52.15  |


## Stratified evaluation and cross-cohort transfer (mean ± SD over seeds)

| Trained on closures of   | Evaluated on closures of   | ROC-AUC       | PR-AUC (chance)       |
|:-------------------------|:---------------------------|:--------------|:----------------------|
| both cohorts             | 2012-2015                  | 0.894 ± 0.017 | 0.200 ± 0.022 (0.041) |
| both cohorts             | 2022-2025                  | 0.847 ± 0.013 | 0.145 ± 0.018 (0.040) |
| 2012-2015                | 2012-2015                  | 0.871 ± 0.011 | 0.183 ± 0.023 (0.041) |
| 2012-2015                | 2022-2025                  | 0.807 ± 0.019 | 0.125 ± 0.013 (0.040) |
| 2022-2025                | 2012-2015                  | 0.848 ± 0.030 | 0.183 ± 0.028 (0.041) |
| 2022-2025                | 2022-2025                  | 0.800 ± 0.016 | 0.134 ± 0.017 (0.040) |


## Transfer gap: training on the target cohort vs on the other cohort

| Evaluated on   |   Own cohort − other cohort (ROC-AUC) | 95 % CI          |
|:---------------|--------------------------------------:|:-----------------|
| 2012-2015      |                                 0.023 | [+0.005, +0.040] |
| 2022-2025      |                                -0.007 | [-0.020, +0.006] |

