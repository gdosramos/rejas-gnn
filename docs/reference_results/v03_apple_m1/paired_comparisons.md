# Paired comparisons over seeds (same spatial partitions)

A difference is supported when its 95 % CI excludes 0. Ten seeds give limited power.

| Comparison (A vs B)                          | Metric   |   Mean diff A−B | 95 % CI          |   Wilcoxon p | A wins   |
|:---------------------------------------------|:---------|----------------:|:-----------------|-------------:|:---------|
| Linear model vs MLP (block A)                | PR-AUC   |           0.076 | [+0.061, +0.092] |        0.002 | 10/10    |
| Linear model vs MLP (block A)                | ROC-AUC  |           0.084 | [+0.053, +0.125] |        0.002 | 10/10    |
| Linear model vs MLP (block A)                | BSS cal  |           0.124 | [+0.084, +0.172] |        0.002 | 10/10    |
| Linear model vs GraphSAGE (block A)          | PR-AUC   |           0.081 | [+0.056, +0.105] |        0.002 | 10/10    |
| Linear model vs GraphSAGE (block A)          | ROC-AUC  |           0.116 | [+0.087, +0.151] |        0.002 | 10/10    |
| Linear model vs GraphSAGE (block A)          | BSS cal  |           0.136 | [+0.087, +0.200] |        0.002 | 10/10    |
| MLP vs GraphSAGE (block A)                   | PR-AUC   |           0.005 | [-0.018, +0.027] |        0.695 | 5/10     |
| MLP vs GraphSAGE (block A)                   | ROC-AUC  |           0.032 | [-0.001, +0.064] |        0.084 | 7/10     |
| MLP vs GraphSAGE (block A)                   | BSS cal  |           0.013 | [-0.031, +0.055] |        0.492 | 5/10     |
| MLP vs 2-variable table                      | PR-AUC   |           0.009 | [-0.002, +0.019] |        0.16  | 7/10     |
| MLP vs 2-variable table                      | ROC-AUC  |          -0.018 | [-0.049, +0.006] |        0.492 | 4/10     |
| MLP vs 2-variable table                      | BSS cal  |          -0.078 | [-0.142, -0.020] |        0.064 | 3/10     |
| Block A vs A+B, MLP                          | PR-AUC   |           0.006 | [-0.016, +0.030] |        0.77  | 4/10     |
| Block A vs A+B, MLP                          | ROC-AUC  |           0.05  | [+0.008, +0.089] |        0.049 | 8/10     |
| Block A vs A+B, MLP                          | BSS cal  |           0.015 | [-0.023, +0.059] |        0.922 | 4/10     |
| Block A vs A+B, GraphSAGE                    | PR-AUC   |           0.025 | [+0.004, +0.047] |        0.084 | 6/10     |
| Block A vs A+B, GraphSAGE                    | ROC-AUC  |           0.064 | [+0.025, +0.094] |        0.027 | 9/10     |
| Block A vs A+B, GraphSAGE                    | BSS cal  |           0.046 | [-0.023, +0.118] |        0.322 | 6/10     |
| GraphSAGE vs GCN (block A)                   | PR-AUC   |           0.084 | [+0.059, +0.107] |        0.002 | 10/10    |
| GraphSAGE vs GCN (block A)                   | ROC-AUC  |           0.144 | [+0.112, +0.174] |        0.002 | 10/10    |
| GraphSAGE vs GCN (block A)                   | BSS cal  |           0.036 | [-0.001, +0.072] |        0.16  | 7/10     |
| Linear: with vs without road class           | PR-AUC   |           0.089 | [+0.079, +0.100] |        0.002 | 10/10    |
| Linear: with vs without road class           | ROC-AUC  |           0.099 | [+0.093, +0.105] |        0.002 | 10/10    |
| Linear: with vs without road class           | BSS cal  |           0.059 | [+0.030, +0.091] |        0.004 | 9/10     |
| Linear: with vs without distance to centroid | PR-AUC   |           0.068 | [+0.061, +0.075] |        0.002 | 10/10    |
| Linear: with vs without distance to centroid | ROC-AUC  |           0.049 | [+0.045, +0.053] |        0.002 | 10/10    |
| Linear: with vs without distance to centroid | BSS cal  |           0.059 | [+0.033, +0.079] |        0.004 | 9/10     |
