# Rejas-GNN · Predicting residential street closures in Pedro Aguirre Cerda

[![CI](https://github.com/gdosramos/rejas-gnn/actions/workflows/ci.yml/badge.svg)](https://github.com/gdosramos/rejas-gnn/actions)
[![python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](#reproduce)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

In Pedro Aguirre Cerda (Santiago de Chile), residents have gated hundreds of alleys and streets
(*rejas*). Using a survey of all 2 790 street segments of the comuna (217 social closures, 7.8 %),
this repository asks a falsifiable question: **can a model that has never seen a neighbourhood
predict which of its segments are gated, and which kind of information makes that possible?**
Models are evaluated with spatial block cross-validation with 200 m buffers, so that no score can
be obtained by copying the label of a neighbouring alley.

> **Status (v0.3, in progress).** Non-neural baselines, paired statistical comparisons and a
> Brier skill score were added; [`REPORT.md`](REPORT.md) states the findings they support. Pending:
> confirming the provenance of the closure dates and results stratified by cohort.

## Current results

Test blocks, pooled out-of-fold predictions, mean ± SD over seeds 42–51 (same folds for every row; Apple M1).
BSS = Brier skill score of the calibrated probabilities against predicting the base rate
(BSS ≤ 0: the probabilities are not more informative than the prevalence).

| Model | Features | PR-AUC | ROC-AUC | BSS (calibrated) |
|---|---|---|---|---|
| Constant prevalence | – | 0.071 | 0.440 | ≈ 0 by construction |
| Lookup table P(y \| road class, dead end) | 2 | 0.202 ± 0.025 | 0.805 ± 0.018 | 0.011 |
| Logistic regression | 28 (A) | **0.286 ± 0.024** | **0.871 ± 0.012** | **0.056** |
| Logistic regression, no road class | 18 | 0.197 ± 0.018 | 0.772 ± 0.013 | −0.003 |
| Logistic regression, no distance to centroid | 27 | 0.218 ± 0.018 | 0.822 ± 0.016 | −0.002 |
| MLP | 28 (A) | 0.211 ± 0.023 | 0.787 ± 0.061 | −0.067 |
| GraphSAGE | 28 (A) | 0.206 ± 0.031 | 0.755 ± 0.048 | −0.080 |
| GCN / GATv2 | 28 (A) | 0.12–0.13 | 0.61–0.64 | −0.12 / −0.17 |

Chance PR-AUC is 0.078. Feature block A = 28 topological and geometric attributes of the segment;
see [`REPORT.md`](REPORT.md) for the zonal socio-economic block B and the full ablation.

## Reproduce

```bash
git clone https://github.com/gdosramos/rejas-gnn.git && cd rejas-gnn
conda create -n rejas-gnn python=3.11 -y && conda activate rejas-gnn
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt   # exact versions of the reference run
python -m pip install -e ".[dev]" --no-deps
python -m pytest                       # graph, no-leakage features, buffer, determinism, calibration
make baselines                         # ~1 min  → results/baselines/baselines_results.md
make run-all                           # 20–40 min on CPU → results/battery/battery_results.md
make paired                            # paired comparisons with CI → results/paired/
make figures                           # after run-all → results/figures/
```

Every run writes its configuration, library versions and metrics to JSON; results are
bit-for-bit reproducible on one machine and agree to the third decimal across machines
(Linux x86 and Apple M1 references in [`docs/reference_results/`](docs/reference_results/)).

## Where things are

| Path | Content |
|---|---|
| [`REPORT.md`](REPORT.md) | the narrative report: question, data, validation design, findings |
| [`docs/guia_paso_a_paso.md`](docs/guia_paso_a_paso.md) | step-by-step guide in Spanish, with the reading of every output |
| [`docs/spatial_cv.md`](docs/spatial_cv.md) | why spatial block CV with buffers |
| `configs/` | one YAML per experiment; every parameter lives here |
| `src/rejas_gnn/data/` | graph construction, features, spatial splits |
| `src/rejas_gnn/models/` | MLP, logistic, GCN, GraphSAGE, GATv2 |
| `src/rejas_gnn/training/pipeline.py` | the battery: ablation × models × seeds, calibration, maps |
| `scripts/run_baselines.py` | non-neural baselines under the identical protocol |
| `data/raw/pac.csv` | the field survey (versioned) |
| `data/external/` | Census 2024 and MBHT 2017 covariates per segment (versioned) |

## Citation

See [`CITATION.cff`](CITATION.cff). MIT License.
