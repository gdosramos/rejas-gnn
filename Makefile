# Deterministic pipeline automation. Every target fixes seeds through configs/*.yaml.
PYTHON ?= python3
CONFIG_DIR := configs
MODELS := logreg mlp gcn sage gat
GRAPH_VARIANTS := sage_union sage_radius
export PYTHONHASHSEED := 0
export CUBLAS_WORKSPACE_CONFIG := :4096:8

.PHONY: help install install-dev test lint data run all graphs compare run-all baselines paired figures clean

help:
	@echo "make install      install package and dependencies"
	@echo "make test         run unit tests"
	@echo "make lint         ruff + mypy"
	@echo "make data         build and cache the PAC line-graph dataset"
	@echo "make run MODEL=sage   run one model (configs/<MODEL>.yaml)"
	@echo "make all          run all models on the line graph"
	@echo "make graphs       run GraphSAGE on radius and union graphs"
	@echo "make baselines    non-neural baselines, same folds as the battery"
	@echo "make run-all      full battery (ablation x models x seeds 42-51)"
	@echo "make paired       paired comparisons over seeds (needs run-all + baselines)"
	@echo "make figures      figures; requires make run-all first"
	@echo "make compare      aggregate results/*/summary.json into results/comparison.md"
	@echo "make clean        remove caches and results"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests
	mypy src

data:
	$(PYTHON) scripts/build_dataset.py --config $(CONFIG_DIR)/base.yaml

run:
	rejas-gnn run --config $(CONFIG_DIR)/$(MODEL).yaml

all: data
	@for m in $(MODELS); do rejas-gnn run --config $(CONFIG_DIR)/$$m.yaml || exit 1; done
	$(MAKE) compare

graphs:
	@for m in $(GRAPH_VARIANTS); do rejas-gnn run --config $(CONFIG_DIR)/$$m.yaml || exit 1; done
	$(MAKE) compare

run-all:            ## Full Phase-2 battery: ablation A/B x {mlp,gcn,sage,gat} x seeds 42-51
	$(PYTHON) scripts/run_battery.py --config $(CONFIG_DIR)/base.yaml --out results/battery

baselines:          ## Non-neural baselines under the identical spatial protocol
	$(PYTHON) scripts/run_baselines.py --config $(CONFIG_DIR)/base.yaml --out results/baselines

paired:             ## Paired per-seed comparisons with bootstrap CI and Wilcoxon test
	$(PYTHON) scripts/paired_comparison.py --battery results/battery --baselines results/baselines --out results/paired

figures:            ## Steps 2, 3 and 6 of docs/guia_paso_a_paso.md (needs results/battery from run-all)
	@test -d results/battery || (echo "results/battery missing: run 'make run-all' first" && exit 1)
	$(PYTHON) scripts/fig_01_graph.py && $(PYTHON) scripts/fig_02_splits.py && $(PYTHON) scripts/fig_03_map.py

compare:
	$(PYTHON) scripts/compare_results.py --results results --out results/comparison.md

clean:
	rm -rf data/processed/*.pt results/* .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
