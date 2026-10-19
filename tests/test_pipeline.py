"""Phase-3 pipeline tests: tensor dimensions, absence of NaNs, seed conservation, export."""
from pathlib import Path

import numpy as np
import pytest
import torch

from rejas_gnn.config import DataConfig, ExperimentConfig, GraphConfig, ModelConfig, SplitConfig, TrainConfig
from rejas_gnn.data.graph_builder import build_urban_graph, load_geometries
from rejas_gnn.evaluation.metrics_and_calibration import (
    expected_calibration_error,
    export_probability_map,
    fit_isotonic,
    full_report,
)
from rejas_gnn.training.pipeline import run_seed

RAW = Path("data/raw/pac.csv")
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="PAC data not available")


@pytest.fixture(scope="module")
def ug_geo():
    return build_urban_graph(DataConfig(feature_set="geometry", covariates_csv=None), GraphConfig(type="line"))


def test_geometry_layer_valid():
    gdf = load_geometries(DataConfig())
    assert len(gdf) == 2790 and (gdf.geom_type == "LineString").all() and gdf.crs.to_epsg() == 32719
    assert gdf["y"].sum() == 217


def test_tensor_dimensions_and_no_nan(ug_geo):
    d = ug_geo.data
    assert d.x.shape[0] == d.y.shape[0] == d.pos.shape[0] == 2790
    assert d.x.shape[1] == len(ug_geo.feature_names)
    assert not torch.isnan(d.x).any() and d.edge_index.shape[0] == 2
    assert not any(n.startswith(("cpv24_", "uai_")) for n in ug_geo.feature_names)


def test_ablation_b_has_more_features(ug_geo):
    ug_b = build_urban_graph(DataConfig(feature_set="geometry_socio"), GraphConfig(type="line"))
    assert ug_b.data.x.shape[1] > ug_geo.data.x.shape[1]
    assert any(n.startswith("uai_") for n in ug_b.feature_names)


def test_seed_conservation(ug_geo):
    cfg = ExperimentConfig(split=SplitConfig(n_blocks=8, n_folds=2, buffer_m=100.0),
                           model=ModelConfig(name="sage", hidden_dim=16),
                           train=TrainConfig(epochs=12, patience=5, min_epochs=1, seeds=(42,)))
    a, b = run_seed(ug_geo, cfg, 42), run_seed(ug_geo, cfg, 42)
    assert np.array_equal(a["oof_raw"], b["oof_raw"]) and np.isfinite(a["oof_cal"]).all()
    assert 0 <= a["calibrated"]["ece"] <= 1


def test_calibration_helpers():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.08  # perfectly calibrated data → small ECE
    iso = fit_isotonic(y, p)
    assert np.all(np.diff(iso.predict(np.linspace(0, 1, 50))) >= 0)  # monotone
    r = full_report(y, p, 0.5)
    assert {"pr_auc", "roc_auc", "brier", "ece"} <= r.keys()


def test_export_probability_map(ug_geo, tmp_path):
    p = np.full(len(ug_geo.segments), 0.1)
    out = export_probability_map(ug_geo.segments, p, tmp_path / "map.geojson")
    import geopandas as gpd
    g = gpd.read_file(out)
    assert len(g) == 2790 and g.crs.to_epsg() == 4326 and "p_closure" in g.columns
    with pytest.raises(AssertionError):
        export_probability_map(ug_geo.segments, p[:-1], tmp_path / "bad.gpkg")
