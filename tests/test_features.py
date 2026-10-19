"""Feature engineering: no leakage, finite values, network metrics sanity."""
import numpy as np

from rejas_gnn.data.features import (
    assemble_feature_matrix,
    build_primal_graph,
    geometry_features,
    network_features,
)
from rejas_gnn.data.load import LABEL_LEAK_COLUMNS


def test_primal_graph_and_network_features(toy_segments):
    g = build_primal_graph(toy_segments)
    assert g.number_of_nodes() == 6 and g.number_of_edges() == 4
    nf = network_features(toy_segments, g)
    # segments 0 and 2 touch a degree-1 node -> cul-de-sac; segment 1 is in the middle
    assert nf.loc[0, "es_culdesac"] == 1.0 and nf.loc[2, "es_culdesac"] == 1.0
    assert nf.loc[1, "es_culdesac"] == 0.0
    assert nf["es_puente"].tolist() == [1.0, 1.0, 1.0, 1.0]  # a path is all bridges
    assert np.isfinite(nf.to_numpy()).all()


def test_no_label_columns_in_features(toy_segments):
    toy_segments["uai_bht"] = [0.5, 0.6, np.nan, 0.7]
    toy_segments["cpv24_prop_ninos"] = [0.1, 0.2, 0.3, 0.4]
    g = build_primal_graph(toy_segments)
    blocks = [network_features(toy_segments, g), geometry_features(toy_segments)]
    X, names = assemble_feature_matrix(toy_segments, blocks)
    assert not LABEL_LEAK_COLUMNS.intersection(names)
    assert "uai_bht__missing" in names  # missingness indicator added
    assert np.isfinite(X).all() and X.shape[0] == 4


def test_geometry_orientation_encoding(toy_segments):
    gf = geometry_features(toy_segments)
    assert np.allclose(gf["orient_sin"] ** 2 + gf["orient_cos"] ** 2, 1.0)
    assert gf["via_living_street"].tolist() == [0.0, 1.0, 0.0, 0.0]
