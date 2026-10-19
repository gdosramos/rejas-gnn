Notebooks are for exploration and figures only. They must import from `rejas_gnn` and read
`results/<experiment>/predictions_oof.csv`; they never compute results that are reported.

Suggested notebooks:
- `01_data_overview.ipynb` – label prevalence, degree distribution of line vs radius graph, Moran's I.
- `02_results_maps.ipynb` – map OOF probabilities, rank unclosed segments by `p_oof` (candidates).
- `03_attention.ipynb` – `GAT.attention` on a trained model to inspect influential neighbours.
