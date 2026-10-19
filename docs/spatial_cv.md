# Why spatial block cross-validation with a buffer

Closures cluster at the block scale: the original study measured a pair-correlation ratio of
9× over a network-constrained null at r ≈ 40 m, and Moran's *I* of the label on the line graph is
0.17 (`summary.json → label_morans_i`), i.e. positive autocorrelation at the segment scale. Under a random node split, a held-out segment almost
always has labelled neighbours 50 m away; a GNN then reaches high scores by *copying* neighbour
labels, which says nothing about predicting closures in a neighbourhood that was never surveyed.

The protocol in `rejas_gnn.data.splits` follows Roberts et al. (2017):

1. **Blocks**: k-means on UTM coordinates → 20 contiguous blocks of ~140 segments.
2. **Folds**: blocks assigned to 5 folds by a seeded permutation.
3. **Buffer**: non-test nodes within 200 m of any test node are stripped of supervision. With a
   2-layer GNN on the line graph the receptive field is 2 hops (~150 m of street), so 200 m
   guarantees no labelled node sits inside the receptive field of a test node.
4. **Validation**: 25 % of the non-test blocks (never individual nodes) for early stopping and
   threshold selection.
5. **Repetition**: 5 seeds → different block partitions; 25 runs per model.

Trade-offs: block CV is pessimistic (it estimates extrapolation, not interpolation) and reduces
training size by ~35 % per fold. Both are preferable to an optimistic estimate.
