% 3 - RingFormer architecture

# A lightweight Transformer for borehole sequences

For every interval, numerical features and the reduced text embedding are concatenated and projected to the Transformer dimension. Positional encoding represents order along the borehole. A lightweight Transformer encoder then exchanges information among intervals, and a classification head produces six class probabilities at each valid depth.

The reference configuration uses a 128-dimensional model, one encoder layer, 16 attention heads, dropout, and a dense classification layer. Cross-entropy is optimized only on valid labelled intervals.

The architecture is deliberately small. The purpose is to test whether contextual sequence modelling improves the use of heterogeneous borehole information, not to maximize model capacity on a limited dataset.
