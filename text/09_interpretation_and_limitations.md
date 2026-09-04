% 9 - Interpretation and limitations

# What the benchmark supports

The geological descriptions provide complementary predictive information. Their value becomes clearest when they are combined with XYZ and depth rather than used as a substitute for spatial context. The benefit persists across validation protocols, although it is smaller for unseen campaigns.

No uncertainty strategy dominates every score and shift. Confidence and predictive entropy provide moderate error ranking in ordinary LOBO. Their discrimination weakens under unseen campaigns and approaches random ranking when a large spatial buffer removes local support.

LLLA and the Training Subsample Ensemble show the strongest increase in epistemic proxy under decreasing spatial support. This is a useful sensitivity response, but it does not solve error localization at the largest buffers.

## Limits of the available evidence

The input table does not contain every source used by a geotechnical expert. Pocket-penetrometer measurements, piezometric information, and prior knowledge of the area can influence interface placement. Missing evidence can produce irreducible ambiguity from the model's perspective.

The entropy decomposition is also model-dependent. Total entropy, expected entropy, and mutual information summarize the probability vectors produced by each approximation; they do not uniquely identify physical aleatoric and epistemic uncertainty.

The practical conclusion is therefore cautious: grouped validation and spatial stress tests reveal when reported accuracy is optimistic, while the tested uncertainty scores provide useful but incomplete warning under distribution shift.
