% 13 - Interpretation and limitations

# What the experiments tell us about geological reliability

## Text complements spatial information

XYZ and depth provide a strong baseline in the study area. Adding text embeddings improves the evaluated feature comparisons, although the gain varies with representation, encoder and validation protocol.

This supports using descriptions as complementary observations. It does not imply that the model has reconstructed every source of evidence used by a geotechnical expert.

## Context and transfer are different questions

A contextual representation may help classify intervals when their surroundings are informative. Generalization to another campaign additionally requires those learned relationships to remain useful under changed spatial and descriptive conditions.

LOBO and LOCO therefore answer different practical questions. Their performance gap should accompany any claim about deployment to unseen investigations.

## An uncertainty response is useful but incomplete

No tested UQ strategy dominates every setting and score. Confidence and entropy offer useful but imperfect error ranking under ordinary validation. Ranking becomes weaker under unseen campaigns and the largest spatial buffers.

LLLA and Subsample Ensemble show marked changes in mean MI as spatial support decreases. This sensitivity is informative, but it should not be equated with accurate localization of individual mistakes.

## Limits imposed by the evidence

Pocket-penetrometer measurements, piezometric information and part of the expert's prior knowledge are absent from the input features. Some errors may reflect that missing evidence or ambiguity in the annotations.

The entropy decomposition describes the model-generated probability vectors. Its aleatoric and epistemic proxies do not uniquely identify physical causes of uncertainty.

The embedding sweep also reuses the same geological observations. Consistent gains across models are descriptive evidence; significance assessment must respect borehole and campaign dependence and the exploratory selection of configurations.

## Practical interpretation

Use the predicted profile together with its uncertainty, observed descriptions and spatial support. Low uncertainty is not sufficient reason to accept a prediction when the available training data poorly represent the location or campaign.

The principal contribution of the validation study is to make those conditions visible: a model can perform well in a familiar setting and still offer limited warning when its support deteriorates.

Next: [Reproducing the figures](14_reproduction.md).
