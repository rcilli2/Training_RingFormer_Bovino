% 9 - Five uncertainty strategies

# Where do the repeated predictions come from?

The common calculation starts with several class-probability vectors for the same interval. What differs is how those vectors are produced.

## Monte Carlo dropout

During training, dropout randomly removes activations and discourages dependence on a particular set of features. MC dropout also enables dropout at prediction time.

Repeated forward passes use different masks with the same fitted weights. The reference configuration uses 30 passes. Their average supplies the final probability vector; their disagreement supplies MI.

This is relatively economical because it requires one training run. Its variability is constrained by one fitted network and its dropout mechanism: stochastic passes can share the same mistaken prediction.

## Deep Ensemble

Several networks with the same architecture are trained independently using different random initializations. Each member learns its own input projection, encoder and classification head.

We average their probability vectors, not their predicted class indices. Disagreement reflects differences between fitted solutions. Training several full networks costs more than training one model and repeating its forward pass.

Independent initialization does not guarantee independence of errors: every member may learn similar patterns from the same limited data.

## Training Subsample Ensemble

Each member is trained on a randomly selected subset of training boreholes. Sampling whole boreholes preserves their internal sequences.

The method probes sensitivity to the available training evidence as well as differences between fitted models. It can respond when alternative training subsets support different interpretations.

Subsampling without replacement is not classical bootstrap bagging, which samples with replacement. The distinction matters when describing how members were constructed.

## Last-Layer Laplace Approximation

LLLA keeps the fitted Transformer representation fixed and approximates uncertainty in the final classification weights.

Around a fitted maximum-a-posteriori solution, local curvature of the negative log posterior defines a Gaussian approximation. Schematically, its covariance is the inverse of a curvature matrix plus prior precision. The exact approximation may simplify that matrix.

Sampling last-layer weights produces multiple class-probability vectors. This is cheaper than repeatedly training a complete Transformer, but it does not propagate uncertainty about all encoder parameters.

The prior precision and curvature approximation influence predictive spread. A larger MI response is not automatically a better error-detection score.

## TabICLv2 as a classification head

TabICLv2 receives labelled training representations as its in-context examples and predicts labels for held-out representations. It replaces the ordinary linear head for this comparison.

The supplied representation can be model input, encoder input or hidden output. For the hidden comparison, the Transformer has already learned its representation from training boreholes.

The pretrained tabular predictor uses in-context prediction configurations to produce an ensemble. In this experiment their probability vectors provide an empirical disagreement score. They are not samples from a Bayesian posterior over all Transformer weights.

A single averaged probability vector is enough to calculate predictive entropy, but not enough to reconstruct MI. MI requires the constituent predictions or an already computed disagreement score.

## Comparing the five approaches

| Method | What varies between predictions? | What is shared? |
|---|---|---|
| MC dropout | Dropout masks | Fitted weights |
| Deep Ensemble | Independently fitted networks | Architecture and training dataset |
| Subsample Ensemble | Training subsets and fitted networks | Architecture and sampling procedure |
| LLLA | Sampled classification-layer weights | Fitted encoder |
| TabICLv2 head | In-context prediction configurations | Pretrained predictor and supplied representation |

All five can be confidently wrong. The experiments therefore measure both predictive performance and how successfully uncertainty identifies errors.

Next: [Reliability metrics](10_reliability_metrics.md).
