% 12 - Decreasing spatial support

# What happens when nearby boreholes disappear?

## The experiment

Ordinary LOBO removes the test borehole but can leave close neighbours in training. Buffered LOBO additionally removes training boreholes inside a radius around it.

The MiniLM PCA16 experiment compares the five UQ methods over buffers from 0 to 250 m. The same type of query is evaluated with progressively less nearby training evidence.

This changes both distance to the remaining data and training-set size. The experiment measures their combined practical effect.

## Read performance first

Without a buffer, the five methods reach approximately 0.77-0.79 accuracy. At 250 m, accuracy is approximately 0.31-0.36.

The decline occurs across all five strategies. It indicates that performance is sensitive to the training support available in this area. It does not establish that changing the uncertainty approximation can recover the missing evidence.

## Then read the average uncertainty

LLLA's mean MI increases from roughly 0.16 to 0.54. The Training Subsample Ensemble also shows a substantial increase. MC dropout and the other approaches change less on their respective MI scales.

These are averages over predictions. Their absolute magnitudes depend on how the method generates variability. Comparing the response within each method is therefore useful alongside comparing their error-detection metrics.

## Finally ask whether the score locates errors

Predictive-entropy Error AUROC drops from approximately 0.77 without a buffer to approximately 0.51-0.54 at 250 m.

Thus, an increasing mean uncertainty can coexist with weak separation between correct and incorrect predictions. The score can change across scenarios without providing an effective ordering of intervals for review within the difficult scenario.

For example, assigning higher uncertainty to almost every interval raises the mean. It only improves error detection if incorrect predictions tend to receive higher scores than correct ones.

## Connection to the notebook

Training_03_Decreasing_Spatial_Support.ipynb plots the packaged buffer_uq_method_metrics.csv table. It loads the saved metrics for each method and radius rather than retraining the buffer sweep.

Read each panel with the others: accuracy describes the failure, mean MI describes the response, and Error AUROC describes whether an uncertainty score helps locate errors.

Next: [Interpretation and limitations](13_interpretation.md).
