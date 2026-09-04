% 8 - Spatial support experiment

# What happens when nearby boreholes disappear?

Standard leave-one-borehole-out validation removes the query borehole but can retain close neighbours in training. This tests transfer to an unseen sequence, but it may still benefit from dense local spatial support.

Buffered LOBO removes every training borehole inside a radius around the query. The radii used here range from 0 to 250 m. Increasing the radius changes the information available to the model in a controlled and geographically meaningful way.

## Predictive performance

With no exclusion buffer, the five methods obtain accuracies of approximately 0.77 to 0.79. At 250 m, accuracy falls to about 0.31 to 0.36. The similarity of this decline across methods shows that the main limitation comes from missing local evidence rather than from one particular uncertainty approximation.

## Average response and error discrimination

LLLA produces the clearest epistemic response: mean mutual information rises from roughly 0.16 to 0.54. The Training Subsample Ensemble also reacts substantially. Other methods show smaller changes in their MI scale.

An increasing mean uncertainty does not guarantee useful warning at the level of individual predictions. Predictive-entropy Error AUROC decreases from around 0.77 at 0 m to nearly 0.50 at 250 m for every method. Under severe spatial shift, uncertainty rises globally but becomes unable to separate correct and incorrect intervals.

This distinction is central. A method can recognize that the overall scenario is unfamiliar while failing to identify which specific predictions require review.
