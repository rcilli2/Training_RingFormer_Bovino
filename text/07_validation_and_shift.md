% 7 - Validation and distribution shift

# How far can the model generalize?

## Predicting an unseen borehole

A model can reproduce its training labels without learning a relationship that transfers to a new borehole. Validation separates the observations used to learn that relationship from those used to evaluate it.

Because intervals within a borehole share location and context, we exclude complete boreholes rather than randomly separating depth positions. The held-out labels are used to score predictions, not to train the classifier.

## Four complementary experiments

| Protocol | What is excluded or changed? | Question |
|---|---|---|
| Leave one borehole out (LOBO) | One entire borehole | Can we interpret an unseen log when neighbouring boreholes and its campaign may remain available? |
| Leave one campaign out (LOCO) | All boreholes from one campaign | Can the model transfer to a campaign absent from training? |
| Buffered LOBO | The test borehole and training boreholes inside a spatial radius | How much does prediction depend on nearby training support? |
| Feature perturbation | Selected input features are permuted | How do predictions and uncertainty react when feature relationships are disrupted? |

These are different stress tests, not a single guaranteed ordering of difficulty.

## Why campaigns matter

Campaigns can differ in description style, acquisition procedures, spatial coverage and geology. LOCO tests their combined effect. A performance drop cannot be attributed solely to writing style because the spatial distribution changes too.

Ordinary LOBO is closer to the setting of interpreting another borehole in a well-investigated area. It is not a guarantee that training and test distributions are identical.

## Spatial support and sample size

Increasing an exclusion buffer removes local evidence, but also reduces the training dataset. Both can affect accuracy. To isolate the effect of distance from the effect of training-set size, a further comparison would remove equally many randomly selected training boreholes.

The buffer experiment remains useful without that additional comparison: it tests what happens when local observations are unavailable in practice.

## Perturbations and out-of-distribution inputs

Permuting embeddings preserves the collection of vectors but changes their association with intervals. Permuting coordinates changes spatial associations. Neither necessarily creates a vector that is individually unusual; the disrupted relationship between features can be the important change.

A decrease in accuracy shows sensitivity to the perturbation. Whether uncertainty identifies the resulting mistakes is a separate question.

## Keeping preprocessing separate

For an inductive test, PCA and other transformations learned from data are fitted on training observations and then applied to the held-out group. A PCA fitted on all available covariates instead defines a transductive preprocessing setting.

Scores should also identify their observational level. Pooling evaluated intervals and averaging per-borehole scores answer different questions.

Next: [Probabilities and uncertainty](08_probabilities_and_uncertainty.md).
