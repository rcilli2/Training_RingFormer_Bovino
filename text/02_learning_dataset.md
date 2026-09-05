% 2 - From logs to a dataset

# What is one training example?

## Campaigns, boreholes and intervals

A campaign contains boreholes; each borehole is an ordered sequence of depth intervals. The model assigns one of six classes to each position. This is sequence labelling, not assigning one class to an entire borehole.

The dataset contains 95 boreholes from 11 survey campaigns.

## From variable intervals to a regular sequence

Original descriptions cover variable thicknesses. The exported workflow pools a regular depth representation into a common sequence grid of 1.6 m resolution and 48 positions per borehole.

A pooled position may not correspond to one original description. Pooling features, labels and validity can blur thin layers or interfaces. Those rules belong in the experiment's preprocessing specification.

## Validity and padding

A common grid may contain positions beyond a borehole's documented depth or without a valid label. The workflow carries validity information; the supplied launcher uses a pooled validity threshold of 0.5.

Two operations must be distinguished:

- Loss and evaluation masks select the positions contributing to optimization and scores.
- Attention masks determine which positions can exchange information inside the encoder.

The supplied MiniTransformer does not pass a padding mask to its encoder. Excluding invalid positions from loss therefore does not automatically exclude their representations from attention.

## Classes and aggregation

The class order is Topsoil, Elu (4), Debris (3), Transitional Flysch (2b), Clay Flysch (2a), and Fyr Flysch (1). Preserve that order when interpreting probability columns and confusion matrices.

Class support is uneven. Overall accuracy can hide failure on rare units.

## Why grouping matters

Intervals from one borehole share position, context and succession. A random interval split can allow closely related observations into both training and testing. LOBO excludes an entire borehole; LOCO excludes an entire campaign.

Next: [Turning descriptions into features](03_text_features.md).
