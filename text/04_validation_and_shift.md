% 4 - Validation and distribution shift

# A reliable model should recognize when it does not know

Spatial predictions inevitably encounter observations that differ from the training data. Four protocols probe increasingly challenging forms of transfer.

## Leave one borehole out

One complete borehole is held out, while neighbouring boreholes and descriptions from the same campaign may remain available. This is the closest protocol to near-distribution validation.

## Leave one campaign out

All boreholes from one survey campaign are held out together. This changes writing style, acquisition context, and spatial support, and is therefore a stronger distribution-shift test.

## Buffered LOBO

Boreholes within an increasing radius of the test borehole are removed from training. The experiment measures performance and uncertainty as local spatial support disappears.

## Feature perturbation

Text embeddings, coordinates, or both are permuted to create synthetic invalid inputs. A useful uncertainty method should react when predictive accuracy deteriorates, although permutation is not identical to a naturally occurring out-of-distribution sample.

All preprocessing that learns from data, including PCA and scaling, must be fitted without the held-out group.
