% 1 - Problem and dataset

# From borehole descriptions to geotechnical units

The task is supervised sequence labelling. Each borehole is represented as an ordered sequence of depth intervals, and the model assigns one of six geotechnical units to every valid interval.

The dataset contains 95 boreholes acquired in 11 survey campaigns. Descriptions were written by different geologists and at different times, so terminology, detail, and sentence style are not uniform. The reference labels were assigned by one geotechnical expert using these descriptions together with prior knowledge of the area.

This distinction matters: the model receives only the exported variables. Pocket tests, piezometric pressures, additional investigations, and part of the expert's contextual knowledge are unavailable. The learning problem is therefore informative but intentionally incomplete.

## Why sequence labelling?

Adjacent intervals are not independent. Geological units tend to persist with depth, and transitions are constrained by the surrounding sequence. A sequence model can use this context instead of classifying every description in isolation.
