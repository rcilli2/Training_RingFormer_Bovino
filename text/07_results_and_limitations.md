% 7 - Results and limitations

# What the experiments support

The spatial baseline using coordinates and depth is highly informative. Adding compressed MiniLM embeddings, especially PCA16, provides a modest improvement in the controlled comparisons, indicating that descriptions contribute information beyond location alone.

Error-detection AUROC is moderate for all uncertainty methods, and no method is consistently superior across every shift. Confidence is informative but insufficient for robust error detection. LLLA shows the clearest uncertainty response in the tested perturbation experiments.

Performance decreases under unseen campaigns and as the spatial exclusion buffer grows. This is evidence that both acquisition context and nearby training support matter. It is not evidence that one uncertainty score can identify every unfamiliar input.

The available inputs do not contain all evidence used by a geotechnical expert. Consequently, even a well-calibrated model cannot resolve ambiguity caused by missing measurements or missing prior geological knowledge.
