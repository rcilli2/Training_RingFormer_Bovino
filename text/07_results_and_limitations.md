% 7 - Controlled representation experiments

# What do coordinates and descriptions contribute?

The representation study changes the input features while preserving the target, architecture, and validation protocol. It combines 14 sentence-embedding models with five uncertainty strategies and multiple PCA dimensions. The purpose is not to find one lucky embedding, but to determine whether the same feature conclusion survives many modelling choices.

## Depth, text, and spatial position

Depth alone performs poorly. In the aggregate benchmark, its mean accuracy is about 0.51 in LOBO and 0.43 in leave-one-campaign-out validation. Text embeddings combined with depth improve these values to approximately 0.63 and 0.48 for PCA16. Geological language therefore contains useful information even without coordinates.

XYZ and depth form a much stronger baseline, reaching about 0.72 in LOBO and 0.63 under unseen campaigns. This confirms that local spatial structure is highly informative in the BOVINO study area.

## Does text add information beyond XYZ?

Combining XYZ, depth, and PCA16 text embeddings reaches approximately 0.75 in LOBO and 0.65 in leave-one-campaign-out validation. The corresponding gains over XYZ and depth are about four and two percentage points when averaged over embedding models and uncertainty methods.

For MiniLM specifically, the gain is larger and appears for every selected method in both protocols. This makes MiniLM PCA16 a useful compact operating point, but not proof of universal superiority over every sentence encoder.

## Why compare LOBO and unseen campaigns?

In LOBO, native and PCA-compressed embeddings perform similarly. Under unseen campaigns, native embeddings fall below the spatial baseline, while 8 to 32 PCA components retain a modest improvement. PCA appears to regularize campaign-specific linguistic variation rather than merely accelerate computation.

The benchmark cells share observations and validation folds. Agreement across models is robust descriptive evidence, not a collection of independent statistical replicates.
