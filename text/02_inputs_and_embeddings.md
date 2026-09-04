% 2 - Inputs and embeddings

# Spatial variables and geological language

The principal configuration combines coordinates $(X,Y,Z)$, depth, and sentence embeddings derived from the geological descriptions. Coordinates provide strong information about the local geological structure; the text supplies observations that coordinates alone cannot encode.

MiniLM maps each description to a 384-dimensional vector. Principal component analysis reduces these vectors to 16 components before they enter the sequence model. PCA is fitted on the training portion of each validation fold and then applied to the held-out data, preventing information leakage.

The experiments compare spatial-only inputs with spatial and textual inputs. The relevant question is not whether embeddings replace the spatial baseline, but whether they add reproducible information beyond that already contained in location and depth.

PCA16 is a pragmatic operating point: it preserves useful variation while limiting dimensionality and computational cost. Results for other PCA dimensions are sensitivity analyses rather than separately tuned models.
