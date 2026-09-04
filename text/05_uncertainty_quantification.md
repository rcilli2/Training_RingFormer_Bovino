% 5 - Uncertainty quantification

# Five routes to a predictive distribution

- **MC Dropout** retains dropout during inference and repeats stochastic forward passes.
- **Deep Ensemble** independently trains models with different random initializations.
- **Training Subsample Ensemble** trains members on different random subsets of the training boreholes.
- **Last-Layer Laplace Approximation** approximates the posterior around the fitted classification head with a Gaussian informed by local loss curvature.
- **TabICLv2 as classification head** applies a pretrained tabular in-context learner to Transformer representations and uses its internal predictive ensemble.

Despite their different origins, each method yields probability vectors $p_m$ that can be summarized consistently. With $\bar p=M^{-1}\sum_m p_m$:

$$H[\bar p] = -\sum_c \bar p_c\log\bar p_c$$

is total predictive uncertainty,

$$\mathbb{E}[H[p_m]] = -\frac{1}{M}\sum_m\sum_c p_{m,c}\log p_{m,c}$$

is the expected conditional entropy, used as an aleatoric proxy, and

$$MI = H[\bar p]-\mathbb{E}[H[p_m]]$$

measures disagreement and is used as an epistemic proxy. These quantities are model-dependent scores, not direct measurements of physical uncertainty.
