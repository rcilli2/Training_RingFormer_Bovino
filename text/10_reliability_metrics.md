% 10 - Reliability metrics

# Accuracy, calibration and error detection answer different questions

## Accuracy and the confusion matrix

Overall accuracy (OA) is the fraction of evaluated intervals whose predicted class matches the reference label.

A confusion matrix shows observed classes in rows and predicted classes in columns. Diagonal counts are correct predictions; off-diagonal counts show which units are confused. Row normalization converts counts into fractions within each observed class.

A class represented by very few intervals has little influence on OA. Its row in the confusion matrix is therefore important even when the overall score looks satisfactory.

Cohen's kappa adjusts agreement for the chance agreement implied by observed and predicted class frequencies:

$$\kappa=\frac{p_o-p_e}{1-p_e}.$$

Here, $p_o$ is accuracy and $p_e$ is the sum, across classes, of observed frequency times predicted frequency. Kappa is not simply accuracy rescaled by the number of classes.

## Calibration: does stated confidence match observed accuracy?

Group predictions into confidence bins. If a bin has mean confidence 0.8, a calibrated predictor should be correct on approximately 80% of the predictions in that bin.

A reliability diagram plots mean confidence on the horizontal axis and empirical accuracy on the vertical axis. Points below the diagonal indicate overconfidence; points above it indicate underconfidence.

Expected Calibration Error summarizes the weighted absolute gaps:

$$ECE=\sum_b\frac{|B_b|}{n}
\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|.$$

For illustration, if half the predictions have a gap of 0.10 and the other half a gap of 0.04, ECE is 0.07. Bin boundaries and bin occupancy affect the estimate.

Good aggregate calibration does not ensure correct predictions for every class, borehole or campaign. It also does not ensure that the model's uncertainty orders individual errors well.

## Error detection: make the error the positive class

Define an error indicator equal to one when the predicted geological label is wrong. A larger uncertainty score should ideally indicate an error.

We test three scores: $1-\max_c\bar p_c$, predictive entropy and MI. Using confidence itself would reverse the intended ranking.

At a chosen threshold, flag intervals whose score exceeds the threshold. Then:

- True-positive rate, or recall, is the fraction of all errors flagged.
- False-positive rate is the fraction of correct predictions flagged.
- Precision is the fraction of flagged predictions that are errors.

For 20 errors and 80 correct predictions, flagging 15 errors and 10 correct predictions gives recall 0.75, false-positive rate 0.125 and precision 0.60. These numbers describe one threshold.

## ROC and AUROC

A receiver operating characteristic (ROC) curve plots true-positive rate against false-positive rate as the threshold varies. AUROC is the area under this curve.

An AUROC of 0.5 corresponds to random ranking. A value of 1 corresponds to perfect separation. Equivalently, AUROC measures how often a randomly selected error receives a higher score than a randomly selected correct prediction, with half credit for ties.

Error AUROC is not geological classification accuracy: it evaluates the ranking produced by an uncertainty score after predictions have been made.

If an evaluated subset contains only errors or only correct predictions, AUROC is undefined. It should not be replaced by an invented zero.

## Precision-recall and AUPRC

A precision-recall curve plots precision against recall as the threshold changes. Its random-ranking reference is the fraction of predictions that are errors.

If one model makes 20% errors and another makes 40%, their reference precision levels differ. A higher AUPRC can partly reflect more errors being available to find, so it must be interpreted alongside OA and error prevalence.

Average precision (AP) and trapezoidal area under a precision-recall curve are related but not identical summaries. The plotting code should name the convention used.

## Correlations across boreholes

A scatterplot of mean uncertainty against accuracy assigns one point to each borehole. Pearson correlation measures linear association; Spearman correlation measures association between ranks.

Mean uncertainty may be negatively associated with accuracy, while confidence may be positively associated. Neither relationship is guaranteed. A strong borehole-level association also does not prove that uncertainty finds the incorrect intervals inside each borehole.

## How to read the figures together

OA and the confusion matrix describe the geological predictions. Reliability diagrams and ECE describe probability calibration. ROC and precision-recall curves describe error ranking. Borehole scatterplots describe variation between logs.

A monotonic transformation of an uncertainty score preserves its ranking and AUROC. Calibration procedures can change numerical probabilities, but they do not automatically improve error ranking.

Next: [Controlled representation experiments](11_representation_experiments.md).
