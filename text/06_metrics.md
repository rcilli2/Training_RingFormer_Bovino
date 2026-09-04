% 6 - Metrics

# Prediction, calibration, and error detection

Overall accuracy measures the fraction of correctly labelled intervals. Cohen's kappa additionally accounts for agreement expected from class frequencies. Both should be accompanied by the confusion matrix because rare and neighbouring geological units can behave very differently.

Expected Calibration Error partitions predictions by confidence and averages the absolute difference between mean confidence and empirical accuracy:

$$ECE=\sum_b\frac{|B_b|}{n}\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|.$$

ECE measures calibration, not whether uncertainty ranks individual errors effectively. Post-hoc recalibration can change ECE without changing that ranking.

Error AUROC and Error AUPRC treat incorrect predictions as positives and rank them using $1-\max_c\bar p_c$, predictive entropy, or mutual information. AUROC is threshold-independent; AUPRC is sensitive to the error prevalence and must be interpreted against its baseline.
