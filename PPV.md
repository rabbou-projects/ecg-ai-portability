# Positive predictive value of an ECG model at an external site

The measurement of the positive predictive value recomputed by Bayes' theorem against the one observed, across transfers between hospitals, care settings and corpora, is part of the article [REPORT.md](REPORT.md): sections 2.5 and 3.1 for the measurement, 3.5 for the predictions that need no diagnosis, and 3.6 for the three repairs judged on net benefit. The methods in statistical terms are in Part 3 of [SUPPLEMENT.md](SUPPLEMENT.md).

## Correcting an ECG model's positive predictive value at an external site with few local labels

### Summary

A site adopting an ECG model receives a positive predictive value (PPV) measured in the developer's population, and in the article Bayes' theorem at the local prevalence did not reliably recover the PPV observed at other hospitals and care settings. How many local labels, patients in whom the site has confirmed the diagnosis, an update of the stated PPV requires, and what sensitivity it retains, has not been measured. We compared five updating methods in 105 transfers, each a model calibrated in one population and applied to another, at three external sites (Columbia, Beth Israel Deaconess and Chongqing), with label budgets of 25 to 400 patients and 200 draws per budget. Without local labels, a prevalence correction erred by a median of 13.1, 11.4 and 51.7 points. With 100 labels, logistic recalibration, which refits the intercept and slope of the model's logit to the local labels, or the intercept alone when the rarer class has fewer than ten events, erred by 3.1, 5.7 and 3.8 points, and a threshold refitted on half of the labels erred by 2.6, 2.2 and 3.5 points. The refitted threshold flagged 99%, 99% and 98% of cases, above the 90% it targeted: 100 labels held a mean of 9.9, 13.2 and 14.8 cases, a half of fewer than nine cases cannot place a threshold, and where a threshold was placed it flagged every patient in 0.79, 0.84 and 0.71 of draws. Its net benefit at a threshold probability of 10%, true positives per patient minus false positives per patient weighted by the odds of that threshold, was 0.001 at Columbia, against −0.001 for flagging every patient and 0.044 for recalibration. A threshold certified by exact binomial tests to hold the source PPV was found in 0.168, 0.251 and 0.010 of draws. These results suggest that 100 local labels allow a site to state its PPV within a few points by recalibration, but that a threshold refitted on so few cases seldom separates cases from non-cases, and that 100 labels did not place a threshold holding the source PPV at the sensitivity fixed at the source.

### Objective

A site adopting an ECG model receives a positive predictive value (PPV) measured in the developer's population and adjusts it to its own prevalence by Bayes' theorem. In the article, this adjustment did not reliably predict the observed PPV across hospitals and care settings, because the operating point of a fixed threshold, the sensitivity and specificity it yields, shifted between populations. A site can instead update the model on local labels, but the number of labels such an update requires, and the sensitivity it retains, have not been measured. A PPV held constant across sites requires a threshold that moves with the site, and a higher threshold trades sensitivity for PPV. We compared five updating methods, referred to here as repairs, by the agreement between the PPV each states and the PPV observed in the target population, as a function of the number of local labels, and report the sensitivity each attains.

### Methods

The unit of analysis is the transfer, from a source population to a target population. Three external sites form the primary analysis. At Columbia, the four EchoNext models, calibrated on inpatients, were applied to emergency and outpatient ECGs (76 transfers). At Beth Israel Deaconess, the models of the seven-source rotation were applied to MIMIC-IV-ECG (28 transfers), whose labels are the ECG machine's statements, so that the observed PPV there measures agreement with the machine rather than with a clinical reference standard. At Chongqing, the PTB-XL myocardial infarction model was applied to the hospital's ECGs (1 transfer). Transfers within a site share data and are not independent, and the conclusions rest on three sites. Shandong, the other rotation corpora, CODE-15%, the CODE test set and ZZU form a secondary reserve, reported in [`results/ppv_repairs_reserve.json`](results/ppv_repairs_reserve.json), and transfers in which a model is evaluated in its own source population serve as in-distribution controls.

For each transfer, the target PPV v is the PPV attained in the source population at the threshold that gives 90% sensitivity there. It was fixed from the source scores alone, before any repair was run, and the results file records the digest of the target table. The five repairs were:

1. prevalence correction without labels, in which the target prevalence was estimated from the model's scores by expectation-maximisation, an iterative maximum-likelihood estimate of the class proportions (Saerens et al., 2002), and the PPV at the source threshold was recomputed by Bayes' theorem;
2. logistic recalibration on n labels, the stated PPV being the mean recalibrated probability among flagged patients;
3. threshold refitting on split labels, in which the threshold was set on one half of the n labels to give 90% sensitivity and the stated PPV was the PPV observed at that threshold in the other half, with its one-sided Clopper-Pearson bound, an exact binomial confidence bound;
4. threshold certification by Learn-then-Test (Angelopoulos et al.), in which thresholds from a grid fixed by the source scores were tested in fixed sequence from the highest down, each by an exact binomial test of PPV ≤ v at δ = 0.1, and the lowest threshold rejected before the first non-rejection was retained, so that a retained threshold has a PPV at or below v with probability at most δ; the stated PPV is v;
5. weighted conformal selection without labels (Jin and Candès, arXiv:2307.09291, Algorithm 1 with deterministic pruning), which flags patients so that the false discovery rate, the expected proportion of false positives among flagged patients, stays below q = 1 − v; covariate-shift weights, the estimated ratio of target to source density of the model's score, adjust the source patients to the target population.

Repairs 3 and 4 place thresholds by the conformal rule used at the source: among m cases, the threshold is the score of the case ranked ⌈(m + 1) × 0.9⌉ from the highest. Labels were sampled by patient at budgets of n = 25, 50, 100, 200 and 400, with 200 draws per budget; target patients not sampled formed the evaluation set, and repairs 1 and 5 used the whole target. The primary outcome was the absolute error, in percentage points, between the stated PPV and the PPV observed in the evaluation set. Sensitivity in the target was averaged over the draws in which a repair returned a threshold, then over the transfers of a site. Discrimination (AUROC) and net benefit at a threshold probability of 10% are reported per transfer in [`results/ppv_repairs.json`](results/ppv_repairs.json).

### Results

Without local labels, the prevalence correction erred by a median of 13.1 points at Columbia, 11.4 points at Beth Israel and 51.7 points at Chongqing; it understated the observed PPV in every Columbia transfer and overstated it at Chongqing. At the source threshold, which this correction retains, sensitivity in the target was 82% at Columbia, 75% at Beth Israel and 84% at Chongqing, against 90% in the source. The published label-free estimate for Columbia, over that study's selection of transfers, erred by 12.6 points. In the in-distribution controls the correction erred by 2.4 points, and with the true prevalence in place of the estimate Bayes' theorem erred by 0.5 points, against 0.2 points over the EchoNext controls of the published study.

Local labels reduced the error. With 25 labels, recalibration erred by 5.8, 8.5 and 9.3 points at Columbia, Beth Israel and Chongqing, and threshold refitting by 6.0, 5.7 and 6.6 points. With 100 labels the two errors were 3.1 and 2.6 points at Columbia, 5.7 and 2.2 points at Beth Israel, and 3.8 and 3.5 points at Chongqing; with 400 labels they were 1.9 and 1.6 points, 3.5 and 1.7 points, and 1.8 and 2.6 points. From 100 labels onwards the stated PPV exceeded the observed PPV in 0.44 to 0.55 of the draws, whereas with fewer labels recalibration overstated it in up to 0.72 of the draws.

Recalibration retains the source threshold, and its sensitivity therefore equalled that of the prevalence correction at every budget. Refitted thresholds flagged more cases than the 90% they targeted: 99%, 99% and 98% with 100 labels, and 94%, 95% and 92% with 400. When a half-sample contains fewer than nine cases, the rank ⌈(m + 1) × 0.9⌉ exceeds m, no threshold can be placed and all patients are flagged; the stated and observed PPV then both approach the prevalence, and the error remains small although the threshold does not separate cases from non-cases.

Certification preserved its guarantee but rarely returned a threshold. At n = 100, the proportion of draws in which the observed PPV fell below v was 0.022 at Columbia, 0.014 at Beth Israel and 0.010 at Chongqing, all below δ, and a threshold was certified in 0.168, 0.251 and 0.010 of draws; at most targets, no threshold in the grid attains the source PPV. Where a threshold was certified, sensitivity was 43% at Columbia, 87% at Beth Israel and 38% at Chongqing, the last over two draws.

Weighted conformal selection did not control the false discovery rate at Chongqing, where the mean false discovery proportion, the realised share of false positives among flagged patients, was 0.75 against a nominal level of 0.39; it was 0.64 against 0.77 at Columbia and 0.46 against 0.60 at Beth Israel. Its sensitivity was 64%, 68% and 86% at the three sites.

### Post hoc analyses

Three measures were added after the protocol was fixed, which named sensitivity and net benefit as secondary criteria: the mean number of cases in each labelled sample, the proportion of draws in which the refitted threshold flagged every patient, and two net-benefit references per transfer, flagging every patient and the target's own threshold for 90% sensitivity fitted on all its labels. The second reference is an operating point, not the highest attainable net benefit. Intervals on the pooled median errors are the 2.5th and 97.5th centiles of 200 bootstrap redraws, resampling with replacement the transfers of a site and then the draws within each transfer; the interval on the label-free median also resamples patients, as for the falsification criteria. With 200 redraws, each bound is approximate to about one decimal place. The references are read on the whole target, the repairs on the evaluation patients of each draw, and the refitted threshold's net benefit on the draws in which it was placed. Table 1 gives the errors and Table 2 the net benefit at n = 100 for every site, the secondary reserve and the in-distribution controls included.

Table 1. Median absolute error of the stated PPV, in percentage points, with 95% intervals, at n = 100 labels.

| Site | Transfers | Cases in 100 labels | Prevalence correction, no labels | Recalibration | Refitted threshold | Draws with a refitted threshold | Of these, flagging every patient |
|---|---|---|---|---|---|---|---|
| Columbia | 76 | 9.9 | 13.1 (10.7 to 17.1) | 3.1 (2.9 to 3.5) | 2.6 (2.3 to 2.8) | 0.89 | 0.79 |
| Beth Israel | 28 | 13.2 | 11.4 (6.7 to 30.3) | 5.7 (4.3 to 7.8) | 2.2 (1.8 to 2.3) | 0.87 | 0.84 |
| Chongqing | 1 | 14.8 | 51.7 (50.4 to 52.8) | 3.8 (3.3 to 4.3) | 3.5 (3.1 to 4.3) | 1.00 | 0.71 |
| Shandong | 25 | 1.6 | 22.3 | 4.9 (2.7 to 7.4) | 1.2 (1.0 to 1.3) | 0.48 | 1.00 |
| Chapman-Ningbo | 28 | 5.0 | 17.3 | 5.1 (3.8 to 7.0) | 1.9 (1.8 to 2.2) | 0.75 | 0.88 |
| CPSC | 28 | 10.3 | 12.2 | 5.1 (3.9 to 6.6) | 2.9 (2.5 to 3.2) | 0.93 | 0.80 |
| Georgia | 28 | 6.9 | 20.0 | 5.6 (4.3 to 7.5) | 2.2 (2.1 to 2.5) | 0.91 | 0.93 |
| PTB-XL | 28 | 15.4 | 14.8 | 4.7 (3.1 to 6.8) | 2.6 (2.6 to 2.7) | 0.83 | 0.82 |
| CODE-15% | 24 | 1.7 | 93.3 | 3.3 (2.2 to 5.4) | 1.6 (1.5 to 1.6) | 0.58 | 1.00 |
| CODE test set | 21 | 3.7 | 54.0 | 5.1 (3.5 to 8.0) | 1.9 (1.7 to 2.0) | 0.85 | 1.00 |
| ZZU | 14 | 1.2 | 22.5 | 6.9 (2.5 to 13.0) | 1.3 (0.8 to 1.6) | 0.44 | 1.00 |
| In-distribution controls | 77 | 12.4 | 2.4 | 3.0 (2.5 to 3.4) | 2.4 (2.2 to 2.8) | 0.87 | 0.75 |

Table 2. Net benefit at a threshold probability of 10%, at n = 100 labels. Recalibration keeps the source threshold and shares the prevalence correction's column; the certified threshold is averaged over the draws in which one was found.

| Site | Flag every patient | Target's own threshold | Prevalence correction and recalibration | Refitted threshold | Certified threshold | Weighted conformal selection |
|---|---|---|---|---|---|---|
| Columbia | −0.001 | 0.032 | 0.044 | 0.001 | 0.033 | 0.035 |
| Beth Israel | 0.036 | 0.101 | 0.090 | 0.031 | 0.118 | 0.073 |
| Chongqing | 0.055 | 0.081 | 0.085 | 0.060 | 0.050 | 0.084 |
| Shandong | −0.093 | 0.003 | 0.006 | −0.093 | 0.009 | −0.014 |
| Chapman-Ningbo | −0.055 | 0.013 | 0.026 | −0.052 | 0.027 | −0.001 |
| CPSC | 0.004 | 0.057 | 0.055 | 0.009 | 0.067 | 0.048 |
| Georgia | −0.034 | 0.037 | 0.037 | −0.032 | 0.034 | 0.005 |
| PTB-XL | 0.061 | 0.124 | 0.088 | 0.052 | 0.120 | 0.047 |
| CODE-15% | −0.092 | −0.077 | −0.041 | −0.092 | −0.036 | −0.032 |
| CODE test set | −0.070 | −0.009 | 0.000 | −0.070 | −0.009 | −0.011 |
| ZZU | −0.098 | −0.015 | −0.001 | −0.098 | 0.004 | −0.012 |
| In-distribution controls | 0.027 | 0.070 | 0.069 | 0.027 | 0.051 | 0.049 |

The labelled samples were small in cases: with 100 labels they held fewer than two cases on average at Shandong, CODE-15% and ZZU, and fewer than 16 at every other site. The refitted threshold flagged every patient in 0.71 to 1.00 of the draws in which it was placed, and in all of them at Shandong, CODE-15%, the CODE test set and ZZU; its net benefit stayed within 0.006 of that of flagging every patient at every site except PTB-XL, where it was 0.009 lower. Its small PPV error at 100 labels therefore reflects a threshold that seldom discriminates, at which the stated and observed PPV both approach the prevalence. The source threshold, which the prevalence correction and recalibration keep, had a higher net benefit than flagging every patient at every site. Without labels, the prevalence correction erred by 12.2 to 93.3 points at the eight reserve sites, against 2.4 points in the controls.

### Falsification criteria

The label-free correction would have been judged sufficient outside Columbia had the upper bound of the bootstrap interval of its median error, obtained by resampling transfers and patients with replacement, remained below two points at both Beth Israel and Chongqing. The bounds were 30.3 points and 52.8 points, which indicates that the need for local labels is not specific to Columbia; with a single transfer, the Chongqing bound reflects the resampling of patients only.

Certification would have been judged invalid under realistic sampling had its violation rate, the proportion of draws in which the observed PPV fell below v, exceeded 0.15 at n = 100 at any site; it did not. The budget would have been judged insufficient to certify v had a threshold been returned in fewer than half of draws, which was the case at all three sites: 100 labels did not certify the source PPV at these targets.

### Limitations

The comparison rests on three sites with correlated transfers within each, and the Beth Israel labels are machine statements. The observed PPV of each draw is estimated on a finite evaluation set, which adds sampling variance to the measured error and to the violation rate. Repair 5 uses the model's score as its only covariate and therefore assumes that the diagnosis depends on the ECG only through that score; the false discovery proportion at Chongqing suggests that this assumption does not hold there. The target v is the source PPV, and a site that sets another target would obtain other certification rates and other sensitivities. Chongqing contributes a single transfer. Calibration curves are not shown; the calibration of the recalibrated model at 100 labels is in [`results/repairs.json`](results/repairs.json).
