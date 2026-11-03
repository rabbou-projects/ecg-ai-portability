# Label-free recalibration of an ECG threshold for structural heart disease across care settings: weighted conformal prediction in EchoNext

Ruben Abbou · November 2026

## Abstract

**Background.** ECG models for structural heart disease (SHD) are applied at a threshold set for a target sensitivity in the derivation population. At Columbia, the threshold set for 90% sensitivity among inpatients detected 71.6% of SHD among outpatients for a ResNet trained on EchoNext. Recalibrating the threshold conventionally requires outpatients with echocardiographic diagnoses. Weighted conformal prediction recalibrates without target diagnoses under covariate shift, provided that the probability of SHD given the ECG is the same in both settings.

**Methods.** Retrospective study in EchoNext v1.1.1, reported following TRIPOD+AI. Seven models were calibrated on 1,903 inpatient ECGs and evaluated on 1,059 outpatients, 271 of them with SHD, and on 1,971 emergency patients, 788 with SHD. Class-conditional conformal thresholds were reweighted by a domain classifier fitted on age and sex, on seven ECG measurements, or on the frozen ECGFounder embedding. The pre-specified criterion required the 95% Wilson interval of composite sensitivity, at the effective sample size, to lie within 85% to 95%, and a majority of judged echocardiographic findings to exceed 80%.

**Results.** Weighting on age and sex gave outpatient sensitivities of 72.7% to 81.5% across models, against 71.6% to 78.6% at the inpatient threshold; weighting on the embedding gave 87.1% to 89.7%, with lower bounds of 81.7% to 84.9% at 213 effective cases. No model met the criterion at that size, at which it is met only for an observed sensitivity of 90.1% to 92.0%, and a recalibration exactly at 90% meets it with probability 0.40; at the 271 outpatient cases, two models met it. Among emergency patients, weighting on the embedding gave 88.5% to 91.5%, and all seven models met the criterion. The weighted thresholds flagged 60.0% to 75.3% of outpatients, against 36.4% to 51.8% at the inpatient threshold, and specificity fell from 57.4%–76.0% to 29.1%–49.7%. ECGFounder fine-tuned on EchoNext reached 72.0% to 75.6% at the inpatient threshold.

**Conclusions.** Covariate-shift weighting on an ECG foundation-model embedding brought sensitivity for SHD close to 90% among outpatients and emergency patients without target diagnoses, at the cost of flagging most outpatients. Among outpatients, the sample was too small to establish whether the recalibrated threshold reached its target.

## 1. Introduction

ECG models for SHD are derived in patients with an echocardiogram and applied at a threshold set for a target sensitivity in the derivation population. The EchoNext study reports the model's AUROC by care setting but not the sensitivity of a fixed threshold in each setting [1]. In a re-analysis of EchoNext, a threshold set for 90% sensitivity among Columbia inpatients detected 71.6% of outpatients with SHD while the AUROC was preserved [2]: the outpatients' scores were lower, and the operating point moved. An external validation of the EchoNext mini-model in MIMIC-IV recalibrated it on local labels [3]. Recalibration on labelled patients restores the operating point but requires echocardiograms in the target setting.

Split conformal prediction sets a threshold at a quantile of calibration scores and guarantees at least the stated coverage when calibration and target patients are exchangeable [4]; computed within each class, it guarantees a sensitivity [5]. Weighted conformal prediction extends the guarantee to covariate shift by weighting each calibration patient by the likelihood ratio of target to source covariates, which can be estimated from unlabelled target ECGs [6]. The guarantee requires the probability of the outcome given the covariates to be the same in both populations. Searches of arXiv and Europe PMC on 20 October 2026 identified no study applying conformal prediction to EchoNext.

We examined whether weighted conformal thresholds, set without outpatient diagnoses, restore 90% sensitivity for SHD among Columbia outpatients, in seven models and three covariate sets.

## 2. Methods

This is a retrospective study of ECG classification models transferred between care settings of one hospital system, reported following the TRIPOD+AI guideline [7]. Its decision criteria were fixed in a protocol dated 16 October 2026, before any result was read.

### 2.1 Participants and outcome

EchoNext v1.1.1 holds 100,000 ECGs from Columbia, each paired with a transthoracic echocardiogram within one year, which yields eleven echocardiographic findings and their composite, moderate or greater SHD [1]. The published splits were used unchanged. Models were trained on 72,475 ECGs and calibrated on the 1,903 inpatient ECGs of the validation split. The test split gave three target cohorts, with one ECG per patient: 2,203 inpatients, 1,971 emergency patients, of whom 788 had SHD, and 1,059 outpatients, of whom 271 had SHD. No patient appears in two cohorts.

### 2.2 Models

Seven models were compared. Four come from the earlier re-analysis [2]:

- a ResNet trained on EchoNext;
- the same ResNet frozen at a seeded random initialisation, with one logistic regression per finding fitted on its features (a linear probe), as a negative control;
- ECGFounder (Net1D, 30.8 million parameters) [8], frozen, with the same probes;
- the published EchoNext mini-model, with its authors' weights [1].

The other three fine-tune ECGFounder end to end on EchoNext from its published weights, with a twelve-output head: AdamW at a learning rate of 1e-4, weight decay 0.01, a cosine schedule, batches of 64 and ten epochs, under seeds 0, 1 and 2. The epoch retained maximised the mean AUROC on the tenth of training patients held out by the ResNet. All three seeds retained epoch 2 (holdout mean AUROC 0.781, 0.780 and 0.779), after which the holdout AUROC declined while the training loss continued to fall. Training used PyTorch 2.2.2 on one GPU, in 2.4 to 4.0 hours per seed.

### 2.3 Covariate-shift weights

The weights come from a domain classifier trained to distinguish calibration inpatients from the unlabelled ECGs of the target setting: an L2-penalised logistic regression on standardised covariates, with the penalty chosen by five-fold cross-validation on log loss and each ECG's prediction taken from a model fitted without it (five-fold cross-fitting). The weight w(x) is the classifier's odds rescaled by the cohort sizes and capped at the 99th centile of the calibration weights. All models share the same weights and differ only in their scores.

Three covariate sets were compared:

- age and sex, a lower reference, since the probability of SHD given age and sex alone is unlikely to be invariant across care settings;
- the seven ECG measurements distributed with EchoNext: age, sex, ventricular and atrial rates, and PR, QRS and QTc intervals;
- the frozen ECGFounder embedding, the 1,024 features the pre-trained network computes from each tracing.

If the probability of SHD given the ECG is the same in both settings, the ratio of target to inpatient covariate densities among cases equals w(x) up to a constant. The conformal quantile of each class is therefore weighted by w(x) alone, and the constant cancels. The AUROC of the domain classifier measures the covariate shift between each target and the calibration inpatients.

### 2.4 Thresholds and flags

For each finding and for the composite, class-conditional conformal thresholds were set for 90% sensitivity on the calibration inpatients, either unweighted (the inpatient threshold) or weighted by w(x). An ECG is flagged when SHD is in its conformal set, including sets that hold both classes. Sensitivity, specificity, the proportion of ECGs flagged and the PPV are reported with 95% Wilson intervals [9].

### 2.5 Pre-specified criteria

A finding was judged when it counted at least 50 cases in the target. Among outpatients four did: LVEF at most 45% (70 cases), LV wall thickness at least 13 mm (146), moderate or greater aortic stenosis (51) and PASP at least 45 mmHg (63).

Recalibration with weights on the embedding meets the primary criterion when two conditions hold: the 95% Wilson interval of composite sensitivity lies within 85% to 95%, and a strict majority of judged findings exceeds 80% sensitivity in a one-sided score test at 0.025, with Holm's correction [10].

Both conditions are evaluated at the effective sample size, which the protocol does not define. Weighting concentrates the threshold on few calibration cases; the Kish effective sample size is the number of equally weighted cases to which the weighted sample is equivalent [11]. Two readings are reported. The primary reading takes the smaller of the Kish size and the target case count, since the weighted threshold rests on the calibration cases: 213 among outpatients. The second takes the target case count, on which sensitivity is measured: 271. The composite condition has a floor and a ceiling: at 213 cases it is met only for an observed sensitivity of 90.1% to 92.0%, since the lower bound reaches 85% only above 90% and the upper bound exceeds 95% above 92.0%. A recalibration exactly on target therefore meets it with probability 0.40, and with probability 0.61 at 271 cases (Table 5).

The protocol judged the criterion among outpatients. The emergency cohort, whose weights and sensitivities were computed in the same analysis, is reported as a secondary target, judged by the same rules after the outpatient results were read. Eight findings counted at least 50 emergency cases: LVEF at most 45% (310), LV wall thickness at least 13 mm (392), moderate or greater aortic stenosis (66), mitral regurgitation (119), tricuspid regurgitation (131) and RV dysfunction (145), PASP at least 45 mmHg (275) and peak TR velocity at least 3.2 m/s (147). The effective size under the primary reading is 571, at which the probability above is 0.94.

Fine-tuning was to halve the loss of composite sensitivity among outpatients at the inpatient threshold: for each of the three seeds, the upper bound of the loss, 90% minus the lower Wilson bound, had to be at most 9 points.

### 2.6 Threshold recalibration on local labels

As a labelled reference, the per-label thresholds were recalibrated on n labelled outpatients drawn from one half of the outpatients and evaluated on the other half, for n from 25 to 400, over 200 draws.

### 2.7 Calibration and selective prediction

For each model, finding and care setting, calibration is reported as the calibration slope, calibration-in-the-large, the integrated calibration index (ICI) [12], here with a Gaussian-kernel smoother on logit p in place of loess, and the Brier score. Confidence is the distance from the inpatient threshold on the logit scale. Abstaining on the least confident ECGs traces a risk-coverage curve, summarised by its generalised area (AUGRC) [13] and by the sensitivity retained at 80% coverage. Intervals are the 2.5th and 97.5th percentiles of 500 bootstrap resamples of target ECGs, with the threshold held fixed.

## 3. Results

### 3.1 The covariate shift was largest among outpatients and on the embedding

Table 1. AUROC of the domain classifier separating calibration inpatients from each target, by covariate set. The inpatient row compares two samples of one population and is a negative control.

<!-- table:domain -->
| Target | age, sex | ECG measurements | embedding |
|---|---|---|---|
| inpatient | 0.499 | 0.504 | 0.485 |
| emergency | 0.552 | 0.590 | 0.768 |
| outpatient | 0.576 | 0.728 | 0.836 |
<!-- /table -->

### 3.2 Weighting on the embedding brought outpatient sensitivity close to 90%

Table 2. Composite sensitivity among outpatients with its 95% Wilson interval. For the weighted thresholds, the interval and n are at the effective size under the primary reading.

<!-- table:composite -->
| Arm | Inpatient threshold | Weighted on age, sex | Weighted on ECG measurements | Weighted on embedding |
|---|---|---|---|---|
| ResNet trained on EchoNext | 71.6% (65.9%–76.6%) | 74.2% (68.6%–79.0%, n=271) | 79.3% (74.1%–83.7%, n=271) | 89.3% (84.3%–92.7%, n=213) |
| ResNet at random initialisation, probes | 78.6% (73.3%–83.1%) | 81.5% (76.5%–85.7%, n=271) | 83.8% (78.9%–87.7%, n=271) | 87.8% (82.7%–91.5%, n=213) |
| ECGFounder frozen, probes | 71.6% (65.9%–76.6%) | 72.7% (67.1%–77.7%, n=271) | 78.6% (73.3%–83.1%, n=271) | 89.7% (84.9%–93.1%, n=213) |
| EchoNext mini-model | 72.7% (67.1%–77.7%) | 73.8% (68.3%–78.7%, n=271) | 82.3% (77.3%–86.4%, n=271) | 89.7% (84.9%–93.1%, n=213) |
| ECGFounder fine-tuned, seed 0 | 72.7% (67.1%–77.7%) | 73.8% (68.3%–78.7%, n=271) | 80.1% (74.9%–84.4%, n=271) | 87.1% (81.7%–90.7%, n=213) |
| ECGFounder fine-tuned, seed 1 | 72.0% (66.3%–77.0%) | 73.4% (67.9%–78.3%, n=271) | 81.5% (76.5%–85.7%, n=271) | 88.2% (83.2%–91.9%, n=213) |
| ECGFounder fine-tuned, seed 2 | 75.6% (70.2%–80.4%) | 77.9% (72.5%–82.4%, n=271) | 81.2% (76.1%–85.4%, n=271) | 89.3% (84.3%–92.7%, n=213) |
<!-- /table -->

Weighting on age and sex changed sensitivity by at most 3 points. Weighting on the ECG measurements recovered part of the loss, and weighting on the embedding brought every model within 3 points of 90%, with lower bounds of 81.7% to 84.9%.

Table 3. Sensitivity among outpatients for each judged finding, weighted on the embedding, with the effective size under the primary reading. Findings with an asterisk are those whose Holm-corrected test rejects a sensitivity of 80%. Criterion: both conditions of section 2.5.

<!-- table:anomalies -->
| Arm | LVEF ≤ 45% | LV wall ≥ 13 mm | Aortic stenosis | PASP ≥ 45 mmHg | Criterion |
|---|---|---|---|---|---|
| ResNet trained on EchoNext | 88.6% (n=70) | 89.7%* (n=101) | 100.0%* (n=29) | 82.5% (n=63) | not met |
| ResNet at random initialisation, probes | 87.1% (n=70) | 91.1%* (n=101) | 94.1% (n=29) | 95.2%* (n=63) | not met |
| ECGFounder frozen, probes | 92.9%* (n=70) | 95.9%* (n=101) | 100.0%* (n=29) | 90.5%* (n=63) | not met |
| EchoNext mini-model | 84.3% (n=70) | 91.8%* (n=101) | 98.0%* (n=29) | 84.1% (n=63) | not met |
| ECGFounder fine-tuned, seed 0 | 94.3%* (n=70) | 91.8%* (n=101) | 100.0%* (n=29) | 81.0% (n=63) | not met |
| ECGFounder fine-tuned, seed 1 | 91.4% (n=70) | 89.0% (n=101) | 100.0%* (n=29) | 82.5% (n=63) | not met |
| ECGFounder fine-tuned, seed 2 | 92.9%* (n=70) | 89.7%* (n=101) | 100.0%* (n=29) | 85.7% (n=63) | not met |
<!-- /table -->

### 3.3 The outpatient verdict depended on the reading of the effective size

Table 4. Bounds of the 95% Wilson interval of composite sensitivity weighted on the embedding, the size at which both conditions are judged, the number of judged findings whose test rejects 80%, and the criterion, under each reading.

<!-- table:readings -->
| Arm | Outpatient, smaller of cases and Kish size | Outpatient, target cases | Emergency, smaller of cases and Kish size | Emergency, target cases |
|---|---|---|---|---|
| ResNet trained on EchoNext | 84.3%–92.7% (n=213), 2 of 4 findings, not met | 85.1%–92.4% (n=271), 2 of 4 findings, not met | 88.8%–93.4% (n=571), 6 of 8 findings, met | 89.3%–93.2% (n=788), 6 of 8 findings, met |
| ResNet at random initialisation, probes | 82.7%–91.5% (n=213), 2 of 4 findings, not met | 83.4%–91.2% (n=271), 3 of 4 findings, not met | 86.9%–91.9% (n=571), 5 of 8 findings, met | 87.3%–91.5% (n=788), 6 of 8 findings, met |
| ECGFounder frozen, probes | 84.9%–93.1% (n=213), 4 of 4 findings, not met | 85.5%–92.8% (n=271), 4 of 4 findings, met | 85.6%–90.8% (n=571), 5 of 8 findings, met | 86.0%–90.5% (n=788), 5 of 8 findings, met |
| EchoNext mini-model | 84.9%–93.1% (n=213), 2 of 4 findings, not met | 85.5%–92.8% (n=271), 2 of 4 findings, not met | 86.9%–91.9% (n=571), 5 of 8 findings, met | 87.3%–91.5% (n=788), 5 of 8 findings, met |
| ECGFounder fine-tuned, seed 0 | 81.7%–90.7% (n=213), 3 of 4 findings, not met | 82.6%–90.6% (n=271), 3 of 4 findings, not met | 85.9%–91.1% (n=571), 7 of 8 findings, met | 86.3%–90.7% (n=788), 7 of 8 findings, met |
| ECGFounder fine-tuned, seed 1 | 83.2%–91.9% (n=213), 1 of 4 findings, not met | 83.8%–91.5% (n=271), 3 of 4 findings, not met | 88.1%–92.8% (n=571), 7 of 8 findings, met | 88.5%–92.6% (n=788), 7 of 8 findings, met |
| ECGFounder fine-tuned, seed 2 | 84.3%–92.7% (n=213), 3 of 4 findings, not met | 85.1%–92.4% (n=271), 3 of 4 findings, met | 86.3%–91.4% (n=571), 5 of 8 findings, met | 86.7%–91.1% (n=788), 6 of 8 findings, met |
<!-- /table -->

Among outpatients, ECGFounder frozen (85.5%–92.8%, four of four findings) and ECGFounder fine-tuned under seed 2 (85.1%–92.4%, three of four) met the criterion at 271 cases; no model met it at 213. Among emergency patients, every model met it under both readings.

Table 5. Observed composite sensitivities for which the 95% Wilson interval lies within 85% to 95%, at the size of each reading, and the probability of such an observation when the true sensitivity is 90%.

<!-- table:power -->
| Target | Reading | n | Sensitivity meeting the condition | Probability |
|---|---|---|---|---|
| outpatient | smaller of cases and Kish size | 213 | 90.1%–92.0% | 0.40 |
| outpatient | target cases | 271 | 89.3%–92.3% | 0.61 |
| emergency | smaller of cases and Kish size | 571 | 88.1%–93.2% | 0.94 |
| emergency | target cases | 788 | 87.6%–93.4% | 0.99 |
<!-- /table -->

### 3.4 Among emergency patients, the criterion was met for every model

Table 6. Composite sensitivity among emergency patients, as in Table 2.

<!-- table:composite_emergency -->
| Arm | Inpatient threshold | Weighted on age, sex | Weighted on ECG measurements | Weighted on embedding |
|---|---|---|---|---|
| ResNet trained on EchoNext | 86.2% (83.6%–88.4%) | 87.1% (84.5%–89.2%, n=788) | 87.9% (85.5%–90.0%, n=788) | 91.5% (88.8%–93.4%, n=571) |
| ResNet at random initialisation, probes | 84.4% (81.7%–86.8%) | 85.8% (83.2%–88.1%, n=788) | 86.7% (84.1%–88.9%, n=788) | 89.6% (86.9%–91.9%, n=571) |
| ECGFounder frozen, probes | 82.9% (80.1%–85.3%) | 83.6% (80.9%–86.0%, n=788) | 85.0% (82.4%–87.3%, n=788) | 88.5% (85.6%–90.8%, n=571) |
| EchoNext mini-model | 85.5% (82.9%–87.8%) | 86.2% (83.6%–88.4%, n=788) | 87.2% (84.7%–89.3%, n=788) | 89.6% (86.9%–91.9%, n=571) |
| ECGFounder fine-tuned, seed 0 | 82.6% (79.8%–85.1%) | 83.1% (80.3%–85.6%, n=788) | 84.8% (82.1%–87.1%, n=788) | 88.7% (85.9%–91.1%, n=571) |
| ECGFounder fine-tuned, seed 1 | 85.3% (82.6%–87.6%) | 85.7% (83.0%–87.9%, n=788) | 87.6% (85.1%–89.7%, n=788) | 90.7% (88.1%–92.8%, n=571) |
| ECGFounder fine-tuned, seed 2 | 84.9% (82.2%–87.2%) | 85.4% (82.8%–87.7%, n=788) | 87.6% (85.1%–89.7%, n=788) | 89.1% (86.3%–91.4%, n=571) |
<!-- /table -->

At the inpatient threshold, sensitivity among emergency patients was 82.6% to 86.2%, closer to 90% than among outpatients, consistent with the smaller covariate shift (domain AUROC 0.768 on the embedding, against 0.836). Weighting on the embedding gave 88.5% to 91.5%, with lower bounds of 85.6% to 88.8% at 571 effective cases.

Table 7. Judged findings among emergency patients, as in Table 3.

<!-- table:anomalies_emergency -->
| Arm | LVEF ≤ 45% | LV wall ≥ 13 mm | Aortic stenosis | Mitral regurgitation | Tricuspid regurgitation | RV dysfunction | PASP ≥ 45 mmHg | TR velocity ≥ 3.2 m/s | Criterion |
|---|---|---|---|---|---|---|---|---|---|
| ResNet trained on EchoNext | 91.3%* (n=273) | 93.4%* (n=235) | 87.9% (n=66) | 92.4%* (n=85) | 93.1%* (n=99) | 94.5%* (n=129) | 95.3%* (n=174) | 85.0% (n=89) | met |
| ResNet at random initialisation, probes | 90.0%* (n=273) | 93.4%* (n=235) | 87.9% (n=66) | 89.1% (n=85) | 94.7%* (n=99) | 91.0%* (n=129) | 95.3%* (n=174) | 75.5% (n=89) | met |
| ECGFounder frozen, probes | 87.4%* (n=273) | 91.3%* (n=235) | 84.8% (n=66) | 83.2% (n=85) | 92.4%* (n=99) | 91.0%* (n=129) | 92.4%* (n=174) | 87.8% (n=89) | met |
| EchoNext mini-model | 86.8%* (n=273) | 91.3%* (n=235) | 84.8% (n=66) | 84.9% (n=85) | 95.4%* (n=99) | 89.7%* (n=129) | 94.9%* (n=174) | 87.1% (n=89) | met |
| ECGFounder fine-tuned, seed 0 | 88.7%* (n=273) | 95.4%* (n=235) | 92.4%* (n=66) | 90.8%* (n=85) | 91.6%* (n=99) | 91.0%* (n=129) | 91.6%* (n=174) | 83.7% (n=89) | met |
| ECGFounder fine-tuned, seed 1 | 89.7%* (n=273) | 95.7%* (n=235) | 86.4% (n=66) | 89.9%* (n=85) | 90.1%* (n=99) | 94.5%* (n=129) | 92.4%* (n=174) | 90.5%* (n=89) | met |
| ECGFounder fine-tuned, seed 2 | 88.1%* (n=273) | 91.3%* (n=235) | 87.9% (n=66) | 89.1% (n=85) | 90.1%* (n=99) | 93.1%* (n=129) | 92.0%* (n=174) | 83.0% (n=89) | met |
<!-- /table -->

### 3.5 Weighted thresholds flagged more patients at lower specificity

Table 8. Sensitivity, specificity, proportion of ECGs flagged and PPV of the composite flag at the inpatient threshold and at the threshold weighted on the embedding. Intervals for sensitivity are in Tables 2 and 6.

<!-- table:flags -->
| Arm | Target | Threshold | Sensitivity | Specificity | Flagged | PPV |
|---|---|---|---|---|---|---|
| ResNet trained on EchoNext | outpatient | inpatient | 71.6% | 71.1% (67.8%–74.1%) | 39.8% | 46.0% (41.3%–50.7%) |
| ResNet trained on EchoNext | outpatient | weighted on embedding | 89.3% | 42.5% (39.1%–46.0%) | 65.6% | 34.8% (31.4%–38.4%) |
| ResNet at random initialisation, probes | outpatient | inpatient | 78.6% | 57.4% (53.9%–60.8%) | 51.8% | 38.8% (34.8%–42.9%) |
| ResNet at random initialisation, probes | outpatient | weighted on embedding | 87.8% | 29.1% (26.0%–32.3%) | 75.3% | 29.9% (26.8%–33.1%) |
| ECGFounder frozen, probes | outpatient | inpatient | 71.6% | 71.3% (68.1%–74.4%) | 39.7% | 46.2% (41.5%–51.0%) |
| ECGFounder frozen, probes | outpatient | weighted on embedding | 89.7% | 42.5% (39.1%–46.0%) | 65.7% | 34.9% (31.5%–38.5%) |
| EchoNext mini-model | outpatient | inpatient | 72.7% | 71.1% (67.8%–74.1%) | 40.1% | 46.4% (41.7%–51.1%) |
| EchoNext mini-model | outpatient | weighted on embedding | 89.7% | 37.4% (34.1%–40.9%) | 69.5% | 33.0% (29.7%–36.5%) |
| ECGFounder fine-tuned, seed 0 | outpatient | inpatient | 72.7% | 76.0% (72.9%–78.9%) | 36.4% | 51.0% (46.1%–56.0%) |
| ECGFounder fine-tuned, seed 0 | outpatient | weighted on embedding | 87.1% | 49.1% (45.6%–52.6%) | 60.2% | 37.0% (33.4%–40.9%) |
| ECGFounder fine-tuned, seed 1 | outpatient | inpatient | 72.0% | 75.8% (72.6%–78.6%) | 36.4% | 50.5% (45.5%–55.5%) |
| ECGFounder fine-tuned, seed 1 | outpatient | weighted on embedding | 88.2% | 49.7% (46.3%–53.2%) | 60.0% | 37.6% (34.0%–41.5%) |
| ECGFounder fine-tuned, seed 2 | outpatient | inpatient | 75.6% | 72.6% (69.4%–75.6%) | 39.8% | 48.7% (44.0%–53.5%) |
| ECGFounder fine-tuned, seed 2 | outpatient | weighted on embedding | 89.3% | 45.8% (42.4%–49.3%) | 63.2% | 36.2% (32.6%–39.9%) |
| ResNet trained on EchoNext | emergency | inpatient | 86.2% | 61.0% (58.2%–63.8%) | 57.8% | 59.6% (56.7%–62.4%) |
| ResNet trained on EchoNext | emergency | weighted on embedding | 91.5% | 49.0% (46.2%–51.9%) | 67.2% | 54.5% (51.8%–57.1%) |
| ResNet at random initialisation, probes | emergency | inpatient | 84.4% | 54.5% (51.7%–57.3%) | 61.0% | 55.3% (52.5%–58.1%) |
| ResNet at random initialisation, probes | emergency | weighted on embedding | 89.6% | 41.0% (38.2%–43.8%) | 71.2% | 50.3% (47.7%–52.9%) |
| ECGFounder frozen, probes | emergency | inpatient | 82.9% | 60.0% (57.2%–62.8%) | 57.1% | 58.0% (55.1%–60.8%) |
| ECGFounder frozen, probes | emergency | weighted on embedding | 88.5% | 49.6% (46.8%–52.5%) | 65.6% | 53.9% (51.2%–56.6%) |
| EchoNext mini-model | emergency | inpatient | 85.5% | 57.7% (54.8%–60.4%) | 59.6% | 57.4% (54.5%–60.2%) |
| EchoNext mini-model | emergency | weighted on embedding | 89.6% | 49.5% (46.6%–52.3%) | 66.2% | 54.1% (51.4%–56.8%) |
| ECGFounder fine-tuned, seed 0 | emergency | inpatient | 82.6% | 65.4% (62.7%–68.1%) | 53.8% | 61.4% (58.4%–64.3%) |
| ECGFounder fine-tuned, seed 0 | emergency | weighted on embedding | 88.7% | 52.9% (50.1%–55.7%) | 63.7% | 55.7% (52.9%–58.4%) |
| ECGFounder fine-tuned, seed 1 | emergency | inpatient | 85.3% | 61.3% (58.5%–64.0%) | 57.3% | 59.5% (56.6%–62.3%) |
| ECGFounder fine-tuned, seed 1 | emergency | weighted on embedding | 90.7% | 48.7% (45.9%–51.5%) | 67.1% | 54.1% (51.4%–56.8%) |
| ECGFounder fine-tuned, seed 2 | emergency | inpatient | 84.9% | 61.9% (59.1%–64.6%) | 56.8% | 59.7% (56.8%–62.6%) |
| ECGFounder fine-tuned, seed 2 | emergency | weighted on embedding | 89.1% | 51.1% (48.3%–54.0%) | 64.9% | 54.8% (52.1%–57.6%) |
<!-- /table -->

Among outpatients, weighting raised the proportion flagged from 36.4%–51.8% to 60.0%–75.3% across models; specificity fell from 57.4%–76.0% to 29.1%–49.7% and PPV from 38.8%–51.0% to 29.9%–37.6%. Among emergency patients, specificity fell from 54.5%–65.4% to 41.0%–52.9% and PPV from 55.3%–61.4% to 50.3%–55.7%. No model gained sensitivity at constant specificity.

### 3.6 Threshold recalibration on local labels

Table 9. Composite sensitivity in one half of the outpatients when the per-label thresholds were recalibrated on n labelled outpatients from the other half: mean and 10th to 90th percentiles over 200 draws. The mean can lie outside that range when a tenth of the draws flag nearly every ECG; with 25 or 50 outpatients, few cases are drawn and the threshold often flags every ECG.

<!-- table:ladder -->
| Arm | 25 labels | 50 labels | 100 labels | 200 labels | 400 labels |
|---|---|---|---|---|---|
| ResNet trained on EchoNext | 98.9% (99.3%–100.0%) | 94.5% (89.1%–100.0%) | 93.4% (89.7%–99.3%) | 92.7% (90.5%–95.6%) | 92.1% (92.0%–92.0%) |
| ResNet at random initialisation, probes | 98.5% (94.2%–100.0%) | 90.7% (81.8%–100.0%) | 88.8% (83.9%–94.2%) | 87.3% (84.7%–89.1%) | 86.7% (85.4%–87.6%) |
| ECGFounder frozen, probes | 99.3% (100.0%–100.0%) | 96.0% (88.2%–100.0%) | 95.3% (88.3%–100.0%) | 95.2% (91.2%–99.3%) | 94.8% (92.7%–95.6%) |
| EchoNext mini-model | 99.2% (99.3%–100.0%) | 96.8% (93.5%–100.0%) | 97.6% (94.9%–99.3%) | 97.3% (94.9%–99.3%) | 97.3% (94.9%–99.3%) |
| ECGFounder fine-tuned, seed 0 | 98.8% (97.8%–100.0%) | 93.7% (87.6%–100.0%) | 91.5% (88.3%–97.8%) | 90.6% (88.3%–94.9%) | 89.7% (89.1%–89.8%) |
| ECGFounder fine-tuned, seed 1 | 98.9% (96.4%–100.0%) | 93.8% (83.2%–100.0%) | 92.8% (83.9%–99.3%) | 91.5% (83.9%–95.6%) | 91.1% (90.7%–92.0%) |
| ECGFounder fine-tuned, seed 2 | 98.6% (97.7%–100.0%) | 92.5% (83.9%–100.0%) | 89.4% (83.9%–97.8%) | 88.2% (83.9%–92.0%) | 87.2% (86.9%–87.6%) |
<!-- /table -->

### 3.7 Calibration and selective prediction

Table 10. Calibration and selective prediction for the composite among inpatients and outpatients, with bootstrap intervals.

<!-- table:calibration -->
| Arm | Target | Slope | Intercept | ICI | Brier | AUGRC | Sens. at 80% kept |
|---|---|---|---|---|---|---|---|
| ResNet trained on EchoNext | inpatient | 0.74 (0.68–0.81) | -0.25 (-0.36–-0.15) | 0.076 (0.061–0.092) | 0.180 (0.170–0.190) | 0.123 (0.114–0.133) | 95.1% (93.7%–96.2%) |
| ResNet trained on EchoNext | outpatient | 0.82 (0.71–0.94) | -0.41 (-0.59–-0.23) | 0.069 (0.052–0.090) | 0.138 (0.125–0.153) | 0.108 (0.094–0.120) | 76.3% (71.6%–82.2%) |
| ResNet at random initialisation, probes | inpatient | 0.98 (0.88–1.10) | -0.11 (-0.20–-0.02) | 0.023 (0.014–0.043) | 0.192 (0.184–0.200) | 0.138 (0.128–0.149) | 94.8% (93.4%–96.0%) |
| ResNet at random initialisation, probes | outpatient | 1.08 (0.89–1.27) | -0.74 (-0.89–-0.62) | 0.139 (0.120–0.162) | 0.173 (0.164–0.185) | 0.168 (0.152–0.184) | 81.3% (76.4%–86.0%) |
| ECGFounder frozen, probes | inpatient | 0.88 (0.80–0.95) | -0.16 (-0.26–-0.06) | 0.045 (0.032–0.062) | 0.182 (0.174–0.191) | 0.124 (0.115–0.135) | 93.8% (92.1%–95.2%) |
| ECGFounder frozen, probes | outpatient | 0.90 (0.77–1.04) | -0.43 (-0.59–-0.28) | 0.069 (0.053–0.093) | 0.148 (0.136–0.161) | 0.112 (0.098–0.125) | 76.8% (71.4%–82.5%) |
| EchoNext mini-model | inpatient | 0.98 (0.89–1.07) | 0.18 (0.08–0.26) | 0.032 (0.022–0.049) | 0.184 (0.176–0.194) | 0.131 (0.122–0.142) | 95.5% (93.9%–96.6%) |
| EchoNext mini-model | outpatient | 1.04 (0.91–1.20) | -0.16 (-0.33–-0.02) | 0.035 (0.023–0.058) | 0.141 (0.129–0.155) | 0.118 (0.104–0.133) | 78.4% (72.6%–83.3%) |
| ECGFounder fine-tuned, seed 0 | inpatient | 0.94 (0.86–1.02) | -0.39 (-0.49–-0.30) | 0.068 (0.056–0.085) | 0.179 (0.172–0.188) | 0.117 (0.108–0.128) | 93.8% (92.2%–95.2%) |
| ECGFounder fine-tuned, seed 0 | outpatient | 1.00 (0.86–1.18) | -0.66 (-0.82–-0.51) | 0.106 (0.085–0.127) | 0.146 (0.135–0.159) | 0.096 (0.084–0.111) | 77.7% (71.9%–83.1%) |
| ECGFounder fine-tuned, seed 1 | inpatient | 0.87 (0.80–0.94) | -0.13 (-0.24–-0.03) | 0.054 (0.041–0.070) | 0.175 (0.166–0.184) | 0.119 (0.111–0.130) | 93.9% (92.4%–95.4%) |
| ECGFounder fine-tuned, seed 1 | outpatient | 0.93 (0.80–1.07) | -0.31 (-0.47–-0.15) | 0.053 (0.036–0.075) | 0.136 (0.124–0.150) | 0.099 (0.086–0.113) | 78.1% (73.2%–83.7%) |
| ECGFounder fine-tuned, seed 2 | inpatient | 0.83 (0.76–0.90) | -0.05 (-0.15–0.05) | 0.050 (0.037–0.066) | 0.176 (0.168–0.186) | 0.122 (0.113–0.132) | 93.2% (91.6%–94.7%) |
| ECGFounder fine-tuned, seed 2 | outpatient | 0.97 (0.83–1.12) | -0.39 (-0.56–-0.23) | 0.063 (0.045–0.084) | 0.136 (0.124–0.149) | 0.114 (0.101–0.128) | 79.6% (74.6%–84.9%) |
<!-- /table -->

At 80% coverage, sensitivity remained above 93% among inpatients for every model and ranged from 76.3% to 81.3% among outpatients. Confidence derived from inpatients did not identify the outpatient cases missed at the threshold.

### 3.8 Fine-tuning ECGFounder did not reduce the loss

Table 11. Composite sensitivity among outpatients at the inpatient threshold, with its lower Wilson bound, and the loss from 90% with its upper bound.

<!-- table:a1 -->
| Arm | Outpatient sensitivity | Loss, points | Loss at most 9 points |
|---|---|---|---|
| ResNet trained on EchoNext | 71.6% (low 65.9%) | 18.4 (up to 24.1) | no |
| ResNet at random initialisation, probes | 78.6% (low 73.3%) | 11.4 (up to 16.7) | no |
| ECGFounder frozen, probes | 71.6% (low 65.9%) | 18.4 (up to 24.1) | no |
| EchoNext mini-model | 72.7% (low 67.1%) | 17.3 (up to 22.9) | no |
| ECGFounder fine-tuned, seed 0 | 72.7% (low 67.1%) | 17.3 (up to 22.9) | no |
| ECGFounder fine-tuned, seed 1 | 72.0% (low 66.3%) | 18.0 (up to 23.7) | no |
| ECGFounder fine-tuned, seed 2 | 75.6% (low 70.2%) | 14.4 (up to 19.8) | no |
<!-- /table -->

## 4. Discussion

### 4.1 Principal findings

Weighting on the ECGFounder embedding raised composite sensitivity for SHD from 71.6%–78.6% to 87.1%–89.7% among outpatients and from 82.6%–86.2% to 88.5%–91.5% among emergency patients, without target diagnoses. Among outpatients, no model met the pre-specified criterion under the primary reading, but at 213 effective cases the criterion is met only for an observed sensitivity of 90.1% to 92.0%, a range that excludes the 90% target itself, and a recalibration exactly on target meets it with probability 0.40; the outpatient result is therefore inconclusive rather than negative. At the 271 outpatient cases two models met it, and among emergency patients, where that probability is 0.94, all seven did. The weighted thresholds reached these sensitivities by flagging more patients: 60.0% to 75.3% of outpatients, at a specificity of 29.1% to 49.7%. Fine-tuning ECGFounder on EchoNext did not reduce the loss of sensitivity at the inpatient threshold: 14.4 to 18.0 points across seeds, against 18.4 for the ResNet.

### 4.2 Comparison with prior work

The external validation of the EchoNext mini-model in MIMIC-IV reported its AUROC by care setting and recalibrated it on local labels, without conformal thresholds [3]. Here, thresholds recalibrated on 100 to 200 labelled outpatients reached a mean sensitivity of at least 90% for five of the seven models, at the cost of obtaining their echocardiograms. A study of uncertainty-aware triage of SHD on EchoNext reports, in its abstract, neither conformal thresholds nor a comparison of care settings [14]. The bound of Han and Qu on label-free class-conditional coverage under shift [15] is consistent with the residual shortfall among outpatients but is not demonstrated by it. Two explanations are compatible with that shortfall: the probability of SHD given the ECG may differ between care settings, which violates the covariate-shift assumption [6], or the embedding may not encode the features on which the settings differ. The larger gain from the embedding than from age and sex suggests that the shift lies mainly in ECG morphology rather than in demographics.

### 4.3 Strengths and limitations

The criterion was fixed before any result was read. Seven models, including an untrained negative control and three fine-tuned seeds, were evaluated on the same patients with one reference standard, against three covariate sets and a labelled reference.

The result comes from one hospital system, and its magnitude in a second system is unknown; a transfer from Columbia to Boston awaits credentialed access to MIMIC-IV-Echo. The protocol does not define the effective size, and neither reading reported is the protocol's own. The emergency cohort was judged after the outpatient results were read, and its verdict is secondary. The protocol fixed no bound on specificity or PPV, so the cost of the flag is described, not judged. The weights use the frozen ECGFounder embedding for every model, including models that do not read it. The audit of label-free coverage on target labels named in the protocol was not implemented; the thresholds recalibrated on labelled outpatients serve as the labelled reference instead. Three TRIPOD+AI items are not covered: the justification of the sample size, the handling of missing data (none in the covariates used) and a protocol for updating the model after deployment.

### 4.4 Implications for practice and research

A site holding unlabelled ECGs can move a delivered threshold towards its target sensitivity by weighting on a foundation-model embedding, but the weighted threshold flags more patients, and the resulting number of echocardiograms per case detected has to be weighed against the cases missed at the delivered threshold. In this cohort, label-free recalibration did not replace a labelled sample of 100 to 200 outpatients. A confirmatory study would pre-specify the effective size and a bound on specificity, and would require several hundred target cases for the interval criterion to have adequate power.

This is a retrospective study on public, de-identified data. Nothing here is intended to guide the care of any patient.

## Declarations

**Competing interests.** The author has worked at Idoven, which develops ECG analysis software, and at Anumana, which commercialises Mayo Clinic's AI-ECG work. Neither company had any part in the design, conduct, funding or reporting of this study, and none of their data, models or software were used.

**Funding.** None.

**Data.** EchoNext is distributed by PhysioNet under its Restricted Health Data License; its tracings, and the scores, embeddings, weights and fine-tuned models derived from them, stay outside this repository, which holds counts and aggregate figures only.

**Ethics.** This is a secondary analysis of publicly released, de-identified ECG data, and it involved no contact with patients. The DISCOVERY trial and its analyses were approved by the Institutional Review Board at NewYork-Presbyterian Hospital/Columbia University Irving Medical Center, as EchoNext's PhysioNet page states; no approval number is given there. No attempt was made to re-identify any patient.

**Code and numbers.** `scripts/structure.py` computes the result files under `results/` from the scores and renders every table of this report from them; `tests/test_structure_results.py` fails if a table or a percentage of the text departs from those files. The SHA-256 digests of the fine-tuned models are recorded with their scores.

## References

1. Poterucha TJ, Jing L, Ricart RP, et al. Detecting structural heart disease from electrocardiograms using AI. *Nature* 2025;644(8075):221-230. doi:10.1038/s41586-025-09227-0.
2. Abbou R. Predicting an ECG model's positive predictive value from a clinic's prevalence: a retrospective measurement across hospitals and care settings. 2026. https://github.com/rabbou/ecg-ai-portability.
3. Otabor E, Hassan A, Okunlola A, et al. Transportability of an artificial intelligence electrocardiography model for structural heart disease: external validation and recalibration of EchoNext-Mini in MIMIC-IV. *Eur Heart J Digit Health* 2026;7(8):ztag147. doi:10.1093/ehjdh/ztag147.
4. Angelopoulos AN, Bates S. Conformal prediction: a gentle introduction. *Found Trends Mach Learn* 2023;16(4):494-591. doi:10.1561/2200000101.
5. Vovk V. Conditional validity of inductive conformal predictors. In: *Proceedings of the Asian Conference on Machine Learning*. PMLR 2012;25:475-490.
6. Tibshirani RJ, Foygel Barber R, Candès EJ, Ramdas A. Conformal prediction under covariate shift. In: *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. arXiv:1904.06019.
7. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378. doi:10.1136/bmj-2023-078378.
8. Li J, Aguirre AD, Moura Junior V, et al. An electrocardiogram foundation model built on over 10 million recordings. *NEJM AI* 2025;2(7). doi:10.1056/AIoa2401033.
9. Wilson EB. Probable inference, the law of succession, and statistical inference. *J Am Stat Assoc* 1927;22(158):209-212. doi:10.1080/01621459.1927.10502953.
10. Holm S. A simple sequentially rejective multiple test procedure. *Scand J Stat* 1979;6:65-70. doi:10.2307/4615733.
11. Kish L. *Survey Sampling*. New York: Wiley; 1965.
12. Austin PC, Steyerberg EW. The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models. *Stat Med* 2019;38(21):4051-4065. doi:10.1002/sim.8281.
13. Traub J, Bungert TJ, Lüth CT, et al. Overcoming common flaws in the evaluation of selective classification systems. arXiv:2407.01032 [preprint]. 2024.
14. Colebank MJ. Uncertainty-aware classification and triage of structural heart disease using electrocardiography and echocardiography metrics. arXiv:2605.22968 [preprint]. 2026.
15. Han W, Qu L. The label complexity of class-conditional coverage under distribution shift. arXiv:2607.18088 [preprint]. 2026.
