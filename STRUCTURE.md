# Label-free recalibration of an ECG threshold for structural heart disease across care settings: weighted conformal prediction in EchoNext

## Abstract

**Background.** ECG models for structural heart disease (SHD) are applied at a threshold set for a target sensitivity in the derivation population. At Columbia, the threshold set for 90% sensitivity among inpatients detected 71.6% of SHD among outpatients for the published study's trained ResNet ([README](README.md)). Recalibrating the threshold for outpatients conventionally requires outpatients with echocardiographic diagnoses. Weighted conformal prediction recalibrates without target diagnoses under covariate shift, provided that the probability of SHD given the ECG is invariant between settings; this assumption has not been examined for any ECG model of SHD.

**Methods.** Retrospective study in EchoNext v1.1.1, reported following TRIPOD+AI. Seven models were calibrated on 1,903 inpatient ECGs and evaluated on 1,059 outpatients, 271 of them with SHD. Class-conditional conformal thresholds were reweighted by a domain classifier fitted on age and sex, on seven ECG measurements, or on the frozen ECGFounder embedding. The pre-specified criterion required the lower bound of the 95% Wilson interval of composite sensitivity, at the effective sample size, to reach 85%.

**Results.** Weighting on age and sex gave outpatient sensitivities of 72.7% to 81.5% across models, against 71.6% to 78.6% at the inpatient threshold; weighting on the embedding gave 87.1% to 89.7%, at an effective size of 213 cases. No model met the pre-specified criterion. ECGFounder fine-tuned on EchoNext under three seeds reached 72.0% to 75.6% at the inpatient threshold. Thresholds recalibrated on 100 to 200 labelled outpatients reached a mean sensitivity of at least 90% for five of the seven models.

**Conclusions.** Covariate-shift weighting on an ECG foundation-model embedding recovered most, but not all, of the sensitivity lost between inpatients and outpatients. In this cohort, label-free recalibration did not substitute for a labelled outpatient sample of 100 to 200 patients.

## Introduction

A threshold on an ECG model's score is usually set on a calibration sample of the derivation population so that a stated proportion of cases scores above it. Split conformal prediction formalises this choice and guarantees at least the stated sensitivity in a new patient when calibration and target patients are exchangeable, that is, drawn from the same distribution. Between care settings of one hospital this condition fails: in EchoNext the threshold set for 90% sensitivity among inpatients detected 71.6% of outpatient cases for the trained ResNet. Recalibration on labelled outpatients restores the operating point, but requires echocardiograms in the target setting.

Weighted conformal prediction (Tibshirani et al., arXiv:1904.06019) extends the guarantee to covariate shift by weighting each calibration patient by the likelihood ratio of target to source covariates, which can be estimated from unlabelled target ECGs. The guarantee holds when the conditional probability of SHD given the covariates is the same in both settings. Whether this holds between care settings has not been examined for any ECG model of SHD.

We examined whether weighted conformal thresholds, fitted without outpatient diagnoses, restore 90% sensitivity among Columbia outpatients, for seven models and three covariate sets, against a criterion fixed before any result was read.

## Methods

### Participants and outcome

EchoNext v1.1.1 (PhysioNet, restricted licence) holds 100,000 Columbia ECGs, each paired with a transthoracic echocardiogram within one year, which yields eleven findings and their composite, SHD. The published splits were used unchanged. Models were trained on 72,475 ECGs and calibrated on the 1,903 inpatient ECGs of the validation split. The test split gave three target cohorts, with one ECG per patient: 2,203 inpatients, 1,971 emergency patients and 1,059 outpatients, of whom 271 had SHD. No patient appears in two roles (`ecs.echonext.transfer_cohorts` raises otherwise). The study is reported following TRIPOD+AI (Collins et al., BMJ 2024). Under clause 3 of the licence, no ECG, score or weight derived from EchoNext left the analysis machine.

### Models

Seven models were compared. Four come from the published study:

- the study's ResNet, trained on EchoNext;
- the same ResNet frozen at a seeded random initialisation, with one logistic regression per finding fitted on its features (a linear probe), as a negative control;
- ECGFounder (Net1D, 30.8 million parameters), frozen, with the same probes;
- the published EchoNext mini-model, with its authors' weights.

The other three fine-tune ECGFounder end to end on EchoNext (arm A1, `scripts/echonext_finetune.py`), from the published weights with a twelve-output head: AdamW at a learning rate of 1e-4, weight decay 0.01, a cosine schedule, batches of 64 and ten epochs, under seeds 0, 1 and 2. The epoch retained maximised the mean AUROC on the tenth of training patients held out by the ResNet. All three seeds retained epoch 2 (holdout mean AUROC 0.781, 0.780 and 0.779), after which the holdout AUROC declined while the training loss continued to fall. Training ran on an Apple M5 GPU under torch 2.2.2, in 2.4 to 4.0 hours per seed. The weights are stored outside the repository, under `~/data/echonext-derived/models`, with their SHA-256 digests in the score records.

### Covariate-shift weights

The weights come from a domain classifier, a model trained to distinguish calibration inpatients from the unlabelled ECGs of the target setting: an L2-penalised logistic regression on standardised covariates, with the penalty chosen by five-fold cross-validation on log loss and each ECG's prediction taken from a model fitted without it (five-fold cross-fitting). The weight w(x) is the classifier's odds rescaled by the cohort sizes and capped at the 99th centile of the calibration weights. All models share the same weights and differ only in their scores.

Three covariate sets were compared:

- age and sex, a lower reference, since the probability of SHD given age and sex alone is unlikely to be invariant across care settings;
- the seven ECG measurements distributed with EchoNext: age, sex, ventricular and atrial rates, and PR, QRS and QTc intervals;
- the frozen ECGFounder embedding, the 1,024 features the pre-trained network computes from each tracing.

If the probability of SHD given the ECG is the same in both settings, the ratio of outpatient to inpatient covariate densities among cases equals w(x) up to a constant. The conformal quantile of each class is therefore weighted by w(x) alone, and the constant cancels.

The AUROC of the domain classifier quantifies the covariate shift between each target and the calibration inpatients; 0.5 indicates none.

<!-- table:domain -->
| Target | age, sex | ECG measurements | embedding |
|---|---|---|---|
| inpatient | 0.499 | 0.504 | 0.485 |
| emergency | 0.552 | 0.590 | 0.768 |
| outpatient | 0.576 | 0.728 | 0.836 |
<!-- /table -->

The inpatient row compares two samples of the same population; it is a negative control.

### Pre-specified decision rules

The rules were fixed in the protocol of 16 October 2026, before any result was read. A finding was judged when it counted at least 50 cases among test outpatients. Four did: LVEF at most 45% (70 cases), LV wall thickness at least 13 mm (146), moderate or greater aortic stenosis (51) and PASP at least 45 mmHg (63).

Recalibration with weights on the embedding (A2) succeeds when two conditions hold. The 95% Wilson interval of composite sensitivity, a score interval for a proportion that remains valid in small samples, lies within [85%, 95%]. A strict majority of judged findings exceeds 80% sensitivity in a one-sided score test at 0.025, with Holm's correction for four tests.

Both conditions are evaluated at the effective sample size. Weighting concentrates the threshold on few calibration cases; the Kish effective sample size is the number of equally weighted cases to which the weighted sample is equivalent. The effective size is the smaller of that number and the target case count, since the weighted threshold rests on the calibration cases.

Fine-tuning (A1) halves the published loss when, for each of the three seeds, the upper bound of the loss of composite sensitivity among outpatients is at most 9 points, the upper bound being 90% minus the lower Wilson bound.

### Calibration and selective prediction

For each model, finding and care setting, calibration is reported as the calibration slope, calibration-in-the-large, the integrated calibration index (ICI, the mean absolute difference between predicted and observed risk; Austin and Steyerberg, Stat Med 2019, here with a Gaussian-kernel smoother on logit p in place of loess) and the Brier score. The decision is the flag at the threshold set for 90% sensitivity among inpatients, and confidence is the distance from that threshold on the logit scale. Abstaining on the least confident ECGs traces a risk-coverage curve, summarised by its generalised area (AUGRC; Traub et al., arXiv:2407.01032) and by the sensitivity retained at 80% coverage. Intervals are the 2.5th and 97.5th percentiles of 500 bootstrap resamples of target ECGs, with the threshold held fixed.

## Results

### Composite sensitivity among outpatients

Sensitivity with its 95% Wilson interval; for the weighted thresholds, the interval and n are at the effective size.

<!-- table:composite -->
| Arm | Single threshold | Weighted on age, sex | Weighted on ECG measurements | Weighted on embedding |
|---|---|---|---|---|
| ResNet trained on EchoNext | 71.6% (65.9%–76.6%) | 74.2% (68.6%–79.0%, n=271) | 79.3% (74.1%–83.7%, n=271) | 89.3% (84.3%–92.7%, n=213) |
| ResNet at random initialisation, probes | 78.6% (73.3%–83.1%) | 81.5% (76.5%–85.7%, n=271) | 83.8% (78.9%–87.7%, n=271) | 87.8% (82.7%–91.5%, n=213) |
| ECGFounder frozen, probes | 71.6% (65.9%–76.6%) | 72.7% (67.1%–77.7%, n=271) | 78.6% (73.3%–83.1%, n=271) | 89.7% (84.9%–93.1%, n=213) |
| EchoNext mini-model | 72.7% (67.1%–77.7%) | 73.8% (68.3%–78.7%, n=271) | 82.3% (77.3%–86.4%, n=271) | 89.7% (84.9%–93.1%, n=213) |
| ECGFounder fine-tuned, seed 0 | 72.7% (67.1%–77.7%) | 73.8% (68.3%–78.7%, n=271) | 80.1% (74.9%–84.4%, n=271) | 87.1% (81.7%–90.7%, n=213) |
| ECGFounder fine-tuned, seed 1 | 72.0% (66.3%–77.0%) | 73.4% (67.9%–78.3%, n=271) | 81.5% (76.5%–85.7%, n=271) | 88.2% (83.2%–91.9%, n=213) |
| ECGFounder fine-tuned, seed 2 | 75.6% (70.2%–80.4%) | 77.9% (72.5%–82.4%, n=271) | 81.2% (76.1%–85.4%, n=271) | 89.3% (84.3%–92.7%, n=213) |
<!-- /table -->

Weighting on age and sex changed sensitivity by at most 3 points. Weighting on the ECG measurements recovered part of the loss, and weighting on the embedding brought every model within 3 points of 90%, at an effective size of 213 calibration cases.

### Judged findings, weighted on the embedding

Sensitivity among outpatients, effective size, and the Holm-corrected test against 80%.

<!-- table:anomalies -->
| Arm | LVEF ≤ 45% | LV wall ≥ 13 mm | Aortic stenosis | PASP ≥ 45 mmHg | Repair |
|---|---|---|---|---|---|
| ResNet trained on EchoNext | 88.6%, n=70, fails | 89.7%, n=101, passes | 100.0%, n=29, passes | 82.5%, n=63, fails | fails |
| ResNet at random initialisation, probes | 87.1%, n=70, fails | 91.1%, n=101, passes | 94.1%, n=29, fails | 95.2%, n=63, passes | fails |
| ECGFounder frozen, probes | 92.9%, n=70, passes | 95.9%, n=101, passes | 100.0%, n=29, passes | 90.5%, n=63, passes | fails |
| EchoNext mini-model | 84.3%, n=70, fails | 91.8%, n=101, passes | 98.0%, n=29, passes | 84.1%, n=63, fails | fails |
| ECGFounder fine-tuned, seed 0 | 94.3%, n=70, passes | 91.8%, n=101, passes | 100.0%, n=29, passes | 81.0%, n=63, fails | fails |
| ECGFounder fine-tuned, seed 1 | 91.4%, n=70, fails | 89.0%, n=101, fails | 100.0%, n=29, passes | 82.5%, n=63, fails | fails |
| ECGFounder fine-tuned, seed 2 | 92.9%, n=70, passes | 89.7%, n=101, passes | 100.0%, n=29, passes | 85.7%, n=63, fails | fails |
<!-- /table -->

### Recalibration on labelled outpatients

Composite sensitivity in one half of the outpatients when the per-label thresholds were recalibrated on n labelled outpatients drawn from the other half: mean and 10th to 90th percentiles over 200 draws. The mean can lie outside that range when a tenth of the draws flag nearly every ECG; with 25 or 50 outpatients, few cases are drawn and the threshold often flags every ECG.

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

### Calibration and selective prediction on the composite

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

### Fine-tuned ECGFounder (arm A1)

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

## Discussion

Weighting on the ECGFounder embedding raised composite sensitivity among outpatients to 87.1% to 89.7%, but for no model did the lower Wilson bound at 213 effective cases reach 85%, and the pre-specified criterion (A2) was not met. Fine-tuning did not reduce the loss sufficiently: fine-tuned ECGFounder lost 14.4 to 18.0 points of composite sensitivity among outpatients across seeds, against 18.4 for the trained ResNet, with upper bounds of 19.8 to 23.7 points against the 9 required (A1).

Two explanations are compatible with the residual shortfall: the probability of SHD given the ECG may differ between care settings, which violates the covariate-shift assumption, or the embedding may not encode the features on which the settings differ. The bound of Han et al. (arXiv:2607.18088) on label-free class-conditional coverage is consistent with this result but is not demonstrated by it. The larger gain from the embedding than from age and sex suggests that the shift lies mainly in ECG morphology rather than in demographics. Thresholds recalibrated on 100 to 200 labelled outpatients reached a mean sensitivity of at least 90% for five of the seven models, at the cost of obtaining their echocardiograms.

## Limitations

The audit of Han et al. on target labels, named in the protocol as a third reference, was not implemented; the thresholds recalibrated on labelled outpatients are the labelled reference instead. The protocol states the decision rule « calculé sur la taille effective »; its implementation as the smaller of the Kish effective size and the target case count is our interpretation. The weights use the frozen ECGFounder embedding for every model, including models that do not read it. The Columbia to Boston arm (A4) awaits credentialed access to MIMIC-IV-Echo. Otabor et al. (Eur Heart J Digit Health 2026, doi:10.1093/ehjdh/ztag147) report the EchoNext mini-model's discrimination and recalibration in 45,878 MIMIC-IV patients, without conformal thresholds or care settings. Three TRIPOD+AI items are not covered: the sample size justification, the handling of missing data (none in the covariates used) and a model-updating protocol for deployment.

## Literature search

Searched on 20 October 2026. arXiv API: "EchoNext AND conformal", 0 records; "EchoNext AND calibration", 1 (arXiv:2607.24553); "EchoNext", 6. medRxiv's search refused the request (HTTP 403). Europe PMC, which indexes medRxiv preprints: "EchoNext AND conformal", 0 records; "EchoNext AND calibration", 14, of which Otabor et al. above is the only external validation with recalibration. Colebank (arXiv:2605.22968) studies Bayesian uncertainty for triage on EchoNext, without conformal thresholds or care settings in its abstract.

## Reproducibility

```bash
# On the machine holding EchoNext and the four arms' scores (~/data/echonext-derived/scores)
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/bin/lockf ~/.claude/locks/mac-gpu.lock \
  .venv/bin/python scripts/echonext_finetune.py --seeds 0 1 2
.venv/bin/python scripts/structure.py           # results/structure*.{csv,json}, then the tables here
.venv/bin/python scripts/structure.py --pages   # the tables alone, from the committed results
```
