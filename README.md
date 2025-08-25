# Royal-Perth-Hospital---Prediction-of-Sepsis-Risk-and-survival-probability

This project, developed in collaboration with Royal Perth Hospital (RPH) and the Health in a Virtual Environment (HIVE), focuses on building machine learning models for early detection of sepsis during hospital admissions.

Sepsis is a life-threatening condition characterized by vague early symptoms, rapid progression, and high mortality rates. Early identification is critical, as each hour of delayed treatment significantly increases the risk of death.

### Our models aim to:

-Support clinicians with standardized, interpretable risk scores.

-Optimize hospital resource allocation through automated early risk prediction.

-Reduce mortality and healthcare costs with timely interventions.

### Objectives

Develop predictive models using structured EHR data (demographics, comorbidities, lab test results).

Compare traditional ML, deep learning (LSTMs), and survival analysis approaches.

Prioritize interpretability alongside accuracy to ensure clinical trust.

### Methodology
1. Data Preparation
Sources: MIMIC-III dataset and RPH data.
Excluded: patients <18 years, unreasonable ages, or missing admission IDs.
Structured early-stage data up to 8 hours post-admission.
Addressed class imbalance (sepsis ~10.9%) with stratified sampling.

2. Feature Selection
Guided by literature, clinical input, and causal inference.
Focused on lab test derangements, comorbidities, and SOFA score.
Feature engineering using Neo4j to identify comorbidity-lab test links.

3. Models Implemented
Traditional ML: Logistic Regression, Random Forest, Gradient Boosting.
Deep Learning: LSTM networks for time-series patient data (0–8 hours).
Survival Analysis: Cox Proportional Hazards, CoxNet, Random Survival Forests, Gradient Boosted Survival models.

4. Evaluation Metrics
ML & DL: Balanced Accuracy, Precision, Recall, F1-score, AUROC.
Survival Models: Concordance Index (C-index), Integrated Brier Score (IBS).

### Results
Traditional ML (Best: Random Forest @ t2, top 20 features)

Balanced Accuracy: 0.747

Recall: 0.758

AUROC: 0.828

Key features: Lactate, Neutrophils, Lymphocytes, WBC, Bilirubin.

### Deep Learning (Best: LSTM @ t8, top 20 features)

Balanced Accuracy: 0.766

Recall: 0.747

AUROC: 0.842

TimeSHAP identified Lymphocytes, Neutrophils, Urea Nitrogen, and Platelet Count as most influential.

### Survival Analysis (Best: Random Survival Forest)

C-index: 0.795

IBS: 0.102

Survival models provided time-dependent risk probabilities, aiding in patient prioritization.

### Key Insights

LSTM models excelled in predictive performance for early detection.

Random Forests balanced accuracy with interpretability, making them suitable for clinical deployment.

Survival models added value by predicting likelihood of sepsis onset over time.

Feature importance consistently highlighted biomarkers such as Lactate, Neutrophils, Lymphocytes, and SOFA score.

### Conclusion

The study demonstrates that ML and DL models can identify sepsis early, improving patient outcomes.

Random Forest + Survival Analysis strike a strong balance of accuracy and interpretability for clinical adoption.

The approach provides a standardized risk score, reducing reliance on subjective clinical judgment.
