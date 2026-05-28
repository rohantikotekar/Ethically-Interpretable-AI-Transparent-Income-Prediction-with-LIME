# FairLens: Locally Interpretable Model-Agnostic Explanantions for Income Predictions
---

## Problem

Modern machine learning models often optimize heavily for predictive accuracy while sacrificing interpretability. In high-impact domains such as income classification, black-box predictions create major concerns around transparency, fairness, and accountability.

Traditional ensemble models like XGBoost can achieve strong performance on tabular datasets, but they do not provide interpretable reasoning for individual predictions. This makes it difficult to:

* Audit model behavior
* Detect hidden bias
* Validate decision consistency
* Explain outcomes to stakeholders

The objective of this project was to design an interpretable ML pipeline capable of maintaining high classification accuracy while generating locally explainable predictions.

---

## Solution

This project implements an explainable binary classification pipeline using **XGBoost** for predictive modeling and **LIME (Local Interpretable Model-Agnostic Explanations)** for post-hoc interpretability.

The system predicts whether an individual's annual income exceeds $50K using demographic and employment-related attributes from the UCI Adult Income dataset.

To address model opacity, LIME generates local surrogate explanations around individual predictions by approximating the complex decision boundary with an interpretable linear model. This enables feature-level attribution for every prediction and provides transparency into the model's reasoning process.

The project focuses on balancing:

* Predictive performance
* Local interpretability
* Ethical transparency
* Explanation coverage

---

## Implementation

### Dataset

* **Dataset:** UCI Adult Income Dataset
* **Records:** ~48,000 instances
* **Features:** 14 structured attributes
* **Target Variable:** Income class (`<=50K` or `>50K`)

### Feature Engineering & Preprocessing

A custom preprocessing pipeline was developed to preserve semantic consistency across categorical features while maintaining compatibility with tree-based learning.

#### Preprocessing Steps

* Missing value handling
* Label encoding / categorical transformation
* Numerical feature normalization where required
* Train-test split using stratified sampling

#### Key Features

* Age
* Education
* Occupation
* Marital status
* Hours-per-week
* Capital gain/loss
* Workclass

### Model Architecture

#### XGBoost Classifier

The core predictive model uses gradient-boosted decision trees through XGBoost.

#### Training Objectives

* Binary logistic classification
* Optimized for generalization on structured tabular data
* Reduced overfitting using boosting regularization

#### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

### Explainability Layer

#### LIME Explanations

LIME was integrated to generate instance-level explanations by:

1. Sampling perturbed data points near a prediction
2. Evaluating local model behavior
3. Fitting an interpretable surrogate model
4. Ranking feature contributions

This allowed direct interpretation of why a prediction was classified as high-income or low-income.

#### Submodular Pick (SP-LIME)

SP-LIME was used to identify a diverse subset of representative explanations across the dataset. This improved global understanding of model behavior while minimizing redundant explanations.

### Tech Stack

* Python
* XGBoost
* LIME
* scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Results and Output

### Model Performance

* Achieved approximately **86% test accuracy**
* Strong classification performance on imbalanced socioeconomic data
* Stable generalization across unseen samples

### Interpretability Results

LIME explanations consistently identified influential features such as:

* Education level
* Occupation category
* Weekly working hours
* Age
* Capital gain
* Marital status

Each prediction included weighted feature contributions, enabling transparent inspection of local decision boundaries.

### Global Explanation Coverage

Using SP-LIME:

* 25 representative instances were selected
* ~80% explanation coverage was achieved
* Redundancy across explanations was significantly reduced

### Ethical Analysis

An interpretability-focused ethical audit was conducted to evaluate:

* Transparency
* Accountability
* Explainability quality
* Potential bias exposure

The addition of explainability mechanisms improved the overall interpretability and auditability of the pipeline compared to a standalone black-box classifier.

---

## Getting Started

### Clone Repository

```bash id="x1a2bc"
git clone https://github.com/rohantikotekar/Ethically-Interpretable-AI-Transparent-Income-Prediction-with-LIME.git
cd Ethically-Interpretable-AI-Transparent-Income-Prediction-with-LIME
```

### Install Dependencies

```bash id="n82klm"
pip install -r requirements.txt
```

### Launch Notebook

```bash id="p0q1rs"
jupyter notebook
```

Run the notebook:

```bash id="w7t8uv"
Copy_of_CS_STAT_108_212_Assignment2 (2).ipynb
```

---

## Future Scope

### SHAP Integration

Integrate SHAP (SHapley Additive Explanations) to compare explanation consistency, stability, and computational efficiency against LIME.

### Counterfactual Explanations

Extend the framework with counterfactual reasoning to answer:

> "What minimum feature changes would alter the prediction outcome?"

### Fairness Evaluation

Incorporate demographic fairness metrics across race, gender, and socioeconomic subgroups to detect bias amplification.

### Interactive Explainability Dashboard

Develop a real-time explainability interface for:

* Prediction visualization
* Feature attribution analysis
* User-driven inference inspection

### Production Deployment

Containerize and deploy the pipeline using Flask/FastAPI with scalable inference endpoints and explanation APIs.

---
