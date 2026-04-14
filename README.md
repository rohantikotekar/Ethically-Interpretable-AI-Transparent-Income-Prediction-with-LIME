# Ethically Interpretable AI
### Transparent Income Prediction with LIME

---

## Problem

Most high-performing ML models are black boxes. They produce predictions with impressive accuracy — but offer no explanation for why a decision was made. In high-stakes domains like income classification, this opacity is a serious problem. Stakeholders — whether they're auditors, policymakers, or the individuals being classified — have no way to scrutinize the model's reasoning. Without that transparency, bias can go undetected, decisions can't be challenged, and trust is nearly impossible to build. Accuracy alone isn't enough when the decisions affect people's lives.

---

## Solution

A transparent income prediction pipeline that pairs a high-accuracy XGBoost classifier with LIME (Local Interpretable Model-agnostic Explanations) — so every prediction comes with a human-readable reason. LIME works by approximating the complex model locally around each prediction using a simpler, interpretable model, answering the question: *"Why did the model make this decision for this specific person?"* The goal was to match black-box performance without sacrificing explainability, auditability, or ethical accountability — proving that transparency and accuracy aren't a tradeoff.

---

## Implementation

- **Dataset:** UCI Adult Income dataset (~48K records, 14 features)
- **Model:** XGBoost gradient boosting classifier — 86% test accuracy
- **Custom Preprocessing Pipeline:** Built a categorical encoding pipeline that retains feature semantics across 9+ categorical variables, keeping explanations meaningful and human-interpretable rather than reducing features to abstract numbers
- **Local Explanations with LIME:** Generated instance-level explanations for individual predictions, pinpointing the features — like `age`, `hours-per-week`, occupation, and education level — that drove each outcome
- **Global Insight via Submodular Pick:** Applied LIME's submodular pick algorithm to select 25 maximally diverse instances, achieving 80% explanation coverage across the dataset with minimal redundancy
- **Ethical Audit:** Conducted a post-hoc interpretability assessment using an AI ethics rubric, identifying transparency gaps in the black-box model and improving the overall explainability score by 30%
- **Stack:** Python, XGBoost, LIME, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook

---

## Future Goals

- Integrate SHAP (SHapley Additive Explanations) and run a head-to-head comparison with LIME across accuracy, stability, and interpretability
- Add counterfactual explanations to answer: *"What would need to change for a different prediction?"* — making the model actionable, not just explainable
- Build an interactive dashboard where users can query predictions and view explanations in real time, without needing to read code
- Extend the ethical audit to evaluate demographic fairness across race and gender subgroups, bridging interpretability with bias detection

---
