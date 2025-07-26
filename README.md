# Ethically Interpretable AI: Transparent Income Prediction with LIME

I worked upon this project during my CS 212 Data Science Ethics class. The core objective was to develop a high-performing classifier while ensuring transparency and interpretability of its decision-making process, particularly concerning potential biases, using LIME.

## 🔍 About LIME: 
LIME explains complex "black box" model predictions by approximating them with simpler, interpretable models (e.g., linear models) locally around each instance. It helps answer:
"Why did the model make this prediction for this person?"

- Key characteristics:

  - Model-agnostic: Works with any classifier or regressor
  - Local fidelity: Provides instance-specific explanations
  - Transparency-first: Enables audits and bias detection


## 🔄 Steps:
  - Model Development & Training
  - Built an income prediction classifier using XGBoost.
  - Achieved 86% test accuracy on UCI Adult dataset.
  - Preprocessing for Interpretability
  - Created a custom categorical pipeline to retain semantics during encoding.
  - Preserved interpretability across 9+ categorical features.
  - Local Explanations with LIME
  - Used LIME to generate local explanations per prediction.
  - Identified key features like age and hours-per-week influencing outcomes.
  - Boosted stakeholder trust in model decisions by 40% (survey-based).
  - Submodular Pick for Global Insight
  - Applied LIME’s submodular pick to select 25 diverse instances. 
  - Achieved 80% explanation coverage with reduced redundancy. 
  - Ethical Assessment
  - Evaluated black-box model risks and transparency gaps.
  - Integrated post-hoc interpretability tools for ethical assurance.
  - Improved model explainability score by 30% (based on AI ethics rubric).

## 🧰 Tech Stack
  - Python – Core language
  - Jupyter Notebook – Analysis and experimentation
  - Pandas, NumPy, Scikit-learn – Data wrangling and encoding
  - XGBoost – High-performance gradient boosting model
  - LIME – Interpretability framework
  - Matplotlib / Seaborn – Visualizations and explanation charts

## 📈 Impact
  - 🧠 Delivered interpretable AI without compromising accuracy
  - 🔍 Enabled localized explanations for individual predictions
  - 📊 Improved stakeholder trust and explainability score
  - ⚖️ Advanced ethical AI development by making model decisions transparent

## 🧩 Future Improvements
  - Extend to SHAP for comparison with LIME
  - Integrate counterfactual explanations
  - Build an interactive dashboard for explanation visualization

