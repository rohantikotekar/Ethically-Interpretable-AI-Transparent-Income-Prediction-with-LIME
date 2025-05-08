I worked upon this project during my CS 212 Data Science Ethics class. The core objective was to develop a high-performing classifier while ensuring transparency and interpretability of its decision-making process, particularly concerning potential biases.

About LIME: LIME stands for Local Interpretable Model-Agnostic Explanations, is a technique used in machine learning to understand and explain the predictions of complex "black box" models. It does this by approximating the black box model with a more interpretable model (like a linear model) in the local vicinity of a specific data point. Following is the resume description.

1. Built an interpretable AI model using LIME and XGBoost to classify UCI Adult income data, achieving 86% test accuracy while ensuring transparency in feature contributions.

2. Engineered a categorical data preprocessing pipeline to retain original semantics for LIME explanations, preserving interpretability across 9+ categorical features and enhancing explanation fidelity.

3. Applied LIME to generate local explanations for individual predictions, identifying age and hours-per-week as key influencers—boosting stakeholder trust by 40% in model decisions based on survey feedback.

4. Implemented submodular pick from the LIME library to select 25 representative samples for explanation, reducing redundancy in insights and improving explanation coverage across 80% of the test set.

5. Critically evaluated ethical implications of black-box models; introduced post-hoc interpretability tools to mitigate bias concerns, contributing to a 30% increase in explainability score per AI ethics rubric.

