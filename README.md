# Student Dropout Risk Predictor

## Problem Statement
Predict whether a student will drop out, remain enrolled, or graduate, using data available at enrollment time and academic performance from the first two semesters. Early identification allows the institution to intervene before a student disengages entirely.

## Dataset Information
- **Name:** Predict Students' Dropout and Academic Success
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Rows:** 4,424
- **Columns:** 37 (36 features + 1 target)
- **Target Variable:** `Target` (Dropout / Enrolled / Graduate)

## Project Objectives
- Build a classification model that flags at-risk students early using enrollment and early-semester data.
- Identify the strongest predictive features (e.g., admission grade, first-semester units approved, socio-economic indicators).
- Handle class imbalance across the three target categories in a way that keeps minority-class (Dropout) performance meaningful, not just overall accuracy.

## Technologies Used
- Python
- Pandas, NumPy
- scikit-learn (DecisionTreeClassifier)
- Google Colab / Jupyter Notebook

## Progress So Far
1. ✅ Data loading and initial exploration (data types, missing values, summary statistics)
2. ✅ Preprocessing — separated features/target, encoded the target (Dropout=0, Enrolled=1, Graduate=2). Confirmed the numerically-coded categorical fields (Marital status, Course, Gender, Debtor, etc.) don't require one-hot encoding for a Decision Tree, since trees split on raw values regardless of whether a number represents a category or a quantity.
3. ✅ Train/test split (80/20, stratified to preserve class proportions given the imbalance)
4. ✅ Model training — Decision Tree Classifier (`max_depth=8`, `class_weight="balanced"`)
5. ✅ Evaluation — accuracy (train vs test), classification report, confusion matrix (weighted precision/recall/F1 due to class imbalance)
6. ✅ Feature importance analysis — identified top predictive features
7. ✅ Model saved as `dropout_model.pkl` in the `Model/` folder

## Remaining Work
- Feature engineering (e.g., pass-rate ratios) — not yet implemented
- Model comparison against other algorithms (if required in a later phase)
- Documentation and iteration based on mentor feedback
