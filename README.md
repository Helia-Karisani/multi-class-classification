# Multi-Class Classification with Logistic Regression

This project performs **multi-class classification of obesity levels** using logistic regression.  
The task is to predict a person’s obesity category based on demographic, physical, and lifestyle features.

---

## Problem Definition

- **Type**: Supervised learning
- **Task**: Multi-class classification
- **Target variable**: `NObeyesdad` (obesity level)
- **Model**: Logistic Regression (OvR and Multinomial)

---

## Dataset Overview

Each row represents one individual.  
Features include:

- Demographics: `Gender`, `Age`
- Physical: `Height`, `Weight`
- Lifestyle: `FAF`, `TUE`, `CH2O`
- Eating habits: `FCVC`, `NCP`, `CAEC`, `CALC`
- Health & behavior: `SMOKE`, `SCC`
- Transportation: `MTRANS`

The target variable `NObeyesdad` contains **multiple mutually exclusive classes** representing obesity levels.

---

## Preprocessing Steps

### 1. Feature Scaling

Continuous numerical features are standardized using **z-score normalization**:

`x' = (x - μ) / σ`

where:
- `μ` = mean of the feature
- `σ` = standard deviation

This ensures all features have:
- Mean = 0
- Standard deviation = 1

---

### 2. Target Encoding

The target variable is label-encoded:

`{class_1, class_2, ..., class_K} → {0, 1, ..., K-1}`

This is done using:
```
astype('category').cat.codes
```

---

## Models Used

### 1. One-vs-Rest Logistic Regression (OvR)

Configured as:
```
LogisticRegression(multi_class='ovr')
```

#### How it works
- Trains **K binary classifiers** (one per class)
- Each classifier learns:
  
  `P(y = k | x)` vs `P(y ≠ k | x)`

- Uses the **sigmoid function**:

  `σ(z) = 1 / (1 + e^{-z})`

- Final prediction = class with the highest score

#### Loss function (binary cross-entropy):
```
L = -[ y log(p) + (1 - y) log(1 - p) ]
```

---

### 2. Multinomial Logistic Regression (Softmax)

Configured as:
```
LogisticRegression(multi_class='multinomial')
```

#### How it works (jointly trained)
- Trains **one single model**
- Computes one score per class:

  `z_1, z_2, ..., z_K`

- Applies **softmax**:

  `P(y = k | x) = exp(z_k) / Σ_j exp(z_j)`

- Probabilities **sum to 1**
- Classes **compete with each other**

#### Loss function (categorical cross-entropy):
```
L = - Σ_k y_k log(P(y = k | x))
```

---

## Key Difference: OvR vs Multinomial

| Aspect | OvR | Multinomial |
|------|-----|-------------|
| Number of models | K binary models | 1 model |
| Probability function | Sigmoid | Softmax |
| Training | Independent per class | Joint over all classes |
| Probability sum | Not constrained | Sums to 1 |
| Best for | Simplicity | True multiclass problems |

---

## Optimization

- Solver uses gradient-based optimization
- `max_iter = 1000` sets the **maximum number of parameter update steps**
- Each iteration updates model weights to minimize loss

---

## Output

- Predicted obesity class for each individual
- Model evaluation via accuracy and classification metrics
- Feature importance visualized via model coefficients

---

## Summary

This notebook demonstrates:
- Proper preprocessing for ML
- Two approaches to multi-class logistic regression
- The mathematical difference between OvR and softmax-based models
- A full end-to-end classification pipeline

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn (for visualization)

---

## Notes

- All continuous features are standardized
- Categorical features are encoded before training
- Multinomial logistic regression is preferred when classes are mutually exclusive
