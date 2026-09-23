# Multi-Class Classification with Logistic Regression

This project classifies **obesity levels** with logistic regression. The task is to predict a person's obesity category from demographic, physical, and lifestyle features.

---

## Problem

- **Type**: supervised learning
- **Task**: multi-class classification
- **Target**: `NObeyesdad` (obesity level)
- **Models**: logistic regression, One-vs-Rest and Multinomial

---

## Dataset

Each row is one person. Features:

- Demographics: `Gender`, `Age`
- Physical: `Height`, `Weight`
- Lifestyle: `FAF`, `TUE`, `CH2O`
- Eating habits: `FCVC`, `NCP`, `CAEC`, `CALC`
- Health and behavior: `SMOKE`, `SCC`
- Transportation: `MTRANS`

The target has several mutually exclusive classes.

---

## Preprocessing

1. Continuous features are standardized: `x' = (x - μ) / σ`
2. Categorical features are encoded.
3. The target is label-encoded with `astype('category').cat.codes`.

---

## Models

### One-vs-Rest (OvR)

`LogisticRegression(multi_class='ovr')`

- Trains K binary classifiers, one per class, each predicting class k vs. the rest.
- Uses the sigmoid `σ(z) = 1 / (1 + e^{-z})`.
- Predicts the class with the highest score.
- Loss: `L = -[ y log(p) + (1 - y) log(1 - p) ]`

### Multinomial (Softmax)

`LogisticRegression(multi_class='multinomial')`

- Trains one model with one score per class.
- Uses softmax: `P(y = k | x) = exp(z_k) / Σ_j exp(z_j)`
- Probabilities sum to 1, so classes compete with each other.
- Loss: `L = - Σ_k y_k log(P(y = k | x))`

### Comparison

| | OvR | Multinomial |
|------|-----|-------------|
| Number of models | K binary models | 1 model |
| Probability function | Sigmoid | Softmax |
| Training | Separate per class | Joint over all classes |
| Probabilities sum to 1 | No | Yes |

Both use `max_iter = 1000`.

---

## Output

- Predicted obesity class for each person
- Accuracy and classification metrics
- Feature importance from the model coefficients

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib, seaborn
