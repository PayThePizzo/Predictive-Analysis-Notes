# Notation and Assumptions of the models

--- 

## SLR
* Y, target
* y
  * $y_{i}$
* X, predictor (or vector of predictors)
* n, number of observations
* x, predictor
  * $x_{i}$, predictor
* p, number of predictors
* p-1, number of beta-parameters

$\beta$-Parameters: 

Predictor variables:
* Single $x_{1}$
* Multiple $x_{1}, x_{2}$
* p-1 $x_{1}, x_{2}, ..., x_{p-1}$

## SLR - Assumptions
The Simple Linear Regression Model is built as:

$$Y = m(X) + \varepsilon$$

To favor a high-level of interpretability: 
1. We focus on Supervised Learning approaches, in particular for prediction ($y_{i} is a continuous numeric value$)
2. In particular we only consider linear models, where the functions $m(X)$ built are linear combinations
3. The error $\varepsilon$ identifies the stochasticity of the domain, the measurement errors and other discrepancies between Y and the model m
4. We want to build generalizable 

---

## MLR
* $Y$, the vector containing the $Y_{i}$ 
* $Y_{i}$, 
* $y$, the vector containing the observed data
* $y_{i}$, i-th observed data object
* $\hat{y}$, the vector
* $X$, the
* $X^{T}$, the transpose of 
* $\beta$,
* $\hat{\beta}$, the vector of estimates for the $\beta$-parameters
* $\varepsilon$, the vector of the errors
* $e$, the vector of the residual values
* $H$, hortogonal projection matrix or design matrix


## MLR - Assumptions

---

## Validation Based Selection 
* $\hat{y_{[i]}}$, the value of $\hat{y_{i}}$ obtained when we remove it from the sample.