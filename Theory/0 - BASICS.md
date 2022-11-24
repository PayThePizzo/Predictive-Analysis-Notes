# Notation and Assumptions of the models

--- 

## 1 - SLR (Least Squares Method)

Our estimate for the Simple Linear Regression Model is built as:

$$\hat{Y} = \hat{\beta}_{0} + \hat{\beta}_{1}X + \varepsilon$$

## 1.1 - SLR, Meaning of parameters and variables (1 response, 1 predictor)
* Y, random variable, target/response/regressand
    + $\hat{Y} = \hat{m}(X)$, our estimated model
* y, vector of observed values of the response
    + $y_{i}$ is the i-th observed value
    + $\hat{y}_{i}$, our estimated i-th observed value
* X, random variable, input/predictor/explanatory variable
* x, vector of observed values of the predictor
    + $x_{i}$, is the i-th predictor
* n, number of observations
    + number of rows of the dataset
* p, number of predictors (from $x_{1}$ to $x_{p}$)
    + count of different predictors (in this case it's 1)
* p+1, number of beta-parameters (from $\beta_{0}$ to $\beta_{p}$)
    + $\beta_{0}$ and $\beta_{1}$ (in this case it's 2)
    +  $\hat{\beta}_{0}$ and $\hat{\beta}_{1}$, our estimates


## 1.2 - SLR, Assumptions



## 1.3 - SLR, Interpretation



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