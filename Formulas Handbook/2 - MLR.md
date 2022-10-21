# MLR 
We restrict the usage to linear and additive multivariate regression models.

## Multiple Predictors
It is rarely the case that a response variable will only depend on a single variable.
To avoid over complicating the formulas we will show the case with 2 predictors but there could
be many.

$$Y_{i} = \beta_{0} + \beta_{1}x_{i1} + \beta_{2}x_{i2} + \varepsilon_{i}$$
 
* With i=1,2, ...,n
* Where $\varepsilon_{i} \thicksim \mathcal{N}(0, \sigma^{2})$ 
* Now the $\hat{MSE}$ depends on $(\beta_{0}, \beta_{1}, \beta_{2})$

So we want to minimize 

$$f(\beta_{0}, \beta_{1}, \beta_{2}) = \sum_{i=1}^{n}(y_{i} - (\beta_{0} + \beta_{1}x_{i1} + \beta_{2}x_{i2}))^{2}$$

We could take the derivate with respect to each parameter and set them equal to zero, to obtain the estimating equations and then apply the plug-in principle to work with sample data.

Else we can use the built in function class `lm()` and find the coefficients
```r
fit <- lm(formula = y ~ x1 + x2, data = dataset)
coef(fit)
```

---

## Matrix approach to regression
Since the model is additive, we could find ourselves to face an arbitrary model with
* p-1 predictor variables
* Total of p $\beta$-parameters
* A single $\sigma^{2}$ for the variance of the errors

$$Y_{i} = \beta_{0} + \beta_{1}x_{i1} + \beta_{2}x_{i2} + ... + \beta_{(p-1)}x_{i(p-1)} + \varepsilon_{i}$$

* With i=1,2, ...,n
* Where $\varepsilon_{i} \thicksim \mathcal{N}(0, \sigma^{2})$
* Now the $\hat{MSE}$ depends on $(\beta_{0}, \beta_{1}, \beta_{2}, ..., \beta_{(p-1)})$

We could still optimize the function with respect to the p $\beta$-parameters, which would require computing p derivatives
and set them to 0, and then computing the estimating equations. Alternatively, we can stack the n linear equations to exploit the matricial version of the estimating equations, easier to represent. The model becomes:

$$Y = X\beta + \varepsilon$$

![Matrix Approach](matrix.png?raw=TRUE)

$\hat{\beta}$, the vector of estimates for the $\beta$-parameters: $\hat{\beta} = (X^{T}X)^{-1}y$

---

## MLR Gaussian 

---

## Single Parameter Tests

---

## Confidence Intervals

### Parameter

### Multivariate Parameters


#### Credits
[PayThePizzo](https://github.com/PayThePizzo/)

[Repository where I play with probability and statistics in R](https://github.com/PayThePizzo/Probability-Statistics)