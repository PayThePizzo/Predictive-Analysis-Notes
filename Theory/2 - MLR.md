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
# And use the same lm class functions ...
```

---

## Matrix approach to regression
Since the model is additive, we could find ourselves to face an arbitrary model with
* p-1 predictor variables
* Total of p $\beta$-parameters
* A single $\sigma^{2}$ for the variance of the errors

$$Y_{i} = $$

* With i=1,2, ...,n
* Where $\varepsilon_{i} \thicksim \mathcal{N}(0, \sigma^{2})$
* Now the $\hat{MSE}$ depends on $(\beta_{0}, \beta_{1}, \beta_{2}, ..., \beta_{(p-1)})$

We could still optimize the function with respect to the p $\beta$-parameters, which would require computing p derivatives
and set them to 0, and then computing the estimating equations. Alternatively, we can stack the n linear equations to exploit the matricial version of the estimating equations, easier to represent. 

The model becomes:

$$Y = X\beta + \varepsilon$$

![Matrix Approach](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/Formulas%20Handbook/resources/matrix.png?raw=TRUE)

```r
# Gets rows, the number of the observed values
n <- nrows(dataset)

# Union of the vectors by column
X <- cbind( rep(1,n), dataset$predictor1, dataset$predictor2)

# Gets columns, the number of parameters
p <- ncols(X)
```

We now have a vector $y = [y_{1}, y_{2}, ..., y_{n}]$ containing the observed data. By optimizing the function with respect to the p $\beta$-parameters, we obtain the estimating equations $X^{T}X\beta = X^{T}y$

$\hat{\beta}$, the vector of estimates for the $\beta$-parameters: $\hat{\beta} = (X^{T}X)^{-1}y$

```r
y <- dataset$target

# First apply the transpose, then apply multiplication between matrix
beta_hat <- solve(t(X) %*% X) %*% t(X) %*% y

# To show it in a row format
t(beta_hat)
```

Alternatively we can use the lm class, and forget about manually computing everything

```r
fit <- lm(formula, data)

# To check the first rows of the matrix generated by R
head(model.matrix(fit))
```

The predicted values can be written $\hat{y} = X\hat{\beta}(=X(X^{T}X)^{-1}X^{T}y)$
```r
y_hat <- X %*% solve(t(X) %*% X) %*% t(X) %*% y

res_val <- y - y_hat
```
The estimate for $\sigma^{2}$ is $s_{e} = \frac{ \sum_{i=1}^{n} (y_{i}-\hat{y_{i}})^2}{n-p} =\frac{e^{T}e}{n-p}$. Which is unbiased.
```r
s2e <- (t(e) %*% e)/(n-p)

# Residual SE in R 
se <- sqrt(s2e)
# which is equal to
sqrt(sum((y-y_hat)^2)/(n-p))
#or we can access it directly through
summary(fit)$sigma
```

---

## MLR Gaussian 

---

## Single Parameter Tests

---

## Confidence Intervals



### Parameter

### Multivariate Parameters

---

#### Slide Sections 
* 1 - 216

#### Credits and Warning
* [PayThePizzo](https://github.com/PayThePizzo/) for maintining this handbook/summary of the course.
* The material of the course [PREDICTIVE ANALYTICS](https://www.unive.it/data/course/339919) is provided by Professor [Ilaria Prosdocimi](https://www.unive.it/data/persone/19166744) and is only intended for personal use.
* Material in these slides was heavily influenced by [David Dalpiaz Applied Statistics with R!](https://book.stat420.org/)

I take no credit from any of the material that is included and I highly recommend purchasing the textbooks cited.