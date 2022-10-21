# Formulas
Formulas handbook in R and Latex, for the empirical version


### Notation

## SLR - Empirical
Model: $Y_{i} = \beta_{0} + \beta_{1}X_{i} + \varepsilon_{i}$

```r
# Define the data
x <- dataset$predictor
y <- dataset$target
n <- lenght(x)
```

We can obtain the optimal result through the following:

```r
s2x <- sum((x-mean(x))^2)/n
s2y <- sum((y-mean(y))^2)/n
covxy <- cov(x,y) 
rxy <- cor(x,y)
mx <- mean(x)
my <- mean(y)

# Parameters 
(beta1 <- rxy * sqrt(s2y/s2x))
(beta0 <- my - beta1 *mx)

# Estimated Values
yhat <- beta0 +  beta1 * x 

# Empirical MSE
mse_hat <- sum((y-yhat)^2)
```

```r
# Define model
fit <- lm(formula = y ~ x, data = dataset)

coef(fit)
summary(fit)
fitted(fit)
residuals(fit)

```

## SLR Gaussian


## MLR 

```r


lm(formula = y ~ x1 + x2, data = dataset)
```

#### Credits
[PayThePizzo](https://github.com/PayThePizzo/)

[Repository where I play with probability and statistics in R](https://github.com/PayThePizzo/Probability-Statistics)