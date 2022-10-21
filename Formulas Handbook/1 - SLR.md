# SLR
Simple empirical formuals of the SLR Model

## SLR Model and assumptions
Model: $Y_{i} = \beta_{0} + \beta_{1}X_{i} + \varepsilon_{i}$


## Data Loading
```r
# Define the data
x <- dataset$predictor
y <- dataset$target
n <- lenght(x)
```

## Least Squares Estimates - 1
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

## lm Class
The [lm class](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm) follows
exactly the same approach we used above but it automates it. 

```r
# Define model
fit <- lm(formula = y ~ x, data = dataset)

coef(fit)
summary(fit)
fitted(fit)
residuals(fit)

```

## SLR Gaussian



#### Credits
[PayThePizzo](https://github.com/PayThePizzo/)

[Repository where I play with probability and statistics in R](https://github.com/PayThePizzo/Probability-Statistics)