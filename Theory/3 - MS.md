# Model Selection and Nested Models
Model selection is the process of analyzing what predictors should be included in a model
in order to balance its complexity and adaptability. In ML, this is called feature selection and it is acceptable to have
a number of observations lower than the number of predictors $n < p$.

In statistics we want to have $n > p$ and **p should be ideally the smallest subset of features**, while
we hope the observations to be $n \rightarrow \infty$. 
This is because $(X^{T}X)^{-1}$ is a matrix $p \times p$, so if $n < p$ it is not possible invert it.

Another aspect we shall consider is that we want to examine the trade-off between different values of p to catch
the **best ratio predictive-ability/computational-cost**

Usually we ask ourserlves:
* Is at least 1 predictor useful in target prediction?
* Do all predictors help? Should we consider a subset only?
* How well does the model fit the data?

## Significance of Regression
We ask ourselves whether a model with p predictors is better than just taking the expected value $\bar{y}$, since the latter is merely the simplest possible model. 
We want to be able to distinguish what predictors are useful and how well the model in question fits the data. 

Since we can still make a use of the decomposition of variance of the SLR, we can define our goal: find models that minimize $SS_{RES}$ and
maximize $SS_{REG}$. As they are complementary, we can try achieving either one of those points. 

One way is to try and maximize the r-squared $R^{2} \rightarrow 1$! However it does not supply any relevant info:
* Problem 1: from this metric we have no clue which predictor explains better the observed variability
* Problem 2: it is not a clear indication that the regression captures a considerably good proportion of the variability. There is no measure to understand what $(a * 100)%$ means or if it is good. 

Since $R^{2}$ does not supply a clear answer for any of our questions, so we need to find another approach with metrics that:
1. Declares the model is useful
2. Tests whether a model is significantly different/better from taking a simpler one: is it worth to build a more complex model?

To obtain some answers with one simple way we can test the significance of the regression which explains
whether building a certain model is better than sticking to a "null model" or a simpler one.
* $H_{0} : Y_{i} = \beta_{0} + \varepsilon_{i} \rightarrow Y = \bar{y}$
  * Where the model is $\hat{y_{0,i}} = \bar{y}$ and represents the null model
  * It indicates no predictor has a significant linear relation with Y, namely $\beta_{1} = \beta_{2} = ... = \beta_{(p-1)} = 0$
* $H_{A} : Y_{i} =\beta_{0} + \beta_{1}x_{i1} + \beta_{2}x_{i2} + ... + \beta_{(p-1)}x_{i(p-1)} + \varepsilon_{i}$
  * Where the model is $\hat{y_{A,i}}$
  * It indicates at least one of the predictors has a significant linear relation with Y, namely $\beta_{0} \neq 0, or \beta_{1} \neq 0, or \beta_{2}x_{i2} \neq 0, ... or \beta_{(p-1)} \neq 0$

```r
# Firstly we specify the two model and save them in two different variables
H_null<- lm(target ~ 1, data = dataset)
H_alt <- lm(target ~ x1 + x2, data = dataset)
``` 

### 1) ANOVA
We decompose the $SS_{TOT}$ into an **AN**alysis **O**f **VA**riance table

![ANOVA](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/ANOVA.png?raw=TRUE)

```r
# Sum of Squares
SS_reg <- sum((fitted(H_alt) - fitted(H_null))^2)
SS_res <- sum(resid(H_alt)^2)
SS_tot <- sum(resid(H_null)^2)

#Degrees of Freedom
DoF_reg <- length(coef(H_alt)) - length(coef(H_null))
DoF_err <- length(resid(H_alt)) - length(coef(H_alt))
DoF_tot <- length(resid(H_null)) - length(coef(H_null))
```

### 2) F-Statistic and p-value
We calculate the F-statistic:

$$F = \frac{SS_{REG}/(p-1)}{SS_{RES}/(n-p)} = \frac {\sum_{i=1}^{n}(\hat{Y_{A,i}}-\bar{Y})^{2}/(p-1)}{\sum_{i=1}^{n}(Y_{i}-\hat{Y_{A,i}})^2/(n-p)} \thicksim F_{p-1, n-p}$$

In R
```r
# TODO: to check
F <- (SS_reg/DoF_reg)/(SS_res/DoF_res)
```

We notice $F \in \mathbb{R^{+}}$
* If F is **LARGE**, we can reject $H_{0}$ as the estimated y $\hat{y_{A,i}}$ are very different from $\bar{y}$ 
  * The regression explains a large portion of the variance
* If F is small, we cannot reject it!

Manually, **p-value** is calculated as $Pr(F > F_{obs})$, where an extreme value would only be found the right tail of the distribution
```r
# Since we only care about the right side of the distribution, we put lower.tail = FALSE
pf(f_obs, df1 = DoF_reg, df2 = DoF_err, lower.tail = FALSE)

# Or
1-pf(f_obs, df1 = DoF_reg, df2 = DoF_err)
```
We see that the value of the F statistic is large, and the p-value `Pr(>F)` is extremely low, so we
reject the null hypothesis at any reasonable $\alpha$ and say that the regression is significant.

More formally, rejecting $H_{0}$ implies that at least one beta parameter in the model we are testing
explains the variance of Y such that the model $\hat{y_{A,i}}$'s values are significantly different from 
the null model.

### R shortcut
```r
# This basically does the job for us
anova(H_null, H_alt)
```
Which returns, for each model:
* `Res.Df`
* `RSS`, Sum of Squares
* `Df`, Degrees of freedom (because of the parameters)
* `Sum of sq`
* `F`, F statistic value
* `Pr(>F)`, p-value

As a shortcut, the `summary()` function reports the `F-statistic`, the degrees of freedom (df1, df2) and the `p-value`
```r
fit <- H_alt
summary(fit)
```
Where:
* `Residual Standar error`
* `Multiple R-squared`
* `F-statistics` on `df1` and `df2` DF
* `p-value` 

However, we still have no clue to assess which predictor is useful and which is not!

---

## Nested Models
Imagine starting from a model that includes all the features and going through the removal of
the ones which are less helpful to achieve a simpler but effective model at each step. This is the core idea here: the significance of regression is a special case of the nested models concept.

If you revise the two models we just saw, you can tell they are the same model! The null model is hidden inside the $H_{A}$ thanks to the constraints on the beta-parameters. In fact, by imposing constraints on every beta-parameter from $\beta_{1} to \beta_{p-1}$ we obtain a model containing a subset of predictors.

```r
# Prints the predictors
names(dataset)
```

If we consider a general linear additive linear model with p-1 beta-parameters:

$$Yi = \beta_{0} + \beta_{1}x_{i1} + ... + \beta_{(q)}x_{i(q)} + \beta_{(q+1)}x_{i(q+1)} + ... + \beta_{(p-1)}x_{i(p-1)} + \varepsilon_{i}$$

We can proceed by comparing different subsets of predictors to achieve the best model for our goals, which is the same process as we did before. Now the two options are:
* $H_{0} : \beta{q} = \beta{q+1} = ... = \beta{p-1} = 0$
  * None of the predictors (from q+1 to p-1) show significant linear relationship with Y
  * Where the model is $\hat{y_{0,i}}$ and represents the **null model**
  * It has q beta-parameters where **q < p** and q-1 predictors
* $H_{A} :$ At least one of $\beta{j}\neq 0$, with $j = q,...,p-1$
  * At least of the predictors (from q+1 to p-1) shows a significant linear relationship with Y
  * Where the model is $\hat{y_{A,i}}$ and represents the **full model**

```r
# Firstly we specify the two model and save them in two different variables
H_null<- lm(target ~ x1 + x2, data = dataset)
H_alt <- lm(target ~ x1 + x2 + x3 + x4, data = dataset)
``` 

Let's denote
* $SS_{RES}(H_{0})$, the Sum of squared residuals under $H_{0}$
* $SS_{RES}(H_{A})$, the Sum of Squared residuals under $H_{A}$ 

As a simpler model implies **less uncertainty**, we might choose it when the difference between the estimates
of $H_{0}$ and $H_{A}$ is small:

$$ SS_{RES}(H_{0})- SS_{RES}(H_{A}) = \sum_{i=1}^{n}(\hat{y_{A,i}} - \hat{y_{0,i}})^{2}$$

We will use a scaled version as a good statistic:

$$\frac{SS_{RES}(H_{0}) - SS_{RES}(H_{A})}{SS_{RES}(H_{A})} $$

And now we repeat the same procedure.

### 1) ANOVA

![ANOVANM](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/ANOVANS.png?raw=TRUE)

The degrees of freedom is the difference in the number of beta-parameters estimated in the two models

```r
# Sum of Squares
SS_diff <- sum((fitted(H_alt) - fitted(H_null))^2)
SS_null <- sum(resid(H_null)^2)
SS_full <- sum(resid(H_alt)^2)

#Degrees of Freedom
DoF_diff <- length(coef(H_alt)) - length(coef(H_null))
DoF_null <- length(resid(H_null)) - length(coef(H_null))
DoF_full <- length(resid(H_alt)) - length(coef(H_alt))
```

### 2) F-Statistic and p-value

$$F = \frac{(SS_{RES}(H_{0}) - SS_{RES}(H_{A}))/(p-q)}{SS_{RES}(H_{A})/(n-p)} = \frac {\sum_{i=1}^{n}(\hat{y_{A,i}}-\hat{y_{0,i}})^{2}/(p-q)}{\sum_{i=1}^{n}(y_{i}-\hat{y_{A,i}})^2/(n-p)} \thicksim F_{p-q, n-p}$$

```r
F <- (SS_diff/DoF_diff)/(sum(resid(H_alt)^2)/H_alt$df.resid)
```

Manually, **p-value** is calculated as
```r
# Since we only care about the right side of the distribution, we put lower.tail = FALSE
# TODO
```

We see that the value of the F statistic is large, and the p-value `Pr(>F)` is pretty small, so we can
reject the null hypothesis at any reasonable $\alpha$ and say that one of the predictor `x3` or`x4` is significant when the predictors `x1` and `x2` are already in the model.


### R shortcut
As before, we can have quick insights through:
```r
anova(H_null, H_alt)
```
```r
fit <- H_alt
summary(fit)
```

---

## To sum up
We found ways to compare two nested models through ANalysis Of VAriance. The goodness of fit here is strictly related to the Sum of Squares of the residuals ($SS_{RES}$), for which a test is built
through the F distribution.

---

## Conclusions

However, this **only works to compare nested models**, while we want to have a competition between different kinds of models. 

We are unable to compare, through the F-statistic, two different models such as:
* `target ~ pred1 + pred2`
* `target ~ pred3`

We might use $R^{2}$ which indicates how much of the variance one model estimates.
The fact that this statistic always increases as the number of predictors increase can result in overfitting. In fact, when we add more predictors it increases regardless of the fact they are not significant!

```r
summary(H_alt)$r.squared
summary(H_null)$r.squared

n <- nrow(dataset)

# If we create a matrix of n normal rand. var., with mean=0 and var=1
# It has no predictive value whatsoever
useless_covariates <- matrix(rnorm(n*(n-1)), ncol = n-1)

# But we still get 1 for r-squared
summary(lm(dataset$target ̃ useless_covariates))$r.squared # =1
```

Moreover, these measures do not represent well the goodness of fit, nor the adaptability of the model. In fact, they do not say anything about the validity of the assumptions so if the assumption of data normality is not valid, the models built here are useless! This is something
we want to be able to test when facing comparisons of different models.
