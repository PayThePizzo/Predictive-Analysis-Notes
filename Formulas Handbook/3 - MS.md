# Model Selection
Model selection is the process of analyzing what predictors should be included in a model
in order to balance its complexity and adaptability. In ML this is called feature selection.

In statistics we want to have $n > p$ as the number of feature should be ideally small, while
we hope the observations to be $n \rightarrow \infty$ (which is not always the case in ML).

## Significance of Regression
We ask ourselves whether a model with p predictors is better than just taking the expected value $\bar{y}$, since the
latter is merely the simplest possible model. We want to be able to distinguish what predictors are useful and how well
the model in question fits the data. 

However $R^{2}$ does not supply a clear answer for any of our questions, so we need to find another approach with metrics 
that:
1. Declares the model is useful
2. Tests whether the model is significantly different from taking a simpler one

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

### 1 - ANOVA
We decompose the $SS_{TOT}$ into an **AN**alysis **O**f **VA**riance table

![ANOVA](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/Formulas%20Handbook/resources/ANOVA.png?raw=TRUE)

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

### 2 - Calculate F-statistic and p-value
We calculate the F-statistic:

$$F = \frac {\sum_{i=1}^{n}(\hat{Y_{A,i}}-\bar{Y})^{2}/(p-1)}{\sum_{i=1}^{n}(Y_{i}-\hat{Y_{A,i}})^2/(n-p)} \thicksim F_{p-1, n-p}$$

We notice $F \in \mathbb{R^{+}}$
* If F is **LARGE**, we can reject $H_{0}$ as the estimated y $\hat{y_{A,i}}$ are very different from $\bar{y}$ 
  * The regression explains a large portion of the variance
* If F is small, we cannot reject it!

In R this is already include in the ANOVA table, along with the p-value.
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

Manually, **p-value** is calculated as $Pr(F > F_{obs})$, where an extreme value would only be found the right tail of the distribution
```r
df_null <- p-1
df_alt <- n-p

# Since we only care about the right side of the distribution, we put lower.tail = FALSE
pf(f_obs, df1 = df_null, df2 = df_alt, lower.tail = FALSE)
# Or
1-pf(f_obs, df1 = df_null, df2 = df_alt)
```
We see that the value of the F statistic is large, and the p-value `Pr(>F)` is extremely low, so we
reject the null hypothesis at any reasonable $\alpha$ and say that the regression is significant.


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