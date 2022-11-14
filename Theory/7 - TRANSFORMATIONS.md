# Transformations
We need to always be aware of the assumptions of the model, since linear models are often applied to data for which they cannot work (i.e. asymmetric data)

Models checks can highlight the issues with linearity and homoschedasticity assumptions (among others). These can be (not always) solved by applying transformations of the target and/or of the predictors:
* Linearity
  * When we plot the residuals against the predictor variable, we may see a curved or stepped pattern. This strongly suggests that the relationship between Y and X is not linear. At this point, it is often useful to fall back to, in fact, plotting the yi against the xi, and try to guess at the functional form of the curve.[1]
  * One possible approach is to transform X (transformation of predictor variables) or Y (target transformation) to make the relationship linear.
* For homoskedasticity problems, transform Y to make the variance more constant


### Real life example
```r
initech_fit <- lm(salary ~ years, data = initech)
summary(initech_fit)

Call:
lm(formula = salary ̃ years, data = initech)

Residuals:
Min     1Q      Median  3Q      Max
-57225 -18104   241     15589   91332

Coefficients:
                Estimate    Std. Error t value  Pr(>|t|)
(Intercept)     5302        5750        0.922   0.359
years           8637        389         22.200  <2e-16 ***
---
...
Residual standard error: 27360 on 98 degrees of freedom
Multiple R-squared: 0.8341, Adjusted R-squared: 0.8324
F-statistic: 492.8 on 1 and 98 DF, p-value: < 2.2e-16
```

It seems like a good model until we check the residuals

![badresex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/badresex.png?raw=TRUE)

This is a typical problem of heteroskedasticity, when $\hat{Y}$ grows so do the residuals

---
## 1 - Target Transformation
Usually we have a constant variance $Var[Y|X=x] = \sigma^{2}$, while here we see the variance is a function of the mean $Var[Y|X=x] = h(\mathbb{E}[Y|X=x])$ , for some increasing function $h$

In order to correct this, one first approach we can use is to change the model, namely the response variable.

We choose and apply some invertible function to $Y$ called **variance stabilizing function** whose goal is to achieve a variance $Var[g(Y)|X=x]=c$, where c is a constant that does not depend on the mean $\mathbb{E}[Y|X=x]$. 

$$g(Y) = \beta_{0} + \beta_{1}x + \varepsilon_{i} \rightarrow Y = g^{-1}(\beta_{0} + \beta_{1}x + \varepsilon_{i})$$

### 1.2 - Variance-Stabilizing Transformation
A common variance stabilizing transformation when we see increasing variance in a fitted versus residuals plot is $\log(Y)$

Also, if the values of a variable range over more than one order of magnitude and the **variable is strictly positive**, then replacing the variable by its logarithm is likely to be helpful.

$$\log(Y_{i}) = \beta_{0} + \beta_{1}x_{i} + \varepsilon_{i}$$

```r
initech_fit_log <- lm(log(salary) ~ years, data = initech)
```

![logvstex]((https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/logvstex.png?raw=TRUE)

n the original scale of the data we have:

$$Y_{i} = exp(\beta_{0} + \beta_{1}x_{i}) \cdot exp(varepsilon_{i})$$

which has the errors entering the model in a multiplicative fashion.

We turn the additive model into a multiplicative model

![origscex]((https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/origscex.png?raw=TRUE)

And we check the residuals

![resvstex]((https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/resvstex.png?raw=TRUE)

The fitted versus residuals plot looks much better. It appears the constant variance assumption is no longer violated.

It has been demonstrated that this application, stops the variance from growing.

## 2 - Box-Cox Transformation
The great statisticians G. E. P. Box and D. R. Cox introduced a family of transformations which includes powers of Y and taking the logarithm of Y , parameterized by a number $\lambda$

This is implemented in R through the function `boxcox` in the package MASS.

It is important to remember that this family of transformations is just somet hing that Box and Cox made up because it led to tractable math. There is absolutely no justification for it in probability theory, general considerations of mathematical modeling, or scientific theories. There is also no reason to think that the correct model will be one where some transformation of Y is a linear function of the predictor variable plus Gaussian noise. Estimating a Box-Cox transformation by maximum likelihood does not relieve us of the need to run all the diagnostic checks after the transformation. Even the best Box-Cox transformation may be utter rubbish.

---
## 4 - Transformations of predictor variables

## 5 - Polynomials 

---

## 6 - Case Study: Melting Artic

---

#### Credits
* [1 - Lecture 7: Diagnostics and Modifications for Simple Regression](http://www.stat.cmu.edu/~cshalizi/mreg/) by [Cosma Shalizi](https://www.stat.cmu.edu/~cshalizi/)