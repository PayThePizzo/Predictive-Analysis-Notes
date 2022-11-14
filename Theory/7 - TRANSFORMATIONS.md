# Transformations
We need to always be aware of the assumptions of the model, since linear models are often applied to data for which they cannot work (i.e. asymmetric data)

Models checks can highlight the issues with linearity and homoschedasticity assumptions (among others). These can be (not always) solved by applying transformations of the target and/or of the predictors:
* Linearity
  * When we plot the residuals against the predictor variable, we may see a curved or stepped pattern. This strongly suggests that the relationship between Y and X is not linear. At this point, it is often useful to fall back to, in fact, plotting the yi against the xi, and try to guess at the functional form of the curve.[1]
  * One possible approach is to transform X (transformation of predictor variables) or Y (target transformation) to make the relationship linear.
* For homoskedasticity problems, transform Y to make the variance more constant

Real life example
```r
initech_fit <- lm(salary ̃ years, data = initech)
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


---
## 1 - Target Transformation
One first approach we can use to face the issue of linearity is to change the model, namely the response variable. That is, one imagines the model is:

$$g(Y) = \beta_{0} + \beta_{1}x + \varepsilon_{i}$$

for some invertible function g. In more old-fashioned sources, this is advocated as a way of handling non-constant variance, or non-Gaussian noise. 

A better rationale is that it might in fact be true. Since the transfor- mation g has an inverse, we can write

$$Y = g^{-1}(\beta_{0} + \beta_{1}x + \varepsilon_{i})$$

Even if $\varepsilon_{i} \thicksim \mathcal{N}(0, \sigma^{2})$, this implies hat Y will have a non-Gaussian distribution,
with a non-linear relationship between $\mathbb{E}[Y|X=x]$ and x, and a non-constant variance. 

If that’s actually the case, we’d like to incorporate that into the model.

## 2 - Variance-Stabilizing Transformation


## 3 - Box-Cox Transformation
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