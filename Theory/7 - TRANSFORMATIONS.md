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
## 1 - Target Transformation - Heteroskedasticity problems
For the assumptions made until now, we want to achieve constant variance $Var[Y|X=x] = \sigma^{2}$. 

Instead, here we see the variance is a function of the mean $Var[Y|X=x] = h(\mathbb{E}[Y|X=x])$ , for some increasing function $h$. In order to correct this, one first approach we can use is to change the model.

We choose and apply some invertible function to $Y$ called **variance stabilizing function** whose goal is to achieve a variance $Var[g(Y)|X=x]=c$, where c is a constant that does not depend on the mean $\mathbb{E}[Y|X=x]$. 

$$g(Y) = \beta_{0} + \beta_{1}x + \varepsilon_{i} \rightarrow Y = g^{-1}(\beta_{0} + \beta_{1}x + \varepsilon_{i})$$

### 1.2 - Variance-Stabilizing Transformation
A common variance stabilizing transformation when we see increasing variance in a fitted versus residuals plot is $\log(Y)$

Also, if the values of a variable range over more than one order of magnitude and the **variable is strictly positive**, then replacing the variable by its logarithm is likely to be helpful.

$$\log(Y_{i}) = \beta_{0} + \beta_{1}x_{i} + \varepsilon_{i}$$

```r
initech_fit_log <- lm(log(salary) ~ years, data = initech)
```

![logvstex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/logvstex.png?raw=TRUE)

In the original scale of the data we have:

$$Y_{i} = exp(\beta_{0} + \beta_{1}x_{i}) \cdot exp(varepsilon_{i})$$

which has the errors entering the model in a multiplicative fashion.

Let's check the results

```r
plot(salary ~ years, data = initech, col = "grey", pch = 20, cex = 1.5, 
      main = "Salaries at Initech, By Seniority")

curve(exp(initech_fit_log$coef[1] + initech_fit_log$coef[2] * x),
      from = 0, to = 30, add = TRUE, col = "darkorange", lwd = 2)
```

![origscex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/origscex.png?raw=TRUE)

And we check the residuals 

![resvstex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/resvstex.png?raw=TRUE)

The fitted versus residuals plot looks much better. <mark>It appears the constant variance assumption is no longer violated, but we pay the price of having different model.</mark>

Let's compare errors
```r
# Sigma^2 under the original model
sqrt(mean(resid(initech_fit) ˆ 2))
[1] 27080.16

# Sigma^2 under the log model
sqrt(mean(resid(initech_fit_log) ˆ 2))
[1] 0.1934907
```

But wait, that isn’t fair, this difference is simply due to the different scales being used.

Since we changed the model, <mark>we also changed the scale of the data!</mark>

We cannot compare them like this, we need to rescale.
```r
sqrt(mean((initech$salary - fitted(initech_fit)) ˆ 2))
[1] 27080.16
sqrt(mean((initech$salary - exp(fitted(initech_fit_log))) ˆ 2))
[1] 24280.36
```

The transformed response is a linear combination of the predictors:

$$log(\hat{y}(x)) = \hat{\beta_{0}}+ \hat{\beta_{1}}x$$

If we re-scale the data from a log scale back to the original scale of the data, we now have

$$\hat{y}(x) = exp(\hat{\beta_{0}}) \cdot exp(\hat{\beta_{1}}x)$$

The average salary increases $exp(\hat{\beta_{1}}x)$ times for one additional year of experience.

Comparing the RMSE using the original and transformed response, we also see that the log transformed model simply fits better, with a smaller average squared error

### 1.2.1 - Conclusions for VST
The model has changed, we are now considering $\log(Y)$:
* Y's distribuiton is a log-Normal distribution, and we are merely modeling its transformation $g(\mathbb{E}[X])$.
* We turned the additive model of the noise into a **multiplicative model** of the noise. 
  * It makes sense, since it takes into account the increasing error.
* Reinterpretation of **the beta parameters** :
  * For each year $x_{i}$ of experience (unit of measure) the median changes accordingly to $x_{i} \cdot \hat{\beta_{1}}$
* Since the logarithm is a monotonic transformation, the median is the exponential of Y's log will be the same, in fact: $\text{median}(\log(X)) = \log(\text{median}(X))$
* The expected value is not the same, in fact: $\mathbb{E}[(g(X))] \neq g(\mathbb{E}[X])$
* The scale is different, we are modeling a different variable and back-transforming needs to be done with care.

If the data is negative, this cannot be applied!

---
## Power Transformation
>In statistics, a power transform is a family of functions applied to create a monotonic transformation of data using power functions. It is a data transformation technique used to stabilize variance, make the data more normal distribution-like, improve the validity of measures of association (such as the Pearson correlation between variables), and for other data stabilization procedures.

Definition: 
> The power transformation is defined as a continuously varying function, with respect to the power parameter $\lambda$, in a piece-wise function form that makes it continuous at the point of singularity ($\lambda = 0$). 

For data vectors $y_{1},...,y_{n}$ in which each $y_{i} > 0$, the power transform is

```math
y_i^{(\lambda)} =
\begin{cases}
\dfrac{y_i^\lambda-1}{\lambda(\operatorname{GM}(y))^{\lambda -1}} , &\text{if } \lambda \neq 0 \\[12pt]
\operatorname{GM}(y)\ln{y_i} , &\text{if } \lambda = 0
\end{cases}
```
where: 
```math
\operatorname{GM}(y) = (\prod_{i=1}^{n}y_{i})^{\frac{1}{n}} = \sqrt[n]{y_1 y_2 \cdots y_n} \, 
```
is the geometric mean of the observations $y_{1},..., y_{n}$. 

The case for $\lambda =0$ is the limit as $\lambda$ approaches 0.[2]

---
## 1.3 - Box-Cox Transformation
The great statisticians G. E. P. Box and D. R. Cox introduced a family of transformations which includes powers of Y and taking the logarithm of Y, parameterized by a number $\lambda$

$$y_i^{(\lambda)} =
\begin{cases}
\dfrac{y^\lambda-1}{\lambda} , &\text{if } \lambda \neq 0 \\[12pt]
\log(y) , &\text{if } \lambda = 0
\end{cases}$$

> Idea: The transformation is defined for positive data only, so the possible remedy could be translating the data after to the positive domain by adding a constant. 

It is important to remember that this family of transformations is just something that Box and Cox made up because it led to tractable math. There is absolutely no justification for it in probability theory, general considerations of mathematical modeling, or scientific theories. 

There is also no reason to think that the correct model will be one where some transformation of Y is a linear function of the predictor variable plus Gaussian noise. 

Estimating a Box-Cox transformation by maximum likelihood does not relieve us of the need to run all the diagnostic checks after the transformation. Even the best Box-Cox transformation may be utter rubbish.

We choose $\lambda$ that specifies how much we are changing the data:
* $\lambda < 1$ for positively skewed data
* $\lambda > 1$ for negatively skewed data

The $\lambda$ is chosen by numerically maximizing the log-likelihood:

$$L(\lambda) = -\frac{n}{2} \cdot \log(\frac{SS_{err}(\lambda)}{n}) + (\lambda-1)\sum_{i=1}^{n}\log(y_{i})$$

where $SS_{err}(\lambda) = \sum_{i=1}^{n}(y_{\lambda,i} - \hat{y}_{\lambda, i})^{2}$

### Procedure 
1. We need to estimate a linear model (and eventually re-iterate if we change the model)
   1. Choose the predictors 
2. We compute many $\lambda$ values
3. We compute the likelihood of the transformation
4. We choose the $\lambda$ value that "stabilize" the values of Y, so that its variability is reduced. That is a $\lambda$ value that maximizes the function $\operatorname{L}(\lambda)$

### In R - Confidence Intrerval for $\lambda$
This is implemented in R through the function `boxcox` in the package MASS. We then use the boxcox function to find the best transformation of the form considered by the Box-Cox method, and builds a confidence interval for the best lambda values. 

We can build a $100(1−\alpha)$% confidence interval for $\lambda$ is:

$$\lambda: L(\lambda) >  L(\hat{\lambda})-\frac{1}{2}\chi_{1,\alpha}^{2}$$

R will plot for us to help quickly select an appropriate $\lambda$ value
```r
# Box-Cox transform
boxcox(initech_fit, plotit = TRUE)
```

![boxcoxex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/boxcoxex.png?raw=TRUE)

We often choose a ”nice” value from within the confidence interval, instead of the best value of $\lambda$, so that it is more interpretable.
* Ex: $\hat{\lambda} = 0.061$, that truly maximizes the likelihood. In this case we choose $\lambda = 0$, which takes us to the logarithmic function, very interpretable.

---
## 2 - Transformations of predictor variables - Linearity problems
Sometimes we can focus our attention on the x: the linear model is termed linear not because the regression curve is a plane, but because the effects of the parameters are linear.

Rather than working with the sample $(x_{1}, y_{1}),...,(x_{n}, y_{n})$, we consider the transformed sample  $(\tilde{x}_{1}, y_{1}),...,(\tilde{x}_{n}, y_{n})$

Practically this adds a new column to the design matrix but does not change the fact that the model is still linear $Y = X\beta +\varepsilon$. Furthermore, the interpretation does not change that much.

For example consider these linear models and the possible transformed samples:
1. $Y = \beta_{0} + \beta_{1}x^{2} + \varepsilon$ 
   1. where we can work with $\tilde{x}_{i}=x_{i}^{2}$
2. $Y = \beta_{0} + \beta_{1}\log(x) + \varepsilon$ 
   1. where we can work with $\tilde{x}_{i}=\log(x_{i})$
3. $Y = \beta_{0} + \beta_{1}(x^{3}-\log(|x|)+ 2^{x}) + \varepsilon$
   1. where we can work with $\tilde{x}_{i}=x_{i}^{3}-\log(|x_{i}|)+ 2^{x_{i}}$

Sometimes these transformations can help with violation of model assumptions and other times they can be used to simply fit a more flexible model!

### Car Dataset Example
Let's use the car dataset to to model `mpg` as a function of `hp`

We first attempt a SLR, but we see a rather obvious pattern in the fitted versus residuals
plot, which includes increasing variance.

![failedslrex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/failedslrex.png?raw=TRUE)

We attempt a log transform of the response (rough approximation)

![failedtrsex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/failedtrsex.png?raw=TRUE)

After performing the log transform of the response, we still have some of the same issues with the fitted versus response. We try also log transforming the predictor.

![improvementex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/improvementex.png?raw=TRUE)

Here, our fitted versus residuals plot looks good.

### Tip
1. If you apply a nonlinear transformation, namely $f()$, and fit the linear model $Y = \beta_{0} + \beta_{1}f(x) + \varepsilon$, then there is no point in fit also the model resulting from the negative transformation $-f()$.
   1. The model with $-f()$ is exactly the same as the one with $f()$ but with the sign of $\beta_{1}$ flipped!
2. As a rule of thumb, use the next figure with the transformations to compare it with the data pattern, then choose the most similar curve, and finally apply the corresponding function with **positive sign**.

---

## 2.1 - Polynomials 
A common ”transformation” of a predictor variable is the polynomial transformation Polynomials are very useful as they allow for more flexible models, but do not change the units of the variables.

Consider the non-linear transformation, namely $f$:
$$Y = f(x) + \varepsilon$$

We make no global assumptions about the function $f$ but assume that locally it can be well approximated with a member of a simple class of parametric function, e.g. a constant or straight line

This is related to Taylor's theorem that says that any continuous function can be approximated with polynomial.

> Taylor teorem
> Suppose $f$ is a real function on [$a,b$], $f^{K-1}$ is continuous on [$a,b$], $f^{K}(x)$ is bounded for $x\in(a,b)$ then for any distinct points $x_{0}< x_{1}$ in [$a,b$] there exists a point $\tilde{x}$ between $x_{0}< \tilde{x} < x_{1}$ such that

$$f(x_{1})= f(x_{0}) + \sum_{k=1}^{K-1} \frac{f^{k}(x_{0})}{k!}(x_{1}-x_{0})^{k} + \frac{f^{K}(\tilde{x})}{K!}(x_{1}-x_{0})^{K}$$

Notice: if we view $f(x_{0})+\sum_{k=1}^{K-1} \frac{f^{k}(x_{0})}{k!}(x_{1}-x_{0})^{k}$ as function of $x_{1}$, it's a polynomial in the family of polynomials

$$\mathcal{P}_{K+1} = \{ f(x) = a_{0} + a_{1}x + ... + a_{K}x^{K}, (a_{0},...,a_{K})^{\text{'}} \in \mathbb{R}^{K+1} \}$$

Using polynomials to approximate the predictors can be useful when we are not in high dimensional space. The only problems comes with extrapolation, since the estimate becomes more variable.

We could fit a polynomial of an arbitrary order $K$,

$$Y_{i} = \beta_{0} + \beta_{1}x_{i} + \beta_{2}x_{i}^{2}+...+ \beta_{K}x_{i}^{K} + \varepsilon_{i}$$

and we can think of the polynomial model as the Taylor series expansion of the unknown function

### Example Polynomials
Suppose you work for an automobile manufacturer which makes a large luxury sedan. You would like to know how the car performs from a fuel efficiency standpoint when it is driven at various speeds.

Instead of testing the car at every conceivable speed (which would be impossible) you create an experiment where the car is driven at speeds of interest in increments of 5 miles per hour (Response surface designs)

Our goal then, is to fit a model to this data in order to be able to predict fuel efficiency when driving at certain speeds

![polyyex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/polyex.png?raw=TRUE)

We see a pattern but it is no linear pattern! Let's say we are very stubborn and wish to fit a SLR to this data

```r
econ <- read.csv("data/fuel-econ.csv")
fit1 <- lm(mpg ̃ mph, data = econ)
summary(fit1)

Call:
lm(formula = mpg ̃ mph, data = econ)

Residuals:
Min     1Q    Median 3Q     Max
-8.337 -4.895 -1.007 4.914 9.191

Coefficients:
              Estimate  Std. Error  t value Pr(>|t|)
(Intercept)   22.74637  2.49877     9.103   1.45e-09 ***
mph           0.03908   0.05312     0.736   0.469
...
Residual standard error: 5.666 on 26 degrees of freedom
Multiple R-squared: 0.02039, Adjusted R-squared: -0.01729
F-statistic: 0.5411 on 1 and 26 DF, p-value: 0.4686
```

![polyslrex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/polyslrex.png?raw=TRUE)

Pretty clearly we can do better. Yes fuel efficiency does increase as speed increases, but only up to a certain point.

We will now add polynomial terms until we fit a suitable fit

To add the second order term we need to use the `I()` function in the model specification.

`I()` is the identity function, which tells R “leave this alone”

```r
fit2 <- lm(mpg ̃ mph + I(mph ˆ 2), data = econ)
summary(fit2)

Call:
lm(formula = mpg ̃ mph + I(mphˆ2), data = econ)

Residuals:
Min     1Q      Median  3Q      Max
-2.8411 -0.9694 0.0017  1.0181  3.3900

Coefficients:
            Estimate    Std.Error   t value   Pr(>|t|)
(Intercept) 2.4444505   1.4241091   1.716     0.0984 .
mph         1.2716937   0.0757321   16.792    3.99e-15 ***
I(mphˆ2)    -0.0145014  0.0008719   -16.633   4.97e-15 ***
---
...
Residual standard error: 1.663 on 25 degrees of freedom
Multiple R-squared: 0.9188, Adjusted R-squared: 0.9123
F-statistic: 141.5 on 2 and 25 DF, p-value: 2.338e-14
```
![poly](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/poly.png?raw=TRUE)

---

## 6 - Case Study: Melting Artic

---

#### Credits
* [1 - Lecture 7: Diagnostics and Modifications for Simple Regression](http://www.stat.cmu.edu/~cshalizi/mreg/) by [Cosma Shalizi](https://www.stat.cmu.edu/~cshalizi/)
* [2 - Power Transformations on Wikipedia](https://en.wikipedia.org/wiki/Power_transform)