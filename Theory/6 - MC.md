# Model Checking
A great problem when dealing with any kind of model is to check whether our assumptions are correct.

## Recap
The least square estimate is “optimal” if the relationship betweem Y and $(X_{1},..., X_{p})$ is approximately linear.

We have discussed methods to test for significance and for estimating the variability of
estimates/predictions which is based on the assumption of iid normal errors

This is typically expressed as:

$$Y_{i} \thicksim \mathcal{N}(\beta_{0} + \beta_{1}x_{1,i} + ... + \beta_{p}x_{p,i}, \sigma)$$

Which can be rewritten as 

$$\varepsilon_{i} = (Y_{i} - (\beta_{0} + \beta_{1}x_{1,i} + ... + \beta_{p}x_{p,i}))\thicksim \mathcal{N}(0,\sigma)$$

If the assumptions are not valid we can not rely on the theory to do inference.

---

## Model Assumptions
Often, the assumptions of linear regression, are stated as:
* **L**inearity: the response can be written as a linear combination of the predictors. (With noise about this true linear relationship.)
  * Goal: We verify that there is **no variable left that could be relevant for the final prediction**.
  * There could be some quadratic/polynomial relation or interaction and we should transform the predictors.
* **I**ndependence: the errors are independent.
  * Very hard to test
* **N**ormality: the distribution of the errors should follow a normal distribution.
* **E**qual Variance: the error variance is the same at any set of predictor values.
  * Homoscedasticity

There are a number of statistical tests and graphical approaches to verify the validity of
these assumptions.

Notice that other things can go wrong and make the model not valid: statistical modeling
is a craft more than a science

---
## Residuals & Residuals-based displays
We use the residuals to prove our assumptions.

They are not exactly the errors, they represent the part of the model which has not be included and they explain part of the variance. 

We define the residuals $r_{i}$ (often indicated also as $e_{i}$) which are a sample estimate of the errors $\varepsilon_{i}$: 
$$r_{i} = (y_{i} - \hat{y_{i}})$$

By definition:
* $\sum(r_{i}/n) = 0$
* $\hat{\sigma} = \sum(r_{i}^2/(n-p))$

*How do we use the residuals? To verify our assumptions!*
1. We use the residuals to **verify whether assumptions are met** (like for simple linear models, but now its harder to separate out the effect of each variable on the quality of the fit)
2. We use residual plots to **check** both the **linearity** and **constant variance assumptions**.
   1. We do not want the residuals to grow when $\hat{Y}$ grows
3. We use the qqplot of residuals to verify the **normality assumption**.
4. We can use residuals to verify the **independence assumption** - but we should also control for this when designing the data collection

*How do we tell when this is the case?*
If the model is well specified, the sample of ( $r_{1}, ..., r_{n}$ ) should be i.i.d. normally distributed and with a constant variance.

## Plotting
Typically we graphically check the residuals to hope an empirical proof of how the errors of the model behave.

For each model we fit to a dataset we have a different vector of residuals $r_{i}$

### First residuals plot: plot $r_{i}$ against $\hat{y_{i}}$
Goal: There should not be a discernible patter in the data (no systematic under or over estimation) and constant variance
* If the variance of the data is constant there should be a scatter around the mean (which is 0)
* If the linear terms capture the entire relationship between $X$ and $Y$, $r_{i}$ should be no systematic under or over estimation for some values of $X$

Let's use an example:

![residualplotex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/residualplotex.png?raw=TRUE)

1. In the first case (leftmost graph) we see that for larger estimated values, the residual's variance grows. This is a problem of **heteroscedasticity** and it is typical to see this funnel-like shape for the data
2. In the second case (center graph), we observe a **non-linear relation** since the residuals. It is evident that we are underestimating for the values of the residuals that are really low or really high, and we are overestimating for the values around the mean.
3. The third case (rightmost graph), is what we would like to observe on average.

When observing the first two graphs we want to go back and, understand what we got wrong and we can modify.

### QQplots
The residuals should be normally distributed. If that was true the empirical cdf/pdf should look like the cdf/pdf of a normal. 

The qqplot help us comparing the empirical residuals to the theoretical quantiles of a normal distribution.

We compare them:
1. Option 1: plot a histogram of residuals.
2. Option 2 (a better option): compare empirical and theoretical quantiles via a qqplot.
   1. Can help identify long tails (excessive variance)


Sometimes for qqplot and summary statistics it is easier to use the standardized residuals (see rstandard):
$$\frac{y_{i}-\hat{y_{i}}}{\hat{\sigma}} \thicksim \mathcal{N}(0,\sigma)$$

Which should follow a standard normal. This means for example that 95% of the residuals should have values between (-2,2).

![qqplotex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/qqplotex.png?raw=TRUE)

We see we can have:
* A tail that is heavier (asymmetric data. i.e.: salary cannot be negative)
* Two heavy tails
* A good normal distributions for the errors, which is our goal

Altough the $n$ can improve the variability.

### Try it in R - qqplot and qqline
In R see `qqPlot()`, `qqplot()` and `qqline()`

Practically:
1. Sort the residuals from the smallest $r_{1}$ to the largest $r_{n}$
2. Assign the observation in position $i$ the empirical cdf value 
   1. for example $(i-0.5)/n$. 
3. Compare the valeue of $r_{i}$ to the theoretical value of a normal sample of size $n$

It is very useful to identify specific points with particularly high residuals, and deviations from normality.

---
## To sum up
The baseline is, if model assumptions are not valid the inference might be dubious.

But nothing is set in stone: models which deviate from the assumptions can be still be OK.

Nevertheless if keeping a non-significance variable in the model improves the model-checks it might be a good reason to keep the variable. The same hold for transformations discussed below. 

Model building is an iterative process: check the data, fit a model or two, check residuals and iterate. Some **subjective** decisions on what is a “good fit”

In R plot(lm.object) gives some useful plots for model-checking.

There can be other problematic cases which we do not discuss - it is a wide field of research