# Influential Points
> An outlier is a data point which is very far, somehow, from the rest of the data. They are often worrisome, but not always a problem.[1]

When we are doing regression modeling, in fact, we don’t really care about whether some data point is far from the rest of the data, but whether **it breaks a pattern the rest of the data seems to follow**.

> An influential point is an outlier that greatly affects the slope of the regression line[2]

<p float="center">
  <img src="https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/infpoint1ex.png" width="450" />
  <img src="https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/infpoint2ex.png" width="450" /> 
</p>

As you can see, some points can be particularly influential in the estimation of $(\beta_{0}, \beta_{1})$ and even if their might be not pattern-breaking when considering the $x_{i}$ value or $y_{i}$ value, their combination $(x_{i}, y_{i})$ value might be at odds with the rest of the data.

On the other hand points which have extreme values of $X$ and $Y$ but which fall in line with the rest of the data do not greatly affect the estimation of $(\beta_{0}, \beta_{1})$

![infpoint4ex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/infpoint4ex.png?raw=TRUE)

If we are worried that outliers might be messing up our model, **we would like to quantify how much the estimates change** if we add or remove individual data points. 

Fortunately, we can quantify this using only quantities we estimated on the complete data, especially the design matrix.

## SLR
Let’s think about what happens with simple linear regression for a moment:

$$Y = \beta_{0} + \beta_{1}X + \varepsilon$$

With a single real-valued predictor variable $X$. When we estimate the coefficients by least squares, we know that:

$$\hat{\beta}_{0} = \bar{y} - \hat{\beta}_{1}\bar{x}$$

Let us turn this around. The fitted value at $X = \bar{x}$ is 

$$\bar{y} = \hat{\beta}_{0} + \hat{\beta}_{1}\bar{x}$$

Suppose we had a data point, say the $i^{th}$ point, where $X = \bar{x}$. Then the actual
value of $y_{i}$ almost wouldn't mater for the fitte value there.
* The regression line **HAS** to go through $(\bar{x},\bar{y})$ never mind wheter $y_{i}$ is close to $\bar{y}$ or far way

![infpoint3ex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/infpoint3ex.png?raw=TRUE)

If $x_{i} = \bar{x}$, we say that $y_{i}$ *has little average* over $\hat{m_{i}}$ or little influence on $\hat{m_{i}}$.
* It has *SOME* influence because $y_{i}$ is part of what we average to get $\bar{y}$,but that's not a lot of influence

Again, with SLR we know that:

$$\hat{\beta_{1}} = \frac{c_{XY}}{s_{X}^2}$$

The ratio between the sample covariance of X and Y and the sample variance of X. How does yi show up in this? It's

$$\hat{\beta_{1}} = \frac{n^{-1}\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{s_{X}^{2}}$$

Notice that 
* When $x_{i} = \bar{x}$, $y_{i}$ doesn't actually matter at all to the slope.
* If $x_{i}$ is far from $\bar{x}$, then $x_{i}=\bar{x}$ will contibute to the slope and its contribution will get bigger (whether positive or negative) as $x_{i}=\bar{x}$ grows.
* $y_{i}$ will also make a big contribution to the slope when $y_{i}-\bar{y}$ (unless, again $x_{i} = \bar{x}$ )

Let’s write a general formula for the predicted value, at an arbitrary point $X=x$

$$\hat{m}(x) = \hat{\beta}_{0} + \hat{\beta}_{1}(x)$$

$$\hat{m}(x) = \bar{y} - \hat{\beta}_{1}\bar{x} + \hat{\beta}_{1}(x)$$

$$\hat{m}(x) = \bar{y} + \hat{\beta}_{1}(x - \bar{x})$$

$$\hat{m}(x) = \bar{y} + \frac{n^{-1} \sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{s_{X}^2}(x-\bar{x})$$

So, in words: 
* The **predicted value is always a weighted average of all the** $y_{i}$
* As $x_{i}$ moves away from $\bar{x}$, $y_{i}$ gets more weight (possibly a large negative weight). When $x_{i} = \bar{x}$, $y_{i}$ only matters because it contributes to the global mean $\bar{y}$ (little leverage)
* The weights on all data points increase in magnitude when the point $x$ where we’re trying to predict is far from $\bar{x}$. 
  * If $x =\bar{x}$ only $\bar{y}$ matters.

All of this is still true of the fitted values at the original data points:
* if $x_{i} = \bar{x}$, $y_{i}$ only matter for the fitt because it contributes to $\bar{y}$
* As $x_{i}$ moves away from $\bar{x}$, in either direction, it makes a bigger contribution to **ALL** the fitte values

Why is this happening? We get the coefficient estimates by minimizing the MSE  which treats all data points equally:

$$\frac{1}{n} \sum_{i=1}^{n}(y_{i}-\hat{m}(x_{i}))^{2}$$

But we’re not just using any old function $\hat{m}(x)$; we’re using a linear function.

This has only two parameters, so we can’t change the predicted value to match each data point — altering the parameters to bring $\hat{m}(x_{i})$ closer to $y_{i}$ might actually increase the error elsewhere. 

By minimizing the over-all MSE with a linear function, we get two constraints
* First, $\bar{y} = \hat{\beta_{0}} + \hat{\beta_{1}}\bar{x}$
  * It makes the regression line insensitive to $y_{i}$ values when $x_{i}$ is close to $\bar{x}$
* Second, $\sum_{i}e_{i}(x_{i} - \bar{x}) = 0$
  * It makes the regression line very sensitive to residuals when $x_{i} - \bar{x}$ is big
  * When $x_{i} - \bar{x}$ is large, a big residual $e_{i}$ far from 0 is harder to balance out than if $x_{i} - \bar{x}$ were smaller.

To sum up
1. Least squares estimation tries to bring all the predicted values closer to $y_{i}$, but it can’t match each data point at once, because the fitted values are all functions of the same coefficients.
2. If $x_{i}$ is close to $\bar{x}$, $y_{i}$ makes little difference to the coefficients or fitted values — they’re pinned down by needing to go through the mean of the data.
3. As $x_{i}$ moves away from $\bar{x}$, $y_{i} − \bar{y}$ makes a bigger and bigger impact on both the coefficients and on the fitted values.

### Conclusions
If we worry that some point isn’t falling on the same regression line as the others, we’re really worrying that including it will **throw off our estimate** of the line.

This is going to be a concern either when $x_{i}$ is far from $\bar{x}$, or when the combination of $x_{i} -\bar{x}$ and $y_{i}-\bar{y}$ makes that point have a disproportionate impact on the estimates. 

We should also be worried if the **residual values are too big** (particularly when they correspond to values with large $(x_{i} -\bar{x})$ value).

However, when asking what’s “too big”, we need to take into account the fact that the model will try harder to fit some points than others. A big residual at a point of high leverage is more of a red flag than an equal-sized residual at point with little influence.

All of this also holds for multiple regression models where things become more complicated because of the high dimensionality (harder to know when  $x_{i}$ is far from $\bar{x}$) and we need to deal with matrices

## MLR

Recall that our least-squares coefficient estimator is:

$$\hat{\beta} = (X^{T}X)^{-1}X^{T}y$$

from which we get our fitted values as:

$$\hat{m} = X\hat{\beta} = X(X^{T}X)^{-1}X^{T} \cdot y = H \cdot y$$

with the design matrix $H = X(X^{T}X)^{-1}X^{T}$. This leads to a very natural sense in which one observation might be more or less influential than another:

$$\frac{\partial \hat{\beta}_{k}}{\partial y_{i}} = ((X^{T}X)^{-1}X^{T})_{ki}$$

and 

$$\frac{\partial \hat{m}_{k}}{\partial y_{i}} = H_{ii}$$

Comment:
* If $y_{i}$ were different, it would change the estimates for all the coefficients and for all the fitted values. 
* The rate at which the $k^{th}$ coefficient or fitted value changes is given by the kith entry in these matrices — matrices which, notice, are completely defined by the design matrix $X$. Plus we need to consider the interactions, the collinearity etc...

---
## Leverage
> $H_{ii}$ is the influence of $y_{i}$ on its own fitted value; it tells us how much of $\hat{m_{i}}$ is just $y_{i}$

This turns out to be a key quantity in looking for outliers, so we’ll give it a special name, the leverage $h_{i}$. 

Once again, the leverage of the $i^{th}$ data point doesn’t depend on $y_{i}$, only on the design matrix.

Because the general linear regression model doesn’t assume anything about the distribution of the predictors, other than that they’re not collinear, we can’t say definitely that some values of the leverage break model assumptions, or even are very unlikely under the model assumptions.

But we can say some things about the leverage.

> The trace of a matrix $A$, $\operatorname{tr}(A)$ is defined to be the sum of elements on the main diagonal (from the upper left to the lower right) of A. 
* The trace is only defined for a square matrix (n × n).

Let's consider some aspects:
1. The trace of the design matrix equals to the sum of its diagonal entries;
   1. The diagonal entries are the $i$ leverages!
2. At the same time it is equal to the number $p$ of coefficients we estimate 
3. Therefore, the **trace of the design matrix is the sum of each point’s leverage** and is equal to p, the number of regression coefficients.

$$tr(H) = \sum_{i=1}^{n}h_{ii} = h_{11} + h_{22} +...+ h_{nn} = p + 1 $$

### Average Leverages
$$\text{average leverage} = \frac{p+1}{n}$$

This represents the typical value, the leverage of a point $x_{i}$ should take.

We don’t expect every point to have exactly the same leverage, but if some points have much more than others, the regression function is going to be pulled towards fitting the high-leverage points, and the function will tend to ignore the low-leverage points.

![inflev1ex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/inflev1ex.png?raw=TRUE)

Notice one curious feature of the leverage is, and of the hat matrix in general, is that it doesn’t care what we are regressing on the predictor variables. The **values of the leverage only depend** on $X$

To sum up:
1. The leverage of a data point just depends on the value of the predictors there
2. It increases as the point moves away from the mean of the predictors. 
3. It increases more if the difference is along low-variance coordinates, and less for differences along high-variance coordinates.

When we are in a low dimensional space is easier to think in 2D, an influential point is something that does not follow the trend and impacts our estimates. However, when translating the same concept in a dimension that is higher than 3D, it is hard to notice.

---
## Standardized and Studentized residuals
We return once more to the hat matrix. The residuals, too, depend only on the hat matrix:

$$e = y - \hat{m} = (I-H)y$$

We know that the residuals vary randomly with the noise, so let’s re-write this
in terms of the noise:

$$e = (I-H)\varepsilon$$

Since $\mathbb{E}[\varepsilon] = 0$ and $Var[\varepsilon] = \sigma^{2}I$, we have 

$$\mathbb{E}[e] = 0 \;\; \text{and} \;\; Var[e]=\sigma^{2}(I-H)(I-H)^{T} = \sigma^{2}(I-H)$$ 

If we also assume that the noise is Gaussian, the residuals are Gaussian, with the stated mean and variance. What does this imply for the residual at the $i^{th}$ data point? It has expectation 0, $\mathbb{E}[e_{i}]=0$ and it has a variance which depends on $i$ through the design matrix:

$$Var[e_{i}]= \sigma^{2}(I-H)_{ii} = \sigma^{2}(1-H_{ii})$$

In other words: the **bigger the leverage** of a point $i$, the **smaller the variance of the residual** there (the model tries very hard to fit these points). If a point is strange, the model will try to minimize the error of that observation, focusing on that instance rather than the "regular" ones.

Previously, when we looked at the residuals, we expected them to all be of roughly the same magnitude. This rests on the leverages $H_{ii}$ being all about the same size. If there are substantial variations in leverage across the data points, it’s better to scale the residuals by their expected size.

The usual way to do this is through the standardized or studentized residuals

$$r_{i} \equiv \frac{e_{i}}{\hat{\sigma}\sqrt{1-H_{ii}}}$$

Why “studentized”? Because **we’re dividing by an estimate of the standard error**, just like in “Student’s” t-test for differences in means
* The distribution here is however not quite a t-distribution, because, while $e_{i}$ has a Gaussian distribution and $\hat{\sigma}$ is the square root of a $\chi^{2} distributed variable, $e_{i}$ is actually used in computing $\hat{\sigma}$, hence they’re not statistically independent. 
* Rather, $\frac{r_{i}^{2}}{n-p-1} \thicksim \beta(\frac{1}{2}, \frac{1}{2}(n-p-2))$ This gives us studentized residuals which all have the same distribution, and that distribution does approach a Gaussian as $n \rightarrow \inf$ with $p$ fixed.

All of the residual plots we’ve done before can also be done with the studentized residuals. In
particular, the studentized residuals should look flat, with constant variance, when plotted
against the fitted values or the predictors.

Indeed the plots we get when applying plot to an `lm` object in R are based ont eh Standardzed residuals, which can be obtained using `rstandard`

## Leave-One-Out - Cross-Validated/ Externally Studentized residuals
In defining the Leave-one-out cross validation (LOOCV) we considered what would be our estimate of $y_{i}$ if $y_{i}$ was not included in the dataset.

To recap a bit, the design matrix is a $n \times n$ matrix, if we deleted the $i^th$ observation when estimating the model, but still asked for a prediction at $x_{i}$, we would get a different $n \times (n-1)$ matrix which we call $H^{-i}$

This in turn would lead to a new fitted value:

$$\hat{y_{\text{[i]}}} = \hat{m}_{[i]} =\hat{m}^{(-i)}(x_{i}) = \frac{(Hy)_{i}-H_{ii}y_{i}}{1-H_{ii}}$$

Basically, this is saying we can take the old fitted value, and then subtract off the part of it which came from having included the observation yj in the first place. Because each row of the hat matrix has to add up to 1, we need to include the denominator

Now we can define, $e_{\text{[i]}}$ or $e_{i}^{(-i)}$ as the leave-one-out residual for the $i^{th}$ observation, when that observation is not used to fit the model:

$$e_{\text{[i]}} = e_{i}^{(-i)} \equiv y_{i} - \hat{y_{\text{[i]}}} = y_{i} - \hat{m}^{(-i)}(x_{i})$$

That is, this is how far off the model’s prediction of $y_{i}$ would be if it didn’t actually get to see $y_{i}$ during the estimation, but had to honestly predict it.

Leaving out the data point $i$ would give us an MSE of $\hat{\sigma_{\text{[i]}}}$

A little work says that:

$$t_{i} \equiv \frac{e_{i}^{(-i)}}{\hat{\sigma_{\text{[i]}}}\cdot \sqrt{1+X_{i}^{T}(X_{-i}^{T}X_{-i})^{-1}X_{i}} } \thicksim t_{n-p-2}$$

Fortunately, we can compute this without having to actually re-run the regression:

$$t_{i} = r_{i} \sqrt{\frac{n-p-2}{n-p-1-r_{i}^{2}}}$$

---
## Cook’s Distance
Omitting point $i$ will generally change all of the fitted values, not just the fitted value at that point. We go from the vector of predictions $\hat{m}$ to $\hat{m}^{(-i)}$

How big a change is this? It’s natural (by this point!) to use the squared length of the
difference vector,

$$||\hat{m} - \hat{m}^{(-i)}||^{2} = (\hat{m} - \hat{m}^{(-i)})^{T} (\hat{m} - \hat{m}^{(-i)}) $$

To make this more comparable across data sets, it’s conventional to divide this by $(p+1)\sigma^{2}$, since there are really only $p + 1$ independent coordinates here, each of which might contribute something on the order of $sigma^{2}$. This is called the Cook’s distance or Cook’s statistic for point $i$:

$$D_{i} = \frac{(\hat{m} - \hat{m}^{(-i)})^{T} (\hat{m} - \hat{m}^{(-i)})}{(p+1)\sigma^{2}}$$

As usual, there is a simplified formula, which evades having to re-fit the regression:

$$D_{i} = \frac{1}{p+1}e_{i}^{2} \cdot \frac{H_{ii}}{(1-H_{ii})^{2}}$$

The total influence of a point over all the fitted values grows with both its leverage $H_{ii}$ and the size of its residual when it is included $e_{i}^{2}$

This tells us how much the estimate $\hat{y}$ changes whether or not we have that particular point inside that particular model.
* If large, the information is influent
* However what we do not know is the reason! 

---

## Case Study: CYG OB1 stars
The data stars contains information on the log of the surface temperature and the log of the light intensity of 47 stars in the star cluster CYG OB1, which is in the direction of Cygnus.

```r
data(star, package = "faraway")
plot(star)
fit_star <- lm(light ̃ temp, data = star)
```

![cybdataex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/cybdataex.png?raw=TRUE)

```r
plot(star[,"temp"], hatvalues(fit_star), pch = 16)
```

![cybplot1](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/cybplot1.png?raw=TRUE)

```r
par(mfrow=c(2,2)); plot(fit_star)
```

![cybplot2](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/cybplot2.png?raw=TRUE)

```r
plot(star[,"temp"], cooks.distance(fit_star), pch = 16)
```

![cybplot3](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/cybplot3.png?raw=TRUE)

Fitted lines with and without giant stars

![cybplot4](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/cybplot4.png?raw=TRUE)

---

## What to do about influential points
1. Automatic detection of influential points can be useful to find points which are problematic
2. Sometimes these methods help us identify data recording issues or problematic measurement and we can discard the points 
3. But sometime they are the symptom of something more complicated going on in the process: need to question what is the origin of these outlying points

---

#### Credits
* [1 - Lecture 20: Outliers and Influential Points](https://www.stat.cmu.edu/~cshalizi/mreg/15/) by Cosma Shalizi
* [2 - Berman H.B., "Influential Points in Regression"](https://stattrek.com/regression/influential-points#:~:text=An%20influential%20point%20is%20an,with%20and%20without%20the%20outlier.)