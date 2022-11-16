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

## Example with SLR
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

If $x_{i} = \bar{x}$, we say that $y_{i}$ *has little average over* $\hat{m}_{i}$ or little influence on $\hat{m}_{i}$
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
* $\hat{m}(x) = \hat{\beta}_{0} + \hat{\beta}_{1}(x)$
* $\hat{m}(x) = \bar{y} - \hat{\beta}_{1}\bar{x} + \hat{\beta}_{1}(x)$
* $\hat{m}(x) = \bar{y} + \hat{\beta}_{1}(x - \bar{x})$

$$\hat{m}(x) = \bar{y} + \frac{n^{-1} \sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{s_{X}^2}(x-\bar{x})$$

---
## Leverage


---
## Standardized and Studentized residuals



## Cook’s Distance

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