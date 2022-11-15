# Influential Points
> An outlier is a data point which is very far, somehow, from the rest of the data. They are often worrisome, but not always a problem.[1]

> An influential point is an outlier that greatly affects the slope of the regression line[2]

## Leverage

## Standardized and Studentized residuals

## Cook’s Distance

---

## Case Study: CYG OB1 stars

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