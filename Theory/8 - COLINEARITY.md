# Collinearity

## Why Collinearity is a Problem
Remember our formula for the estimated coefficients in a multiple linear regression, with $p$ derivatives and $n$ observations:

$$\hat{\beta} = (X^{T}X)^{-1}X^{T}y$$

Where:
* $X^{T}$ is a $p \operatorname{x} n$ matrix
* $X$ is a $n \operatorname{x} p$ matrix
* y is a $n \operatorname{x} 1$ vector

This is obviously going to lead to problems if $X^{T}X$ isn’t invertible. Similarly, the variance of the estimates:

$$Var[\hat{\beta}] = \sigma^{2}(X^{T}X)^{-1}$$

will blow up when $X^{T}X$ is singular. If that matrix isn’t exactly singular, but is close to being non-invertible, the variances will become huge.

### Quick Algebra Review
There are several equivalent conditions for any square matrix, say $u$, to be singular or non-invertible:
1. $\operatorname{det} u = 0$ or $|u| = 0$
   1. The determinant of $u$ is 0, 
   2. We cannot invert the matrix since $u^{-1} = \frac{\operatorname{adj} u}{\operatorname{det} u}$ is not defined when $\operatorname{det}u = 0$.
   3. Hence, the **inverse of a singular matrix is NOT defined**
2. At least one eigenvalue of u is 0.
   1. This is because the determinant of a matrix is the product of its eigenvalues
   2. Again the $\operatorname{det}u = 0$
3. $u$ is rank deficient, meaning that one or more of its columns (or rows) is equal to a linear combination of the other rows.
   1. When the columns of $X$ are not linearly independent and we cannot use the matrix notation formula above.

The last explains why we call this problem collinearity: it looks like we have $p$ different predictor variables (**considering as an additional predictor variable the column defining the constant term**), but really some of them are linear combinations of the others, so they don’t add any information. 

The real number of distinct variables is $q < p$, the column rank of $X$, which is number of linearly independent columns it has.

If $X$, the desing matrix, has column rank $q < p$, then the data vectors are confined to a q-dimensional subspace. It looks like we’ve got $p$ different variables, but really by a change of coordinates we could get away with just $q$ of them.

> If the exact linear relationship holds among more than two variables, we talk about multicollinearity;

>Collinearity can refer either to the general situation of a linear dependence among the predictors, or, by contrast to multicollinearity, a linear relationship among just two of the predictors.

Again, if there isn’t an exact linear relationship among the predictors, but they’re close to one
* $X^{T}X$ will be invertible
* $(X^{T}X)^{-1}$ will be huge and the variances of the estimated coefficients will be enormous.

This can make it very hard to say anything at all precise about the coefficients, but that’s not
necessarily a problem if all we care about is prediction.

## Dealing with collinearity by deleting predictor variables
Since not all of the $p$ predictor variables are actually contributing information, a natural
way of dealing with collinearity is to **drop some predictor variables** from the model.

If you want to do this, you should think very carefully about which predictor variable to delete.

Ideally you would use knowledge on the measuring processes and about the process under study.
* Ex: As a concrete example: if we try to include all of a student’s grades as predictors, as well as their over-all GPA, we’ll have a problem with collinearity (since GPA is a linear function of the grades). 
  * But depending on what we want to predict, it might make more sense to use just the GPA, dropping all the individual grades, or to include the individual grades and drop the average.

---

## Diagnosing collinearity among pairs of predictor variables
Linear relationships between pairs of variables are fairly easy to diagnose:
* Few variables: we make the pairs plot of all the variables, and we see if any of them fall on a straight line, or close to one. Unless the number of variables is huge, this is by far the best method. 
* Lots of variables: If the number of variables is huge, look at the correlation matrix, and worry about any entry off the diagonal which is (nearly) $\plusmn 1$

Example from the car dataset
```r
fit_big <- lm(mpg ̃., data = autompg); summary(fit_big)
# Omitted

signif(cor(autompg),4)
        mpg         disp        hp          wt          acc         year
mpg     1.0000      -0.8176     -0.7800     -0.8424     0.4224      0.5786
disp    -0.8176     1.0000      0.9027      0.9352      -0.5632     -0.3720
hp      -0.7800     0.9027      1.0000      0.8687      -0.6947     -0.4154
wt      -0.8424     0.9352      0.8687      1.0000      -0.4337     -0.3127
acc     0.4224      -0.5632     -0.6947     -0.4337     1.0000      0.2922
year    0.5786      -0.3720     -0.4154     -0.3127     0.2922      1.0000
```
Strong relationships to mpg but lots of correlation between variables

### Why Multicollinearity is Harder
However a multicollinear relationship involving three or more predictor variables might be
totally invisible on a pairs plot. 

![multicex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/multicex.png?raw=TRUE)

### Some signals to identify issues
We use some checks and metrics which can help identify issues.
1. Red flag 1: Large changes in estimated coefficients when one other predictor variable is included or removed
2. Red flag 2: non-significant results in individual tests on $\beta_{j}$ for variables $X_{j}$ which appear to be important when taken individually
3. Red flag 3: estimated value of $\beta_{j}$ with opposite sign from what we see in scatterplot of $(X_{j}, Y)$ or that we expect from theoretical considerations.
4. Red flag 4: large sample correlations between $X_{i}s$ and $X_{j}s$
5. Red flag 5: $X^{T}X$ is almost singular, then check if any eigenvalues are $\approx 0$

---

## Variance inflation factors (VIF)
If the predictors are correlated with each other, the standard errors of the coefficient estimates will be bigger than if the predictors were uncorrelated.

If the predictors were UNCORRELATED 

$$Var\left[ \hat{\beta}_{i} \right] = \frac{\hat{\sigma^{2}}}{ns_{X_{i}}^{2}}$$

If they were CORRELATED 

$$Var\left[ \hat{\beta}_{i} \right] = \sigma^{2}(X^{T}X)_{i+1,i+1}^{-1}$$

The ratio between the two variances is the **variance inflation factor** for the $i^{th}$ coefficient, $VIF_{i}$:

$$ VIF_{i} =(X^{T}X)_{i+1,i+1}^{-1} \cdot ns_{X_{i}}^{2}$$

It tells us how much the variance of $\hat{\beta_{i}}$ is inflated if we have problems of collinearity.

The average of the variance inflation factors across all predictors is often written $\bar{VIF}$ or just $VIF$

It is possible to show that variance inflation factor for $X_{i}$ can be found by regressing $X_{i}$ all of the other $X_{j}$, computing the $R^{2}$ of this regression say $R_{i}^{2}$, and setting 

$$VIF_{i} = \frac{1}{1-R_{i}^{2}}$$

Consequence:
* If $VIF_{i} \geq 1$ i.e. the predictors are correlated with each other, the standard errors of the coefficient estimates will be bigger than if the predictors were uncorrelated
  * The variance inflation factor increases as $X_{i}$ becomes more correlated with some linear combination of other predictors $VIF_{i} \geq 10 \Leftrightarrow R_{i}^{2} \geq 0.9$
* Folklore (rules of thumb) says 
  * that $VIF_{i} > 10$ indicates “serious” multicollinearity for the predictor.
  * that $VIF >> 1$  indicates “serious” multicollinearity.

## In R 
```r
car::vif(fit_big)
disp        hp          wt          acc         year
11.492407   9.357184    10.894270   2.631376    1.240917

eigen(cor(model.matrix(fit_big)[,-1]))$values
[1] 3.43467889 0.80685624 0.62848320 0.07994727 0.05003440

# We try to drop disp and wt :
car::vif(lm(mpg ̃ hp + acc + year, data = autompg))
hp          acc         year
2.136251    1.932649    1.208566

eigen(cor(model.matrix(lm(mpg ̃ hp + acc + year, data = autompg))[,-1]))$values
[1] 1.9573993 0.7515672 0.2910334
```

We se many of the red flags that we highlighted before, so we try and remove predictors that cause them.

```r
summary(lm(mpg ̃ hp + acc + year, data = autompg))

Call:
lm(formula = mpg ̃ hp + acc + year, data = autompg)

Residuals:
Min         1Q          Median      3Q      Max
-11.7092    -2.8513     -0.6334     2.1798  15.5216

Coefficients:
                Estimate    Std.Error   t value     Pr(>|t|)
(Intercept)     1.006697    5.636870    0.179       0.858
hp              -0.164426   0.008104    -20.289     < 2e-16
acc             -0.662701   0.108492    -6.108      2.5e-09
year            0.657771    0.064254    10.237      < 2e-16

Residual standard error: 4.208 on 379 degrees of freedom
Multiple R-squared: 0.7148, Adjusted R-squared: 0.7125
F-statistic: 316.6 on 3 and 379 DF, p-value: < 2.2e-16
```

$\beta_{hp}$ now is negative and significant. However we lost predictive ability

---
## Is multicollinearitya problem?
Multicollinearity is another instance of the model correctness vs. usefulness

A model with multicollinearity might be perfectly valid in the sense of respecting the assumptions of the model. It does not matter whether the predictors are related or not. At least for the verification of the assumptions

But the model will be useless if the multicollinearity is high, since it can inflate the variability of the estimation without any kind of bound

Some more advanced methods do exist to deal with multicollinearity, for example **ridge regression** or **Priciple Components Regression**, we do not discuss these in the course

---

#### Credits
1. [17 - Multicollinearity Notes](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/lecture-17.pdf) by Cosma Shalizi