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

The last explains why we call this problem collinearity: it looks like we have $p$ different predictor variables, but really some of them are linear combinations of the others, so they don’t add any information. 

The real number of distinct variables is $q < p$, the column rank of $X$. 

> If the exact linear relationship holds among more than two variables, we talk about multicollinearity;

>Collinearity can refer either to the general situation of a linear dependence among the predictors, or, by contrast to multicollinearity, a linear relationship among just two of the predictors.

Again, if there isn’t an exact linear relationship among the predictors, but they’re close to one
* $X^{T}X$ will be invertible
* $(X^{T}X)^{-1}$ will be huge and the variances of the estimated coefficients will be enormous.

This can make it very hard to say anything at all precise about the coefficients, but that’s not
necessarily a problem if all we care about is prediction.

## Dealing with collinearity by deleting predictor variables

## Diagnosing collinearity among pairs of predictor variables

## Variance inflation factors (VIF)

## Is multicollinearitya problem?

---

#### Credits
1. [17 - Multicollinearity Notes](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/lecture-17.pdf) by Cosma Shalizi