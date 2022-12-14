# GLM - Generalized Linear Models 
The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

### Classical Linear Model - Issues

$$[Y | X_{p-1}= x_{p-1}] \thicksim \mathbb{N}(\beta_{0}+\beta_{1}x_{1}+...+ \beta_{p-1}x_{p-1}, \sigma^2) $$

Ordinary linear regression predicts the expected value of a given unknown quantity (the response variable, a random variable) as a linear combination of a set of observed values (predictors). This implies that a **constant change in a predictor leads to a constant change in the response variable** (i.e. a linear-response model). This is appropriate when the response variable can vary, to a good approximation, indefinitely in either direction, or more generally for any quantity that only varies by a relatively small amount compared to the variation in the predictive variables, e.g. human heights.

<p style="color:red">However, these assumptions are inappropriate for some types of response variables. For example, in cases where the response variable is expected to be always positive, and where varying over a wide range constant input changes, lead to geometrically  varying output changes (rather than constantly varying)</p>

As an example, suppose a linear prediction model learns from some data (perhaps primarily drawn from large beaches) that a 10 degree temperature decrease would lead to 1,000 fewer people visiting the beach. This model is unlikely to generalize well over different sized beaches. More specifically, the problem is that if you use the model to predict the new attendance with a temperature drop of 10 for a beach that regularly receives 50 beachgoers, you would predict an impossible attendance value of −950.

---

## What is new with the GLM?
The GLM consists of three elements:
1. A response variable $Y$ which distribution is arbitrary and comes from the exponential families of probability distributions
2. A linear predictor $\eta = X \beta$, and
3. A link function $g$ such that $\mathbb{E}[Y|X] = \mu = g^{-1}(\eta)$.

### 1 - Exponential Family
$Y$ is assumed to be generated from a particular distribution in an exponential family (rather than simply normal distributions), such as:
* Normal
* Exponential
* Gamma
* Inverse Gaussian
* Poisson
* Bernoulli
* Binomial
* Categorical
* Multinomial

$$Y_{i} \thicksim \mathbb{EF}(\theta_{i}, \phi_{i})$$
* $\theta$, the canonical parameter which represents the **location** 
  * It is related to where the distribution is centered;
* $\phi$, the dispersion or **scale** parameter which defines how the distribution is dispersed.

One of the characteristics of this family of functions is that, it is possible to write the **probability density/mass function** as:
```math
f(y; \theta, \phi) = exp \left\{\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \theta)\right\}
```
Where
* $a(\cdot)$, $b(\cdot)$ and $c(\cdot, \cdot)$ are specific functions
* $\mu = \mathbb{E}[Y]= b'(\theta)$
* $Var[Y] = a(\phi)b''(\theta)$
  * The Variance of the distribution is also written as a function of $\theta$ and $\phi$
  * The $b''(\theta)$ function is called the variance function: specifies how the variance depends on the location parameter.

### 2 - Linear Predictor
The linear predictor is the quantity which incorporates the information about the independent variables into the model. It is related to the expected value of the data through the link function.

$$\eta = X \beta$$

### 3 - Link Function
GLM allow for an **arbitrary function of the response variable (the link function) to vary linearly with the predictors** (rather than assuming that the response itself must vary linearly).[1](https://en.wikipedia.org/wiki/Generalized_linear_model)

#### Why is this important?
Some of these distributions have constraints for the values of their parameters, so we cannot directly model the parameters of the distributions as a function of the linear predictors since we risk to estimate values that make no sense in this framework. The solution is to transform the parameters (which are usually related to the expected value) and to model a transformation of the expected value, to respect the constraints of a particular random variable.

The link function's goal is to pass from the domain of the linear predictors to the domain of the response variables. It provides the relationship between the linear predictor and the mean of the distribution function.

$$\mathbb{E}[Y|X=x] = \mu(x) = g(\beta_{0}+\beta_{1}x_{i}) = g(X \beta) = g(\eta_{i})$$
* Instead of saying that the expected values changes through a linear relationship with $x$, we say the **expected value changes through a function of the linear predictor**, the link function (or its inverse).
* In some cases it makes sense to try to match the domain of the link function to the range of the distribution function's mean.
* The interpretation of the beta parameters changes as the link function changes.

### 3.1 - Canonical Link Function 
The canonical link function $g$ is the one that transforms

$$\mathbb{E}[Y]=b'(\theta) \rightarrow \theta$$

This happens if:
* $\theta = g(\mathbb{E}[Y])$ or,
* $\theta = b^{'-1}(\mathbb{E}[Y])$

---

## Common distributions with typical use and canonical link functions

GLM is a broad class of models. We can use many different functions $g()$ for each such
function, we have a different GLM. In fact, starting from their original distribution functions we can rewrite them in the form:

```math
f(y; \theta, \phi) = exp \left\{\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \theta)\right\}
```

Here we consider the following ones:

| Distribution 	| Typical uses                                       	| $\theta$       	| $\phi$       	| $a(\phi)$ 	| $b(\theta)$             	| $c(y, \phi)$                                               	| Expected Value 	| Variance                   	| Canonical Link Function    	|
|--------------	|----------------------------------------------------	|----------------	|--------------	|-----------	|-------------------------	|------------------------------------------------------------	|----------------	|----------------------------	|----------------------------	|
| Gaussian     	| Linear-response data                               	| $\mu$          	| $\sigma^{2}$ 	| $\phi$    	| $\frac{1}{2}\theta^{2}$ 	| $-\frac{1}{2}(\frac{y^{2}}{\phi} + log(\sqrt{2\pi \phi}))$ 	| $\mu$          	| $\sigma^{2}$               	| $g(\xi) = \xi$             	|
| Gamma        	| Exponential-response data                          	| $-1/\mu$       	| $\nu$        	| $1/\nu$   	| $log(-1/\theta)$        	| $-\log (\Gamma(\nu))+\nu \log(\nu)+(\nu -1)\log(y)$        	| $\mu$          	| $1/\theta^{2} \cdot 1/\nu$ 	| $g(\xi) = -1/\xi$          	|
| Bernoulli    	| Outcome of single yes/no occurrence                	| $log(p/(1-p))$ 	| -            	| $1$       	| $log(1+exp(\theta))$    	| $0$                                                        	| $p$            	| $p(1-p)$                   	| $g(\xi) = \log(\xi/1-\xi)$ 	|
| Poisson      	| Count of occurrences in fixed amount of time/space 	| $log(\lambda)$ 	| -            	| $1$       	| $exp(\theta)$           	| $-\log(y!)$                                                	| $\lambda$      	| $\lambda$                  	| $g(\xi) = \log(\xi)$       	|

Note that:
* Poisson and Bernoulli are 1-parameter distribution: the variance is related to the mean.
  * It is harder to estimate.
* In the Normal distribution we also have a nuisance parameter $\sigma^2$

---
## Binary Response and Logistic Regression
Categorical variables with two classes such as yes/no, cat/dog, sick/healthy, etc. can be coded in a binary variable, Y , using 0 and 1. With a binary (Bernoulli) response, we’ll mostly focus on the case when Y = 1, since we can obtain probabilities of Y = 0 with:

$$\mathbb{Pr}[Y=0] = 1−\mathbb{Pr}[Y =1]=1−p$$

Moreover: 

$$E[Y]= \mathbb{Pr}[Y=1]= p$$

> Probability Odd, the probability for a positive event (Y = 1) divided by the probability of a negative event (Y = 0).

$$odd = \frac{\mathbb{Pr}[Y=1]}{\mathbb{Pr}[Y=0]} = \frac{p}{1-p} \in (0,1)$$
* When the odd is 1, the two events have equal probability. Odds greater than 1 favor a positive event. Odds smaller than 1 favor a negative event.

> The log odd is the **logit** transform applied to p

$$logit(\xi) = \log( \frac{\xi}{1 - \xi}) \in (-\infty, +\infty)$$

> The inverse logit, also known as the logistic or sigmoid function

$$logit^{-1}(\xi) = \frac{e^{\xi}}{1+e^{\xi}} = \frac{1}{1+e^{\xi}}-1$$
Note that for $\xi \in (−\infty, \infty)$ , the logistic takes values between 0 and 1. The logistic function is used to pass $(0, 1) \rightarrow (−\infty, \infty)$

### Logistic regression
We have a binary response $Y$: this is the variable we wish to model as function of a vector of $p−1$ predictors $X_{1}, . . . , X_{p−1}$. We denote

$$p(x_{1},..., x_{p-1}) = Pr[Y=1| X_{1}=x_{1}, ..., X_{p-1}=x_{p-1}] = \mathbb{E}[Y| X_{1}=x_{1}, ..., X_{p-1}=x_{p-1}]$$

A logistic regression model for Y is a model such that the logarithm of the odd is a linear combination of the predictors

$$log \left( \frac{p(x_{1},...,x_{p-1})}{1-p(x_{1},...,x_{p-1})} = X\beta \right) \rightarrow  p(x_{1},...,x_{p-1}) = \frac{e^{X\beta}}{1+e^{X\beta}}$$

Then 

$$\mathbb{E}[Y| X_{1}=x_{1}, ..., X_{p-1}=x_{p-1}] = \frac{e^{X\beta}}{1+e^{X\beta}}$$

Note that the link function g such that $g(\mathbb{E}[Y| X_{1}=x_{1}, ..., X_{p-1}=x_{p-1}]) = X\beta$ is the logit transformation $logit(\xi) = \log(\frac{\xi}{1-\xi})$

---
## Poisson response and log-linear regression
Suppose we have count responses $Y$ that we wish to model in terms of a vector of predictors $x_{1},...,x_{p-1}$.
Assume that Y is Poisson with mean $\mathbb{E}[Y] = \mu > 0$, then:

$$Pr[Y=y] = \frac{\mu^{y}}{y!}exp(-\mu)$$

Model $\mu$ in terms of the predictors:

$$Y| X_{1}=x_{1}, ..., X_{p-1}=x_{p-1} \thicksim Poisson(\mu(x_{1},...,x_{p-1}))$$
* We need to specify the function $\mu(\cdot) \geq 0$

We use a linear combination of the $x_i$ to form the linear predictor.

$$\mu(x_{1},...,x_{p-1}) = exp\{X\beta\} \rightarrow \log(\mu(x_{1},...,x_{p-1})) = X \beta$$

The link function g $g(\mathbb{E}[Y| X_{1}=x_{1}, ..., X_{p-1}=x_{p-1}]) = X\beta$ is the logarithm and the model is termed log-linear model. This corresponds to the canonical link: it is a natural way to construct the model

---

## Estimation of a GLM
In the sequel we denote $Y \thicksim \mathbf{EF}(θ, ϕ, a, b, c)$ if Y has a distribution that belongs to an
exponential family. With n observations, we write the model indexed with i to note that it is being applied to
each observation $Y \thicksim \mathbf{EF}(θ_{i}, ϕ, a, b, c)$

We define a link between the mean and expected value of the distribution:

$$\eta_{i} = g(\mu_{i}) = X\beta$$

Often we choose the **canonical link** $\eta = \theta = g(\mu)$

The beta parameters, of a GLM can be estimated using **maximum likelihood**. Then the log-likelihood is:

```math
l(\beta) = \sum^{n}_{i=1} \left\{ \frac{y_{i}\theta_{i}-b(\theta_{i})}{a(\phi)}+c(y_{i},\phi) \right\}
```

Unfortunately, unlike ordinary linear regression, there is no analytical solution for this maximization problem. Instead, it will need to be solved using **numerical approximations** via an iteratively reweighted least squares algorithm (IRLS).

![GLM1](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/glm1.png?raw=TRUE)


![GLM2](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/glm2.png?raw=TRUE)


![GLM3](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/glm3.png?raw=TRUE)


![GLM4](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/glm4.png?raw=TRUE)


![GLM5](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/glm5.png?raw=TRUE)


![GLM6](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/glm6.png?raw=TRUE)
* $V(\beta^{(t)})$ is a matrix where the diagonal is composed of weights ($\phi V_{i}$) deriving from the variance of each observation.
* $z^{(t)}$ working variable, which is a fake y that is updated at each iteration. 

![GLM7](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/glm7.png?raw=TRUE)

---

## Glm in R
The `glm` function fit a GLM model using the maximum likelihood method. For ordinary usage, the call is the same as for lm, except for one extra argument, family. The default value of family is gaussian, for each family we can specify a link function
* In the Poisson regression case the call looks like `glm(y~x, family = poisson)`
* In the logistic regression case, for example, the call looks like `glm(y~x, family = binomial)`

## glm Bernoulli example
```R
logitout <- glm(yesVote ~ . , data = chileElection, family = binomial)
```
The summary here prints out:
* Deviance Residuals, with the min, max and the quantiles
* Coefficients, the usual table with the estimations and significances.
  * Estimate 
  * Std. Error 
  * `z value` and `Pr(>|z|)`, which are test statistics on the single predictors. They are based on the distribution of the MLE, which is a gaussian distribution.
* A note `Dispersion parameter for binomial family taken to be 1` 
* Null deviance and the DoF, similar to R-Squared
* Residual deviance and the DoF, similar to the F-Statistic
* Number of observations deleted `54 observations deleted due to missingness`
* AIC
* Number of Fisher Scoring iterations

### Link
As a default R uses the canonical link, for the binomial we check `binomial()$link` which returns `"logit"`. The link can be changed - see ?family

### Variance 
Specifying the family implies the specification of the variance function:

```R
binomial()$variance

function (mu)
mu * (1 - mu)
<bytecode: 0x000001ee75be2310>
<environment: 0x000001ee764122d8>
```

### Names
Specifying the family and the link implies a number of relationships:

```R
names(binomial())

[1] "family" "link" "linkfun" "linkinv" "variance"
[6] "dev.resids" "aic" "mu.eta" "initialize" "validmu"
[11] "valideta" "simulate"
```

---

## Fitting issues for the logistic regression
We should note that, if there exists some $\beta^{*}$ such that $X\beta > 0 \Rightarrow y_{i}=1$ and $X\beta < 0 \Rightarrow y_{i}=0$, for all observations, then the MLE is not unique. Which is usually what happens when we have binary response variable.

![separabledata](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/Separabledataglm.png?raw=TRUE)

Such data is said to be separable. There are many ways to proceed with the estimations. 

This, and similar numeric issues related to estimated probabilities near 0 or 1, will return a warning in R.

```R
Warning messages:
1: glm.fit: algorithm did not converge
2: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

When this happens, the model is still “fit”, but there are consequences, namely, the estimated coefficients are highly suspect.
* This is an issue when then trying to interpret the model.
* However it could be useful for creating a classifier

---

### Credits
* [1 - Generalized Linear Models at Wikipedia](https://en.wikipedia.org/wiki/Generalized_linear_model)
  * Some rephrasing has been done to clarify the issues.
  * All the [intro](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/Theory/10%20-%20GLM.md#classical-linear-model) and the first part of [What is new with the GLM?](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/Theory/10%20-%20GLM.md#what-is-new-with-the-glm)