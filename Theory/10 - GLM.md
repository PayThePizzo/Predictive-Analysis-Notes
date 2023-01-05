# GLM - Generalized Linear Models 

### Classical Linear Model - Issues

$$[Y | X_{p-1}= x_{p-1}] \thicksim \mathbb{N}(\beta_{0}+\beta_{1}x_{1}+...+ \beta_{p-1}x_{p-1}, \sigma^2) $$

Ordinary linear regression predicts the expected value of a given unknown quantity (the response variable, a random variable) as a linear combination of a set of observed values (predictors). This implies that a **constant change in a predictor leads to a constant change in the response variable** (i.e. a linear-response model). This is appropriate when the response variable can vary, to a good approximation, indefinitely in either direction, or more generally for any quantity that only varies by a relatively small amount compared to the variation in the predictive variables, e.g. human heights.

<p style="color:red">However, these assumptions are inappropriate for some types of response variables. For example, in cases where the response variable is expected to be always positive, and where varying over a wide range constant input changes, lead to geometrically  varying output changes (rather than constantly varying)</p>

As an example, suppose a linear prediction model learns from some data (perhaps primarily drawn from large beaches) that a 10 degree temperature decrease would lead to 1,000 fewer people visiting the beach. This model is unlikely to generalize well over different sized beaches. More specifically, the problem is that if you use the model to predict the new attendance with a temperature drop of 10 for a beach that regularly receives 50 beachgoers, you would predict an impossible attendance value of −950.

---

## What is new with the GLM?

Generalized linear models cover all these situations by **allowing for response variables that have arbitrary distributions** (rather than simply normal distributions) which needs to be from the exponential family of distributions.

$$Y_{i} \thicksim \mathbb{EF}(\theta_{i}, \phi_{i})$$
* $\theta$, the **location** parameter which is related to where the distribution is centered;
* $\phi$, the **scale** parameter defines how the distribution is dispersed.

Furthermore, they allow for an arbitrary function of the **response variable (the link function) to vary linearly with the predictors** (rather than assuming that the response itself must vary linearly).[1](https://en.wikipedia.org/wiki/Generalized_linear_model)

$$\mathbb{E}[Y|X=x] = \mu(x) = g(\beta_{0}+\beta_{1}x_{i}) = g(X \beta) = g(\eta_{i})$$
* Instead of saying that the expected values changes through a linear relationship with $x$, we say the **expected value changes through a function of the linear predictor**, the link function (or its inverse).
* The link function's goal is to pass from the domain of the linear predictors to the domain of the response variables. 
* The interpretation of the beta parameters changes as the link function changes.

### GLM components
1. A **response variable** $Y$ that follows a distribution from the $\mathbb{EF}$ family $Y_{i} \thicksim \mathbb{EF}(\theta_{i}, \phi_{i})$
2. A linear predictor $\eta = X \beta$
   1. The linear predictor is the quantity which incorporates the information about the independent variables into the model.
3. A link function $g$ such that $\mathbb{E}[Y|X=x] = \mu(x) = g(\beta_{0}+\beta_{1}x_{i}) = g(X \beta) = g(\eta_{i})$ 
   1. The link function provides the relationship between the linear predictor and the mean of the distribution function.

---
## Exponential Family
One of the characteristics of this family of functions is that, it is possible to write the **pdf** as:

```math
f(y; \theta, \phi) = exp \left\{\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \theta)\right\}
```
* $\mathbb{E}[Y]= b'(\theta)$
* $Var[Y] = a(\phi)b''(\theta)$

To this family, belong the normal, inverse gaussian, binomial, exponential, gamma, poisson, bernoulli distributions.

These functions can only be defined on the positive axis.


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
Note that for $\xi \in (−\infty, \infty)$ , the logistic takes values between 0 and 1.

---

## Poisson response and log-linear regression

---

### Credits
* [1 - Generalized Linear Models at Wikipedia](https://en.wikipedia.org/wiki/Generalized_linear_model)
  * Some rephrasing has been done to clarify the issues.
  * All the [intro](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/Theory/10%20-%20GLM.md#classical-linear-model) and the first part of [What is new with the GLM?](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/Theory/10%20-%20GLM.md#what-is-new-with-the-glm)
* 