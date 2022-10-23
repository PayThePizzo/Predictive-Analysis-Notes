# Quality Criterion

#### Recap
Criteria for assessing quality of fit such as $R^{2}$ and rooted mean squared error (RMSE) have a fatal flaw: it is impossible to add a predictor to a model and make $R^{2}$ or RMSE worse.

This suggests that we need a quality criteria that **takes into account the size of the model**, since our preference is for small models that still fit well. This translates into sacrificing a small amount of ”goodness-of-fit” to obtain a smaller model.

These measures do not represent well the **goodness of fit**, nor the **adaptability** of the model. Plus, they do not say anything about the **validity of the assumptions** which are crucial for our models to work. These are some other aspects to take into account while testing models.

What we cover here: 
* We will look at three criteria that do this explicitly: **AIC, BIC, and Adjusted $R^{2}$**
* We will also look at one, **Cross-Validated RMSE**, which implicitly considers the size of the model.
* We will look at **frameworks** that also prevents overfitting.

---

## Information Criteria
An information criterion balances the fitness of a model with the number of predictors employed.

Given the model $Y = X\beta + \varepsilon$, where:
* $\varepsilon \thicksim \mathcal{N}(0, \sigma^{2})$ and I.I.D.
* $Y \thicksim \mathcal{N}(X\beta, \sigma^{2})$
* $\hat{\beta}$ is an estimator of MLE $\hat{\beta} = argmax_{\beta}(\ell(\beta, y))$ 
  * We are working with estimators of MLE, which can be used as a measure for the goodness of fit.

We can use **IC = goodness of fit + penalty for the complexity** 
* Complexity of a model is a function of number of parameter $p(\mathcal{M})$
* Goodness of fit can be taken to be the **negative log-likelihood** 
  * In this case $-2\log Lik(\mathcal{M})$

We use ICs of the form: 
$$-2\log Lik(\mathcal{M}) + k * p(\mathcal{M})$$

We want to find a model that **minimizes the IC**

### AIC

### BIC

## Adjusted $R^{2}$

## Cross-Validated RMSE
Using Cross-Validation means

## Frameworks