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
* Mind that we also need to estimate $\sigma^{2}$ for the IC computation, so R inserts it inside
the complexity of the model
  * Some other models define them differently!

We use ICs of the form: 
$$-2\log Lik(\mathcal{M}) + k * p(\mathcal{M})$$

>We want to find a model that **minimizes the IC**

For linear regression, the Max Likelihood reduces to a function of the $SS_{RES}$ after fitting the model $\mathcal{M}$, but it is only valid for this type of model:

![IC](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/Formulas%20Handbook/resources/IC.png?raw=TRUE)

Because the MLE of $\sigma$ is $\hat{\sigma} = (\sum_{i=1}^{n}(y - \hat{y_{i}})^{2})/n$

However, they are general and can be applied to models that are out of the scope of linear models!

For the linear regression models:

## Akaikes' Information Criterion
Best used when n is large, and we want to use the model for prediction.

$$AIC(\mathcal{M}) = n * \log MSS_{RES}(\mathcal{M}) + 2p(\mathcal{M})$$

```r
# Two models with different predictors
mod1 <- lm(target ~ x1 + x2, data = dataset) 
mod2 <- lm(target ~ x1 + x2 + x3 + x4, data = dataset) 

#AICs
c(AIC(mod1), AIC(mod2))

```

## Bayesian Information Criterion
It replaces the $2$ with $log(n)$ so it penalizes more complex models. Best used to find parsimonious, highly-interpretable models.

$$BIC(\mathcal{M}) = n * \log MSS_{RES}(\mathcal{M}) + \log (n)p(\mathcal{M})$$

This is one of the reasons why BIC is preferred by some practitioners for model comparison. Also, because it is consistent in selecting the true model: if enough data is provided, the BIC is guaranteed to select the data-generating model among a list of candidate models

```r
#BICs
c(BIC(mod1), BIC(mod2))

#Or
AIC(mod1, k = log(nrow(dataset)))
AIC(mod2, k = log(nrow(dataset)))
```

> ICs: small value indicates a low test error

However, the IC evaluation does not formally supply with any indication whether a model is better than another one! We can just use it to "sort" models.

---

## Adjusted $R^{2}$
In the MLR models and some other cases we can use this version or r-squared. The presence
of unnecessary variables influences negatively this measure!

$$Adjusted \; R^{2} = 1 - \frac{SS_{RES}/(n-p-1)}{SS_{TOT}/(n-1)} $$

> We want to find a model that **maximizes the $Adjusted \; R^{2}$**

This is equals to $minimize(SS_{RES}/(n-p-1))$
* $SS_{RES}$ always decreases as the number of parameter (complexity) increases
* $SS_{RES}/(n-p-1)$ may increase or decrease due to the presence of p

> $R^{2}$: high value indicates a model with a small test error

```r
# Adjusted R Squared
summary(mod1)$adj.r.squared
summary(mod2)$adj.r.squared
```
