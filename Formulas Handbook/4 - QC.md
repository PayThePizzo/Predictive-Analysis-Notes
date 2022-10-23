# Quality Criterion

---

#### Recap
Criteria for assessing quality of fit such as $R^{2}$ and RMSE have a fatal flaw: it is impossible to add a predictor to a model and make $R^{2}$ or RMSE worse.

This suggests that we need a quality criteria that **takes into account the size of the model**, since our preference is for small models that still fit well. This translates into sacrificing a small amount of ”goodness-of-fit” to obtain a smaller model.

These measures do not represent well the **goodness of fit**, nor the **adaptability** of the model. Plus, they do not say anything about the **validity of the assumptions** which are crucial for our models to work. These are some other aspects to take into account while testing models.

---

What we cover here: 
* We will look at three criteria that do this explicitly: AIC, BIC, and Adjusted $R^{2}$ 
* We will also look at one, Cross-Validated RMSE, which implicitly considers the size of the model.
* We will look at frameworks that also prevents overfitting.


