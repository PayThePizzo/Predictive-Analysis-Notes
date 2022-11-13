# Model Checking

## Recap

The least square estimate is “optimal” if the relationship betweem Y and $(X_{1},..., X_{p})$ is approximately linear.

We have discussed methods to test for significance and for estimating the variability of
estimates/predictions which is based on the assumption of iid normal errors

This is typically expressed as:

$$Y_{i} \thicksim \mathcal{N}(\beta_{0} + \beta_{1}x_{1,i} + ... + \beta_{p}x_{p,i}, \sigma)$$

Which can be rewritten as 

$$\varepsilon_{i} = (Y_{i} - (\beta_{0} + \beta_{1}x_{1,i} + ... + \beta_{p}x_{p,i}))\thicksim \mathcal{N}(0,\sigma)$$

If the assumptions are not valid we can not rely on the theory to do inference.

---

## Model Assumptions

---

## Residuals-based displays


### QQplots