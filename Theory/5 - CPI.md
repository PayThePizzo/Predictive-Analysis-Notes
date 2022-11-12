# Categorical Predictors and Interactions
It is normal to find columns filled with non-numerical values when dealing with datasets.
This kinds of variables are called *categorical variables*, and they include nominal variables which we know in computer science as strings.

Until now we have not covered any way to include this kind of information inside the model, even though the many categorical predictors could be useful. How can we use categorical, nominal (strings) variables for the linear model?

To check what we are missing, we just need to plot our usual model $Y = \beta_{0} + \beta_{1}x_{1} + \varepsilon$ and give a color to the points (X,Y) based on one of their categorical variables (i.e. for the penguins' dataset we can try and see female vs male). Adding our linear regression, will highlight the usefulness of adding that categorical variable to the model.

![Intro ex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/introex.png?raw=TRUE)

This looks like a graph of the residuals and we can tell there is a difference! Let's implement it

---
## Dummy Variables 
> A dummy variable is a numerical variable that is used in a regression analysis to code for a binary numerical variable

Let's update the model:

$$Y = \beta_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} + \varepsilon$$
* Y remains the same but,
* $x_{2}$ becomes a dummy variable

For our example the meaning is: 
$$x_{2} = \left\{ \begin{array}{rcl}
1 & male & Y_{male} = (\beta_{0}+\beta_{2}) + \beta_{1}x_{1} + \varepsilon = \beta_{0} + \beta_{1}x_{1} + \beta_{2}\cdot 1 +\varepsilon\\ 
0 & female & Y_{female} = \beta_{0} + \beta_{1}x_{1} + \sout{\beta_{2}x_{2}} +\varepsilon
\end{array}\right.$$

We end up effectively writing and estimating two different models. However, they are not completely different from each other. 
In fact,
1. We are estimating the same angular coefficient $\beta_{1}$, **the relation** (line) that explains how the $x_{1}$ (ex: body_mass) influences Y (ex: flipper_length), **is the same!**
2. We also find the same errors, as they share the residuals they also share the same variability! **Same $\sigma^{2}$**.

In terms of the intercept $\beta_{0}$, they are different, but they still are graphically parallel.

```r
fc_mlr_add <- lm(flipper_length_mm ~ body_mass_g + sex, data = penguins)
# Then we plot
```

![dummyex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/dummyex.png?raw=TRUE)

Since, the models are bound to pass through $(\bar{x}, \bar{y})$, this tells us that there is a difference in the the intercept of the two groups.

We could achieve the same results by estimating two different models, but this would have resulted in a loss in the number of observations used to estimate. In fact, we lose the variability on the full sample and we merely focus on the instances of the same sex, restricting our vision of the sample.

<p style="color:red">Beware that this variable has a binary meaning, such as a boolean, and sometimes it is not to be included in arithmetic operations as it would be misleading</p>

### Verifying Hypothesis
We want to test: $H_{0}:\beta_{2} = 0 vs H_{A}:\beta_{2} \neq 0$, where $H_{0}$ indicates no difference between the two interecepts. 

```r
summary(fc_mlr_add)$coefficients["sexmale",]

[1]
    Estimate        Std. Error      t value             Pr(>|t|)
-3.956995825081     0.801177208007  -4.938977027226     0.000001250047
```

### Anova & F test

```r
anova(f0_slr, fc_mlr_add)

[1]
Analysis of Variance Table

Model 1: flipper_length_mm ̃ body_mass_g
Model 2: flipper_length_mm ̃ body_mass_g + sex

    Res.Df  RSS     Df  Sum of Sq   F       Pr(>F)
1   331     15516
2   330     14448   1   1068        24.393  0.00000125 ***
```

---

## Interactions

---

## Factor Variables


## Factors with more than two levels

---

## Model Assumptions

---

## Residuals-based displays


### QQplots