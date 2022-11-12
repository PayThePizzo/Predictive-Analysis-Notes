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
1 & male & Y_{male} = (\beta_{0}+\beta_{2}) + \beta_{1}x_{1} + \varepsilon\\ 
0 & female & Y_{female} = \beta_{0} + \beta_{1}x_{1}+\varepsilon
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

We could achieve the same results by estimating two different models, but this would have resulted in a loss in the number of observations used to estimate. <mark>In fact, we lose the variability of the full sample </mark>and we merely focus on the instances of the same sex, restricting our vision of the sample. Consequently the estimate of the variance would not match between two different models.

<p style="color:red">Beware that this variable has a binary meaning, such as a boolean, and sometimes it is not to be included in arithmetic operations as it would be misleading</p>

### Verifying Hypothesis
We want to test: $H_{0}:\beta_{2} = 0 vs H_{A}:\beta_{2} \neq 0$, where $H_{0}$ indicates no difference between the two interecepts. 

```r
# Significance of the coefficient relative to sex
summary(fc_mlr_add)$coefficients["sexmale",]

[1]
    Estimate        Std. Error      t value             Pr(>|t|)
-3.956995825081     0.801177208007  -4.938977027226     0.000001250047
```
We can see there is a difference of 3.957 mm in the length of the flipper, which is statistically relevant.

### Anova & F test
It is easy to see that these models can be thought as nested models, so an ANOVA test is
alternative.

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
Notice that the F statistics is the t-test statistic squared.

---
## Interactions
>An interaction occurs when an independent variable has a different effect on the outcome depending on the values of another independent variable. 

Let's consider: 
$$Y = \beta_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} + \beta_{3}x_{1}x_{2}+\varepsilon$$

This model essentially creates two slopes and two intercepts:
* $\beta_{2}$ being the difference in intercepts
* $\beta_{3}$ being the difference in slopes/angular coefficients.

Here we still do not divide the sample based on any of the parameter, so that the estimate of the variability stays consistent between the two groups and we can also build tests for the previous statements.

In terms of regression equations, we have three different models:
1. No effect of sex: 
   * $x_{2}=0 \rightarrow Y = \beta_{0} + \beta_{1}x_{1} +\varepsilon$
2. Sex has an effect but **no interaction**: 
   * $x_{2}=1 \rightarrow Y = \beta_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} +(0 \cdot x_{1}x_{2})+\varepsilon$
3. Sex has an effect with an interaction: 
   * $x_{2}=1 \rightarrow Y = \beta_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} + \beta_{3}x_{1}x_{2}+\varepsilon$


### Fitting the model in R

1) Create a new variable then fit a model like any other
 
```r
penguins$sexInt <- penguins$body_mass_g * (penguins$sex == "Male")

do_not_do_this <- lm(flipper_length_mm ̃ body_mass_g + sex + sexInt, data = penguins)
```
You should only do this as a last resort!

2) Use the existing data with an interaction term such as the `:` operator
 
```r
fi_mlr_int <- lm(flipper_length_mm ̃ body_mass_g + sex + body_mass_g : sex, data = penguins)
```

3) An alternative method uses the `*` operator. This method automatically creates the interaction term, as well as any ”lower order terms” which in this case are the first order terms for body_mass_g and sex

```r
fi_mlr_int2 <-lm(flipper_length_mm ̃ body_mass_g * sex, data = penguins)
```

### Verifying Hypothesis
We consider the first and the third ones, to test for difference in the two groups.
$$H_{0}: \beta_{3}=0 vs H_{A} \neq 0$$




---

## Factor Variables


## Factors with more than two levels

---

## Model Assumptions

---

## Residuals-based displays


### QQplots

---
Credits 
* [1 - Interaction](https://www.medicine.mcgill.ca/epidemiology/joseph/courses/EPIB-621/interaction.pdf) from [Lawrence Joseph](https://www.medicine.mcgill.ca/epidemiology/joseph/)'s Epidemiology Course at [McGill University](https://www.mcgill.ca/)