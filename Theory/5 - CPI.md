# Categorical Predictors and Interactions

## Some Vocabulary First
> Categorical variables (also called qualitative variable) refers to a characteristic that can’t be quantifiable.

Categorical variables can be either nominal or ordinal.

> A nominal variable is one that describes a name, label or category without natural order. 
Sex and type of dwelling are examples of nominal variables 

> An ordinal variable is a variable whose values are defined by an order relation between the different categories. 

In the variable “behaviour” is ordinal because the category “Excellent” is better than the category “Very good,” which is better than the category “Good,” etc. There is some natural ordering, but it is limited since we do not know by how much “Excellent” behaviour is better than “Very good” behaviour.[3](https://www150.statcan.gc.ca/n1/edu/power-pouvoir/ch8/5214817-eng.htm)

> Dichotomous variables (or indicator) are categorical variables with two categories or levels — different groups within the same independent variable.

> Binary variables are a sub-type of dichotomous variable; variables assigned either a 0 or a 1 are said to be in a binary state. For example Male (0) and female (1)

Dichotomous variables can be further described as either a discrete dichotomous variable or a continuous dichotomous variable. The idea is very similar to regular discrete variables and continuous variables. 
* When two dichotomous variables are discrete, there’s nothing in between them
  * “Dead or Alive” is a discrete dichotomous variable. You can be dead. Or you can be alive. But you can’t be both at the same time.
* When they are continuous, there are possibilities in between.
  * “Passing or Failing an Exam” is a continuous dichotomous variable. Grades on a test can range from 0 to 100% with every possible percentage in between. You could get 74% and pass. You could get 69% and fail. Or a 69.5% and pass (if your professor rounds up!).

The line between discrete and continuous dichotomous variables is very thin. For example, it could be argued that a person who has been dead for three days is “more” dead than someone who has been declared brain dead and who is on life support.[4](https://www.statisticshowto.com/dichotomous-variable/)

---

It is normal to find columns filled with non-numerical values (i.e. strings) when dealing with datasets.

Until now we have not covered any way to include this kind of information inside the model, even though the many categorical predictors could be useful. How can we use categorical variables for the linear model?

To check what we are missing, we just need to plot our usual model $Y = \beta_{0} + \beta_{1}x_{1} + \varepsilon$ and give a color to the points (X,Y) based on one of their categorical variables (i.e. for the penguins' dataset we can try and see female vs male). Adding our linear regression, will highlight the usefulness of adding that categorical variable to the model.

![Intro ex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/introex.png?raw=TRUE)

This looks like a graph of the residuals and we can tell there is a difference! Let's find a way to quantify it.

<p style="color:red">Beware that any conversion from categorical variables to numerical values, can be misleading when used for certain arithmetic operations.</p>

## Dummy Variables 
> A dummy variable (or indicator) is a numerical variable that is used in a regression analysis to code for a binary numerical variable

It is exactly what we call a binary variable.

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
>An interaction occurs when an independent variable has a different effect on the outcome depending on the values of another independent variable. [1](https://www.medicine.mcgill.ca/epidemiology/joseph/courses/EPIB-621/interaction.pdf)

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

We consider the first and the third ones.

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
summary (fi_mlr_int)
```

![int2ex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/int2ex.png?raw=TRUE)

$\beta_{2}$ has lost its significance, since we used a more complex model and we estimated a lower variance (the model now explains more). $\beta_{3}$ has high p-value too!

By using a more complex model, we shadow the signal of $\beta_{2}$.

3) An alternative method uses the `*` operator. This method automatically creates the interaction term, as well as any ”lower order terms” which in this case are the first order terms for body_mass_g and sex

```r
fi_mlr_int2 <-lm(flipper_length_mm ̃ body_mass_g * sex, data = penguins)
```

### Verifying Hypothesis
Testing for $\beta_{3}$: 
* Testing two lines with parallel slopes $H_{0}: \beta_{3}=0$
  * Less complex models, same angular coefficient, different intercepts
* Testing two lines with possibly different slopes $H_{A}: \beta_{3}\neq 0$
  * More complex models, different angular coefficients, different intercepts

```r
# Uses a t-test to perform the test
summary(fi_mlr_int)$coefficients["body_mass_g:sexmale",]
[1]
    Estimate        Std. Error      t value         Pr(>|t|)
    -0.0006176503   0.0010129746    -0.6097391962   0.5424554613
```
The t-value is close to 0 and the p-value is very high, so we do not reject the $H_{0}$ with confidence. We can say that, the effect of the total dimension of the penguin on the flipper length is equal in the two groups, even though they have different mean values. We can use test ANOVA again if neeeded.

![estimatex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/estimatex.png?raw=TRUE)

Even if the results differ, they have no statistically relevant difference, which leads us to use a simpler model, with a difference only in the intercept.

---

## Factor Variables
> A "factor" is a vector whose elements can take on one of a specific set of values, called levels (categories).

Factor variables are special variables that R uses to deal with categorical variables.
A factor can also contain ordinal variables.

For example, "Sex" will usually take on only the values "M" or "F," whereas "Name" will generally have lots of possibilities. The set of values that the elements of a factor can take are called its **levels**. If you want to add a new level to a factor, you can do that, but you can't just change elements to have new values that aren't already levels [2]

### Binary Encoding
If we only have two levels for a factor, we can encode a variable with an appropriate name (i.e. `is_x` or `has_`) which takes 0 if the condition (or the feature is not present) is not statisfied, else 1,  and use it to fit a model.

### Introductory example for Penguins
How can we translate factors, usually strings, into numerical values to inculde them into our model?
```r
# Check type
class(penguins$sex)
```

We can encode a variable gender which takes 0 for male and 1 for female penguins.
```r
# Set col to 0 
penguins$gender <- 0
# Set female instances to gender=1
penguins$gender[penguins$sex == "female"] = 1
penguins$gender[1:4]

[1] 0 1 1 1

class(penguins$gender)
[1] "numeric"

# Build the model 
fc_mlr_add_alt <- lm(flipper_length_mm ̃ body_mass_g + gender, data = penguins)

# Same R^2
summary(fc_mlr_add)$r.squared; summary(fc_mlr_add_alt)$r.squared
[1] 0.7784678
[1] 0.7784678
```

However, it seems that it doesn’t produce the same results
* $\beta_{0}$ changes
* $\beta_{1}$ is different, as is the the coefficient in front of sex
* $\beta_{2}$ same magnitude
```r
coef(fc_mlr_add)
    (Intercept)     body_mass_g     sexmale
    134.63634714    0.01624103      -3.95699583

coef(fc_mlr_add_alt)
    (Intercept)     body_mass_g     gender
    130.67935131    0.01624103      3.95699583
```

### Generalizing - Under the hood
What is happening?

When we ask R to use a variable stored as a string, the program turns this into a factor
and from the factor it creates dummy variables: one dummy for each level except the
reference level. The reference level is taken to be the first level, by the order is set alphabetically. 

By having the program create the levels we can avoid doing unreasonable things such as
making prediction for levels we have not observed. Let's avoid the following operations:
```r
# Dumb Prediction, what does gender=0.2 mean???
predict(fc_mlr_add_alt,newdata = data.frame(body_mass_g=45,gender=0.2))
# This gives an error
predict(fc_mlr_add,newdata=data.frame(body_mass_g=45,sex="unknown"))
```
Obviously this defeats the binary nature of the variables, even though R allows us to estimate absurd instances.

<mark>Under the hood, R has turned the string into a dummy variable </mark>, with female as the baseline ($x_{2} = 0$):

```r
# This returns some consistent values
predict(fc_mlr_add,newdata=data.frame(body_mass_g=c(45,45),sex=c("male","female")))
        1          2
131.4102    135.3672

# Not including output here, but remember the naming conventions also matter
head(model.matrix(fc_mlr_add))
```

We can also force this transformation manually by calling the `factor()` function on a column of the dataset.Then to check the levels (possible values) for a factor we use the function `levels()`
```r
penguins$sex2 <- factor(penguins$sex)
penguins$sex2[1:4]

[1] male female female female
Levels: female male

levels(penguins$sex2)

[1] "female" "male"
```
### Ordered Factors
> An "ordered" factor is a factor whose levels have a particular order. 

Ordered variables inherit from factors, so anything that you can to a factor you can do to an ordered factor. Create ordered factors with the `ordered()` command, or by using `factor(...,ordered=TRUE)`. Many R models generally ignore ordering even if it is present.

Remember that R is case sensitive, so "female" and "Female" are seen as two different levels.

---
## Factors with more than two levels
Let’s now consider a factor variable with more than two levels.

Species is an example:
```r
penguins$species <- factor(penguins$species)
# To check the number of instances
table(penguins$species)
[1] Adelie  Chinstrap   Gentoo
    146     68          119

unique(as.numeric(penguins$species))
[1] 1 3 2
```
It would not make much sense to include a numerical values as encoding of the labels from the species (es: `Adelie -> 1`), as that would imply a numerical distance between the species. 

### One-Hot-Encoding
This method is best suited when an element can take a set of values (i.e. movies can be dramatic and romantic).

We use dichotomous variables for each one of the levels.

We create multiple $v_{i}$ dummies binary variables whose levels are either 1 or 0, for each one of the levels $i$ in the factor column representing the categorical variable.

### Penguins example
Let’s define three dummy variables related to the species factor variable. 
* v1, Adelie (1) or not Adelie (0)
* v2, Chinstrap (1) or not Chinstrap (0)
* v3, Gentoo (1) or not Gentoo (0)

We fit an additive model in R, using `flipper_length_mm` as the response (Y), and `body_mass_g` (x) and `species` ($v_{2}, v_{3}$ the dummy variables defined above) as predictors that uses ”three regression lines” to model flipper’s length, one for each of the possible species levels

$$Y = \beta_{0} + \beta_{1}x + \beta_{2}v_{2} +  \beta_{3}v_{3} + \varepsilon$$

Notice we use two dummy variables to codify the three level categorical variable.

```r
fc_mass_species <- lm(flipper_length_mm ̃ body_mass_g + species, data = penguins)
fc_mass_species

Call:
lm(formula = flipper_length_mm ̃ body_mass_g + species, data = penguins)

Coefficients:
    (Intercept)     body_mass_g     speciesChinstrap    speciesGentoo
    158.546071      0.008515        5.491543            15.328938
```
R doesn’t use $v_{1}$ because it doesn’t need to. To create three lines, it only needs two dummy variables since it is using a **reference level**



---

## Linear Models Repurposed


---
#### Credits 
* [1 - Interaction](https://www.medicine.mcgill.ca/epidemiology/joseph/courses/EPIB-621/interaction.pdf) from [Lawrence Joseph](https://www.medicine.mcgill.ca/epidemiology/joseph/)'s Epidemiology Course at [McGill University](https://www.mcgill.ca/)
* [2 - Factors](https://faculty.nps.edu/sebuttre/home/R/factors.html) from [Samuel E. Buttrey](https://faculty.nps.edu/sebuttre/)'s General Applied Statistics Course at [Naval Postgraduate School](https://nps.edu/)
* [3 - Types of Variables](https://www150.statcan.gc.ca/n1/edu/power-pouvoir/ch8/5214817-eng.htm) by [Statistics Canada](https://www.statcan.gc.ca/en/start)
* [4 - Dichotomous Variable](https://www.statisticshowto.com/dichotomous-variable/) by [Statistics How To](https://www.statisticshowto.com/)
