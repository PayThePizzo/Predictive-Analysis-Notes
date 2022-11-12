# Categorical Predictors and Interactions
It is normal to find columns filled with non-numerical values when dealing with datasets.
This kinds of variables are called *categorical variables*, and they include nominal variables which we know in computer science as strings.

Until now we have not covered any way to include this kind of information inside the model, even though the many categorical predictors could be useful. How can we use categorical, nominal (strings) variables for the linear model?

To check what we are missing, we just need to plot our usual model $Y = \beta_{0} + \beta_{1}x_{1} + \varepsilon$ and give a color to the points (X,Y) based on one of their categorical variables (i.e. for the penguins' dataset we can try and see female vs male). Adding our linear regression, will highlight the usefulness of adding that categorical variable to the model.

![Intro ex](https://github.com/PayThePizzo/Predictive-Analysis-Notes/blob/main/resources/introex.png?raw=TRUE)

This looks like a graph of the residuals 