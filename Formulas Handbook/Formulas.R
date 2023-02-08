# Formulas

# ---------------------------------
# Extra

# Avoid missing data
df = df[!is.na(df$predictor),]

# ---------------- SLR -----------------

# SLR - y_i = b0 + b1x + epsilon_i with 
# [target | predictor = xi] = \beta_0 + \beta_1 predictor_i + \epsilon_i 
# \forall i=1,\ldots, n 
# con \epsilon_i \thicksim \mathcal{N}(0, \sigma^2) errori indipendenti.

slr_lse_coefficients <- function(predictor , target){
    
    # Define basic data
    x <- predictor
    y <- target
    n <- length(x)
    
    # Find plugin variance
    s2x <- empirical_variance(x,n)
    s2y <- empirical_variance(y,n)
    
    rxy <- cor(x,y)
    
    mx <- mean(x)
    my<- mean(y)
    
    # Angular coefficient
    # Dictates how the y changes in proportion to x
    beta1_hat <- rxy * sqrt(s2y/s2x)
    # Alternatively
    # covxy <- cov(x,y) 
    # beta1_hat <- covxy/s2x
    
    # Intercept
    # Forces the regression to pass by the sample mean of x and y.
    beta0_hat <- my - beta1_hat *mx
    # Alternatively
    # beta0 _hat <- se * sqrt (1/n + mean (x) ^2/(n * s2x ))
    
    
    # Estimated values
    yhat <- beta0_hat + beta1_hat * x
    
    # Empirical MSE
    mse_hat<-(sum((y-yhat)^2))
    
    # Residuals
    res_hat <- y-yhat
    
    # Std. Error
    se_hat <- sqrt(sum((yhat - y)^2)/(n-2))
    
    c(beta0_hat, beta1_hat, yhat, mse_hat, res_hat, se_hat)
}

fit <- lm(formula = target~predictor, data = df)

y_hat <- fitted(fit) # stima puntuale/ valori stimati

# intercept -> (E[X], E[Y]=E[Y|X=0])
beta0_hat <- fit$coefficients[1]
# beta0_hat <- y _ bar - beta1 _ hat * x _ bar
# SAME AS doing predict(fit, newdata= data.frame(predictor=0))

# angular coefficient
beta1_hat <- fit$coefficients[2] 
# beta1_hat <- rxy * sqrt ( s2y / s2x )

## Residuals 
## epsilon_i \stackrel{iid}\sim (0, sigma^2)
residuals <- fit$residuals # or residuals(fit) or target - y_hat

## R^2
# R^2*100% of the total variability observed in the target
# is explained by the linear relationship with the predictor(s)
r2 <- summary(fit)$r.squared


## Hypothesis testing - TS = EST-HYP/SE ~ t_(alpha/2,n-p)

# 1. Compute TS
TS <- (beta_j - hyp)/se_beta_j
# 2. Check p-value
2*pt(abs(TS), df=nrow(df)-p, lower.tail = FALSE)
# 2*(1-pt(abs(TS)), df=fit$df.residual)
# 3. If TS is large or small (far away from 0, so that the value can be
# on either tails of the t-student distribution), and the p-value is 
# small, REJECT THE NULL HYPOTHESIS
# 3B: We can also try confidence intervals 
# 3C: Check if abs(TS)>qt(1-alpha/2, df=n-p)


## Confidence Intervals Bj

confint(fit, parm=bj, level=0.95)
## Confidence Intervals Beta_j - EST +- CRIT * SE
# If the HYP is inside the interval We DO NOT REJECT THE NULL HYPOTHESIS
est + qt(c(0.0+(alpha/2), 1.0-(alpha/2)), df=n-p) * sqrt(vcov[i,j])
# Or
cbind(beta_hat+qt(0.0+(alpha/2), n-p)*se_beta_hat,
      beta_hat+qt(1.0-(alpha/2), n-p)*se_beta_hat)
# Or
confint(fit, parm=predictor, level=0.95)


## Stima puntuale E[Y|X=x]
predict(fit, newdata = nd)

# se for m(x) 
se <- summary(fit)$sigma
# se <- $se.fit
# se <- sqrt(sum((dataset$target - fitted(fit))^2)/(nrow(dataset)-2))


## Confidence Intervals

nd <- data.frame(predictor=c(1,2,3,...,10))
predict(fit, newdata= nd, interval = "confidence", level=1-alpha)
# E' come fare la stima puntuale e trovare gli intervalli per ogni stima
# Or
s_conf <- summary(fit)$sigma * sqrt(1/n+(((nd$predictor-mean(df$predictor))^2)/sum((df$predictor-mean(df$predictor))^2)))
cbind(est + qt(alpha/2, df=fit$df.residual) * s_conf,
      est + qt(1-(alpha/2), df=fit$df.residual) * s_conf)


## Prediction Intervals 
# More variability 

predict(fit, newdata= nd, interval = "prediction", level=1-alpha)
# Or
s_pred <- summary(fit)$sigma * sqrt(1 +(1/n)+(
    ((nd$predictor-mean(df$predictor))^2)/sum((df$predictor-mean(df$predictor))^2))) 
cbind(est + qt(alpha/2, df=fit$df.residual) * s_pred,
      est + qt(1-(alpha/2), df=fit$df.residual) * s_pred)


## Assumptions - L.I.N.E
# Le assunzioni si possono scrivere in maniera sintetica come
# $Y_i|X=x_i \stackrel{iid}{\sim} N(\beta_0 + \beta_1 x_i, \sigma)$ 
# per ogni $i= 1, \ldots, n$. 
# Si deve quindi verificare che la relazione tra X e Y sia lineare
# e che le osservazioni seguano una normale e abbiano varianza costante. 

### Linearity (of the model) - Residuals vs Predictors
# Plot Y~X and the estimated regression, to see if there's a non-linear pattern
plot(target ~ predictor, data = dataset, xlab=xlabel, ylab=ylabel, main=title)
abline(beta0_hat, beta1_hat, col = 2, lwd = 1.4)
# Plot Residuals vs Predictors
# See if there is a pattern here, if so there's a problem
# See if there are problematic points far from the mean
plot(df$predictor, resid(fit))
abline(h=0)

whichCols <- names(fit$coefficients)[-1]

if(any(whichCols == "predictor")) whichCols[whichCols == "predictor"] <- "predictor"

par(mfrow=c(2, ceiling(length(whichCols)/2)), pch=16)

for(j in seq_along(whichCols)){
    plot(df[,whichCols[j]], residuals(fit)) 
    abline(h=0,lty=2)
}
# Check other predictors vs residuals if possible

### Independence (of observations)
# Hard to test, unless specified (we just hope they are iid and well sampled
# so that they represent the entire population)
# Can we find any reason for the observations not to be independent?
# Try to find collinearity/multicollinearity

### Normality (of errors) - QQplot
# It should not show heavy tails 
qqnorm(resid(fit))
qqline(resid(fit))

### Equal Variance/ Homoskedasticity
# Should not show a pattern, just random, equal distant points from the mean
plot(fitted(fit), resid(fit))
abline(h=0)

### ALL IN ONE
plot(fit)
# Residuals vs Fitted
## Smoother should follow the 0, no clear pattern present
# Normal QQplot
# Scale-Location
## Heteroskedasticity would show growing variance as Y grows
# Residuals vs Leverage
## Shows problematic points

# ---------------- MLR -----------------
#
# Y_i = \beta_0 + \beta_1 predictor1_i + \beta_2 predictor2_i + \epsilon_i,
# \forall i=1,\ldots, n 
# con \epsilon_i \thicksim \mathcal{N}(0, \sigma^2) errori indipendenti.

fit <- lm(target~predictor1+predictor2, data=df)

## Model Matrix = X
model.matrix(fit)

# X =   1       predictor1_{1}      predictor2_{1}
#       1       predictor1_{2}      predictor2_{2}
#       1       ...                 ...
#       1       predictor1_{n}      predictor2_{n}
X <- cbind(rep(1, n), df$predictor1, df$predictor2)

# hatvalues(fit)
H <- X  %*% solve(t(X) %*% X) %*% t(X)

y_hat <- as.vector(H %*% y)

## Variance
est_sigma <- sqrt(sum(fit$residuals^2)/(n-length(fit$coef)))
se_beta_hat <- as.vector(est_sigma * sqrt(diag(solve(t(X) %*% X))))

## Residuals
residuals(fit) ## these are y - fitted(fit)
rstandard(fit) ## standardised residuals 
rstudent(fit)  ## studentized residuals 

## Predictors with p_value lower than 0.05
chosenVars <- rownames(coef(summary(fit))[coef(summary(fit))[,4] < 0.05,])
# Fit them into new model
fit2 <- lm(target~., data = prostate[,c(chosenVars,"target")])


## Hypothesis testing - TS = EST-HYP/SE ~ t_(alpha/2,n-p)

## 1. Compute TS
TS <- (beta_j - hyp)/se_beta_j
## 2. Check p-value
2*pt(abs(TS), df=nrow(df)-p, lower.tail = FALSE)
## 3. If TS is large or small (far away from 0, so that the value can be
## on either tails of the t-student distribution), and the p-value is 
## small, REJECT THE NULL HYPOTHESIS
## 3B: We can also try confidence intervals 
## 3C: Check if abs(TS)>qt(1-alpha/2, df=n-p)


## Confidence Intervals Bj

confint(fit, parm=bj, level=0.95)


## Stima puntuale E[Y|X=x]

predict(fit, newdata = nd) # ritorna una stima puntuale

## Confidence Intervals Y|X=x 
# Must take the right form
x0 <- cbind(rep(1,3), c(1650, 3000, 5000),c(72, 75, 82))
predict(fit, newdata=nd, interval="confidence") # Stima puntuale + Intervalli
# Or 
se_cx0 <- est_sigma * sqrt(diag(x0 %*% solve(t(X) %*% X) %*% t(x0)))
cbind(x0 %*% beta_hat + qt(0.025, n-length(beta_hat)) * se_cx0,
      x0 %*% beta_hat + qt(0.975, n-length(beta_hat)) * se_cx0)


## Prediction Intervals Y

predict(fit, newdata=nd, interval="prediction") # Stima puntuale + Intervalli
# Or
se_px0 <- est_sigma * sqrt(1+diag(x0 %*% solve(t(X) %*% X) %*% t(x0)))
cbind(x0 %*% beta_hat + qt(0.025, n-length(beta_hat)) * se_px0,
      x0 %*% beta_hat + qt(0.975, n-length(beta_hat)) * se_px0)

## Visualize the intervals


## Compare Nested Models 
# REJECT the bigger model for a large F-statics and a small p-value
anova(fit_h0, fit_ha)

Anova_test <- function(fit_h0, fit_ha){
    RSS_h0 <- sum(residuals(fit_h0)^2)
    RSS_ha <- sum(residuals(fit_ha)^2)
    RSS_diff <- RSS_h0 - RSS_h1
    df_res_diff <- fit_h0$df.residual - fit_ha$df.residual
    fobs<- ((RSS_diff/df_res_diff)/(RSS_ha/fit_ha$df.residual))
    p_value<-pf(fobs,
                df1 = df_res_diff, 
                df2=fit_ha$df.residual, 
                lower.tail = FALSE)
    c(fobs, p_value)
}
   
## Model assessment 
## The F-statistic for the model is large - the model is significant, i.e. 
## it explains an amount of variability of the observed data that is considerably 
## larger than using the mean only (Null model vs Model with some predictors).
## This does not automatically mean that the model is better.
## Find a balance between being able to explain some variability but keeping the 
## number of predictors low enough to avoid inflating the model variability.

## Quando usiamo troppi parametri facciamo aumentare l’incertezza 
## nella stima: stiamo inserendo troppe variabili che non spiegano 
## variabilità generale dei dati ma seguono caratteristiche di 
## alcune osservazioni. Dobbiamo bilanciare la necessità di “spiegare bene” 
## i dati con la capacità del modello di fare predizioni non troppo incerte: 
## per ottenere questo bilanciamento cerchiamo di specificare modelli parsimoniosi.

## Model Selection
# Meno parametri -> stima dei parametri meno incerta
# Piu' parametri -> aumento della varianza nella stima (inflazione della stima)

## Adjusted R^2 = 1-((SSres/(n-p-1))/(SStot/(n-1)))
# Si tiene in considerazione la complessità del modello.
# Ha una penalizzazione che tiene conto del numero di gradi di 
# libertà usati dal modello
summary(fit)$adj.r.square

adj_r_squared <-function(y, y_hat, n,p){
    (sum((y-y_hat)**2)/(n-p-1))/(sum((y-mean(y))**2)/(n-1))
}

## IC - Higher is best
## Bontà di adattamento dei modelli in cui si tiene conto 
## della complessità del modello sono i criteri di 
## informazione legati alla verosimiglianza:
logLik(fit) # Lower is best
# logLik <- sum(dnorm(df$target, fit$fitted.values, summary(fit)$sigma,log = TRUE))

## AIC 
## AIC permette di individuare modelli in qualche senso ottimali bilanciando 
# l'aumento della verosimiglianza per modelli più complessi con una 
# penalizzazione basata sulla complessità del modelli 
# (in termini di numero di parametri stimati)
AIC(fit, k=2)
# AIC <-(-2*as.numeric(logLik(fit)))+(2*(1+length(fit$coef)))

## BIC - Prefers less complex models
AIC(fit, k=log(n))
BIC(fit)
# BIC <-(-2*as.numeric(logLik(fit)))+(2*(1+length(fit$coef)))

## LOOCV RMSE - Lower is best
### we assess how well would the model do if used to predict out-of-sample values.
### Only for SLR and MLR
### In the GLM model we need to recompute the model each time
calc_loocv_rmse <- function(model) {
    sqrt(mean((resid(model) / (1 - hatvalues(model)))**2))
}

## k-fold Cross Validation
### Fit the model k times, while leaving out 1/k of the data which is used
### to compute the estimation of the error. The final evaluation will be based
### on the mean of the k estimations of the error
###
### We achieve "on average" evaluation of the model
###
### When k=n, then we perform the same process n times, by leaving out
### just 1 observation at the time, which is more precise but computationally
### expensive. In our framework we can just use RMSE_LOO


## Model Selection
# L'algoritmo forward parte da un modello poco complesso e verifica di volta 
# in volta se aumentare la complessità del modello migliora la bontà di 
# adattamento misurata tramite AIC o BIC. 
# L'algoritmo backward invece parte da un modello complesso e ad 
# ogni passo dell'algoritmo verifica se sottrarre una variabile 
# migliora la bontà di adattamento (misurata tramite AIC o BIC)

### FS - AIC
step(null, 
     scope=list(lower=null, upper=full), 
     direction="forward", k=2, trace=1)

### FS - BIC
step(null, scope=list(lower=null, upper=full), 
     direction="forward", k=log(n), trace=1)

### BS - AIC
step(full, scope=list(lower=null, upper=full), 
     direction="backward", k=2, trace=1)
### BS - BIC
step(full, scope=list(lower=null, upper=full), 
     direction="backward", k=log(n), trace=1)

### Step - AIC
step(intermediate, 
     scope = list(lower = null, upper=full),
     direction="both", trace=1, k=2)

### Step - BIC
step(intermediate, 
     scope = list(lower = null, upper=full),
     direction="both", trace=1, k=log(n))

# -----------------Categorical Predictors and Interactions ----------------

# Categorical Predictors and Interactions

# Interactions

# Verificare se questo parametro (relativo all'interazione tra x1 e x2) è pari a 0
# permette di verificare se l'effetto x1 è lo stesso nei vari livelli di x2


# ---------------- Transformations -----------------

# Transformations

## Predictor Transformation
#
# E[Y|XT = t*xi] = beta0 + (beta1 xi * t)
#
## Logaritmo 
#
# log(Y_{i}) = \beta_{0} +\beta_{1}x_{i} + \varepsilon_{i} \rightarrow Y_{i} 
# = exp \left\{ \beta_{0} + \beta_{1}x \right\} \cdot exp \left\{\varepsilon_{i} 
# \right\}
#
# \hat{y_{i}} = \exp\{\hat{\beta_{0}}\} \cdot \exp\{\hat{\beta_{1}} x_{i}\}
#
## Trasformazione di Box-Cox 
# Da usare se y|X risulta non-normale 
#
MASS::boxcox (y~x, data = df, lambda = seq(-0.5,0.5, by=0.05))

bctransf$x[which.max(bctransf$y)]

# Box-Cox transform
boxcox(fit, plotit = TRUE)


## Polynomials - Y_i = beta0 + beta1x1 + beta2 x^2_i + e_i
#
# Even count of polynomials
lm(formula = target ~ predictor + I(predictor^2) + I(predictor^3) + I(predictor^4), 
   data = df)
# Or
lm(formula = target ~ poly(predictor, 4, raw=TRUE), data = df)

# Odd count of polynomials
lm(formula = target ~ predictor + I(predictor^2) + I(predictor^3),
   data = df)
# Or
lm(formula = target ~ poly(predictor, 3, raw=TRUE), data = df)


# --------------- Collinearity ------------------

# Xj vs Y
par(mfrow= c(3,4))
for(j in 2:13){
    plot(bodyfat[,-which(names(bodyfat) %in% c("Density", "Abdomen"))][,c(j,1)])
    title(main = paste("betahat is", signif(coef(fit_all)[j],3)))
} 
## Check Correlation
signif(cor(df),4)
## Check Covariance
signif(cov(df),4)

## VIF = 1/1-R^2
### Model
vif <- 1/(1-summary(fit)$r.squared) 
### Bj
vif_bj<- diag(solve(t(X) %*% X))[j]*(sum((data$bj-mean(data$bj))^2))

car::vif(fit) ## variance inflation factors

sort(car::vif(fit_all))


# Se vediamo valori di VIF > 10 si tende a dire che ci sono problemi 
# legati a multi-colinearità
# Un altro modo di interpretare il VIF è l’indicazione di quanto sia 
# possibile stimare una delle variabili esplicative come una funzione 
# delle altre variabili presenti nel dataset

# La stima di $\beta_{CashFlow}$ cambia di segno dei due modelli: quando si 
# tengono in considerazione tutte le altre variabili l'effetto del flusso di 
# cassa diventa negativo (e non significativo): le altre variabili incluse nel 
# modello catturano la variabilita spiegata da `CashFlow`. Questo avviene quando 
# i predittori inseriti nel modello sono fortemente correlati tra loro e si ha 
# il problema della multi-colinearità, come evidenziato anche dai valori alti 
# dei variance inflation factors.

# Data la forte correlazione tra i predittori potremmo trovarci in una 
#situazione di multicolinearità, questo crea problemi alla stima dei modello
#inflazionando la varianza della stima (che poi va anche ad influire sulla 
#precisione della stima in termini di stima dell'effetto del predittore sulla
#variabile risposta). Si può quantificare quanto la possibile colinearità dei 
#predittori vada ad inflazionare la variabilità della stima usando i 
#Variance Inflation Factors, che indicano quanto più è grande la varianza nel 
#modello stimato rispetto ad un modello con variabili indipendenti. Valori 
#grandi di VIFs indicano che la variabilità della stima è ben più grande di 
#quella che potrebbe essere usando predittori indipendenti. 
#Un altro controllo che è possibile (ed opportuno) fare è una verifica che le 
#assunzioni del modello (normalità, varianza costante, etc...) siano soddisfatte:
#i grafici dei residui non mostrano forti devianze dalle assunzioni del modello 
#lineare. La normalità dei residui non è totalmente dimostrata ma il qqplot non 
#è particolarmente problematico. Eventuali forti deviazioni dalle assunzioni del
#modello possono inficiare la validità della stima di un modello lineare.

# Includere una variabile esplicativa fortemente correlata ad un predittore 
# già presente nel modello tipicamente riduce la significatività della relazione 
# tra una o più delle variabili esplicative e la variabile risposta: questo 
# avviene perché quando si inseriscono variabili correlate tra lor si inflaziona 
# la variabilità delle stime dei coefficienti di regressione

# --------------- Influence ------------------

## Influential points: outliers that greatly affect the slope of the regression line

cooks.distance(fit) # outliers/punti particolari 

hatvalues(fit) ##  leverages - punti di leva 
plot(dataset$predictor, hatvalues(fit_mul))

# --------------- GLM ------------------

# Valore atteso e' una trasformazione della combinazione lineare
# E[Y|X=x] = g^-1(Xbeta)

## Inferenza ed intervalli di confidenza
#
# Nota: a differenza di quanto fatto per il modelli 
# lineari, per costruire l’intervallo di confidenza 
# usiamo un normale (e non una T di Student) dato 
# che stiamo utilizzando il fatto che le stime dei 
# coefficienti di regressione nei GLM sono ottenute 
# tramite massima verosimiglianza per cui possiamo 
# sfruttare il fatto che per gli stimatori di massima 
# verosimiglianza si ha una distribuzione 
# approssimativamente normale.
# Di conseguenza l’intervallo di confidenza è 
# approssimato e l’approssimazione sarà tanto 
# migliore quanto più è grande il campione.
# 
# L’inferenza per i parametri nei GLM si basa sul fatto che 
# le stime dei coefficienti di regressione sono stime di massima 
# verosimiglianza e sono di conseguenza approssimativamente normalmente 
# distribuite per n -> infinity


## Analysis of Deviance Table/ LRT Test - Confronto tra modelli nested
# La devianza funge la stessa funzione del RSS nei modelli lineari: 
# più variabili si inseriscono nel modello più diminuisce la devianza.
summary(fit)$deviance
summary(fit)$null.dev
# Per verificare la significatività del modello si possono confrontare le devianze 
# nulle e residue nel summary: un confronto formale richiede l’uso di un LRT
#
# anova(small_model, big_model)
anova(small_model, big_model, test="LRT")
# Or 
# LR Test Statistics
tstat <- as.numeric(2*(logLik(small_model) - logLik(big_model)))
diff_df <- length(small_model$coefficients) - length(big_model$coefficients)
pchisq(tstat, df = diff_df, lower.tail = FALSE)-

# AIC, BIC
AIC(fit, k = 2); 
AIC(fit, k = log(n)); BIC(fit);
    
## Cross-Validation
# Every time we need to re-estimate the model

plot(fit)

## Residuals
# type = c("deviance", "pearson", "response"))
# Deviance
residuals(fit) ## default deviance residuals 
# Pearson
residuals(fit, type = "pearson") # ~N(0,1) approximately
# Response
residuals(fit, type = "response")
# Working residuals
fit$residuals
# 
# Deviance residuals vs Bj - check for patterns (none should be there)
plot(residuals(fit)~ data$predictor, ylab="Deviance residuals")
# Dev Res vs mu_hat
plot(residuals(fit)~ predict(fit,type="response"), 
     xlab=expression(hat(mu)), ylab="Deviance residuals")
# Dev Res vs eta_hat - 
plot(residuals(fit)~ predict(fit,type="link"), 
     xlab=expression(hat(eta)), ylab="Deviance residuals")
# Response residuals vs eta_hat - No increasing pattern
plot(residuals(fit ,type="response")~predict(fit,type="link"), 
     xlab=expression(hat(eta)),ylab="Response residuals")
# Working residuals vs linear predictor - should be linear

# Assumptions hard to verify

# --------- GLM Poisson -------------

## Expected value = Variance


### Hypothesis Testing


### Stima puntuale 
# Stima Predittore Lineare  
# X beta
# Or
# preds_link <- predict(fit, newdata = nd) by default is link
preds_link <- predict(fit, newdata = nd, type = "link")

# Stima puntuale su scala della risposta (tra 0 ed 1) 
# E[Y|X=x] = exp{X beta}
# Or
preds_resp <- predict(fit, newdata = nd, type = "response")
# Or
# exp(preds_link)

### Confidence Interval
confint.default(fit, parm="predictor", level= 1-alpha/2)
# Or
# con opzione se.fit si ottiene lo standard error per il predittore lineare 
lpred <- predict(fit, type="link", se.fit=TRUE)
# For linear predictor
cbind(lpred$fit[i:j]+qnorm(0.0+(alpha/2))*lpred$se.fit[i:j],
      lpred$fit[i:j]+qnorm(1-(alpha/2))*lpred$se.fit[i:j])
# For expected value,
exp(cbind(
        lpred$fit[i:j]+qnorm(0.0+(alpha/2))*lpred$se.fit[i:j],
        lpred$fit[i:j]+qnorm(1-(alpha/2))*lpred$se.fit[i:j]))

est_bj + qnorm(c(0.0+(alpha/2), 1-(alpha/2))) * se_bj


# --------- GLM Bernoulli -------------

# Y|X=x ~ Bern(p(x) ~ Bin(1, p(x))

### Data:
# 1 - Factor
# 'success' is interpreted as the factor not having the first level 
# (and hence usually of having the second level)
# WE NEED TO MAKE SURE THE RIGHT LEVEL IS USED TO REPRESENT SUCCESS
df$target <- as.factor(df$target)
glm(target ~ predictor, data = df, family = binomial)

# 2 - Numerical vector 
# with values between 0 and 1, interpreted as the proportion 
# of successful cases (with the total number of cases given by the weights).
df$numsuccess <- as.numeric(df$target == "success") # 1 when 'success'
glm(numsuccess ~ predictor, data = df, 
    family = binomial, 
    weights = rep(1, nrow(df))) # We need to specify the weights

# 3 - As a two-column integer matrix: 
# the first column gives the number of successes and 
# the second the number of failures.
df$numsuccess <- as.numeric(df$target == "success") # successes
df$ntrial <- rep(1, nrow(df)) # failures
glm(cbind(df$numsuccess, df$ntrial-df$numsuccess) ~ df$predictor, 
    family = binomial)

## Intercept: E[Y|X=0] 

### Hypothesis Testing - EST-HYP/SE ~ N(0,1)
## 1. Compute TS
TS <- (beta_j - hyp)/se_beta_j
## 2. Check p-value
2*pnorm(abs(TS), lower.tail = FALSE)
## 3. If TS is large or small (far away from 0, so that the value can be
## on either tails of the t-student distribution), and the p-value is 
## small, REJECT THE NULL HYPOTHESIS
## 3B: We can also try confidence intervals 
## 3C: Check if abs(TS)>qnorm(1-alpha/2, df=n-p)


### Confidence Interval Bj

confint.default(fit, parm="predictor", level= 1-alpha/2)
# Or
est_bj + qnorm(c(0.0+(alpha/2), 1-(alpha/2))) * se_bj


# Stima puntuale 

# Stima Predittore Lineare -> X beta
# Or
# preds_link <- predict(fit, newdata = nd) by default is link
preds_link <- predict(fit, newdata = nd, type = "link") 

# Stima puntuale su scala della risposta (tra 0 ed 1)

# E[Y|X=x] = exp{X beta}/(1 + exp{X beta})
# Or
preds_resp <- predict(fit, newdata = nd, type = "response") 
# Or
# exp(preds_link)/(1+exp(preds_link))
## Odd ratio = preds_resp/(1-preds_resp)

# Confidence Interval
# con opzione se.fit si ottiene lo standard error per il predittore lineare 
preds <- predict(fit, newdata = nd, type = "link", se.fit = TRUE)
pint <- cbind(fit$family$linkinv(preds$fit + qnorm(0.0+(alpha/2)) * preds$se.fit),
              fit$family$linkinv(preds$fit + qnorm(1-(alpha/2)) * preds$se.fit))


# --------- GLM Binomial -------------

# Y|X = x ~ Bin(k,p(x))
# PRO: aggregated info = less space,  
# CON: cannot estimate how single instances are impacted, or do some inference

# Data for one aggregation (ex:by year) 
# - column for the "groupby" (one row per year)
# - column for count of successes (grouped by year)
# - column for total trials (grouped by year)
byYear <- data.frame(year = tapply(dat$year, factor(dat$year), unique), 
                     numopp = tapply(dat$numopp, factor(dat$year), sum), 
                     tpat = tapply(dat$numopp, factor(dat$year), length)) 

# 1 - As a two-column integer matrix: 
# the first column gives the number of successes (numopp) and 
# the second the number of failures.
byYear$n_notopp <- byYear$tpat - byYear$numopp # Columns of failures
glm(cbind(numopp, n_notopp) ~ year, family = binomial, data=byYear)

# 2 - Numerical vector 
# with values between 0 and 1, interpreted as the proportion 
# of successful cases (with the total number of cases given by the weights).
byYear$propOpp <- byYear$num_opp/byYear$tpat
glm(propOpp ~ year, data = byYear, family = binomial,  weights = tpat) 
# We need to specify the weights


### Hypothesis Testing

### Confidence Interval Bj
confint.default(fit, parm = "predictor")

#Or 
# Std. Error + Normal quantile at alpha=0.05 * SE
coef(fit)[2] + qnorm(c(0.025,0.975))*sqrt(vcov(fit)[2,2])

### Confidence Interval

predict(fit, newdata = nd, type = "link")


predict(fit, newdata = nd, type = "response")


# --------------- Classifier ------------------

# GLM Classifier - Logistic Regression

## 1. Split train-test and use test only at the end
# Train-Test Split
# Seed changes the results
set.seed(42)
spam_idx <- sample(nrow(spam), 2000)
# train set, used for model selection and diagnostics 
spam_trn <- spam[spam_idx,] 
# test set, used only at the end for evaluation
spam_tst <- spam[-spam_idx,]

## 2. Define success and cutoff 
ifelse(p > cutoff, 1, 0)

## 3. Define measure of error
### Misclassification rate -> loocv/ k-fold cv
### Confusion Matrix

## loocv-fold cv on misclassification rate
loocv_glm <- function(n, fit, target,  dataset){
    # n observations = k 
    n <- nrow(dataset)
    # save the errors of classification
    errorClass <- rep(NA, n)
    # For each i-th observation (row) in the dataset
    for(i in 1:n){
        # Fit the model without it
        # Check formula and family
        fit <- glm(target ~ ., 
                   family = binomial, 
                   data = dataset, 
                   subset = -i)
        # Compute the misclassfication rate
        # With the new model which excludes the i_th observation
        # On the i_th observation, to see how far we are.
        errorClass[i] <- (dataset$target[i] - 
                              ifelse(predict(fit, 
                                             newdata = dataset[i,], 
                                             type ="r") < 0.5, 0, 1))
    }
}

## k-fold cross validation on misclassification rate
cv_class <- function(K=5, dat, model, cutoff = 0.5){
    assign_group <- rep(seq(1,K), each = floor(nrow(dat)/K))
    ### this ensures we use all points in the dataset
    ### this way we might have subgroups of different size 
    if(length(assign_group) != nrow(dat)) assign_group <- c(assign_group, sample(seq(1,K)))[1:nrow(dat)] 
    assign_group <- sample(assign_group, size = nrow(dat))
    error <- 0
    for(j in 1:K){
        whichobs <- (assign_group == j)
        ## fit a model WITHOUT the hold-out data
        folded_model <- suppressWarnings(glm(model$formula, data = dat[!whichobs,], family = "binomial"))
        ## evaluate the model on the hold-out data
        fitted <- suppressWarnings(predict(folded_model,dat[whichobs,], type="response"))
        observed <- dat[whichobs, strsplit(paste(model$formula), "~")[[2]]]
        error <- error + mean(observed != (fitted>cutoff))/K 
        
        ### in cv.glm the actual error is calculated as (y - p(y=1)) 
        # error <- error + mean((observed - fitted)^2)/K 
        ### the mis-classification rate will depend on 
        ### how we decide what is assigned to each category 
    }
    error
}

set.seed(1)
cv_class(K=k, dat = data_train, model = fit)

# Or

## Cross validation on the actual mis-classification rate
## 1. Define the cost function, which accepts
## y, vector of observed values
## yhat, vector of estimated values from the model
cost_function <- function(y, yhat){
    # misclassification rate on cutoff
    mean((y != (yhat>0.5)))
} 
## 2. CV 
boot::cv.glm(data=df, model= fit, K = k, cost = cost_function)


# This uses the error calculated as (y - p(y=1)) 
## error <- error + mean((observed - fitted)^2)/K 
## NOT THE SAME
set.seed(1)
boot::cv.glm(data=data_train, model= fit, K = k)


## Metrics
predicted <- ifelse(predict(fit, 
                            newdata=dataset_tst, 
                            type="response") > cutoff, 1, 0)
actual <- dataset_tst$target

### Prevalence - Tasso con cui avviene l'evento di interesse
### Se non avessimo alcuna info aggiuntiva, noi assegneremmo
### Che p(x)=1 con una certa probabilita' data da prevalence
get_prevalence <- function(actual, dataset){
    table(actual)/nrow(dataset_tst)
}

## In qualche modo, questo e' pari a come classifica il modello nullo
##
## Null model: Il classificatore meno complesso a cui possiamo 
## pensare, in cui tutto viene allocato alla 
## categoria più frequente (è quello che farebbe un 
## modello logistico con solo l’intercetta)
## table(dataset$target) per controllare

## Questo e' l'errore che otterremmo se avessimo stimato un modello con 
## la sola intercetta
round(as.numeric((table(dataset_tst$target)[2])) / nrow(dataset_tst),3)

### Misclassification
get_misclassification <- function(predicted, dataset, target){
    mean(predicted != dataset_tst$target)
}

## Confusion Matrix
## 1 denotes success
##
## True Positive = Predicted=1, Actual=1
## True Negative = Predicted=0, Actual=0
## False Positive = Predicted=1, Actual=0
## False Negative = Predicted=0, Actual=1

## Predicted           Actual
##              0                   1
##   0      True Negative    | False Negative
##   1      False Positive   | True Positive
make_conf_mat <- function(predicted, actual) {
    table(predicted = predicted, actual = actual)
}

### Sensitivity 
# True Positive Rate, Tasso dei veri positiviti
# TPR = Sens = TP/P = TP/(TP+FN) = 1-FNR

# Note that this function is good for illustrative purposes, 
# but is easily broken. (Think about what happens if there are 
# no "positives" predicted.)

### Specificity
# True Negative Rate
# TNR = Spec = TN/N = TN/(TN+FP) = 1-FPR

### Precision 
# PPV = TP/TP+FP = 1-FDR

### Negative Predictive Value
# NPV = TN/TN+FN = 1-FOR

### Miss Rate
# FNR = FN/P = FN/FN+TP = 1-TNR

### False Positve Rate
# FPR = FP/N = FP/FP+TN

### Accuracy
# Acc = TP+TN/TP + TN + FP + FN = 1 - MISC
mean(spam_tst_pred == spam_tst$type)

### Misclassification rate on Conf Matrix
# Misc = FP+FN/TP + TN + FP + FN = 1 - ACC

### Prevalence
### Tasso con cui avviene l'evento di interesse
# Prev = P/ #Observations = TP + FN/ #Observations


## We can change the cutoff but
# Increasing cutoff -> more FN, less FP
# Decreasing cutoff -> less FN (Specificity), more FP (Sensitivity)
#
# The important aspect is to cut the costs and increase the cutoff
# based on what is less expensive (ex: real mail in the spam vs real spam in inbox)
