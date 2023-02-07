getwd()
setwd()

# ---------------------------------
# Avoid missing data
df = df[!is.na(df$predictor),]

# Confidence intervals: EST +- CRIT * SE
# Hypothesis: EST-HYP/SE

# ---------------------------------

# SLR - y_i = b0 + b1x + epsilon_i with 
# [target | predictor = xi] = \beta_0 + \beta_1 predictor_i + \epsilon_i 
# \forall i=1,\ldots, n 
# con \epsilon_i \thicksim \mathcal{N}(0, \sigma^2) errori indipendenti.

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
## R^2*100% of the total variability observed in the target
## is explained by the linear relationship with the predictor(s)
r2 <- summary(fit)$r.squared

## Hypothesis testing - TS = EST-HYP/SE ~ t_(n-p)
## 1. Compute TS
TS <- (beta_j - hyp)/se_beta_j
## 2. Check p-value
2*pt(abs(TS), df=nrow(df)-p, lower.tail = FALSE)
# 2*(1-pt(abs(TS)), df=fit$df.residual)
## 3. If TS is large or small (far away from 0, so that the value can be
## on either tails of the t-student distribution), and the p-value is 
## small, REJECT THE NULL HYPOTHESIS
## 3B: We can also try confidence intervals 
## 3C: Check if abs(TS)>qt(1-alpha/2, df=n-p)

# se for m(x) 
se <- summary(fit)$sigma
# se <- sqrt(sum((dataset$target - fitted(fit))^2)/(nrow(dataset)-2))

## Confidence Intervals Beta_j - EST +- CRIT * SE
## If the HYP is inside the interval We DO NOT REJECT THE NULL HYPOTHESIS

est + qt(c(0.0+(alpha/2), 1.0-(alpha/2)), df=n-p) * sqrt(vcov[i,j])

# Or
cbind(beta_hat+qt(0.0+(alpha/2), n-p)*se_beta_hat,
      beta_hat+qt(1.0-(alpha/2), n-p)*se_beta_hat)
# Or
confint(fit, parm=predictor, level=0.95)

## Confidence Intervals - Y|X=x
nd <- data.frame(predictor=c(1,2,3,...,10))
predict(fit, newdata= nd, interval = "confidence", level=1-alpha)
# E' come fare la stima puntuale

# Or
s_conf <- summary(fit)$sigma * sqrt(1/n+(((nd$predictor-mean(df$predictor))^2)/sum((df$predictor-mean(df$predictor))^2)))
    
cbind(est + qt(alpha/2, df=fit$df.residual) * s_conf,
      est + qt(1-(alpha/2), df=fit$df.residual) * s_conf)

## Prediction Intervals 
## More variability 
predict(fit, newdata= nd, interval = "prediction", level=1-alpha)

# Or

s_pred <- summary(fit)$sigma * sqrt(1 +(1/n)+(((nd$predictor-mean(df$predictor))^2)/sum((df$predictor-mean(df$predictor))^2))) 

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

# ---------------------------------

# MLR
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


## Hypothesis testing - TS = EST-HYP/SE ~ t_(n-p)
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

# Stima puntuale

## Y|X=x
predict(fit, newdata = nd) # ritorna una stima puntuale

## Confidence Intervals Y|X=x 
# Must take the right form
x0 <- cbind(rep(1,3), c(1650, 3000, 5000),c(72, 75, 82))

predict(fit, newdata=nd, interval="confidence")

# Or 
se_cx0 <- est_sigma * sqrt(diag(x0 %*% solve(t(X) %*% X) %*% t(x0)))
cbind(x0 %*% beta_hat + qt(0.025, n-length(beta_hat)) * se_cx0,
      x0 %*% beta_hat + qt(0.975, n-length(beta_hat)) * se_cx0)

# Or


## Prediction Intervals Y
predict(fit, newdata=nd, interval="prediction")

# Or
se_px0 <- est_sigma * sqrt(1+diag(x0 %*% solve(t(X) %*% X) %*% t(x0)))

cbind(x0 %*% beta_hat + qt(0.025, n-length(beta_hat)) * se_px0,
      x0 %*% beta_hat + qt(0.975, n-length(beta_hat)) * se_px0)



## Visualize the intervals


## Compare Nested Models 
## REJECT the bigger model for a large F-statics and a small p-value
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
# Meno parametri -> stima meno incerta
# Piu' parametri -> aumento della varianza nella stima

## Adjusted R^2 = 1-((SSres/(n-p-1))/(SStot/(n-1)))
# Si tiene in considerazione la complessità del modello.
# Ha una penalizzazione che tiene conto del numero di gradi di 
# libertà usati dal modello
summary(fit)$adj.r.square
# adjr2 <- (sum((y-y_hat)**2)/(n-p-1))/(sum((y-mean(y))**2)/(n-1))

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
calc_loocv_rmse <- function(model) {
    sqrt(mean((resid(model) / (1 - hatvalues(model)))**2))
}

## k-fold Cross Validation
### Fit the model k times, while leaving out 1/k of the data
### to achieve "on average" evaluation

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

# ---------------------------------

# Categorical Predictors and Interactions

# Interactions
# Verificare se questo parametro (relativo all'interazione tra x1 e x2) è pari a 0
# permette di verificare se l'effetto x1 è lo stesso nei vari livelli di x2


# ---------------------------------

# Transformations

## Predictor Transformation
## E[Y|XT = t*xi] = beta0 + (beta1 xi * t)

## trasformazione di Box-Cox 
## da usare se y|X risulta non-normale 
## MASS::boxcox

# ---------------------------------

# Collinearity

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

##

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

# Influence 
## Influential points: outliers that greatly affect the slope of the regression line

cooks.distance(fit) # outliers/punti particolari 

hatvalues(fit) ##  leverages - punti di leva 
plot(dataset$predictor, hatvalues(fit_mul))
# ---------------------------------

# GLM

##in questo esempio usiamo una Poisson 

df <- data.frame(x1 = runif(15), x2 = runif(15), y  = rpois(15, 6))

## stima del modello 
fit <- glm(y~x1+x2, data = df, family = poisson()) 
## di default si usa la funzione legame canonica
poisson()$link
summary(fit) ## varie informazioni riassuntive sulla stima
coef(fit) ## valori stimati dei coefficienti del modello 
confint.default(fit) ## intervalli di confidenza per i coefficienti del modello 

## predizione 
# per i valori osservati delle X
fitted(fit) ## predizione sulla scale di Y (exp(linear.predictor))
predict(fit) ## predict di default mostra il predittore lineare
predict(fit, type = "response") ## predict accetta un'opzione type per mostrare i valori stimati sulla scala delle Y 
## per un oggetto glm predict non può costruire intervalli di confidenza (e non si possono costruire intervalli di predizione)
predict(fit, se.fit = TRUE) # con opzione se.fit si ottiene lo standard error per il predittore lineare 
## per un nuovo set di punti 
nd <- data.frame(x1 = c(0.2,0.8), x2 = c(0.3,0.6))
a <- predict(fit, newdata = nd, se.fit = TRUE); a
# intervalli di confidenza manuali
alpha = 0.05
cbind(a$fit + qnorm(alpha/2) * a$se.fit, 
      a$fit + qnorm(1-alpha/2) * a$se.fit)


# residui 
residuals(fit) ## di default deviance residuals 
residuals(fit, type = "pearson") ## type = c("deviance", "pearson", "response"))

# goodness of fit/ bontà di adattamento 
plot(fit) # grafici riassuntivi
AIC(fit, k = 2); BIC(fit); logLik(fit) ## verosimiglianza e criteri di informazione

## test anova per modelli annidati 
# anova(small_model, big_model)
anova(glm(y~x1, data = df, family=poisson()), fit, test = "LRT")



## Poisson

## Bernoulli


## Binomial



## Deviance Analysis - Confronto tra modelli nested
## La devianza funge la stessa funzione del RSS nei modelli lineari: 
## più variabili si inseriscono nel modello più diminuisce la devianza.

# L’inferenza per i parametri nei GLM si basa sul fatto che 
# le stime dei coefficienti di regressione sono stime di massima 
# verosimiglianza e sono di conseguenza approssimativamente normalmente 
# distribuite per n -> infinity

## LRT - Confronto tra modelli nested
##Per verificare la significatività del modello si possono confrontare le devianze 
## nulle e residue nel summary: un confronto formale richiede l’uso di un LRT
anova(fit_simple, fit_additive, test="LRT")

### LR Test Statistics
LR_stat <- -2 * as.numeric(logLik(fit_simple) - logLik(fit_additive))

## AIC, BIC

## Cross-Validation
### Every time we need to re-estimate the model


## Hypothesis Testing

## Confidence Intervals Bj
confint.default(fit,level=1-alpha/2)

## Confidence Intervals Y

# ---------------------------------

# GLM Classifier



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
        folded_model <- suppressWarnings(glm(model$formula, 
                                             data = dat[!whichobs,], 
                                             family = "binomial"))
        ## evaluate the model on the hold-out data
        fitted <- suppressWarnings(predict(folded_model,
                                           dat[whichobs,], 
                                           type="response"))
        observed <- dat[whichobs, strsplit(paste(model$formula), "~")[[2]]]
        error <- error + mean(observed != (fitted>cutoff))/K 
        ### in cv.glm the actual error is calculated as (y - p(y=1)) 
        # error <- error + mean((observed - fitted)^2)/K 
        ### the mis-classification rate will depend on how we decide what is assigned to each category 
    }
    error
}


make_conf_mat <- function(predicted, actual) {
    table(predicted = predicted, actual = actual)
}

get_sens <- function(conf_mat) {
    conf_mat[2, 2] / sum(conf_mat[, 2])
}
# Note that this function is good for illustrative purposes, but is easily broken. (Think about what happens if there are no "positives" predicted.)


get_spec <-  function(conf_mat) {
    conf_mat[1, 1] / sum(conf_mat[, 1])
}