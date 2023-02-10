# Formulas

# ---------------- SLR -----------------

fit <- lm(formula = target~predictor, data = df)

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
    beta1_hat <- rxy * sqrt(s2y/s2x)
    # beta1_hat <- rxy * sqrt ( s2y / s2x )
    beta0_hat <- my - beta1_hat *mx
    # beta0_hat <- y _ bar - beta1 _ hat * x _ bar
    
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

y_hat <- fitted(fit) # stima puntuale/ valori stimati

# se for m(x) 
se <- summary(fit)$sigma
# se <- $se.fit
se <- sqrt(sum((dataset$target - fitted(fit))^2)/(nrow(dataset)-2))


# intercept 
beta0_hat <- fit$coefficients[1]
# angular coefficient
beta1_hat <- fit$coefficients[2] 
# Residuals 
residuals <- fit$residuals # or residuals(fit) or target - y_hat
# R^2
r2 <- summary(fit)$r.squared

## Hypothesis testing 
hyp_test_t <- function(beta_j, hyp, se_beta_j, n, p){
    # 1. Compute TS
    TS <- (beta_j - hyp)/se_beta_j
    # 2. Check p-value
    p_value <- 2*pt(abs(TS), df=nrow(df)-p, lower.tail = FALSE) 
    # 2*(1-pt(abs(TS)), df=fit$df.residual)
    c(TS, p_value)
}
# 2): We can also try confidence intervals 
# 3)
p_value <- function(TS, alpha, n, p){
    # Check if abs(TS)>qt(1-alpha/2, df=n-p)
    ifelse((abs(TS)>qt(1-(alpha/2), df=n-p)), TRUE, FALSE)
}


# Confidence Intervals Beta_j -> EST +- CRIT * SE
confint(fit, parm=bj, level=1-(alpha/2))
# Or 
est + qt(c((alpha/2), 1.0-(alpha/2)), df=n-p) * sqrt(vcov[j,j])
# Or
cbind(beta_hat+qt((alpha/2), n-p)*se_beta_hat,
      beta_hat+qt(1.0-(alpha/2), n-p)*se_beta_hat)

# Stima puntuale
predict(fit, newdata = data.frame(predictor= value))

# Confidence Intervals
nd <- data.frame(predictor=c(1,2,3,...,10))
predict(fit, newdata= nd, interval = "confidence", level=1-alpha)
# Or
s_conf <- summary(fit)$sigma * sqrt(1/n+(((nd$predictor-mean(df$predictor))^2)/sum((df$predictor-mean(df$predictor))^2)))
cbind(est + qt(alpha/2, df=fit$df.residual) * s_conf,
      est + qt(1-(alpha/2), df=fit$df.residual) * s_conf)

# Prediction Intervals

predict(fit, newdata= nd, interval = "prediction", level=1-alpha)
# Or
s_pred <- summary(fit)$sigma * sqrt(1 +(1/n)+(
    ((nd$predictor-mean(df$predictor))^2)/sum((df$predictor-mean(df$predictor))^2))) 
cbind(est + qt(alpha/2, df=fit$df.residual) * s_pred,
      est + qt(1-(alpha/2), df=fit$df.residual) * s_pred)

# ---------------- MLR -----------------

fit <- lm(target~predictor1+predictor2, data=df)

# Model Matrix = X
model.matrix(fit)
# Or
X <- cbind(rep(1, n), df$predictor1, df$predictor2)

# hatvalues(fit)
H <- X  %*% solve(t(X) %*% X) %*% t(X)

y_hat <- as.vector(H %*% y)

# Variance
est_sigma <- sqrt(sum(fit$residuals^2)/(n-length(fit$coef)))
se_beta_hat <- as.vector(est_sigma * sqrt(diag(solve(t(X) %*% X))))

# Residuals
residuals(fit) ## these are y - fitted(fit)
rstandard(fit) ## standardised residuals 
rstudent(fit)  ## studentized residuals 

# Predictors with p_value lower than 0.05
chosenVars <- rownames(coef(summary(fit))[coef(summary(fit))[,4] < 0.05,])
# Fit them into new model
fit2 <- lm(target~., data = prostate[,c(chosenVars,"target")])


# Hypothesis testing 
## 1. Compute TS
TS <- (beta_j - hyp)/se_beta_j
## 2. Check p-value
2*pt(abs(TS), df=nrow(df)-p, lower.tail = FALSE)

# Confidence Intervals Bj
confint(fit, parm=bj, level=1-(alpha/2))

## Stima puntuale
#
predict(fit, newdata = nd) 

## Confidence Intervals  
# Must take the right form
x0 <- cbind(rep(1,3), c(1650, 3000, 5000),c(72, 75, 82))
predict(fit, newdata=nd, interval="confidence") 
# Or 
se_cx0 <- est_sigma * sqrt(diag(x0 %*% solve(t(X) %*% X) %*% t(x0)))
cbind(x0 %*% beta_hat + qt(0.025, n-length(beta_hat)) * se_cx0,
      x0 %*% beta_hat + qt(0.975, n-length(beta_hat)) * se_cx0)

## Prediction Intervals 
predict(fit, newdata=nd, interval="prediction") 
# Or
se_px0 <- est_sigma * sqrt(1+diag(x0 %*% solve(t(X) %*% X) %*% t(x0)))
cbind(x0 %*% beta_hat + qt(0.025, n-length(beta_hat)) * se_px0,
      x0 %*% beta_hat + qt(0.975, n-length(beta_hat)) * se_px0)

# Compare Nested Models 
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
   
# Adjusted R^2
summary(fit)$adj.r.square
# Or
adj_r_squared <-function(y, y_hat, n,p){
    # Adjusted R^2 = 1-((SSres/(n-p-1))/(SStot/(n-1)
    (sum((y-y_hat)**2)/(n-p-1))/(sum((y-mean(y))**2)/(n-1))
}

# IC 
logLik(fit) 
# logLik <- sum(dnorm(df$target, fit$fitted.values, summary(fit)$sigma,log = TRUE))

# AIC 
AIC(fit, k=2)
# Or
AIC <-(-2*as.numeric(logLik(fit)))+(2*(1+length(fit$coef)))

#BIC 
AIC(fit, k=log(n))
# Or
BIC(fit)
# Or
BIC <-(-2*as.numeric(logLik(fit)))+(2*(1+length(fit$coef)))

# RMSE_LOOCV
calc_loocv_rmse <- function(model) {
    sqrt(mean((resid(model) / (1 - hatvalues(model)))**2))
}

# Model Selection
# FS - AIC
step(object = null, 
     scope=list(lower=null, upper=full), 
     direction="forward", k=2, trace=1)
# FS - BIC
step(object = null, scope=list(lower=null, upper=full), 
     direction="forward", k=log(n), trace=1)
# BS - AIC
step(object = full, scope=list(lower=null, upper=full), 
     direction="backward", k=2, trace=1)
# BS - BIC
step(object = full, scope=list(lower=null, upper=full), 
     direction="backward", k=log(n), trace=1)
# Step - AIC
step(object = intermediate, 
     scope = list(lower = null, upper=full),
     direction="both", trace=1, k=2)
# Step - BIC
step(object = intermediate, 
     scope = list(lower = null, upper=full),
     direction="both", trace=1, k=log(n))


# ----------------- Model Checking  ----------------

# Assumptions - L.I.N.E

# Residuals vs Predictors
plot(target ~ predictor, data = dataset, xlab=xlabel, ylab=ylabel, main=title)
abline(beta0_hat, beta1_hat, col = 2, lwd = 1.4)

whichCols <- names(fit$coefficients)[-1]
if(any(whichCols == "predictor")) whichCols[whichCols == "predictor"] <- "predictor"
par(mfrow=c(2, ceiling(length(whichCols)/2)), pch=16)
for(j in seq_along(whichCols)){
    plot(df[,whichCols[j]], residuals(fit)) 
    abline(h=0,lty=2)
}

# QQplot
qqnorm(resid(fit))
qqline(resid(fit))

# Homoskedasticity
# 1) Residuals vs Fitted
plot(fitted(fit), resid(fit))
abline(h=0)
# 2) Scale-Location
# 3) Residuals vs Leverage

# ALL IN ONE
plot(fit)

# -----------------Categorical Predictors and Interactions ----------------

# Categorical Predictors

# Interactions

# ---------------- Transformations -----------------

# Transformations

# Predictor Transformation
linear_fit <- lm(formula = target ~ predictor + I(predictor-10), data = df)
nonlinear_fit <- lm(formula = target ~ predictor + log(predictor), data = df)

# Logaritmo 

# Trasformazione di Box-Cox 
# Da usare se y|X risulta non-normale 
MASS::boxcox (y~x, data = df, lambda = seq(-0.5,0.5, by=0.05))
# Choose lambda
dataset$boxcoxtarget <- (dataset$target^lmbda)/lambda
hist(residuals(lm(boxcoxtarget ~ predictor, data = dataset)))
#bctransf$x[which.max(bctransf$y)]
# Box-Cox transform
boxcox(fit, plotit = TRUE)

# Polynomials
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

# y vs predictors
par(mfrow= c(3,4))
for(j in 2:13){
    plot(bodyfat[,-which(names(bodyfat) %in% c("Density", "Abdomen"))][,c(j,1)])
    title(main = paste("betahat is", signif(coef(fit_all)[j],3)))
} 

#Check Correlation
signif(cor(df),4) 

# Check Covariance
signif(cov(df),4) 

# Errore 
se1 <- predict(fit_better, newdata = data.frame(p1 = 110, p2 = 90), 
               se.fit = TRUE)$se.fit
se2 <- predict(fit_worse, newdata = data.frame(p1 = 110, p2 = 90), 
               se.fit = TRUE)$se.fit

# 3) Controllo VIF = 1/1-R^2

# VIF for a beta_j
# fit new model:
# formula = focus_predictor ~ [same predictors as before, without the focus predictor]
fit <- lm(target ~ b1 + b2 + b3 + b4 + b5, data=dataset) # Original model
fit_b1 <- lm(b1 ~ b2 + b3 + b4 + b5, data=dataset) # Model for vif_b1
vif_b1 <- 1/(1-summary(fit_b1)$r.squared) 
# Or
# solve(t(X) %*% X))[2] -> for b1 
# solve(t(X) %*% X))[1] -> b0 = cbind(rep(1, n))
vif_b1<- diag(solve(t(X) %*% X))[2]*(sum((data$b1-mean(data$b1))^2))

# For all beta_j
car::vif(fit) 
sort(car::vif(fit_all))

# Controllo assunzioni.
plot(fit)

# --------------- Influence ------------------

# cutoff value
dataset[dataset$predictor > cutoff,]

# 2) Plot Y vs X, see how the regression changes and identifiy outliers
plot(bodyfat[,c("predictor","target")])
points(dataset[dataset$predictor > cutoff ,
               c("predictor","target")], col = 2, pch = 16)
abline(coef(lm(target~predictor, data = dataset)), 
       col = 2)# Fit with outliers
abline(coef(lm(target~predictor, data = dataset, 
               subset = dataset$predictor < cutoff))) # Exclude outliers

# 3)Leverage(fit) vs X
hatvalues(fit) ##  leverages - punti di leva
# H <- diag(X %*% solve(crossprod(X)) %*% t(X))

plot(dataset$predictor, hatvalues(lm(target~predictor, data = dataset)))
# Average Leverage = p+1/n 

influence(fit)
# 6) Look at the residuals
# Standardized Residuals 
rstandard(fit)
# std_r_i = e_i/(est_sigma * sqrt(1-H_ii))
#
residuals(fit)/(summary(fit)$sigma*sqrt(1-hatvalues(fit)))
# Studentized Residuals
rstudent(fit)
# stud_r_i = (r_i**2)/(n-p-1) ~ beta(1/2, 1/2(n-p-2)) ~ Norma as n->inf
#
residuals(fit)/(influence(fit)$sigma * sqrt(1-hatvalues(fit)))
# Or
rstandard(fit)*sqrt((fit$df.residual-1)/((fit$df.residual)-rstandard(fit)^2))

# Externally Studentized Residuals ~ t_{n-p-2}
t_i <- std_r_i * sqrt((n-p-2)/(n-p-1-(std_r_i**2)))

# 7) Cook's distance 
#
cooks.distance(fit) # outliers/punti particolari 
# d_i <- (1/p+1)*(exp(1)**2)*(hatvalues(fit)/((1-hatvalues(fit))**2))
# (resid(fit_mul)^2)*(1/fit_mul$rank)*(hatvalues(fit_mul)/((1-hatvalues(fit_mul))^2))*(1/summary(fit_mul)$sigma^2)

# i is the index of the problematic value
ei <- bodyfat[i,"Pct.BF"] - 
    predict(lm(formula, data = dataset, subset = -i), newdata = bodyfat[i,])
# (residuals(fit_mul)[i]^2*hatvalues(fit_mul)[i]/((1-hatvalues(fit_mul)[i])^2))*(1/length(coef(fit_mul)))
plot(cooks.distance(fit))
# Or
plot(fit, 4)

# --------------- GLM ------------------

## Analysis of Deviance Table/ LRT Test 
summary(fit)$deviance
summary(fit)$null.dev
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
    
# Cross-Validation
plot(fit)

## Residuals
# type = c("deviance", "pearson", "response"))
# Deviance
residuals(fit) ## default deviance residuals 
# Pearson
residuals(fit, type = "pearson") 
residuals(fit, type = "response")
# Working residuals
fit$residuals
# 
# Deviance residuals vs Bj 
plot(residuals(fit)~ data$predictor, ylab="Deviance residuals")
# Dev Res vs mu_hat
plot(residuals(fit)~ predict(fit,type="response"), 
     xlab=expression(hat(mu)), ylab="Deviance residuals")
# Dev Res vs eta_hat 
plot(residuals(fit)~ predict(fit,type="link"), 
     xlab=expression(hat(eta)), ylab="Deviance residuals")
# Response residuals vs eta_hat 
plot(residuals(fit ,type="response")~predict(fit,type="link"), 
     xlab=expression(hat(eta)),ylab="Response residuals")
# Working residuals vs linear predictor

# per un oggetto glm predict non può costruire intervalli di confidenza 
# (e non si possono costruire intervalli di predizione)
# con opzione se.fit si ottiene lo standard error per il predittore lineare 

nd <- data.frame(Assets = c(2000,20000))
preds <- predict(fit1, newdata = nd, type = "link", se.fit = TRUE)
# LOWER BOUNDS 
cbind(binomial()$linkinv(preds$fit + qnorm(0.01) * preds$se.fit),
      # UPPER BOUNDS 
      binomial()$linkinv(preds$fit + qnorm(0.99) * preds$se.fit))
nd <- data.frame(Assets = seq(0,60000))
preds <- predict(fit1, newdata = nd, type = "link", se.fit = TRUE)
plot(dex1$Assets, jitter(as.numeric(dex1$Banks == "Bank"),amount = 0.05), 
     ylab = "P(Azienda = Bank)" ,pch = 16)
lines(nd$Assets, binomial()$linkinv(preds$fit), col = 2)
lines(nd$Assets, binomial()$linkinv(preds$fit + qnorm(0.01) * preds$se.fit), col = 2, lty = 2)
lines(nd$Assets, binomial()$linkinv(preds$fit + qnorm(0.99) * preds$se.fit), col = 2, lty = 2)

# --------- GLM Poisson -------------

### Hypothesis Testing

### Stima puntuale 
# Predittore Lineare 
preds_link <- predict(fit, newdata = nd, type = "link")
# Risposta 
preds_resp <- predict(fit, newdata = nd, type = "response")
# Or
exp(preds_link)

# Confidence Interval
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

# Hypothesis Testing 
# 1. Compute TS
TS <- (beta_j - hyp)/se_beta_j
# 2. Check p-value
2*pnorm(abs(TS), lower.tail = FALSE)

# Confidence Interval Bj
confint.default(fit, parm="predictor", level= 1-alpha/2)
# Or
est_bj + qnorm(c(0.0+(alpha/2), 1-(alpha/2))) * se_bj

# Stima puntuale 

# Predittore Lineare
preds_link <- predict(fit, newdata = nd, type = "link") 

# Risposta
preds_resp <- predict(fit, newdata = nd, type = "response") 
# Or
exp(preds_link)/(1+exp(preds_link))
# Odd ratio = preds_resp/(1-preds_resp)

# Confidence Interval
# con opzione se.fit si ottiene lo standard error per il predittore lineare 
preds <- predict(fit, newdata = nd, type = "link", se.fit = TRUE)
pint <- cbind(fit$family$linkinv(preds$fit + qnorm(alpha/2) * preds$se.fit),
              fit$family$linkinv(preds$fit + qnorm(1-(alpha/2)) * preds$se.fit))

preds <- predict(fit, newdata = nd, type = "link", se.fit = TRUE)
cbind(
    binomial()$linkinv(preds$fit + qnorm(alpha/2)*preds$se.fit),
      binomial()$linkinv(preds$fit + qnorm(1-(alpha/2))*preds$se.fit))

# --------- GLM Binomial -------------

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

# Confidence Interval
predict(fit, newdata = nd, type = "link")

predict(fit, newdata = nd, type = "response")


# --------------- Classifier ------------------

# GLM Classifier - Logistic Regression

# 1. Train-Test Split
set.seed(42)
spam_idx <- sample(nrow(spam), 2000)
spam_trn <- spam[spam_idx,] 
spam_tst <- spam[-spam_idx,]
## 2. success and cutoff 
ifelse(p > cutoff, 1, 0)
## 3. Define measure of error

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
# 1. Cross validation on the actual mis-classification rate
cost_function <- function(y, yhat){
    # misclassification rate on cutoff
    mean((y != (yhat>0.5)))
} 
# 2. CV 
boot::cv.glm(data=df, model= fit, K = k, cost = cost_function)


# This uses the error calculated as (y - p(y=1)) 
# error <- error + mean((observed - fitted)^2)/K 
set.seed(1)
boot::cv.glm(data=data_train, model= fit, K = k)


## Metrics
predicted <- ifelse(predict(fit, 
                            newdata=dataset_tst, 
                            type="response") > cutoff, 1, 0)
actual <- dataset_tst$target

# Misclassification
get_misclassification <- function(predicted, dataset, target){
    mean(predicted != dataset_tst$target)
}

# Confusion Matrix
make_conf_mat <- function(predicted, actual) {
    table(predicted = predicted, actual = actual)
}
# Or
table(as.numeric(predict(fitTot, type = "response") > 0.5),dex2$fail)

# Sensitivity 
# TPR = Sens = TP/P = TP/(TP+FN) = 1-FNR

# Note that this function is good for illustrative purposes, 
# but is easily broken. (Think about what happens if there are 
# no "positives" predicted.)

# Specificity
# TNR = Spec = TN/N = TN/(TN+FP) = 1-FPR

### Accuracy
# Acc = TP+TN/TP + TN + FP + FN = 1 - MISC
mean(spam_tst_pred == spam_tst$type)
mean(as.numeric(predict(fitTot, type = "response") > 0.5) == dex2$fail)

### Misclassification rate on Conf Matrix
# Misc = FP+FN/TP + TN + FP + FN = 1 - ACC
mean(as.numeric(predict(fitTot, type = "response") > 0.5) != dex2$fail)

# Prevalence
# Prev = P/ #Observations = TP + FN/ #Observations
