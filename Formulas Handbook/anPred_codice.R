## Questo file riassume alcune funzioni/elementi di codice usati frequentemente
## nel corso Analisi Predittiva 
## Il file non è esaustivo - indica solo alcune funzioni usate frequentemente 

## posizione della working directory 
getwd()
setwd("my/long/path")

##lm - modelli lineari  -------
df <- data.frame(x1 = runif(15), x2 = runif(15), y  = rnorm(15))

## stima del modello 
fit <- lm(y~x1+x2, data = df)
summary(fit) ## varie informazioni riassuntive sulla stima
coef(fit) ## valori stimati dei coefficienti del modello 
confint(fit) ## intervalli di confidenza per i coefficienti del modello 

## per aggiungere trasformazioni di X come predittori si usa la funzione I(.)
## fit <- lm(y~x1+x2+I(x2^2), data = df) 
## per polinomi 
## fit <- lm(y~x1+poly(x2,2), data = df)
## fit <- lm(y~x1+poly(x2,2, raw=TRUE), data = df)

## la matrice di disegno usata nella stima
model.matrix(fit)

## predizione 
# per i valori osservati delle X
fitted(fit)
predict(fit) ## predict produce anche intervalli di confidenza e predizione
predict(fit, interval = "confidence") 
predict(fit, interval = "prediction") 
## per un nuovo set di punti 
nd <- data.frame(x1 = c(0.2,0.8), x2 = c(0.3,0.6))
predict(fit, newdata = nd)

# residui 
residuals(fit) ## these are y - fitted(fit)
rstandard(fit) ## standardised residuals 
rstudent(fit)  ## studentized residuals 

# goodness of fit/ bontà di adattamento 
plot(fit) # grafici riassuntivi
AIC(fit, k = 2); BIC(fit); logLik(fit) ## verosimiglianza e criteri di informazione
hatvalues(fit) ##  leverages - punti di leva 
car::vif(fit) ## variance inflation factors - ci sono problemi di colinearità? 
cooks.distance(fit) # outliers/punti particolari 

## test anova per modelli annidati 
# anova(small_model, big_model)
anova(lm(y~x1, data = df), fit)


## model selection 
## la funzione step qui usata per un algoritmo forward come esempio 
## opzioni importanti 
# scope per delineare l'ambito della ricerca
# k: per definire la penalizzazione del criterio di informazione
# direction: per la direzione: forward, backward, both

step(object = lm(y~1, data = df), 
     scope = list(lower = lm(y~1, data = df), 
                  upper = fit), 
     direction = "forward",
     k = 2
) ## 


## trasformazione di Box-Cox 
## da usare se y|X risulta non-normale 
## MASS::boxcox


##glm - modelli lineari generalizzati -------

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



## glm as a classifier ------------- 

## funzioni implementate nelle slides/laboratorio 


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
