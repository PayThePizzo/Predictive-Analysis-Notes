
# Esercizio 1 
Due modelli lineari semplici vengono stimati: nel modello A y1 è la
variabile risposta e x1 è la variabile esplicativa, mentre nel modello B
y2 è la variabile risposta e x2 è la variabile esplicativa. Le stime di
intercetta e coefficiente angolare del modello A sono:

```r
x1	            y1	                x2	        y2
Min. :2.05	    Min. : 4.62	    Min. :1.17	    Min. : 3.63
1st Qu.:2.45	1st Qu.: 7.59	1st Qu.:1.98	1st Qu.: 6.89
Median :2.89	Median : 8.71	Median :2.85	Median : 8.77
Mean :2.93	    Mean : 8.85	    Mean :2.93	    Mean : 8.85
3rd Qu.:3.41	3rd Qu.:10.02	3rd Qu.:3.89	3rd Qu.:10.67
Max. :3.90	    Max. :13.05	    Max. :4.87	    Max. :14.58
```

Due modelli lineari semplici vengono stimati: nel modello A y1 è la variabile risposta e x1 è la variabile esplicativa, mentre nel modello B y2 è la variabile risposta e x2 è la variabile esplicativa. Le stime di intercetta e coefficiente angolare del modello A sono:

```r
  (Intercept)          x1 
         2.94        2.02
```
Sapendo che il coefficiente angolare del modello B è $\hat{\beta_{1}} = 2.02$ si indichi il valore dell’intercetta del modello B:

## Solution

$$\hat{\beta_{0}} = \bar{y} - \hat{\beta_{1}}\bar{x} = 8.85 - 2.20 \cdot 2.93$$

```r
(intercept <- 8.85 - (2.20*2.93))

[1] 2.9314
```

---
