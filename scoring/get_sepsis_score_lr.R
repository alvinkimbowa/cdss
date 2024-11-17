#!/usr/bin/Rscript

## And now continue as before.
get_sepsis_score = function(data, myModel){
    # Get the current row we are predicting
    x <- tail(data, n=1)[,1:23]
    x <- (x - myModel$x_mean) / myModel$x_std
    # Missing is normal
    x[is.na(x)] <- 0

    score <- plogis(myModel$const + sum(x * myModel$coeffs))
    score <- min(max(score,0),1)
    label <- score > myModel$thresh
  return(c(score,label))
}

load_sepsis_model <- function(){
  myModel<-list(x_mean = c(22.48, 83.57, 97.24, 36.92, 123.13, 81.84, 62.91, 
                    18.25, -0.46, 0.5, 7.39, 40.32, 20.48, 7.82, 1.34, 131.07, 2.04, 
                    4.14, 31.47, 10.58, 11.46, 196.31, 62.25), 
         x_std = c(15.22, 16.51, 2.88, 0.71, 22.19, 15.71, 13.51, 4.95, 3.39, 0.18, 
                   0.06, 7.28, 16.33, 2.13, 1.57, 42.49, 0.37, 0.56, 5.34, 1.87, 
                   5.37, 93.23, 15.82), 
         const = c(`(Intercept)` = -4.04364), 
         coeffs = c(0.19145, 0.1191, -0.00313, 0.34322, 0.0794, -0.19759, -0.00878, 0.08566,
                    0.02238, 0.23235, -0.00197, 0.00382, 0.14659, -0.04752, 0.07321, 0.07838,
                    -0.04307, -0.16909, -0.10443, 0.09631, 0.09385, 0.07281,-0.03671), 
         thresh = c(threshold = 0.029)
  )
  return(myModel)
}
