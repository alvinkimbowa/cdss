#!/usr/bin/Rscript

## And now continue as before.
get_sepsis_score = function(data, myModel){
    # Get the current row we are predicting
    x <- data[nrow(data),1:23]
    # Make the z-score
    x_norm <- (x - myModel$x_mean) / myModel$x_std
    # Missing is normal
    x_norm[is.na(x_norm)] <- 0
    x_norm<-cbind(x_norm,data[nrow(data),24])
    # Add both normalized and raw values and sex
    #x[is.na(x)]<-myModel$x_mean[is.na(x)]
    #x_norm<-cbind(x,data[nrow(data),24]),x_norm)
    
    score <- predict(myModel$bst,as.matrix(x_norm))
    score <- min(max(score,0),1)
    label <- score > myModel$thresh
  return(c(score,label))
}

load_sepsis_model <- function(){
  library('lightgbm')
  myModel<-list(x_mean = c(22.48, 83.57, 97.24, 36.92, 123.13, 81.84, 62.91, 
                           18.25, -0.46, 0.5, 7.39, 40.32, 20.48, 7.82, 1.34, 131.07, 2.04, 
                           4.14, 31.47, 10.58, 11.46, 196.31, 62.25), 
                x_std = c(15.22, 16.51, 2.88, 0.71, 22.19, 15.71, 13.51, 4.95, 3.39, 0.18, 
                          0.06, 7.28, 16.33, 2.13, 1.57, 42.49, 0.37, 0.56, 5.34, 1.87, 
                          5.37, 93.23, 15.82), 
                thresh = c(threshold = 0.031)
  )
  myModel$bst<-lgb.load("lightgbm.model")
  return(myModel)
}
