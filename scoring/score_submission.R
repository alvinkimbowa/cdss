## Clear workspace
rm(list=ls())
#source("get_sepsis_score.R")
source("get_sepsis_score_lr.R")
#source("get_sepsis_score_gbm.R")

library('data.table')
library('foreach')
library('doParallel')
n.cores <- parallel::detectCores() - 1
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
)
doParallel::registerDoParallel(cl = my.cluster)

source('evaluate_sepsis_score.R')
## Doesn't seem to help much if at all ...
library('compiler')
get_sepsis_score <- cmpfun(get_sepsis_score)
evaluate_sepsis_score <- cmpfun(evaluate_sepsis_score)

## Load all the data so we can quickly combine it and explore it. 
load(file.path("..","Data","training_2024-11-04.RData"))
SEPSISdat<-setDT(sepsis_data)
load(file.path("..","Data","testing_2024-11-04.RData"))
nrow(SEPSISdat) # should be n=200112
SEPSISdat_test<-setDT(sepsis_data)
if (file.exists(file.path("..","Data","evaluation_2024-11-04.RData"))){
  load(file.path("..","Data","evaluation_2024-11-04.RData"))
  nrow(SEPSISdat) # should be n=200112
  SEPSISdat_eval<-setDT(sepsis_data)
}
rm("sepsis_data")

# Load model
model <- load_sepsis_model()

## Obtain the labels in a causal way
score_cohort <- function(cohort){
  patients <- unique(cohort$patient)
  num_patients <- length(patients)
  ct <- 1
  cohort$PredictedProbability<-0
  cohort$PredictedLabel<-F
  starttime = Sys.time()
  # Loop over patients
  for (patno in 1:num_patients){
    sub_cohort <- cohort[cohort$patient==patients[patno],2:(ncol(cohort)-1)]
    num_rows <- nrow(sub_cohort)
    # Make predictions.
    predictions <- foreach (t = 1:num_rows, .combine = 'rbind', .export=c("get_sepsis_score","model")) %dopar% {
    #predictions <- foreach (t = 1:num_rows, .combine = 'rbind', .export=c("get_sepsis_score","model")) %do% {
      current_data <- sub_cohort[1:t,]
      get_sepsis_score(current_data, model)
    }
    cohort$PredictedProbability[ct:(ct+num_rows-1)]<-predictions[,1]
    cohort$PredictedLabel[ct:(ct+num_rows-1)]<-predictions[,2]
    ct <- ct + num_rows
  }
  elapsed = round(Sys.time() - starttime)
  #Calculate the utility score
  util <- evaluate_sepsis_score(cohort)[5]
  return(c(util, elapsed))
}

train_res<-score_cohort(SEPSISdat)
print(train_res)
test_res<-score_cohort(SEPSISdat_test)
print(test_res)
if (file.exists(file.path("..","Data","evaluation_2024-11-04.RData"))){
  eval_res<-score_cohort(SEPSISdat_scoring)
  print(eval_res)
}
#
stopCluster(cl=my.cluster)
