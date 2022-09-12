# predictive retention model

###################################################################################################
# The below script is MSU Denver's predictive retention ensemble model, in which three models are #
# created for designated subgroups in order to predict the likelihood of a student retaining.     #
# The script below has been abstracted in order to allow room for future model development.       #
###################################################################################################

# loading packages / disabling scientific notation for the summary stats
load_packages <- function() {
  library(caret)
  library(ranger)
  library(ggplot2)
  library(adabag)
  library(klaR)
  library(vip)
  library(tictoc)
  library(svDialogs)
  library(dplyr)
  options(scipen = 999)
}
load_packages()

# sets testing working directory and automatically loads in testing data sets if set to TRUE.
# if FALSE prompts you for location of files to load in, and sets working directory to parent folder.
# also modifies the random forest parameters to function on a small testing data set if TRUE.
DEBUG <- F
if (DEBUG) {
  setwd("C:/Users/Laserbeams/Desktop/predictive_model_testing")
  RF_MTRY <- 1
} else {
  setwd("C:/Users/Laserbeams/Desktop/results")
  RF_MTRY <- 3
}

# creating functions

get_data <- function() {
  # importing calibration and prediction data
  if (DEBUG) {
    calib_data <- read.csv("./calibration_testing_data.csv", header = T, stringsAsFactors = T)
    pred_data <- read.csv("./prediction_testing_data.csv", header = T, stringsAsFactors = T)
  } else {
    calib_data <- read.csv(file.choose(), header = T, stringsAsFactors = T)
    pred_data <- read.csv(file.choose(), header = T, stringsAsFactors = T)
  }
  # what to do with treatment population
  intervention_choice <- dlgInput("Do you want to exclude the intervention population (Y or N)?")$res
  if (intervention_choice == 'Y') {
    calib_data <- calib_data %>% 
      subset(INTERVENTION == 'N') %>%
      subset(select = -c(INTERVENTION))
  }
  # option to trim the calibration data
  trim_choice <- dlgInput("At which point in time (i.e. term code) do you want to start the calibration data? (Select cancel if you want all data included.)")$res
  if (length(trim_choice) > 0) {
    calib_data <- calib_data %>% 
      subset(TERM >= as.numeric(trim_choice))
  }
  # formatting calibration / predictive data
  calib_data$MIN_REGIST_YEAR <- as.factor(calib_data$MIN_REGIST_YEAR)
  calib_data$MIN_REGIST_FALLSPRSUM <- as.factor(calib_data$MIN_REGIST_FALLSPRSUM)
  calib_data$CIP_2DIG <- as.factor(calib_data$CIP_2DIG)
  pred_data$MIN_REGIST_YEAR <- as.factor(pred_data$MIN_REGIST_YEAR)
  pred_data$MIN_REGIST_FALLSPRSUM <- as.factor(pred_data$MIN_REGIST_FALLSPRSUM)
  pred_data$CIP_2DIG <- as.factor(pred_data$CIP_2DIG)
  
  return(list(calib_data = calib_data, pred_data = pred_data))
}

# this is the only function that needs to be modified, should you choose to change the groups.
# the functions simply needs to output the column name being used for grouping purposes, and a
# list of the groups themselves.
create_groups <- function(calib_data) {
  calib_data$CLUSTER <- as.factor(calib_data$CLUSTER)
  groups <- as.list(levels(calib_data$CLUSTER))
  groups_column <- "CLUSTER"
  return(list(groups = groups, groups_column = groups_column))
}

prep_data <- function(calib_data, pred_data, group, groups_column) {
  # subsetting the calibration data
  df_calib <- calib_data %>%
    subset(get(groups_column) == group)
  # subsetting the predictive data
  df_pred <- pred_data %>%
    subset(get(groups_column) == group) %>%
    subset(select = -c(MIN_REGIST_YEAR, get(groups_column), PIDM, TERM))
  # splitting into testing and training
  indxTrain <- createDataPartition(y = df_calib$RETAINED, p = 0.7, list = FALSE)    
  training <- df_calib[indxTrain, ] %>%
    subset(select = -c(MIN_REGIST_YEAR, get(groups_column), PIDM, TERM))
  testing_all <- df_calib[-indxTrain, ]
  testing <- df_calib[-indxTrain, ] %>%
    subset(select = -c(MIN_REGIST_YEAR, get(groups_column), PIDM, TERM))
  x = training %>% subset(select = -RETAINED)
  y = training$RETAINED
  return(list(df_calib = df_calib, df_pred = df_pred, training = training, testing = testing, testing_all = testing_all, x = x, y = y))
}

nb_func <- function(x, y, training, testing, group, df_pred) {
  # training and testing
  nb <- train(x, y, method = 'nb', laplace = 1, na.action = na.pass, trControl = trainControl(method = 'cv'))
  nb_result <- predict(nb, newdata = testing, type = 'raw')
  probabilities <- predict(nb, newdata = testing, type = 'prob')
  cm_nb <- confusionMatrix(nb_result, testing$RETAINED, positive = "Y")
  # exporting variable importance graphs
  varperf <- varImp(nb, scale = F)
  png(filename = paste("nb_varperf_", group, ".png", sep = ""), width = 800, height = 600)
  print(plot(varperf, main = paste(group, " Model Variable Importance (NB)", sep = "")))
  dev.off()
  # nb predictions
  nb_binary <- predict(nb, newdata = df_pred, type ='raw')
  nb_prob <- predict(nb, newdata = df_pred, type ='prob')
  nb_list <- list(cm_nb = cm_nb, nb_binary = nb_binary, nb_prob = nb_prob,
                  nb_result = nb_result, probabilities = probabilities)
  print(noquote("Naive Bayes done!"))
  return(nb_list)
}

rf_func <- function(training, testing, group, df_pred) {
  # training and testing
  rf <- ranger(RETAINED ~ ., data = training, write.forest = T, importance = "permutation", num.trees = 2000, mtry = RF_MTRY)
  rf_result <- predict(rf, data = testing)
  cm_rf <- confusionMatrix(rf_result$predictions, testing$RETAINED, positive = "Y")
  # exporting variable importance graphs
  png(filename = paste("rf_varperf_", group, ".png", sep = ""), width = 800, height = 600)
  print(vip(rf) + ggtitle(paste(group, " Variable Importance (RF)", sep = "")))
  dev.off()
  # rf predictions
  rf_binary <- predict(rf, data = df_pred)
  rf_list <- list(cm_rf = cm_rf, rf_binary = rf_binary,
                  rf_result = rf_result)
  print(noquote("Random Forest done!"))
  return(rf_list)
}

ad_func <- function(training, testing, group, df_pred, adabag_threshold) {
  # training and testing
  adabag <- boosting(RETAINED ~ ., data = training, mfinal = 200, boos = T)
  adabag_result <- predict.boosting(adabag, newdata = testing)
  adabag_result$class <- as.factor(adabag_result$class)
  cm_adabag <- confusionMatrix(adabag_result$class, testing$RETAINED, positive = "Y")
  adabag_mod <- as.factor(ifelse(adabag_result$prob[, 2] > adabag_threshold, "Y", "N"))
  cm_adabag_mod <- confusionMatrix(adabag_mod, testing$RETAINED, positive = "Y")
  # exporting variable importance graphs
  ad_imp <- as.data.frame(adabag$importance[order(adabag$importance, decreasing = F)])
  colnames(ad_imp) <- "importance"
  png(filename = paste("ad_varperf_", group, ".png", sep = ""), width = 800, height = 600)
  print(ggplot(data = ad_imp, aes(x = importance, y = reorder(rownames(ad_imp), importance))) +
          geom_bar(stat = "identity") +
          labs(title = paste(group, " Variable Importance (AD)", sep = ""), y = "variable"))
  dev.off()
  # adabag predictions
  ad_binary <- predict.boosting(adabag, newdata = df_pred)
  ad_list <- list(cm_adabag = cm_adabag, cm_adabag_mod = cm_adabag_mod, ad_binary = ad_binary,
                  adabag_result = adabag_result)
  print(noquote("Adabag done!"))
  return(ad_list)
}

export_stats <- function(conf_matrices, file_name) {
  # creates an empty data frame to append summary stats on to
  summary_stats <- data.frame(matrix(NA, nrow = 22, ncol = 1))
  rownames(summary_stats) <- c("Accuracy", "Kappa", "AccuracyLower", "AccuracyUpper", "AccuracyNull",
                               "AccuracyPValue", "McnemarPValue", "Sensitivity", "Specificity",
                               "Pos Pred Value", "Neg Pred Value", "Precision", "Recall", "F1",
                               "Prevalence", "Detection Rate", "Detection Prevalence", "Balanced Accuracy",
                               "true_pos", "false_neg", "false_pos", "true_neg")
  colnames(summary_stats) <- "remove"
  # appending summary stats on from each confusion matrix
  for (cm_label in names(conf_matrices)) {
    stats <- rbind(data.frame(V1 = conf_matrices[[cm_label]]$overall),
                   data.frame(V1 = conf_matrices[[cm_label]]$byClass),
                   data.frame(V1 = t(data.frame(true_pos = conf_matrices[[cm_label]]$table[1], 
                                                false_neg = conf_matrices[[cm_label]]$table[2], 
                                                false_pos = conf_matrices[[cm_label]]$table[3], 
                                                true_neg = conf_matrices[[cm_label]]$table[4]))))
    colnames(stats) <- cm_label
    summary_stats <- cbind(summary_stats, stats)
  }
  # exporting final summary stats csv
  summary_stats %>%
    subset(select = -c(remove)) %>%
    round(6) %>%
    write.csv(file = paste("summary_stats", file_name, sep = "_"), row.names = T) 
}

export_predictions <- function(predictions, pred_data, dr_threshold, file_name) {
  # creating empty data frame for the final data set
  final_predictions <- setNames(data.frame(matrix(ncol = ncol(pred_data) + 7, nrow = 0)),
                                c(colnames(pred_data), "nb_binary", "nb_prob$Y", "rf_binary$predictions", 
                                  "ad_binary$class", "dr1", "dr2", "dr3"))
  for (group in names(predictions)) {
    # decision rules
    dr1 <- as.factor(ifelse(as.numeric(predictions[[group]]$nb_prob$Y > dr_threshold) & (predictions[[group]]$rf_binary$predictions == "Y"), "Y","N"))
    dr2 <- as.factor(ifelse(as.numeric(predictions[[group]]$nb_prob$Y > dr_threshold) | (predictions[[group]]$rf_binary$predictions == "Y"), "Y","N"))
    dr3 <- as.factor(ifelse(as.numeric(predictions[[group]]$nb_prob$Y > dr_threshold) & (predictions[[group]]$rf_binary$predictions == "Y") & 
                              (predictions[[group]]$ad_binary$class == "Y"), "Y","N"))
    # recompiling original feeder data + predictions, and adding to final data frame
    df_pred_subset <- subset(pred_data, STUDENT_CLASSIF == group)
    recombo <- cbind(df_pred_subset, 
                     nb_binary = predictions[[group]]$nb_binary, 
                     nb_prob = predictions[[group]]$nb_prob$Y, 
                     rf_binary = predictions[[group]]$rf_binary$predictions, 
                     ad_binary = predictions[[group]]$ad_binary$class, 
                     dr1, dr2, dr3)
    final_predictions <- rbind(final_predictions, recombo)
  }
  # exporting final predictions
  write.csv(final_predictions, file = paste("final_predictions", file_name, sep = "_"), row.names = F)
}

testing_predictions <- function(testing_all, dr_threshold, nb_out, rf_out, ad_out, group) {
    # decision rules
    dr1 <- as.factor(ifelse(as.numeric(nb_out$probabilities$Y > dr_threshold) & (rf_out$rf_result$predictions == "Y"), "Y","N"))
    dr2 <- as.factor(ifelse(as.numeric(nb_out$probabilities$Y > dr_threshold) | (rf_out$rf_result$predictions == "Y"), "Y","N"))
    dr3 <- as.factor(ifelse(as.numeric(nb_out$probabilities$Y > dr_threshold) & (rf_out$rf_result$predictions == "Y") & 
                                      (ad_out$adabag_result$class == "Y"), "Y","N"))
    # recompiling original feeder data + predictions, and adding to final data frame
    df_testing_subset <- subset(testing_all, STUDENT_CLASSIF == group)
    recombo <- cbind(df_testing_subset, 
                     nb_binary = nb_out$nb_result, 
                     nb_prob = nb_out$probabilities, 
                     rf_binary = rf_out$rf_result$predictions, 
                     ad_binary = ad_out$adabag_result$class, 
                     dr1, dr2, dr3)
    recombo_list <- list(recombo = recombo)
    return(recombo_list)
}

# final predictive function

main <- function() {
  tic("[1] Total run time")
  
  # loading in data
  working_data <- get_data()
  calib_data <- working_data$calib_data
  pred_data <- working_data$pred_data
  
  # creating our model subsets
  groups_data <- create_groups(calib_data)
  groups <- groups_data$groups
  groups_column <- groups_data$groups_column
  
  # setting the decision rule threshold, adabag threshold, and export file names
  dr_threshold <- 0.65
  adabag_threshold <- 0.6
  file_name <- sprintf("%s_%s.csv", unique(pred_data$TERM), format(Sys.Date(), "%m%d%y"))
  
  # creating empty list for confusion matrices, and prediction results
  conf_matrices <- list()
  predictions <- list()
  # creating empty data frame for the final testing predictions
  final_testing_predictions <- setNames(data.frame(matrix(ncol = ncol(pred_data) + 7, nrow = 0)),
                                        c(colnames(pred_data), "nb_binary", "nb_prob$Y", "rf_binary$predictions", 
                                          "ad_binary$class", "dr1", "dr2", "dr3"))
  
  # running the models
  for (group in groups) {
    print(noquote(paste("Running models on group:", group, sep = " ")))
    # bringing in the subsetted data (sd)
    sd <- prep_data(calib_data, pred_data, group, groups_column)
    testing_all <- sd$testing_all
    # running models
    tic(sprintf("[1] %s group run time", group))
    nb_out <- nb_func(sd$x, sd$y, sd$training, sd$testing, group, sd$df_pred)
    rf_out <- rf_func(sd$training, sd$testing, group, sd$df_pred)
    ad_out <- ad_func(sd$training, sd$testing, group, sd$df_pred, adabag_threshold)
    toc()
    # compiling summary stats
    temp_list <- list(nb_out$cm_nb, rf_out$cm_rf, ad_out$cm_adabag, ad_out$cm_adabag_mod)
    names(temp_list) <- c(paste("nb", group, sep = "_"),
                          paste("rf", group, sep = "_"),
                          paste("ad", group, sep = "_"),
                          paste("ad", adabag_threshold, group, sep = "_"))
    conf_matrices <- append(conf_matrices, temp_list)
    # compiling testing prediction results
    testing_results <- testing_predictions(testing_all, dr_threshold, nb_out, rf_out, ad_out, group)
    final_testing_predictions <- rbind(final_testing_predictions, testing_results$recombo)
    # compiling prediction results
    temp_list <- list(list(nb_prob = nb_out$nb_prob, 
                           nb_binary = nb_out$nb_binary, 
                           rf_binary = rf_out$rf_binary, 
                           ad_binary = ad_out$ad_binary))
    names(temp_list) <- group
    predictions <- append(predictions, temp_list)
    print(noquote(""))
  }
  print(noquote("All models and all groups complete!"))
  
  # exporting final csvs
  export_stats(conf_matrices, file_name)
  export_predictions(predictions, pred_data, dr_threshold, file_name)
  # exporting final predictions
  write.csv(final_testing_predictions, file = paste("final_testing_predictions", file_name, sep = "_"), row.names = F)
  
  print(noquote("ALL DONE BITCHES"))
  toc()
}

main() # load calibration first, then prediction
