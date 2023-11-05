rm(list=ls()); cat("\014") # to clear all
if(!is.null(dev.list())) dev.off()

library(Metrics) # conf_mat measures like f1, recall, precision
library(randomForest) # rf
library(pROC) # roc
library(Boruta) # boruta
library(e1071) # svm 
library(caret) # feature selection and model tuning 
library(rsample) # splitting data into train and test 
library(rpart) # building decision trees
library(ROCR) # rod 
library(sampling) # sampling
library(stats) # chisq.test, lm
library(Hmisc) # describe()
library(caTools) # splitting test and train data 
library(MASS) # glm.nb
library(tidyverse) # pipe operator
library(mltools)
library(knitr)

setwd("/Users/adit0106/Desktop/Boston University/SEM 2/MET CS699 Data Mining/Project")
df <- read.csv("Makaan_Properties_Buy.csv", header = TRUE)
head(df, 10)

# Taking a subset of the data
df <- df %>% sample_n(100000)
#df <- df %>% sample_n(1000)
head(df)
# Check the structure of the data
str(df)
dim(df)
# Drop the columns which are not required for our analysis
df <- df %>% dplyr::select(-c("Property_Name",
                              "Property_id",
                              "Posted_On", 
                              "Project_URL", 
                              "City_id",
                              "builder_id",
                              "Builder_name",
                              "Locality_ID", 
                              "Locality_Name", 
                              "Sub_urban_ID", 
                              "Sub_urban_name",
                              "description",
                              "listing_domain_score", 
                              "is_plot", 
                              "is_Apartment", 
                              "is_PentaHouse", 
                              "is_studio", 
                              "Listing_Category",
                              "is_commercial_Listing"))
str(df)
dim(df)

# Remove duplicates
df <- df[!duplicated(df),]

# Identify missing values and remove them 
na_count <- sum(is.na(df)); na_count
df <- na.omit(df); nrow(df)

prop.table(table(df$Property_type))

dim(df)

#################################################### EDA #######################################################

barplot(table(df$Property_type) , main="Property Type Distribution (Before Merging)",
        xlab = "Type", ylab = "Frequency")

# Create horizontal barplot
barplot(table(df$City_name), horiz = TRUE, names.arg = names(df$City_name), xlab = "Count", ylab = "Category",main="Property Distribution based on City")
axis(1, labels = names(df$City_name), las = 1)


# Pie Plot (is_furnished)
data <- table(df$is_furnished)
slice.labels <- names(table(df$is_furnished))
slice.percents <- round(data/sum(data)*100)
slice.labels <- paste(slice.labels, slice.percents)
slice.labels <- paste(slice.labels, "%", sep="")
slice.labels
pie(data, labels = slice.labels, main="Furnishing")


# Pie Plot (Property_status)
data <- table(df$Property_status)
slice.labels <- names(table(df$Property_status))
slice.percents <- round(data/sum(data)*100)
slice.labels <- paste(slice.labels, slice.percents)
slice.labels <- paste(slice.labels, "%", sep="")
slice.labels
pie(data, labels = slice.labels, main="Property Status")

# Pie Plot (is_ready_to_move)
data <- table(df$is_ready_to_move)
slice.labels <- names(table(df$is_ready_to_move))
slice.percents <- round(data/sum(data)*100)
slice.labels <- paste(slice.labels, slice.percents)
slice.labels <- paste(slice.labels, "%", sep="")
slice.labels
pie(data, labels = slice.labels, main="Ready to Move")

# Pie Plot (is_RERA_registered)
data <- table(df$is_RERA_registered)
slice.labels <- names(table(df$is_RERA_registered))
slice.percents <- round(data/sum(data)*100)
slice.labels <- paste(slice.labels, slice.percents)
slice.labels <- paste(slice.labels, "%", sep="")
slice.labels
pie(data, labels = slice.labels, main="RERA Registered")

###########################################################################

###### DATA PRE-PROCESSING #######
head(df)
df$Property_type = ifelse(df$Property_type %in% c("Independent Floor","Independent House", "Residential Plot", "Villa"), "Non-Apartment", "Apartment")
table(df$Property_type)
prop.table(table(df$Property_type))
dim(df)

barplot(table(df$Property_type) , main="Property Type Distribution (After Merging)",
        xlab = "Type", ylab = "Frequency")

# data types
str(df)
df$Price_per_unit_area <- as.numeric(gsub(",","",df$Price_per_unit_area))
new <- gsub(" sq ft","",df$Size)
df$Size <- as.numeric(gsub(",","",new))
df$Price <- as.numeric(gsub(",","",df$Price))
str(df)

# Factors
df$Property_type <- factor(df$Property_type)
df$Property_status <- factor(df$Property_status)
df$City_name <- factor(df$City_name)
df$is_furnished <- factor(df$is_furnished)
df$is_RERA_registered <- factor(df$is_RERA_registered)
df$is_ready_to_move <- factor(df$is_ready_to_move)
df$Property_building_status <- factor(df$Property_building_status)
df$No_of_BHK <- factor(df$No_of_BHK)
str(df)

# custom function to implement min max scaling
minMax <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

#normalise data using custom function
df$Price_per_unit_area <- as.data.frame(lapply(df['Price_per_unit_area'], minMax))
df$Price_per_unit_area <- as.numeric(unlist(df$Price_per_unit_area))

df$Price <- as.data.frame(lapply(df['Price'], minMax))
df$Price <- as.numeric(unlist(df$Price))

df$Size <- as.data.frame(lapply(df['Size'], minMax))
df$Size <- as.numeric(unlist(df$Size))

str(df)
dim(df)

######################################## EVALUATION METRIC FUNCTION ###########################################

## Computing Classification metrics
binary_metrics <- function(y_true, y_pred) {
  # Compute confusion matrix
  cm <- table(y_true, y_pred)
  conf <- confusionMatrix(y_pred, y_true, positive = "Apartment")
  # Compute number of classes
  n_classes <- length(levels(y_true))
  
  # Initialize results vector
  results <- data.frame(class = factor(levels(y_true)))
  results$TP_rate <- numeric(n_classes)
  results$FP_rate <- numeric(n_classes)
  results$precision <- numeric(n_classes)
  results$recall <- numeric(n_classes)
  results$F_measure <- numeric(n_classes)
  results$ROC_area <- numeric(n_classes)
  results$MCC <- numeric(n_classes)
  
  # Compute metrics for each class
  for (i in 1:n_classes) {
    class <- levels(y_true)[i]
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    tn <- sum(diag(cm)) - tp
    fn <- sum(cm[i, ]) - tp
    
    # Compute TP rate
    results$TP_rate[i] <- tp / (tp + fn)
    
    # Compute FP rate
    results$FP_rate[i] <- fp / (fp + tn)
    
    # Compute precision
    results$precision[i] <- tp / (tp + fp)
    
    # Compute recall
    results$recall[i] <- tp / (tp + fn)
    
    # Compute F-measure
    results$F_measure[i] <- 2 * (results$precision[i] * results$recall[i]) / (results$precision[i] + results$recall[i])
    
    # Compute ROC area
    roc <- roc(ifelse(y_true == class, 1, 0), ifelse(y_pred == class, 1, 0))
    results$ROC_area[i] <- auc(roc)
    
    # Compute MCC
    
    results$MCC[i] <- (tp*tn - fp*fn)/sqrt(as.double(tp+fp)*as.double(tp+fn)*as.double(tn+fp)*as.double(tn+fn))
    
  }
  # # Compute weighted averages
  weighted_avg <- data.frame(class = "weighted_avg")
  weighted_avg$TP_rate <- weighted.mean(results$TP_rate, na.rm = TRUE)
  weighted_avg$FP_rate <- weighted.mean(results$FP_rate, na.rm = TRUE)
  weighted_avg$precision <- weighted.mean(results$precision, na.rm = TRUE)
  weighted_avg$recall <- weighted.mean(results$recall, na.rm = TRUE)
  weighted_avg$F_measure <- weighted.mean(results$F_measure, na.rm = TRUE)
  weighted_avg$ROC_area <- weighted.mean(results$ROC_area, na.rm = TRUE)
  weighted_avg$MCC <- weighted.mean(results$MCC, na.rm = TRUE)
  
  #Combine results and weighted averages
  results <- rbind(results, weighted_avg)
  #results <- rbind(results, weighted_avg)
  return(results)
  
} 


##################################### CLASSIFIERS (ALL FEATURES) ##########################################

set.seed(31)
sample <- sample.split(df$Property_type, SplitRatio = 0.66)
train  <- subset(df, sample == TRUE)
test   <- subset(df, sample == FALSE)

##### CLASSIFIER 1: GLM Model (logistic regression) #####

ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
log.model <- train(Property_type ~., data = train, method = "glm", trControl = ctrl,family=binomial)
log.model
#summary(log.model)

log.pred <- predict(log.model, newdata = test)
log.cm <- table(test$Property_type, log.pred)
print(log.cm)

# Confusion Matrix
log.results <- confusionMatrix(log.cm)
log.results

# Accuracy
log.acc <- log.results$overall['Accuracy']
print(log.acc)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
log.metrics <- binary_metrics(test$Property_type, log.pred)
print(log.metrics)

##### CLASSIFIER 2: Naive Bayes #####

# Define the cross-validation method
ctrl <- trainControl(method="cv", number=10, verboseIter = TRUE)

# Train the Naive Bayes model with cross-validation
n.model <- train(Property_type ~., data = train, method = "naive_bayes", trControl = ctrl)
n.model

n.pred <- predict(n.model, newdata = test)
n.cm <- table(test$Property_type, n.pred)
print(n.cm)

# Confusion Matrix
n.results <- confusionMatrix(n.cm)
n.results

# Accuracy
n.acc <- n.results$overall['Accuracy']
print(n.acc)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
n.metrics <- binary_metrics(test$Property_type, n.pred)
print(n.metrics)

##### CLASSIFIER 3: K-Nearest Neighbours #####

# Define the training control (k=5)
kgrid <- expand.grid(k = 1:15)
train_control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
# Train the KNN model
knn.model <- train(Property_type ~ ., data = train, method = "knn", trControl = train_control, tuneGrid = kgrid)
knn.model

knn.pred <- predict(knn.model, newdata = test)
knn.cm <- table(test$Property_type, knn.pred)
knn.cm

# Confusion Matrix
knn.results <- confusionMatrix(knn.cm)
knn.results

# Accuracy
knn.acc <- knn.results$overall['Accuracy']
print(knn.acc)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
knn.metrics <- binary_metrics(test$Property_type, knn.pred)
print(knn.metrics)


##### CLASSIFIER 4: Support Vector Machine #####

# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train SVM model with 10-fold CV
svm.model <- train(Property_type ~ ., data = train, method = "svmRadial", trControl = trainControl)
svm.model

svm.pred <- predict(svm.model, newdata = test)
svm.cm <- table(test$Property_type, svm.pred)
svm.cm

# Confusion Matrix
svm.results <- confusionMatrix(svm.cm)
svm.results

# Accuracy
svm.acc <- svm.results$overall['Accuracy']
print(svm.acc)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
svm.metrics <- binary_metrics(test$Property_type, svm.pred)
print(svm.metrics)


##### CLASSIFIER 5: Random Forest #####

# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train Random Forest model with 10-fold CV
rf.model <- train(Property_type ~ ., data = train, method = "rf", trControl = trainControl)
rf.model

# Print model summary
# summary(rf.model)

rf.pred <- predict(rf.model, newdata = test)
rf.cm <- table(test$Property_type, rf.pred)

# Confusion Matrix
rf.results <- confusionMatrix(rf.cm)
rf.results

# Accuracy
rf.acc <- rf.results$overall['Accuracy']
print(rf.acc)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
rf.metrics <- binary_metrics(test$Property_type, rf.pred)
print(rf.metrics)

################################ CLASSIFIERS (RANDOM FOREST FEATURE SELECTION) ###############################
# Build random forest model
rf_model <- randomForest(Property_type ~ ., data = train, importance = TRUE)
rf_model

# Compute variable importance scores
importance_scores <- importance(rf_model)

# Select variables with highest importance scores
selected_features <- rownames(importance_scores)[order(importance_scores[,1], decreasing =TRUE)][1:4]
selected_features

# Select only the chosen features
df1 <- df %>% dplyr::select(selected_features, "Property_type")

# Split data into train and test sets
set.seed(31)
sample <- sample.split(df1$Property_type, SplitRatio = 0.66)
train1  <- subset(df1, sample == TRUE)
test1   <- subset(df1, sample == FALSE)

##### CLASSIFIER 1: GLM Model (logistic regression) #####

ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
log.model1 <- train(Property_type ~., data = train1, method = "glm", trControl = ctrl,family=binomial)
log.model1
plot(log.model1)
#summary(log.model1)

log.pred1 <- predict(log.model1, newdata = test1)
log.cm1 <- table(test1$Property_type, log.pred1)
print(log.cm1)

# Confusion Matrix
log.results1 <- confusionMatrix(log.cm1)
log.results1

# Accuracy
log.acc1 <- log.results1$overall['Accuracy']
print(log.acc1)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
log.metrics1 <- binary_metrics(test1$Property_type, log.pred1)
print(log.metrics1)

##### CLASSIFIER 2: Naive Bayes #####

# Define the cross-validation method
ctrl <- trainControl(method="cv", number=10, verboseIter = TRUE)
# Train the Naive Bayes model with cross-validation
n.model1 <- train(Property_type ~., data = train1, method = "naive_bayes", trControl = ctrl)
n.model1
#summary(n.model1)
plot(n.model1)

n.pred1 <- predict(n.model1, newdata = test1)
n.cm1 <- table(test1$Property_type, n.pred1)

# Confusion Matrix
n.results1 <- confusionMatrix(n.cm1)
n.results1

# Accuracy
n.acc1 <- n.results1$overall['Accuracy']
print(n.acc1)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
n.metrics1 <- binary_metrics(test1$Property_type, n.pred1)
print(n.metrics1)

##### CLASSIFIER 3: K-Nearest Neighbours #####
kgrid <- expand.grid(k = 1:15)
# Define the training control (k=5)
train_control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
# Train the KNN model
knn.model1 <- train(Property_type ~ ., data = train1, method = "knn", trControl = train_control, tuneGrid = kgrid)
knn.model1
#summary(knn.model1)
plot(knn.model1)

knn.pred1 <- predict(knn.model1, newdata = test1)
knn.cm1 <- table(test1$Property_type, knn.pred1)
knn.cm1

# Confusion Matrix
knn.results1 <- confusionMatrix(knn.cm1)
knn.results1

# Accuracy 
knn.acc1 <- knn.results1$overall['Accuracy']
print(knn.acc1)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
knn.metrics1 <- binary_metrics(test1$Property_type, knn.pred1)
print(knn.metrics1)


##### CLASSIFIER 4: Support Vector Machine #####

# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train SVM model with 10-fold CV
svm.model1 <- train(Property_type ~ ., data = train1, method = "svmRadial", trControl = trainControl)
svm.model1
plot(svm.model1)

svm.pred1 <- predict(svm.model1, newdata = test1)
svm.cm1 <- table(test1$Property_type, svm.pred1)
svm.cm1

# Confusion Matrix
svm.results1 <- confusionMatrix(svm.cm1)
svm.results1

# Accuracy
svm.acc1 <- svm.results1$overall['Accuracy']
print(svm.acc1)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
svm.metrics1 <- binary_metrics(test1$Property_type, svm.pred1)
print(svm.metrics1)


##### CLASSIFIER 5: Random Forest #####
library(randomForest)
library(caret)


# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train Random Forest model with 10-fold CV
rf.model1 <- train(Property_type ~ ., data = train1, method = "rf", trControl = trainControl)
rf.model1
# Print model summary
#summary(rf.model1)
plot(rf.model1)

rf.pred1 <- predict(rf.model1, newdata = test1)
rf.cm1 <- table(test1$Property_type, rf.pred1)
rf.cm1

# Confusion Matrix
rf.results1 <- confusionMatrix(rf.cm1)
rf.results1

# Accuracy
rf.acc1 <- rf.results1$overall['Accuracy']
print(rf.acc1)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
rf.metrics1 <- binary_metrics(test1$Property_type, rf.pred1)
print(rf.metrics1)


################################ CLASSIFIERS (BORUTA FEATURE SELECTION) ###############################
X <- train[,-1]
y <- train[,1]

# Train Boruta model
boruta_model <- Boruta(X, y)

# Print selected features
boruta_model
print(boruta_model$finalDecision)
# Print the selected features
selected_features <- (which(boruta_model$finalDecision == "Confirmed"))
print(selected_features)

# Select only the chosen features
df2 <- df %>% dplyr::select(selected_features, "Property_type")

head(df2)
str(df2)
set.seed(31)
sample <- sample.split(df2$Property_type, SplitRatio = 0.66)
train2  <- subset(df2, sample == TRUE)
test2   <- subset(df2, sample == FALSE)

##### CLASSIFIER 1: GLM Model (logistic regression) #####

ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
log.model2 <- train(Property_type ~., data = train2, method = "glm", trControl = ctrl,family=binomial)
log.model2
plot(log.model2)

#summary(log.model2)
log.pred2 <- predict(log.model2, newdata = test2)
log.cm2 <- table(test2$Property_type, log.pred2)
print(log.cm2)

# Confusion Matrix
log.results2 <- confusionMatrix(log.cm2)
log.results2

# Accuracy
log.acc2 <- log.results2$overall['Accuracy']
print(log.acc2)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
log.metrics2 <- binary_metrics(test2$Property_type, log.pred2)
print(log.metrics2)

##### CLASSIFIER 2: Naive Bayes #####

# Define the cross-validation method
ctrl <- trainControl(method="cv", number=10, verboseIter = TRUE)
# Train the Naive Bayes model with cross-validation
n.model2 <- train(Property_type ~., data = train2, method = "naive_bayes", trControl = ctrl)
n.model2
#summary(n.model2)
plot(n.model2)

n.pred2 <- predict(n.model2, newdata = test2)
n.cm2 <- table(test2$Property_type, n.pred2)

# Confusion Matrix
n.results2 <- confusionMatrix(n.cm2)
n.results2

# Accuracy
n.acc2 <- n.results2$overall['Accuracy']
print(n.acc2)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
n.metrics2 <- binary_metrics(test2$Property_type, n.pred2)
print(n.metrics2)

##### CLASSIFIER 3: K-Nearest Neighbours #####
kgrid <- expand.grid(k = 1:15)
# Define the training control
train_control <- trainControl(method = "cv", verboseIter = TRUE, number = 10)
# Train the KNN model
knn.model2 <- train(Property_type ~ ., data = train2, method = "knn", trControl = train_control, tuneGrid = kgrid)
knn.model2
#summary(knn.model2)
plot(knn.model2)

knn.pred2 <- predict(knn.model2, newdata = test2)
knn.cm2 <- table(test2$Property_type, knn.pred2)
knn.cm2

# Confusion Matrix
knn.results2 <- confusionMatrix(knn.cm2)
knn.results2

# Accuracy
knn.acc2 <- knn.results2$overall['Accuracy']
print(knn.acc2)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
knn.metrics2 <- binary_metrics(test2$Property_type, knn.pred2)
print(knn.metrics2)



##### CLASSIFIER 4: Support Vector Machine #####

# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train SVM model with 10-fold CV
svm.model2 <- train(Property_type ~ ., data = train2, method = "svmRadial", trControl = trainControl)
svm.model2
plot(svm.model2)

svm.pred2 <- predict(svm.model2, newdata = test2)
svm.cm2 <- table(test2$Property_type, svm.pred2)
svm.cm2

# Confusion Matrix
svm.results2 <- confusionMatrix(svm.cm2)
svm.results2

# Accuracy
svm.acc2 <- svm.results2$overall['Accuracy']
print(svm.acc2)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
svm.metrics2 <- binary_metrics(test2$Property_type, svm.pred2)
print(svm.metrics2)


##### CLASSIFIER 5: Random Forest #####
library(randomForest)
library(caret)


# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train Random Forest model with 10-fold CV
rf.model2<- train(Property_type ~ ., data = train2, method = "rf", trControl = trainControl)
rf.model2
# Print model summary
#summary(rf.model2)
plot(rf.model2)

rf.pred2 <- predict(rf.model2, newdata = test2)
rf.cm2 <- table(test2$Property_type, rf.pred2)
rf.cm2

# Confusion Matrix
rf.results2 <- confusionMatrix(rf.cm2)
rf.results2

# Accuracy
rf.acc2 <- rf.results2$overall['Accuracy']
print(rf.acc2)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
rf.metrics2 <- binary_metrics(test2$Property_type, rf.pred2)
print(rf.metrics2)

################################# CLASSIFIERS (CHI-SQUARE FEATURE SELECTION) #################################

# Define a function to calculate the chi-square statistic for a single feature
chi_squared <- function(feature) {
  # Create a contingency table of the feature and the target variable
  table <- table(train[[feature]], train$Property_type)
  
  # Calculate the chi-square statistic
  chisq.test(table)$statistic
}
# Calculate the chi-square statistic for each feature
chi_values <- sapply(names(train)[2:13], chi_squared)

# Print the chi-square values for each feature
print(chi_values)

# Select the top 5 features based on their chi-square values
selected_features <- sub("\\.X-squared", "", names(sort(chi_values, decreasing=TRUE)[1:5]))
selected_features

# Select only the chosen features
df3 <- df %>% dplyr::select(selected_features, "Property_type")
str(df3)
set.seed(31)

sample <- sample.split(df3$Property_type, SplitRatio = 0.66)
train3  <- subset(df3, sample == TRUE)
test3  <- subset(df3, sample == FALSE)

##### CLASSIFIER 1: GLM Model (logistic regression) #####

ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
log.model3 <- train(Property_type ~., data = train3, method = "glm", trControl = ctrl,family=binomial)
log.model3
plot(log.model3)
#summary(log.model3)

log.pred3 <- predict(log.model3, newdata = test3)
log.cm3 <- table(test3$Property_type, log.pred3)
print(log.cm3)

# Confusion Matrix
log.results3 <- confusionMatrix(log.cm3)
log.results3

# Accuracy
log.acc3 <- log.results3$overall['Accuracy']
print(log.acc3)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
log.metrics3 <- binary_metrics(test3$Property_type, log.pred3)
print(log.metrics3)

##### CLASSIFIER 2: Naive Bayes #####

# Define the cross-validation method
ctrl <- trainControl(method="cv", number=10, verboseIter = TRUE)
# Train the Naive Bayes model with cross-validation
n.model3 <- train(Property_type ~., data = train3, method = "naive_bayes", trControl = ctrl)
n.model3
plot(n.model3)
#summary(n.model3)

n.pred3 <- predict(n.model3, newdata = test3)
n.cm3 <- table(test3$Property_type, n.pred3)

# Confusion Matrix
n.results3 <- confusionMatrix(n.cm3)
n.results3

# Accuracy
n.acc3 <- n.results3$overall['Accuracy']
print(n.acc3)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
n.metrics3 <- binary_metrics(test3$Property_type, n.pred3)
print(n.metrics3)

##### CLASSIFIER 3: K-Nearest Neighbours #####

kgrid <- expand.grid(k = 1:15)
# Define the training control 
train_control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
# Train the KNN model
knn.model3 <- train(Property_type ~ ., data = train3, method = "knn", trControl = train_control, tuneGrid = kgrid)
knn.model3
#summary(knn.model3)
plot(knn.model3)

knn.pred3 <- predict(knn.model3, newdata = test3)
knn.cm3 <- table(test3$Property_type, knn.pred3)
knn.cm3

# Confusion Matrix
knn.results3 <- confusionMatrix(knn.cm3)
knn.results3

# Accuracy
knn.acc3 <- knn.results3$overall['Accuracy']
print(knn.acc3)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
knn.metrics3 <- binary_metrics(test3$Property_type, knn.pred3)
print(knn.metrics3)


##### CLASSIFIER 4: Support Vector Machine #####

# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train SVM model with 10-fold CV
svm.model3 <- train(Property_type ~ ., data = train3, method = "svmRadial", trControl = trainControl)
svm.model3
plot(svm.model3)

svm.pred3 <- predict(svm.model3, newdata = test3)
svm.cm3 <- table(test3$Property_type, svm.pred3)
svm.cm3

# Confusion Matrix
svm.results3 <- confusionMatrix(svm.cm3)
svm.results3

# Accuracy
svm.acc3 <- svm.results$overall['Accuracy']
print(svm.acc3)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
svm.metrics3 <- binary_metrics(test3$Property_type, svm.pred3)
print(svm.metrics3)


##### CLASSIFIER 5: Random Forest #####
library(randomForest)
library(caret)


# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train Random Forest model with 10-fold CV
rf.model3 <- train(Property_type ~ ., data = train3, method = "rf", trControl = trainControl)
rf.model3
plot(rf.model3)
# Print model summary
#summary(rf.model3)

rf.pred3 <- predict(rf.model3, newdata = test3)
rf.cm3 <- table(test3$Property_type, rf.pred3)
rf.cm3

# Confusion Matrix
rf.results3 <- confusionMatrix(rf.cm3)
rf.results3

# Accuracy
rf.acc3 <- rf.results3$overall['Accuracy']
print(rf.acc3)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
rf.metrics3 <- binary_metrics(test3$Property_type, rf.pred3)
print(rf.metrics3)


############################### CLASSIFIERS (R-PART IMPORTANCE FEATURE SELECTION) ############################

#Train an rpart model to select features
rPartMod <- train(Property_type ~ ., data=train, method="rpart")
rpartImp <- varImp(rPartMod)
print(rpartImp)
plot(rpartImp)

# Select only the chosen features
df4 <- df %>% dplyr::select("Longitude", "Property_type","Property_building_status","Latitude",
                            "Price_per_unit_area","No_of_BHK","Size","Price")

head(df4)
str(df4)
set.seed(31)

sample <- sample.split(df4$Property_type, SplitRatio = 0.66)
train4  <- subset(df4, sample == TRUE)
test4  <- subset(df4, sample == FALSE)



##### CLASSIFIER 1: GLM Model (logistic regression) #####

ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
log.model4 <- train(Property_type ~., data = train4, method = "glm", trControl = ctrl,family=binomial)
log.model4

#summary(log.model4)

log.pred4 <- predict(log.model4, newdata = test4)
log.cm4 <- table(test4$Property_type, log.pred4)

# Confusion Matrix
log.results4 <- confusionMatrix(log.cm4)
log.results4

# Accuracy
log.acc4 <- log.results4$overall['Accuracy']
print(log.acc4)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
log.metrics4 <- binary_metrics(test4$Property_type, log.pred4)
print(log.metrics4)

##### CLASSIFIER 2: Naive Bayes #####

# Define the cross-validation method
ctrl <- trainControl(method="cv", number=10, verboseIter = TRUE)
# Train the Naive Bayes model with cross-validation
#n.model <- train(Property_type ~ ., data=train, method="nb", trControl=ctrl)
n.model4 <- train(Property_type ~., data = train4, method = "naive_bayes", trControl = ctrl)
n.model4
plot(n.model4)
#n.model <- naiveBayes(Property_type ~ ., data = train)
#summary(n.model4)

n.pred4 <- predict(n.model4, newdata = test4)
n.cm4 <- table(test4$Property_type, n.pred4)

# Confusion Matrix
n.results4 <- confusionMatrix(n.cm4)
n.results4

# Accuracy
n.acc4 <- n.results4$overall['Accuracy']
print(n.acc4)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
n.metrics4 <- binary_metrics(test4$Property_type, n.pred4)
print(n.metrics4)

##### CLASSIFIER 3: K-Nearest Neighbours #####

# Define the training control (k=5)
train_control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
# Train the KNN model
kgrid <- expand.grid(k = 1:15)
knn.model4 <- train(Property_type ~ ., data = train4, method = "knn", trControl = train_control, tuneGrid = kgrid)
knn.model4
plot(knn.model4)
#summary(knn.model4)

knn.pred4 <- predict(knn.model4, newdata = test4)
knn.cm4 <- table(test4$Property_type, knn.pred4)
knn.cm4

# Confusion Matrix
knn.results4 <- confusionMatrix(knn.cm4)
knn.results4

# Accuracy
knn.acc4 <- knn.results4$overall['Accuracy']
print(knn.acc4)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
knn.metrics4 <- binary_metrics(test4$Property_type, knn.pred4)
print(knn.metrics4)


##### CLASSIFIER 4: Support Vector Machine #####

# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train SVM model with 10-fold CV
svm.model4 <- train(Property_type ~ ., data = train4, method = "svmRadial", trControl = trainControl)
svm.model4
plot(svm.model4)

svm.pred4 <- predict(svm.model4, newdata = test4)
svm.cm4 <- table(test4$Property_type, svm.pred4)
svm.cm4

# Confusion Matrix
svm.results4 <- confusionMatrix(svm.cm4)
svm.results4

# Accuracy
svm.acc4 <- svm.results4$overall['Accuracy']
print(svm.acc4)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
svm.metrics4 <- binary_metrics(test4$Property_type, svm.pred4)
print(svm.metrics4)


##### CLASSIFIER 5: Random Forest #####

# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train Random Forest model with 10-fold CV
rf.model4 <- train(Property_type ~ ., data = train4, method = "rf", trControl = trainControl)
rf.model4
# Print model summary
#summary(rf.model4)
plot(rf.model4)

rf.pred4 <- predict(rf.model4, newdata = test4)
rf.cm4 <- table(test4$Property_type, rf.pred4)
rf.cm4

# Confusion Matrix
rf.results4 <- confusionMatrix(rf.cm4)
rf.results4

# Accuracy
rf.acc4 <- rf.results4$overall['Accuracy']
print(rf.acc4)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
rf.metrics4 <- binary_metrics(test4$Property_type, rf.pred4)
print(rf.metrics4)

##################################### CLASSIFIERS (RFE FEATURE SELECTION) ##########################################

set.seed(31)
df <- read.csv("Makaan_Properties_Buy.csv", header = TRUE)
df <- df %>% sample_n(10000)
sample <- sample.split(df$Property_type, SplitRatio = 0.66)
train  <- subset(df, sample == TRUE)
test   <- subset(df, sample == FALSE)

# create formula for the target variable and all other variables
formula <- as.formula("Property_type ~ .")

# set up control parameters for feature selection
control <- rfeControl(functions = rfFuncs,
                      method = "cv",
                      number = 10,
                      verbose = FALSE)

# run RFE with the random forest algorithm
model <- rfe(train[, -1], train$Property_type, sizes = c(1:ncol(train)-1),
             rfeControl = control, method = "rf")

# summarize the results
print(model)

# get selected features
selected_features <- model$optVariables[1:5]; selected_features

# Select only the chosen features
df5 <- df %>% dplyr::select(selected_features, "Property_type")

head(df5)
str(df5)
set.seed(31)

sample <- sample.split(df5$Property_type, SplitRatio = 0.66)
train5  <- subset(df5, sample == TRUE)
test5  <- subset(df5, sample == FALSE)

##### CLASSIFIER 1: GLM Model (logistic regression) #####

ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
log.model5 <- train(Property_type ~., data = train5, method = "glm", trControl = ctrl,family=binomial)
log.model5
plot(log.model5)
#summary(log.model5)

log.pred5 <- predict(log.model5, newdata = test5)
log.cm5 <- table(test5$Property_type, log.pred5)
print(log.cm5)

# Confusion Matrix
log.results5 <- confusionMatrix(log.cm5)
log.results5

# Accuracy
log.acc5 <- log.results5$overall['Accuracy']
print(log.acc5)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
log.metrics5 <- binary_metrics(test5$Property_type, log.pred5)
print(log.metrics5)

##### CLASSIFIER 2: Naive Bayes #####

# Define the cross-validation method
ctrl <- trainControl(method="cv", number=10, verboseIter = TRUE)
# Train the Naive Bayes model with cross-validation
#n.model <- train(Property_type ~ ., data=train, method="nb", trControl=ctrl)
n.model5 <- train(Property_type ~., data = train5, method = "naive_bayes", trControl = ctrl)
n.model5
plot(n.model5)
#n.model <- naiveBayes(Property_type ~ ., data = train)
#summary(n.model5)

n.pred5 <- predict(n.model5, newdata = test5)
n.cm5 <- table(test5$Property_type, n.pred5)
n.cm5

# Confusion Matrix
n.results5 <- confusionMatrix(n.cm5)
n.results5

# Accuracy
n.acc5 <- n.results5$overall['Accuracy']
print(n.acc5)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
n.metrics5 <- binary_metrics(test5$Property_type, n.pred5)
print(n.metrics5)

##### CLASSIFIER 3: K-Nearest Neighbours #####

# Define the training control (k=5)
train_control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
# Train the KNN model
kgrid <- expand.grid(k = 1:15)
knn.model5 <- train(Property_type ~ ., data = train5, method = "knn", trControl = train_control, tuneGrid = kgrid)
knn.model5
plot(knn.model5)
#summary(knn.model5)

knn.pred5 <- predict(knn.model5, newdata = test5)
knn.cm5 <- table(test5$Property_type, knn.pred5)
knn.cm5

# Confusion Matrix
knn.results5 <- confusionMatrix(knn.cm5)
knn.results5

# Accuracy
knn.acc5 <- knn.results5$overall['Accuracy']
print(knn.acc5)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
knn.metrics5 <- binary_metrics(test5$Property_type, knn.pred5)
print(knn.metrics5)


##### CLASSIFIER 4: Support Vector Machine #####

# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train SVM model with 10-fold CV
svm.model5 <- train(Property_type ~ ., data = train5, method = "svmRadial", trControl = trainControl)
svm.model5
plot(svm.model5)

svm.pred5 <- predict(svm.model5, newdata = test5)
svm.cm5 <- table(test5$Property_type, svm.pred5)
svm.cm5

# Confusion Matrix
svm.results5 <- confusionMatrix(svm.cm5)

# Accuracy
svm.acc5 <- svm.results5$overall['Accuracy']
print(svm.acc5)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
svm.metrics5 <- binary_metrics(test5$Property_type, svm.pred5)
print(svm.metrics5)


##### CLASSIFIER 5: Random Forest #####

# Define training control with 10-fold CV
trainControl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Train Random Forest model with 10-fold CV
rf.model5 <- train(Property_type ~ ., data = train5, method = "rf", trControl = trainControl)
rf.model5
plot(rf.model5)
# Print model summary
#summary(rf.model5)

rf.pred5 <- predict(rf.model5, newdata = test5)
rf.cm5 <- table(test5$Property_type, rf.pred5)
rf.cm5

# Confusion Matrix
rf.results5 <- confusionMatrix(rf.cm5)
rf.results5

# Accuracy
rf.acc5 <- rf.results5$overall['Accuracy']
print(rf.acc5)

# Get the TP rate, FP rate, precision, recall, F-measure, ROC area, and MCC for each class and weighted averages of performance measures
rf.metrics5 <- binary_metrics(test5$Property_type, rf.pred5)
print(rf.metrics5)

