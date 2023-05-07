
# Load libraries

library(data.table)
library(datasets)
library(tidyverse)
library(ggbiplot)
library(caret)
library(glmnet)
library(xgboost)



# Import data (OLD)
#df <- read.csv('./data/train.csv') 
#head(df)
#dim(df)
#table(df$author)

#charles <- df[df$author %in% c(2,3), ]
#head(charles)
#dim(charles)

#write.csv(charles, file="charles.csv")


#############################################################

# Get vectorized data created from Python code
tfidf_train <- read.csv('./data/tfidf_train.csv')
tfidf_test <- read.csv('./data/tfidf_test.csv')
tfidf_train_svd <- read.csv('./data/tfidf_train_svd.csv')
tfidf_test_svd <- read.csv('./data/tfidf_test_svd.csv')

y_train <- read.csv('./data/y_train.csv')
y_train <- ifelse(y_train$author == "Darwin", 1, 0)
y_test <- read.csv('./data/y_test.csv')
y_test <- ifelse(y_test == "Darwin", 1, 0)


X_train <- data.matrix(tfidf_train)
X_train_svd <- data.matrix(tfidf_train_svd)

X_test <- data.matrix(tfidf_test)
X_test_svd <- data.matrix(tfidf_test_svd)

###############################################

# Logistic Regression

# Lasso
start <- Sys.time()
cv_model <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial")
end <- Sys.time()
print(end - start)
plot(cv_model)
# Make predictions on the test data
probabilities <- predict(cv_model, newx = X_test)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy
mean(predicted_classes == y_test)
# Model using cross-validated (minimum) lambda
model <- glmnet(X_train, y_train, alpha = 1, lambda = cv_model$lambda.min, family = "binomial")
probabilities <- predict(model, newx = X_test)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
mean(predicted_classes == y_test)
# Model using 1se lambda
model <- glmnet(X_train, y_train, alpha = 1, lambda = cv_model$lambda.1se, family = "binomial")
probabilities <- predict(model, newx = X_test)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
mean(predicted_classes == y_test)

# Ridge
start <- Sys.time()
cv_model <- cv.glmnet(X_train, y_train, alpha = 0, family = "binomial")
end <- Sys.time()
print(end - start)
plot(cv_model)
# Make predictions on the test data
probabilities <- predict(cv_model, newx = X_test)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy
mean(predicted_classes == y_test)
# Model using cross-validated lambda
model <- glmnet(X_train, y_train, alpha = 0, lambda = cv_model$lambda.min, family = "binomial")
probabilities <- predict(model, newx = X_test)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
mean(predicted_classes == y_test)
# Model using 1se lambda
model <- glmnet(X_train, y_train, alpha = 0, lambda = cv_model$lambda.1se, family = "binomial")
probabilities <- predict(model, newx = X_test)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
mean(predicted_classes == y_test)


# Logistic Regression (with SVD)

# Lasso 
start <- Sys.time()
cv_model <- cv.glmnet(X_train_svd, y_train, alpha = 1, family = "binomial")
end <- Sys.time()
print(end - start)
probabilities <- predict(cv_model, newx = X_test_svd)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
mean(predicted_classes == y_test)
# Model using cross-validated lambda
model <- glmnet(X_train_svd, y_train, alpha = 1, lambda = cv_model$lambda.min, family = "binomial")
probabilities <- predict(model, newx = X_test_svd)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
mean(predicted_classes == y_test)
# Model using 1se lambda
model <- glmnet(X_train, y_train, alpha = 1, lambda = cv_model$lambda.1se, family = "binomial")
probabilities <- predict(model, newx = X_test)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
mean(predicted_classes == y_test)

# Ridge
start <- Sys.time()
cv_model <- cv.glmnet(X_train_svd, y_train, alpha = 0, family = "binomial")
end <- Sys.time()
print(end - start)
probabilities <- predict(cv_model, newx = X_test_svd)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
mean(predicted_classes == y_test)
# Model using cross-validated lambda
model <- glmnet(X_train_svd, y_train, alpha = 0, lambda = cv_model$lambda.min, family = "binomial")
probabilities <- predict(model, newx = X_test_svd)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
mean(predicted_classes == y_test)
# Model using 1se lambda
model <- glmnet(X_train, y_train, alpha = 0, lambda = cv_model$lambda.1se, family = "binomial")
probabilities <- predict(model, newx = X_test)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)
mean(predicted_classes == y_test)



#########################################################



