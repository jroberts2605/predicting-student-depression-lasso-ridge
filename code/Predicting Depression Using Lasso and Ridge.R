library(dplyr)
library(glmnet)
library(caret)
library(ggplot2)
library(pROC)
library(reshape2)

data <- read.csv("depression.csv")

data_clean <- data

data_clean$Gender <- ifelse(data_clean$Gender == "Male", 0, 1)

data_clean$SleepDuration <- case_when(
  data_clean$SleepDuration == "Less than 5 hours" ~ 0,
  data_clean$SleepDuration == "5-6 hours" ~ 1,
  data_clean$SleepDuration == "7-8 hours" ~ 2,
  data_clean$SleepDuration == "More than 8 hours" ~ 3
)

data_clean$DietaryHabits <- case_when(
  data_clean$DietaryHabits == "Unhealthy" ~ 0,
  data_clean$DietaryHabits == "Moderate" ~ 1,
  data_clean$DietaryHabits == "Healthy" ~ 2
)

data_clean$SuicidalThoughts <- ifelse(data_clean$SuicidalThoughts == "Yes", 1, 0)
data_clean$FamilyHistory <- ifelse(data_clean$FamilyHistory == "Yes", 1, 0)
data_clean$Depression <- ifelse(data_clean$Depression == "Yes", 1, 0)

data_clean_no_suicidal <- subset(data_clean, select = -SuicidalThoughts)



# LASSO MODEL (with SuicidalThoughts)


set.seed(534)
train_index_lasso <- createDataPartition(data_clean$Depression, p = 0.7, list = FALSE)
train_data_lasso <- data_clean[train_index_lasso, ]
test_data_lasso  <- data_clean[-train_index_lasso, ]

X_train_lasso <- model.matrix(Depression ~ ., data = train_data_lasso)[, -1]
y_train_lasso <- train_data_lasso$Depression

X_test_lasso  <- model.matrix(Depression ~ ., data = test_data_lasso)[, -1]
y_test_lasso  <- test_data_lasso$Depression

lasso_model <- cv.glmnet(
  x = X_train_lasso,
  y = y_train_lasso,
  family = "binomial",
  alpha = 1,
  type.measure = "class"
)

best_lambda_lasso <- lasso_model$lambda.min
cat("LASSO Lambda:", best_lambda_lasso, "\n")

cat("\nLASSO Coefficients:\n")
print(coef(lasso_model, s = "lambda.min"))

pred_probs_lasso <- predict(lasso_model, newx = X_test_lasso, s = "lambda.min", type = "response")
pred_class_lasso <- ifelse(pred_probs_lasso > 0.5, 1, 0)

cat("\nLASSO Confusion Matrix:\n")
print(table(Predicted = pred_class_lasso, Actual = y_test_lasso))

accuracy_lasso <- mean(pred_class_lasso == y_test_lasso)

train_pred_probs_lasso <- predict(lasso_model, newx = X_train_lasso, s = "lambda.min", type = "response")
train_pred_class_lasso <- ifelse(train_pred_probs_lasso > 0.5, 1, 0)
train_accuracy_lasso <- mean(train_pred_class_lasso == y_train_lasso)

cat("Training Accuracy:", round(train_accuracy_lasso, 6), "\n")
cat("Test Accuracy:    ", round(accuracy_lasso, 6), "\n")

null_model_lasso <- glm(Depression ~ 1, data = train_data_lasso, family = "binomial")
null_deviance_lasso <- null_model_lasso$deviance

lasso_pred_train <- predict(lasso_model, newx = X_train_lasso, s = "lambda.min", type = "response")
lasso_deviance <- -2 * sum(y_train_lasso * log(lasso_pred_train) + (1 - y_train_lasso) * log(1 - lasso_pred_train))

mcfadden_r2_lasso <- 1 - (lasso_deviance / null_deviance_lasso)
cat("McFadden's R²:", round(mcfadden_r2_lasso, 6), "\n\n")


# LASSO MODEL OMITTED (without SuicidalThoughts)


set.seed(534)
train_index_lasso_omit <- createDataPartition(data_clean_no_suicidal$Depression, p = 0.7, list = FALSE)
train_data_lasso_omit <- data_clean_no_suicidal[train_index_lasso_omit, ]
test_data_lasso_omit  <- data_clean_no_suicidal[-train_index_lasso_omit, ]

X_train_lasso_omit <- model.matrix(Depression ~ ., data = train_data_lasso_omit)[, -1]
y_train_lasso_omit <- train_data_lasso_omit$Depression

X_test_lasso_omit  <- model.matrix(Depression ~ ., data = test_data_lasso_omit)[, -1]
y_test_lasso_omit  <- test_data_lasso_omit$Depression

lasso_model_omitted <- cv.glmnet(
  x = X_train_lasso_omit,
  y = y_train_lasso_omit,
  family = "binomial",
  alpha = 1,
  type.measure = "class"
)

best_lambda_lasso_omit <- lasso_model_omitted$lambda.min
cat("LASSO OMITTED Lambda:", best_lambda_lasso_omit, "\n")

cat("\nLASSO OMITTED Coefficients:\n")
print(coef(lasso_model_omitted, s = "lambda.min"))

pred_probs_lasso_omit <- predict(lasso_model_omitted, newx = X_test_lasso_omit, s = "lambda.min", type = "response")
pred_class_lasso_omit <- ifelse(pred_probs_lasso_omit > 0.5, 1, 0)

cat("\nLASSO OMITTED Confusion Matrix:\n")
print(table(Predicted = pred_class_lasso_omit, Actual = y_test_lasso_omit))

accuracy_lasso_omit <- mean(pred_class_lasso_omit == y_test_lasso_omit)

train_pred_probs_lasso_omit <- predict(lasso_model_omitted, newx = X_train_lasso_omit, s = "lambda.min", type = "response")
train_pred_class_lasso_omit <- ifelse(train_pred_probs_lasso_omit > 0.5, 1, 0)
train_accuracy_lasso_omit <- mean(train_pred_class_lasso_omit == y_train_lasso_omit)

cat("Training Accuracy:", round(train_accuracy_lasso_omit, 6), "\n")
cat("Test Accuracy:    ", round(accuracy_lasso_omit, 6), "\n")

null_model_lasso_omit <- glm(Depression ~ 1, data = train_data_lasso_omit, family = "binomial")
null_deviance_lasso_omit <- null_model_lasso_omit$deviance

lasso_pred_train_omit <- predict(lasso_model_omitted, newx = X_train_lasso_omit, s = "lambda.min", type = "response")
lasso_deviance_omit <- -2 * sum(y_train_lasso_omit * log(lasso_pred_train_omit) + (1 - y_train_lasso_omit) * log(1 - lasso_pred_train_omit))

mcfadden_r2_lasso_omit <- 1 - (lasso_deviance_omit / null_deviance_lasso_omit)
cat("McFadden's R²:", round(mcfadden_r2_lasso_omit, 6), "\n\n")



# RIDGE MODEL (with SuicidalThoughts)

set.seed(534)
train_index_ridge <- createDataPartition(data_clean$Depression, p = 0.7, list = FALSE)
train_data_ridge <- data_clean[train_index_ridge, ]
test_data_ridge  <- data_clean[-train_index_ridge, ]

X_train_ridge <- model.matrix(Depression ~ ., data = train_data_ridge)[, -1]
y_train_ridge <- train_data_ridge$Depression

X_test_ridge  <- model.matrix(Depression ~ ., data = test_data_ridge)[, -1]
y_test_ridge  <- test_data_ridge$Depression

ridge_model <- cv.glmnet(
  x = X_train_ridge,
  y = y_train_ridge,
  family = "binomial",
  alpha = 0,
  type.measure = "class"
)

best_lambda_ridge <- ridge_model$lambda.min
cat("RIDGE Lambda:", best_lambda_ridge, "\n")

cat("\nRIDGE Coefficients:\n")
print(coef(ridge_model, s = "lambda.min"))

pred_probs_ridge <- predict(ridge_model, newx = X_test_ridge, s = "lambda.min", type = "response")
pred_class_ridge <- ifelse(pred_probs_ridge > 0.5, 1, 0)

cat("\nRIDGE Confusion Matrix:\n")
print(table(Predicted = pred_class_ridge, Actual = y_test_ridge))

accuracy_ridge <- mean(pred_class_ridge == y_test_ridge)

train_pred_probs_ridge <- predict(ridge_model, newx = X_train_ridge, s = "lambda.min", type = "response")
train_pred_class_ridge <- ifelse(train_pred_probs_ridge > 0.5, 1, 0)
train_accuracy_ridge <- mean(train_pred_class_ridge == y_train_ridge)

cat("Training Accuracy:", round(train_accuracy_ridge, 6), "\n")
cat("Test Accuracy:    ", round(accuracy_ridge, 6), "\n")

null_model_ridge <- glm(Depression ~ 1, data = train_data_ridge, family = "binomial")
null_deviance_ridge <- null_model_ridge$deviance

ridge_pred_train <- predict(ridge_model, newx = X_train_ridge, s = "lambda.min", type = "response")
ridge_deviance <- -2 * sum(y_train_ridge * log(ridge_pred_train) + (1 - y_train_ridge) * log(1 - ridge_pred_train))

mcfadden_r2_ridge <- 1 - (ridge_deviance / null_deviance_ridge)
cat("McFadden's R²:", round(mcfadden_r2_ridge, 6), "\n\n")


# RIDGE MODEL OMITTED (without SuicidalThoughts)


set.seed(534)
train_index_ridge_omit <- createDataPartition(data_clean_no_suicidal$Depression, p = 0.7, list = FALSE)
train_data_ridge_omit <- data_clean_no_suicidal[train_index_ridge_omit, ]
test_data_ridge_omit  <- data_clean_no_suicidal[-train_index_ridge_omit, ]

X_train_ridge_omit <- model.matrix(Depression ~ ., data = train_data_ridge_omit)[, -1]
y_train_ridge_omit <- train_data_ridge_omit$Depression

X_test_ridge_omit  <- model.matrix(Depression ~ ., data = test_data_ridge_omit)[, -1]
y_test_ridge_omit  <- test_data_ridge_omit$Depression

ridge_model_omitted <- cv.glmnet(
  x = X_train_ridge_omit,
  y = y_train_ridge_omit,
  family = "binomial",
  alpha = 0,
  type.measure = "class"
)

best_lambda_ridge_omit <- ridge_model_omitted$lambda.min
cat("RIDGE OMITTED Lambda:", best_lambda_ridge_omit, "\n")

cat("\nRIDGE OMITTED Coefficients:\n")
print(coef(ridge_model_omitted, s = "lambda.min"))

pred_probs_ridge_omit <- predict(ridge_model_omitted, newx = X_test_ridge_omit, s = "lambda.min", type = "response")
pred_class_ridge_omit <- ifelse(pred_probs_ridge_omit > 0.5, 1, 0)

cat("\nRIDGE OMITTED Confusion Matrix:\n")
print(table(Predicted = pred_class_ridge_omit, Actual = y_test_ridge_omit))

accuracy_ridge_omit <- mean(pred_class_ridge_omit == y_test_ridge_omit)

train_pred_probs_ridge_omit <- predict(ridge_model_omitted, newx = X_train_ridge_omit, s = "lambda.min", type = "response")
train_pred_class_ridge_omit <- ifelse(train_pred_probs_ridge_omit > 0.5, 1, 0)
train_accuracy_ridge_omit <- mean(train_pred_class_ridge_omit == y_train_ridge_omit)

cat("Training Accuracy:", round(train_accuracy_ridge_omit, 6), "\n")
cat("Test Accuracy:    ", round(accuracy_ridge_omit, 6), "\n")

null_model_ridge_omit <- glm(Depression ~ 1, data = train_data_ridge_omit, family = "binomial")
null_deviance_ridge_omit <- null_model_ridge_omit$deviance

ridge_pred_train_omit <- predict(ridge_model_omitted, newx = X_train_ridge_omit, s = "lambda.min", type = "response")
ridge_deviance_omit <- -2 * sum(y_train_ridge_omit * log(ridge_pred_train_omit) + (1 - y_train_ridge_omit) * log(1 - ridge_pred_train_omit))

mcfadden_r2_ridge_omit <- 1 - (ridge_deviance_omit / null_deviance_ridge_omit)
cat("McFadden's R²:", round(mcfadden_r2_ridge_omit, 6), "\n\n")

# LASSO GRAPHS (with SuicidalThoughts)

plot(lasso_model$glmnet.fit, xvar = "lambda", label = TRUE)
title("LASSO Coefficient Path (with SuicidalThoughts)")
legend("topleft", legend = paste(1:ncol(X_train_lasso), colnames(X_train_lasso), sep = ": "), 
       cex = 0.7, title = "Variables")

plot(lasso_model, main = "LASSO Cross-Validation (with SuicidalThoughts)")

conf_mat_lasso <- as.data.frame(table(Predicted = pred_class_lasso, Actual = y_test_lasso))

ggplot(conf_mat_lasso, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 12, color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "LASSO Confusion Matrix (with SuicidalThoughts)",
       x = "Actual Depression Status",
       y = "Predicted Depression Status") +
  theme_minimal()

roc_lasso <- roc(y_test_lasso, as.vector(pred_probs_lasso))
plot(roc_lasso, col = "blue", lwd = 2, main = "LASSO ROC Curve (with SuicidalThoughts)")
legend("bottomright", legend = paste("AUC =", round(auc(roc_lasso), 3)), col = "blue", lwd = 2)


# LASSO OMITTED GRAPHS (without SuicidalThoughts)

plot(lasso_model_omitted$glmnet.fit, xvar = "lambda", label = TRUE)
title("LASSO Coefficient Path (without SuicidalThoughts)")
legend("topleft", legend = paste(1:ncol(X_train_lasso_omit), colnames(X_train_lasso_omit), sep = ": "), 
       cex = 0.7, title = "Variables")

plot(lasso_model_omitted, main = "LASSO Cross-Validation (without SuicidalThoughts)")

conf_mat_lasso_omit <- as.data.frame(table(Predicted = pred_class_lasso_omit, Actual = y_test_lasso_omit))

ggplot(conf_mat_lasso_omit, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 12, color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "LASSO Confusion Matrix (without SuicidalThoughts)",
       x = "Actual Depression Status",
       y = "Predicted Depression Status") +
  theme_minimal()

roc_lasso_omit <- roc(y_test_lasso_omit, as.vector(pred_probs_lasso_omit))
plot(roc_lasso_omit, col = "blue", lwd = 2, main = "LASSO ROC Curve (without SuicidalThoughts)")
legend("bottomright", legend = paste("AUC =", round(auc(roc_lasso_omit), 3)), col = "blue", lwd = 2)


# RIDGE GRAPHS (with SuicidalThoughts)

plot(ridge_model$glmnet.fit, xvar = "lambda", label = TRUE)
title("Ridge Coefficient Path (with SuicidalThoughts)")
legend("topleft", legend = paste(1:ncol(X_train_ridge), colnames(X_train_ridge), sep = ": "), 
       cex = 0.7, title = "Variables")

plot(ridge_model, main = "Ridge Cross-Validation (with SuicidalThoughts)")

conf_mat_ridge <- as.data.frame(table(Predicted = pred_class_ridge, Actual = y_test_ridge))

ggplot(conf_mat_ridge, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 12, color = "white") +
  scale_fill_gradient(low = "lightcoral", high = "darkred") +
  labs(title = "Ridge Confusion Matrix (with SuicidalThoughts)",
       x = "Actual Depression Status",
       y = "Predicted Depression Status") +
  theme_minimal()

roc_ridge <- roc(y_test_ridge, as.vector(pred_probs_ridge))
plot(roc_ridge, col = "red", lwd = 2, main = "Ridge ROC Curve (with SuicidalThoughts)")
legend("bottomright", legend = paste("AUC =", round(auc(roc_ridge), 3)), col = "red", lwd = 2)


# RIDGE OMITTED GRAPHS (without SuicidalThoughts)

plot(ridge_model_omitted$glmnet.fit, xvar = "lambda", label = TRUE)
title("Ridge Coefficient Path (without SuicidalThoughts)")
legend("topleft", legend = paste(1:ncol(X_train_ridge_omit), colnames(X_train_ridge_omit), sep = ": "), 
       cex = 0.7, title = "Variables")

plot(ridge_model_omitted, main = "Ridge Cross-Validation (without SuicidalThoughts)")

conf_mat_ridge_omit <- as.data.frame(table(Predicted = pred_class_ridge_omit, Actual = y_test_ridge_omit))

ggplot(conf_mat_ridge_omit, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 12, color = "white") +
  scale_fill_gradient(low = "lightcoral", high = "darkred") +
  labs(title = "Ridge Confusion Matrix (without SuicidalThoughts)",
       x = "Actual Depression Status",
       y = "Predicted Depression Status") +
  theme_minimal()

roc_ridge_omit <- roc(y_test_ridge_omit, as.vector(pred_probs_ridge_omit))
plot(roc_ridge_omit, col = "red", lwd = 2, main = "Ridge ROC Curve (without SuicidalThoughts)")
legend("bottomright", legend = paste("AUC =", round(auc(roc_ridge_omit), 3)), col = "red", lwd = 2)

# COMPARISON GRAPHS


lasso_coefs <- coef(lasso_model, s = "lambda.min")[-1, 1]
ridge_coefs <- coef(ridge_model, s = "lambda.min")[-1, 1]

coef_comparison <- data.frame(
  Variable = names(lasso_coefs),
  LASSO = abs(lasso_coefs),
  Ridge = abs(ridge_coefs)
)

coef_long <- melt(coef_comparison, id.vars = "Variable")

ggplot(coef_long, aes(x = reorder(Variable, value), y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(title = "Variable Importance: LASSO vs Ridge (with SuicidalThoughts)",
       x = "Variable",
       y = "Absolute Coefficient Value",
       fill = "Method") +
  theme_minimal()

lasso_coefs_omit <- coef(lasso_model_omitted, s = "lambda.min")[-1, 1]
ridge_coefs_omit <- coef(ridge_model_omitted, s = "lambda.min")[-1, 1]

coef_comparison_omit <- data.frame(
  Variable = names(lasso_coefs_omit),
  LASSO = abs(lasso_coefs_omit),
  Ridge = abs(ridge_coefs_omit)
)

coef_long_omit <- melt(coef_comparison_omit, id.vars = "Variable")

ggplot(coef_long_omit, aes(x = reorder(Variable, value), y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(title = "Variable Importance: LASSO vs Ridge (without SuicidalThoughts)",
       x = "Variable",
       y = "Absolute Coefficient Value",
       fill = "Method") +
  theme_minimal()

plot(roc_lasso, col = "blue", lwd = 2, main = "ROC Curves: LASSO vs Ridge (with SuicidalThoughts)")
lines(roc_ridge, col = "red", lwd = 2)
legend("bottomright", 
       legend = c(paste("LASSO AUC =", round(auc(roc_lasso), 3)),
                  paste("Ridge AUC =", round(auc(roc_ridge), 3))),
       col = c("blue", "red"), lwd = 2)

plot(roc_lasso_omit, col = "blue", lwd = 2, main = "ROC Curves: LASSO vs Ridge (without SuicidalThoughts)")
lines(roc_ridge_omit, col = "red", lwd = 2)
legend("bottomright", 
       legend = c(paste("LASSO AUC =", round(auc(roc_lasso_omit), 3)),
                  paste("Ridge AUC =", round(auc(roc_ridge_omit), 3))),
       col = c("blue", "red"), lwd = 2)



cat("LASSO Lambda:", best_lambda_lasso, "\n")