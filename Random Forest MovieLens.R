set.seed(941)

data = read.table(
  
  "C:\\Users\\colli\\OneDrive/Documents\\movielens_100k.base
",
  header=FALSE,
  sep="\t",
  col.names = c("UserID", "MovieID", "Rating",
                "Timestamp")
)

data = data[,1:3]

#Testing Data
test = read.table(
  
  "C:\\Users\\colli\\OneDrive\\Documents\\movielens_100k.test
",
  header=FALSE,
  sep="\t",
  col.names = c("UserID", "MovieID", "Rating",
                "Timestamp")
)

test = test[,1:3]
install.packages("randomForest")
library(randomForest)
library(caret)


# Fit random forest model on training data
rfmodel <- randomForest(Rating ~ ., data = data, ntree = 500,
                        mtry = sqrt(ncol(data)))

# Predict on testing data
pre <- predict(rfModel, test)

# Calculate RMSE
rmse <- sqrt(mean((test$Rating - pre)^2))

plot_data <- data.frame(actual = test$Rating, predicted = pre)
ggplot(plot_data, aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed")
+
  labs(title = "Random Forest Regression", x = "Actual", y =
         "Predicted")
