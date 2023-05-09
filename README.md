# Movie Rating Recommendation Prediction

In this project, I will use different classification and prediction models to find the best prediction of movie
recommendations based off prior movie and user data. The MovieLens dataset wil be used from https://paperswithcode.com/sota/collaborative-filtering-on-movielens-100k. This is a dataset containing 100,000 observations,
which consists of user rating of movies. A matrix factorization
model and random forest model will be used, and their individual
root mean squared error (RMSE) scores will be used to determine
the model of best fit. Finally, the overall best results will be
compared with models built by others.

# Model Building
Our first model created will be a matrix factorization
model. This model is created by reading in data and splitting it between 80% training data and 20% testing data. Next, the training data is used to create a large sparse matrix by transforming the raw data to a
COO matrix to a CSR matrix and finally to a dense matrix. We
then perform the matrix factorization and multiply the two
matrices together, which gives us our prediction matrix. Next,
I read-in our test data and transform it into a sparse matrix as
well. I have chosen the number of latent features (k) to be 2.
Now using the test sparse matrix, the model can be predicted on the testing data to
obtain a Root Mean Square Error (RMSE) of 0.89821. This process has a long processing
time due to such large data being present.

The next model used is a random forest model. This
model will create decision trees based off the training data
which will classify the data and create a model which will be used to
predict on the test data. Using the randomForest package in R,
this model was created. The number of trees is set to 500. After creating this model and
predicting on the testing data, a RMSE of 1.10301 is obtained.
The below plot shows the relationship between the actual values
and predicted values of the target variable.

![image](https://user-images.githubusercontent.com/50085554/236968702-a8bd272c-b1c8-41ea-8ccb-66c89fccb898.png)

#Results and Conclusion
Now, it can be determined that the matrix factorization model is a
better model than the random forest model since the RMSE for
the matrix factorization model is less than the RMSE for the
random forest model. Therefore, the matrix factorization model will be used as the final model. 
![image](https://user-images.githubusercontent.com/50085554/236968973-5fda4269-bd69-4040-82e9-f7eef71aa9cf.png)

The above image shows RMSE scores over time for models
built by others. It can be noted that the matrix factorization model fits
in well here. My model compares well with most of the models, and performed better than any
model made before mid-2020.

