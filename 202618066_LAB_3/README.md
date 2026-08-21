# Scikit-learn: Data Preprocessing and Model Performance Evaluation

**Name:** Abhikarsh Raj
**Student ID:** 202618066

## Assignment Overview

This assignment explores the impact of different data preprocessing techniques on the performance of classification models using scikit-learn. The dataset used is the Hotel Booking Demand dataset, and the analysis compares model performance across multiple scaling and imputation strategies.

## Dataset

**Source:** [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

The dataset contains booking information for a city hotel and a resort hotel, including details such as booking dates, length of stay, number of guests, and whether a booking was canceled. It is commonly used for classification tasks predicting booking cancellations.

## Preprocessing Techniques Used

1. **KNN Imputer** - Used to handle missing values in the dataset by imputing them based on the values of the nearest neighbors, rather than using simple mean or median imputation.
2. **Standard Scaler** - Standardizes features by removing the mean and scaling to unit variance, resulting in a distribution centered around 0.
3. **Min-Max Normalization** - Rescales features to a fixed range, typically between 0 and 1, preserving the shape of the original distribution.

## Models Evaluated

- Logistic Regression
- Decision Tree Classifier

Each model was trained and evaluated under different preprocessing conditions (no scaling, Standard Scaler, and Min-Max Scaler) to assess how preprocessing choices influence performance.

## Evaluation Metrics

- Testing Accuracy
- F1-Score
- Train-Test Performance Gap (used to assess overfitting)

## Final Observations

1. Decision Tree combined with Standard Scaler gives the best overall result based on testing accuracy and F1-score.
2. Scaling has a very small effect on Logistic Regression. However, when comparing the two scaling methods, Standard Scaler performs slightly better than Min-Max Scaler based on the calculated F1-score.
3. Scaling has a negligible effect on Decision Tree, as the F1-score does not vary significantly after applying scaling.
4. Logistic Regression does not show much overfitting, as evidenced by the very minimal train-test gap compared to Decision Tree.
5. Decision Tree shows overfitting, which can be derived from its large train-test gap.

## Conclusion

While Decision Tree achieves the highest raw performance scores, its tendency to overfit makes it less reliable for generalization compared to Logistic Regression. Preprocessing choices such as scaling have a more noticeable impact on Logistic Regression than on Decision Tree, since Decision Tree-based models are inherently scale-invariant. Standard Scaler was found to be the more effective scaling method overall across both models.

## Tools and Libraries

- Python
- scikit-learn
- pandas
- NumPy