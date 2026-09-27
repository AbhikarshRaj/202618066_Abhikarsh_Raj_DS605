# Lab 5: Machine Learning with Scikit-learn and From Scratch

**Course:** DS605 — Fundamentals of Machine Learning  
**Name:** Abhikarsh Raj  
**Student ID:** 202618066  

## Objective

Build Linear Regression and Logistic Regression models using Scikit-learn, recreate the workflow using NumPy and Pandas, and compare predictive performance and execution time.

The manual Logistic Regression implementation is then optimized using Newton's method.

## Dataset

The project uses the UCI Productivity Prediction of Garment Employees dataset.

- **Observations:** 1,197
- **Original columns:** 15
- **Training samples:** 957
- **Test samples:** 240
- **Input features after preprocessing:** 32

The dataset contains production information such as department, team, targeted productivity, overtime, incentives, work in progress, and number of workers.

### Prediction Tasks

| Task | Model | Target |
|---|---|---|
| Regression | Linear Regression | `actual_productivity` |
| Classification | Logistic Regression | Whether actual productivity meets or exceeds targeted productivity |

The classification target is defined as:

```python
MeetsTarget = (
    actual_productivity >= targeted_productivity
).astype(int)
```

`actual_productivity` is excluded from the input features for both models.

## Data Preparation

The following steps were applied:

1. Inspected data types, missing values, duplicate rows, and target values.
2. Removed extra spaces from department labels.
3. Extracted month from the date and removed the original date column.
4. Treated team identifiers as categorical features.
5. Created a fixed 80:20 train-test split using `random_state=42`, stratified by the classification target.
6. Filled missing numerical values using training-set medians.
7. Standardized numerical features using training means and standard deviations.
8. One-hot encoded categorical features, dropping the first category.

The processed inputs contain:

- **9 numerical features**
- **23 encoded categorical features**
- **32 features in total**, excluding the intercept

All learned preprocessing statistics and categories come from the training data only.

### Missing Work-in-Progress Values

The `wip` column contains 506 missing values, approximately 42.27% of the dataset. These missing values occur in the finishing department.

For the baseline workflow, missing values were filled using the training median of **1038.0**. This is a modeling choice and does not imply that the actual work in progress for these observations was 1038.

## Part A: Scikit-learn Implementation

Preprocessing uses:

- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `ColumnTransformer`

Models:

- `LinearRegression`
- `LogisticRegression(max_iter=2000)`

Training and prediction times are measured separately using `perf_counter`.

Regression is evaluated using MAE, RMSE, and R². Classification is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

## Part B: From-Scratch Implementation

The manual workflow uses NumPy and Pandas for preprocessing, model calculations, predictions, and evaluation metrics.

The same training and test samples are reused from Part A. No Scikit-learn preprocessing, model, metric, or splitting functions are called in the manual section.

### Manual Preprocessing

- Training-median imputation
- Standardization using training statistics and `ddof=0`
- One-hot encoding using categories learned from training data
- Combination of numerical and encoded categorical matrices

### Manual Linear Regression

Linear Regression is implemented using NumPy's least-squares solver:

```python
np.linalg.lstsq(X_train_bias, y_train, rcond=None)
```

A column of ones accounts for the intercept. Predictions and regression metrics are calculated manually.

### Manual Logistic Regression

The baseline implementation includes:

- A numerically stable sigmoid function
- Probability prediction
- Classification using a probability threshold of 0.5
- Vectorized batch gradient descent
- L2 regularization, excluding the intercept
- Manual confusion-matrix counts and classification metrics

The baseline uses a learning rate of **0.1**, a maximum of **20,000 iterations**, and a gradient tolerance of **1e-6**.

It reached the iteration limit without satisfying the convergence criterion.

## Part C: Optimization Using Newton's Method

The manual Logistic Regression optimizer was changed from gradient descent to Newton's method with backtracking line search.

Newton's method uses:

- The **gradient** to describe the slope of the objective.
- The **Hessian** to describe its curvature.
- A linear-system solve to calculate the update direction.
- Backtracking to ensure sufficient reduction in the training objective.

The implementation solves for the Newton direction using:

```python
direction = np.linalg.solve(hessian, gradient)
```

With 32 input features and an intercept, the Hessian is a manageable **33 × 33** matrix.

The data split, input features, regularization, classification threshold, and gradient tolerance remained unchanged. Backtracking decisions used only the training objective.

## Results

The following values are recorded execution results. Timings may change when the notebook is rerun.

### Linear Regression

| Implementation | MAE | RMSE | R² | Training time (ms) | Prediction time (ms) |
|---|---:|---:|---:|---:|---:|
| Scikit-learn | 0.108565 | 0.142306 | 0.302654 | 17.2974 | 0.9759 |
| Manual NumPy — Least Squares | 0.108565 | 0.142306 | 0.302654 | 16.9276 | 0.2119 |

The manual implementation matched the Scikit-learn metrics to six decimal places. Training times were similar in this run.

### Logistic Regression

Precision, recall, and F1-score use **“meets target” as the positive class**.

| Implementation | Accuracy | Precision | Recall | F1-score | Training time (ms) | Prediction time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Scikit-learn | 0.687500 | 0.763158 | 0.828571 | 0.794521 | 19.2523 | 0.4301 |
| Manual NumPy — Gradient Descent | 0.687500 | 0.763158 | 0.828571 | 0.794521 | 1786.8808 | 0.3550 |
| Manual NumPy — Newton | 0.687500 | 0.763158 | 0.828571 | 0.794521 | 10.5305 | 0.2798 |

All three implementations produced the same test classification metrics and confusion matrix.

### Confusion Matrix

| Actual / Predicted | Misses target (0) | Meets target (1) |
|---|---:|---:|
| Misses target (0) | 20 | 45 |
| Meets target (1) | 30 | 145 |

### Optimization Outcome

| Measure | Manual Gradient Descent | Manual Newton |
|---|---:|---:|
| Iterations / updates | 20,000 | 6 |
| Convergence criterion satisfied | No | Yes |
| Training time (ms) | 1786.8808 | 10.5305 |

Newton's method achieved a final maximum absolute gradient of approximately **1.693e-9**, below the specified tolerance of **1e-6**.

The recorded training speedup over manual gradient descent was approximately **169.69×**, corresponding to a **99.41% reduction in training time**.

This optimization improved convergence and computational efficiency. It did not improve the reported test classification metrics.

## Visualizations

The notebook includes:

- Actual versus predicted productivity
- Residuals versus predicted productivity
- Residual distribution
- Classification confusion-matrix heatmap

The regression plots show a tendency to overpredict low productivity and underpredict high productivity. The residual distribution also contains some large negative errors, indicating substantial overpredictions for certain observations.

## Key Observations and Limitations

### Predictive Performance

- Manual Linear Regression matched Scikit-learn's regression metrics.
- Both manual Logistic Regression implementations matched Scikit-learn's classification metrics.
- Identical rounded metrics do not establish identical coefficients or predicted probabilities.

### Classification Limitation

The classifier correctly identified only **20 of 65 cases that missed their target**, giving class-0 recall of approximately **30.77%**.

Always predicting “meets target” would achieve **72.92% accuracy** on this test set, compared with the model's **68.75%**. Therefore, the positive-class recall and F1-score should be interpreted alongside the confusion matrix and class distribution.

### Runtime Interpretation

- NumPy uses optimized numerical routines for matrix calculations.
- Scikit-learn performs additional validation and general-purpose handling.
- Newton's method required fewer updates, making it effective for this relatively small feature set.
- Hessian-based methods can become expensive when the number of features is very large.
- These are single-run timings, not repeated benchmark estimates.
- Timings exclude preprocessing and metric calculation.
- Lower runtime in one run does not establish that an implementation is always faster.

## Project Files

| File / Folder | Purpose |
|---|---|
| `202618066_Lab_05.ipynb` | Analysis, implementations, evaluation, and optimization |
| `Data/garments_worker_productivity.csv` | Original dataset |
| `split_indices.npz` | Split positions generated by Part A |
| `README.md` | Project description and results |

## How to Run

1. Place the dataset at:

   ```text
   Data/garments_worker_productivity.csv
   ```

2. Install the required packages:

   ```bash
   pip install numpy pandas scikit-learn matplotlib jupyter
   ```

3. Open the notebook in Jupyter or VS Code:

   ```text
   202618066_Lab_05.ipynb
   ```

4. Restart the kernel and run all cells from top to bottom.

Part A generates `split_indices.npz`. Run Part A before Part B because the notebook reuses its fixed split and raw training/test variables.

The manual computation sections use NumPy and Pandas; Python's timing utility is used only to measure execution time.

## Conclusion

The project reproduces Scikit-learn's reported predictive metrics using manual NumPy/Pandas implementations.

Replacing batch gradient descent with Newton's method allowed manual Logistic Regression to converge in six updates and substantially reduced its recorded training time while preserving test classification performance.