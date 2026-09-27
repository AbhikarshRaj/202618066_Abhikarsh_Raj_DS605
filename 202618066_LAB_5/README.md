# Lab 5: Machine Learning with Scikit-learn and From Scratch

**Course:** DS605 — Fundamentals of Machine Learning  
**Name:** Abhikarsh Raj  
**Student ID:** 202618066

## Objective

Implement Linear Regression and Logistic Regression using Scikit-learn, recreate the workflow using NumPy and Pandas, and compare predictive performance and execution time.

Optimize the manual implementation using a justified method while maintaining a fair comparison.

## Dataset and Targets

The project uses the UCI Productivity Prediction of Garment Employees dataset.

| Property | Value |
|---|---:|
| Observations | 1,197 |
| Original columns | 15 |
| Training observations | 957 |
| Test observations | 240 |
| Input features before encoding | 14 |
| Input features after preprocessing | 32 |

### Regression

Predict `actual_productivity` using Linear Regression.

### Classification

Predict whether actual productivity meets or exceeds targeted productivity:

```python
MeetsTarget = (
    actual_productivity >= targeted_productivity
).astype(int)
```

- **Class 0:** Misses target
- **Class 1:** Meets target

`actual_productivity` is excluded from the input features for both tasks.

## Data Preparation

The workflow includes:

1. Inspecting data types, missing values, duplicate rows, and unusual values.
2. Removing extra spaces from department labels.
3. Extracting month from the date and removing the original date string.
4. Treating team identifiers as categorical features.
5. Creating one fixed 80:20 train-test split with `random_state=42`, stratified by the classification target.
6. Filling missing numerical values using training medians.
7. Standardizing numerical features using training means and standard deviations.
8. One-hot encoding categorical features and dropping the first category.

The final feature matrix contains:

- **9 numerical features**
- **23 encoded categorical features**
- **32 features in total**, excluding the intercept

Preprocessing statistics and categories are learned from training data only and applied unchanged to test data.

### Missing Values

The `wip` column contains 506 missing values, approximately 42.27% of the dataset. These occur in the finishing department.

Missing values were filled using the training median of **1038.0**. This is a baseline imputation choice, not a claim about the actual work in progress for those observations.

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

## Part B: Manual Implementation

The manual preprocessing, model calculations, predictions, and evaluation metrics use NumPy and Pandas.

The same raw training and test samples are reused from Part A. Scikit-learn preprocessing, model, metric, and splitting functions are not called in the manual section.

### Manual Preprocessing

- Median imputation using training statistics
- Standardization using `ddof=0`
- One-hot encoding using training categories
- Combination of numerical and categorical feature matrices

### Linear Regression: Least Squares

The first implementation uses NumPy's numerical least-squares solver:

```python
weights = np.linalg.lstsq(
    X_train_bias,
    y_train,
    rcond=None
)[0]
```

The design matrix includes a column of ones for the intercept.

### Linear Regression: Normal Equations with Pseudoinverse

The second implementation explicitly calculates the matrix products:

```python
XtX = X_train_bias.T @ X_train_bias
Xty = X_train_bias.T @ y_train

weights = np.linalg.pinv(XtX) @ Xty
```

This computes:

\[
\hat{\beta} = (X^\top X)^+X^\top y
\]

where the superscript \(+\) denotes the pseudoinverse.

The pseudoinverse accommodates possible dependent columns. However, forming \(X^\top X\) can worsen numerical conditioning, so the direct least-squares approach is generally preferable for numerical stability.

Both implementations minimize the sum of squared errors. Predictions and evaluation metrics are calculated manually.

### Logistic Regression: Gradient Descent

The baseline implementation includes:

- A numerically stable sigmoid function
- Probability prediction and a classification threshold of 0.5
- Vectorized batch gradient descent
- L2 regularization, excluding the intercept
- Manual evaluation metrics and confusion-matrix counts

Settings:

| Parameter | Value |
|---|---:|
| Learning rate | 0.1 |
| Maximum iterations | 20,000 |
| Gradient tolerance | 1e-6 |
| Regularization parameter C | 1.0 |

The objective is average binary log loss plus an L2 penalty:

\[
J(\beta)=
-\frac{1}{n}\sum_{i=1}^{n}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
+\frac{1}{2Cn}\sum_{j=1}^{d}\beta_j^2
\]

The intercept is not penalized.

## Part C: Optimization with Newton's Method

The manual Logistic Regression optimizer was changed from batch gradient descent to Newton's method with backtracking line search.

Newton's method uses the gradient and Hessian to calculate an update direction:

```python
direction = np.linalg.solve(hessian, gradient)
```

Backtracking reduces the step size when necessary to obtain sufficient reduction in the training objective.

With 32 features and an intercept, the Hessian is only **33 × 33**, making this approach practical for the dataset.

The optimization retained the same:

- Training and test samples
- Input features
- Regularized objective
- Classification threshold
- Gradient tolerance

Backtracking decisions used training loss only.

## Evaluation Results

The following are recorded single-run measurements from the final reported comparison. Execution times may change when the notebook is rerun.

### Linear Regression

| Implementation | MAE | RMSE | R² | Training time (ms) | Prediction time (ms) |
|---|---:|---:|---:|---:|---:|
| Scikit-learn | 0.108565 | 0.142306 | 0.302654 | 13.755 | 0.628 |
| Manual NumPy — Least Squares | 0.108565 | 0.142306 | 0.302654 | 7.149 | 0.106 |
| Manual NumPy — Normal Equations (Pseudoinverse) | 0.108565 | 0.142306 | 0.302654 | 8.876 | 0.119 |

All three implementations matched on the reported regression metrics to six decimal places.

In this run, manual least squares recorded approximately **1.92× faster training** than Scikit-learn, while the normal-equation implementation recorded approximately **1.55× faster training**.

### Logistic Regression

Precision, recall, and F1-score treat **“meets target” as the positive class**.

| Implementation | Accuracy | Precision | Recall | F1-score | Training time (ms) | Prediction time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Scikit-learn | 0.687500 | 0.763158 | 0.828571 | 0.794521 | 21.126 | 0.472 |
| Manual NumPy — Gradient Descent | 0.687500 | 0.763158 | 0.828571 | 0.794521 | 1009.125 | 0.141 |
| Manual NumPy — Newton | 0.687500 | 0.763158 | 0.828571 | 0.794521 | 3.822 | 0.135 |

All three implementations produced the same reported classification metrics and confusion matrix.

### Confusion Matrix

| Actual / Predicted | Misses target (0) | Meets target (1) |
|---|---:|---:|
| Misses target (0) | 20 | 45 |
| Meets target (1) | 30 | 145 |

## Optimization Outcome

The earlier convergence output showed:

- Gradient descent reached 20,000 iterations without satisfying the gradient tolerance.
- Newton's method converged in 6 updates.
- Newton's final maximum absolute gradient was approximately **1.693e-9**, below the tolerance of **1e-6**.

Using the latest recorded timings:

| Comparison | Training speedup |
|---|---:|
| Newton vs manual gradient descent | Approximately 264× |
| Newton vs Scikit-learn | Approximately 5.53× |

Manual Logistic Regression training decreased from **1009.125 ms to 3.822 ms**, approximately a **99.62% reduction**.

Newton's method requires more work per update but needed far fewer updates on this dataset. The improvement was in computational efficiency and convergence, not in the reported test classification scores.

## Visualizations

The notebook includes:

- Actual versus predicted productivity
- Residuals versus predicted productivity
- Residual distribution
- Confusion-matrix heatmap

These plots help identify prediction errors and classification weaknesses that aggregate metrics alone can hide.

## Interpretation and Limitations

### Predictive Performance

The implementations successfully match Scikit-learn's reported metrics, but predictive performance remains limited.

- Regression R² is approximately **0.303**.
- The regression plots show a tendency to overpredict low productivity and underpredict high productivity.
- Classification accuracy is **68.75%**.
- Always predicting “meets target” would achieve **72.92% accuracy** on this test set.
- Only **20 of 65 cases that missed their target** were correctly identified, giving class-0 recall of approximately **30.77%**.

Therefore, matching the library implementation demonstrates implementation consistency, not a highly accurate predictive model.

Matching rounded metrics also does not establish identical coefficients or probabilities.

### Runtime

- NumPy uses optimized numerical routines for matrix calculations.
- Scikit-learn includes input validation and general-purpose handling.
- Newton's method benefits from the relatively small number of features.
- Hessian-based methods can become expensive for high-dimensional datasets.
- All reported timings are single-run measurements.
- Training and prediction timings exclude preprocessing and evaluation metrics.
- System load, initialization, and execution order can affect measured runtime.

The observed speedups should not be interpreted as guaranteed advantages across all runs or datasets.

## Project Files

| File | Purpose |
|---|---|
| `202618066_Lab_05.ipynb` | Analysis, model implementations, comparisons, and optimization |
| `Data/garments_worker_productivity.csv` | Original dataset |
| `split_indices.npz` | Fixed split positions generated by Part A |
| `README.md` | Project description and results |

## How to Run

1. Place the dataset at:

   ```text
   Data/garments_worker_productivity.csv
   ```

2. Install dependencies:

   ```bash
   pip install numpy pandas scikit-learn matplotlib jupyter
   ```

3. Open `202618066_Lab_05.ipynb` in Jupyter or VS Code.

4. Restart the kernel and run all cells from top to bottom.

Part A creates the fixed split. Part B reuses the raw training/test samples and independently performs preprocessing.

Each manual Linear Regression cell saves its own results immediately after execution so that later calculations do not overwrite the comparison values.

## Conclusion

The project implements and compares Scikit-learn and manual approaches for regression and classification.

Both manual Linear Regression methods match Scikit-learn's reported regression metrics. Manual Logistic Regression also matches the classification metrics, while replacing gradient descent with Newton's method substantially improves training speed and convergence.

The main achievement is a consistent manual implementation with a demonstrated computational optimization. Predictive limitations and runtime variability are reported alongside the results.