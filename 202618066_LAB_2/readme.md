# Lab 2 – NumPy and Pandas: Vectorized Programming and Data Wrangling

## Project Information

- **Assignment:** Lab 2 – NumPy and Pandas
- **Name:** Abhikarsh Raj
- **Student ID:** 202618066
- **Dataset:** Titanic Dataset (`train.csv`)
- **Language:** Python
- **Environment:** Jupyter Notebook

---

## 1. Project Overview

The objective of this assignment is to practice vectorized programming using NumPy and perform basic data manipulation, cleaning, aggregation, and exploratory data analysis using Pandas.

The assignment is divided into two major parts:

**Part A – Vectorized Programming with NumPy**

This section focuses on creating and manipulating arrays, performing statistical operations, implementing vectorized arithmetic, applying linear algebra operations, and generating normally distributed random data.

**Part B – Data Wrangling with Pandas**

This section uses the Titanic dataset to practise data inspection, filtering, querying, grouping, aggregation, handling missing values, detecting outliers, feature engineering, pivot tables, and visualization.

---

## 2. Dataset Description

The Titanic dataset contains information about passengers aboard the Titanic, including their demographic characteristics, ticket details, family relationships, and survival status.

The dataset used in this assignment is `train.csv`.

### Important Dataset Attributes

| Column | Description |
|---|---|
| PassengerId | Unique passenger identifier |
| Survived | Survival status (0 = No, 1 = Yes) |
| Pclass | Passenger class (1, 2, 3) |
| Name | Passenger name |
| Sex | Passenger gender |
| Age | Passenger age |
| SibSp | Number of siblings or spouses aboard |
| Parch | Number of parents or children aboard |
| Ticket | Ticket identification number |
| Fare | Passenger ticket fare |
| Cabin | Cabin identification |
| Embarked | Port of embarkation (C, Q, S) |

---

## 3. Project Tasks and Implementation

### Part A – Vectorized Programming with NumPy

### Task 1: Arrays, Statistics, and Indexing

The following NumPy operations are implemented:

- Generation of 100 random integers with a fixed random seed.
- Calculation of minimum, maximum, median, mean, and standard deviation.
- Creation of arrays using `np.arange()`.
- Implementation of `np.zeros()` and `np.ones()`.
- Creation of evenly spaced values using `np.linspace()`.
- Creation and manipulation of 2D and 3D arrays.
- Array indexing, slicing, shape, and dimensionality.
- Reshaping arrays into matrices and flattening them back into one-dimensional arrays.

### Task 2: Vectorized Arithmetic and Linear Algebra

The following mathematical operations are performed:

- Matrix addition.
- Element-wise multiplication.
- Matrix multiplication using the `@` operator.
- Matrix transpose.
- Determinant calculation.
- Matrix inverse.
- Verification of matrix inversion using `np.allclose()`.

Vectorized NumPy operations are used instead of explicit Python loops.

### Task 3: Normal Distribution and Histogram

- Generation of at least 1,000 random observations from a normal distribution.
- Selection of a specified mean and standard deviation.
- Calculation of sample mean and sample standard deviation.
- Comparison of sample statistics with the theoretical distribution parameters.
- Visualization using a histogram.

---

### Part B – Data Wrangling with Pandas

### Task 4: Load and Inspect Data

The Titanic dataset is loaded using Pandas.

The following inspection methods are implemented:

- `head()`
- `tail()`
- `shape`
- `columns`
- `info()`
- `describe()`

Row and column selection is performed using:

- `loc[]` – Label-based selection.
- `iloc[]` – Integer-position-based selection.

### Task 5: Filtering and Querying

Boolean indexing and Pandas `query()` are used to answer the following questions:

1. Count of male passengers older than 50.
2. Number of female first-class passengers and their survival percentage.
3. Number of passengers aged 20–40, with fares above the overall median, who survived.
4. Number of passengers travelling alone, younger than 30, who did not survive.
5. Number of passengers who embarked at Southampton, travelled in second or third class, and paid above the Southampton median fare.

### Task 6: Grouping and Aggregation

The `groupby()` method is used to analyse passenger survival and other characteristics.

The following aggregations are performed:

- Survival rate by Sex.
- Survival rate by Passenger Class.
- Average Age and Fare by Passenger Class.
- Passenger count and survival rate by Sex and Passenger Class.
- Passenger count, average Fare, and survival rate by Embarkation Port.

### Task 7: Missing Values and Fare Outliers

Missing-value analysis includes:

- Missing-value count for every column.
- Missing-value percentage for every column.
- Visualization of missing values using a bar chart.
- Imputation of missing Age values using the mean.
- Comparison of missing values before and after imputation.
- Comparison of mean, median, mode, and random-value imputation techniques.

Fare outlier detection is performed using the Interquartile Range (IQR) method.

The following are calculated:

- First Quartile (Q1).
- Third Quartile (Q3).
- Interquartile Range (IQR).
- Lower and upper outlier bounds using 1.5 × IQR.
- Total number of detected Fare outliers.

### Task 8: Feature Engineering and Pivot Tables

Two additional features are created:

**FamilySize**

Calculated using:

`FamilySize = SibSp + Parch + 1`

**IsAlone**

A binary feature indicating whether a passenger travelled alone:

- 1 = Travelling alone.
- 0 = Travelling with family.

A pivot table is generated using:

- Rows: Sex
- Columns: Pclass
- Values: Mean Survived

The highest and lowest survival-rate groups are identified from the resulting pivot table.

### Task 9: Visualization and Observations

The following visualizations are generated:

1. Correlation heatmap of relevant numerical variables.
2. Survival rate by Sex.
3. Age versus Fare, differentiated by survival status.

These visualizations are used to identify relationships between passenger characteristics and survival outcomes.

---

## 4. Key Observations and Findings

The following observations are investigated through statistical analysis and visualizations:

1. **Gender and Survival:** Female passengers had a higher survival rate than male passengers, indicating a substantial difference in survival outcomes between the two groups.

2. **Passenger Class:** First-class passengers generally experienced higher survival rates than second- and third-class passengers.

3. **Age and Survival:** Age provides useful information for understanding survival patterns, although survival outcomes vary across different age groups.

4. **Fare and Survival:** Passengers paying higher fares may show different survival patterns compared with passengers paying lower fares. Fare is also associated with passenger class.

5. **Family Size:** FamilySize and IsAlone help investigate whether travelling alone or with family was associated with different survival outcomes.

6. **Missing Values:** Age and Cabin contain missing observations, requiring appropriate data-cleaning and imputation techniques.

7. **Fare Outliers:** The IQR method identifies unusually high or low fare observations that may influence statistical summaries and visualizations.

*Note: These are the analytical patterns to investigate. The final conclusions should be supported by the actual numerical results generated by the notebook.*

---

## 5. Technologies and Libraries Used

| Library | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Vectorized operations and linear algebra |
| Pandas | Data manipulation and wrangling |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Interactive coding and analysis |

---

## 6. Repository Structure

```text
Lab-2-NumPy-and-Pandas/
│
├── data/
│   ├── train.csv
│   └── titanic_cleaned.csv
│
├── notebooks/
│   └── Lab_2_NumPy_Pandas.ipynb
│
├── visualizations/
│   ├── missing_values.png
│   ├── normal_distribution.png
│   ├── correlation_heatmap.png
│   ├── survival_by_sex.png
│   └── age_vs_fare.png
│
├── requirements.txt
└── README.md
```

---

## 7. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Navigate to the Project Directory

```bash
cd Lab-2-NumPy-and-Pandas
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

### Step 4: Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `Lab_2_NumPy_Pandas.ipynb` and execute the cells sequentially.

---

## 8. Conclusion

This assignment demonstrates the practical application of NumPy and Pandas in numerical computing and data analysis.

The NumPy section develops an understanding of array manipulation, vectorized arithmetic, statistical operations, and linear algebra.

The Pandas section applies data-wrangling techniques to the Titanic dataset, including data inspection, conditional filtering, aggregation, missing-value treatment, outlier detection, and feature engineering.

Finally, visualizations and statistical summaries are used to explore relationships between passenger characteristics and survival outcomes.

Overall, the assignment provides practical experience in numerical computation, data preprocessing, and exploratory data analysis using Python.

---

## 9. References

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Titanic Dataset – Kaggle](https://www.kaggle.com/competitions/titanic/data)

**Submitted as part of the NumPy and Pandas Lab Assignment.**