# NYC Airbnb Price Prediction

**Course:** DS605 – Fundamentals of Machine Learning  
**Lab:** Lab Assignment 4 – End-to-End Machine Learning Project  
**Student ID:** 202618066  

**Dataset:** New York City Airbnb Open Data (`AB_NYC_2019.csv`)

**Live Streamlit App:** https://202618066abhikarshrajds605-8p7n65e5vwwn9sdpopm7b2.streamlit.app/  
**GitHub Repository:** https://github.com/AbhikarshRaj/202618066_Abhikarsh_Raj_DS605/tree/main/202618066_LAB_4

---
<p align="center">
  <a href="https://202618066abhikarshrajds605-8p7n65e5vwwn9sdpopm7b2.streamlit.app/">
    <img src="https://img.shields.io/badge/LIVE%20DEMO-OPEN%20STREAMLIT%20APP-success?style=for-the-badge" alt="Open Streamlit App">
  </a>
</p>

<p align="center">
  <strong>Click the button above to try the deployed Airbnb Price Prediction application.</strong>
</p>

---

# NYC Airbnb Price Prediction

**Course:** DS605 – Fundamentals of Machine Learning  
**Lab:** Lab Assignment 4 – End-to-End Machine Learning Project  
**Student ID:** 202618066  

**Dataset:** New York City Airbnb Open Data (`AB_NYC_2019.csv`)

### Live Application

**[Launch the NYC Airbnb Price Predictor](https://202618066abhikarshrajds605-8p7n65e5vwwn9sdpopm7b2.streamlit.app/)**

---

## Project Overview

This project develops an end-to-end machine learning system for predicting the nightly price of Airbnb listings in New York City.

# 1. Project Overview

This project develops an end-to-end machine learning system for predicting the nightly price of Airbnb listings in New York City.

The project covers the complete machine learning workflow:

- Dataset understanding
- Exploratory Data Analysis (EDA)
- Missing-value analysis
- Data cleaning
- Outlier analysis
- Feature engineering
- Feature selection
- Categorical encoding
- Geographic clustering
- Target encoding
- Model comparison
- Overfitting and underfitting analysis
- Hyperparameter tuning
- Final model evaluation
- Model serialization
- Streamlit application development

The final system accepts Airbnb listing information and returns an estimated nightly price.

---

# 2. Problem Statement

The objective is to estimate the nightly price of an Airbnb listing using information such as:

- Location
- Room type
- Minimum stay requirements
- Review activity
- Host listing activity
- Availability
- Geographic patterns
- Neighbourhood pricing patterns

This is a **regression problem** because the target variable, `price`, is continuous.

The main evaluation metrics used are:

- **MAE – Mean Absolute Error**
- **RMSE – Root Mean Squared Error**
- **R² – Coefficient of Determination**

Lower MAE and RMSE are better, while higher R² is better.

---

# 3. Dataset

The project uses the Kaggle **New York City Airbnb Open Data** dataset:

`AB_NYC_2019.csv`

The original dataset contains:

- **48,895 listings**
- **16 columns**

Original columns include:

```text
id
name
host_id
host_name
neighbourhood_group
neighbourhood
latitude
longitude
room_type
price
minimum_nights
number_of_reviews
last_review
reviews_per_month
calculated_host_listings_count
availability_365
```

The prediction target is:

```text
price
```

---

# 4. Data Preservation Strategy

The original dataset was kept unchanged.

All cleaning and feature engineering were performed on a separate working DataFrame / cleaned dataset.

This allowed the original Kaggle data to remain available for reference while model development was performed independently.

---

# 5. Removed Identifier Features

The following columns were removed:

```text
id
host_id
host_name
```

### Why were they removed?

`id` and `host_id` are identifiers rather than meaningful numerical measurements.

For example:

```text
host_id = 2787
```

is not meaningfully smaller than:

```text
host_id = 7192
```

Using identifiers could encourage the model to memorize individual listings or hosts rather than learn generalizable pricing patterns.

`host_name` was also removed because a person's name should not determine Airbnb price and would not provide stable predictive information.

---

# 6. Missing-Value Analysis

Missing values were investigated before performing imputation.

This avoided blindly replacing values without understanding why they were missing.

---

## 6.1 Missing `reviews_per_month`

There were originally:

```text
10,052
```

missing values in:

```text
reviews_per_month
```

Investigation showed that **all missing values occurred when**:

```text
number_of_reviews = 0
```

Therefore the missingness was meaningful rather than random.

For a listing with no reviews:

```text
reviews_per_month = 0
```

is logically appropriate.

The missing values were therefore replaced with zero.

---

## 6.2 Missing `last_review`

Missing `last_review` values also corresponded entirely to listings with:

```text
number_of_reviews = 0
```

A listing with no reviews naturally cannot have a last review date.

Instead of filling the missing dates with an artificial value, two features were engineered:

```text
has_review
days_since_last_review
```

`has_review` was defined as:

```text
1 = listing has at least one review
0 = listing has never been reviewed
```

---

# 7. Review Recency Feature

The original `last_review` column was converted from string format to a datetime variable.

The earliest review in the dataset was:

```text
2011-03-28
```

The latest review date was:

```text
2019-07-08
```

The latest historical review date was selected as the common reference date:

```text
2019-07-08
```

The engineered variable was:

```text
days_since_last_review
```

calculated as:

```text
2019-07-08 - last_review
```

Example:

```text
Last review: 2019-07-05
Days since last review: 3
```

Listings with no reviews were represented as:

```text
has_review = 0
days_since_last_review = -1
```

After extracting the useful information, the original `last_review` column was removed.

---

# 8. Missing Listing Names

Only a very small number of `name` values were missing.

Instead of deleting those rows, missing listing names were replaced with an empty string.

The listing title was initially retained because words such as:

```text
luxury
penthouse
townhouse
studio
room
private
```

could potentially contain price information.

Text modelling was later tested experimentally.

---

# 9. Invalid Price Observations

The target variable contained:

```text
11 listings with price = 0
```

A zero nightly price is not meaningful for the commercial Airbnb price-prediction objective.

These 11 observations were removed.

After removing them:

```text
Final cleaned row count before percentile filtering = 48,884
```

---

# 10. Duplicate Analysis

The dataset was checked for duplicate rows.

Result:

```text
Duplicate rows = 0
```

Therefore no duplicate-removal step was necessary.

---

# 11. Price Distribution

The price variable was strongly right-skewed.

Important percentiles were approximately:

| Percentile | Price |
|---|---:|
| 50th | $106 |
| 75th | $175 |
| 90th | $269 |
| 95th | $355 |
| 97th | $450 |
| 99th | $799 |
| 99.5th | $1,000 |
| 99.9th | $3,000 |
| Maximum | $10,000 |

The mean price was much larger than the median because a small number of very expensive properties created a long right tail.

This later became an important modelling issue because RMSE and R² use squared errors and are highly sensitive to extreme values.

---

# 12. Minimum Nights Distribution

`minimum_nights` was also strongly right-skewed.

Important values included approximately:

```text
Median = 3 nights
90th percentile = 28 nights
95th percentile = 30 nights
99th percentile = 45 nights
99.5th percentile = 90 nights
Maximum = 1,250 nights
```

A simple IQR rule was not used to automatically remove these observations because many long-term rental requirements could still represent legitimate listings.

---

# 13. Why IQR Outlier Removal Was Not Used Directly

For price:

```text
Q1 = 69
Q3 = 175
IQR = 106
```

The standard IQR upper fence would be approximately:

```text
$334
```

This would classify every Airbnb above approximately $334 as an outlier.

However, a $400–$500 listing in New York City may be completely legitimate.

Similarly, the IQR rule for `minimum_nights` would incorrectly classify many legitimate 28-day and 30-day minimum-stay listings as outliers.

Therefore:

> An IQR outlier was not automatically treated as an invalid observation.

The extreme price tail was instead investigated experimentally later in the modelling process.

---

# 14. Initial Numerical Correlation Analysis

A Pearson correlation matrix was calculated for the numerical variables.

Most individual numerical variables showed relatively weak linear correlations with price.

Examples included approximately:

```text
longitude                  -0.15
availability_365            0.08
calculated_host_count       0.06
minimum_nights              0.04
number_of_reviews          -0.05
reviews_per_month          -0.05
```

This showed that simple linear correlation alone was insufficient to explain Airbnb price.

However:

> Low Pearson correlation does not mean a feature is useless for nonlinear models.

Tree-based models can still learn nonlinear relationships and interactions between features.

---

# 15. Categorical Price Analysis

Categorical analysis showed much stronger pricing patterns.

---

## 15.1 Room Type

Typical median prices were approximately:

| Room Type | Median Price |
|---|---:|
| Entire home/apt | $160 |
| Private room | $70 |
| Shared room | $45 |

This demonstrated that room type is a major pricing factor.

---

## 15.2 Neighbourhood Group

Median prices were approximately:

| Borough | Median Price |
|---|---:|
| Manhattan | $150 |
| Brooklyn | $90 |
| Queens | $75 |
| Staten Island | $75 |
| Bronx | $65 |

Manhattan clearly had higher typical Airbnb prices.

---

## 15.3 Neighbourhood

Large price differences were also found between individual neighbourhoods.

Examples included:

```text
Midtown
Financial District
West Village
Chelsea
Hell's Kitchen
Upper West Side
East Village
Harlem
Bedford-Stuyvesant
Flatbush
```

This demonstrated that detailed neighbourhood information was substantially more predictive than simple latitude or longitude correlation alone.

---

# 16. Initial Text Feature Experiment

The Airbnb listing `name` column was initially retained.

TF-IDF was used to convert title text into numerical features.

The model detected relationships such as:

```text
text__luxury
text__townhouse
text__private
text__room
```

For example:

```text
"luxury"      → positive relationship with price
"townhouse"   → positive relationship with price
"room"        → negative relationship with price
```

This confirmed that listing titles contained some real pricing information.

---

# 17. Text Privacy Artifact Discovery

The TF-IDF analysis also revealed suspicious features such as:

```text
hidden by
by airbnb
hidden by airbnb
```

Inspection showed listing names such as:

```text
East 7th Street III by (Hidden by Airbnb)
Website hidden by Airbnb
Email hidden by Airbnb
Phone number hidden by Airbnb
```

These were privacy-redaction artifacts rather than real property characteristics.

These artificial phrases were removed while genuine phrases such as:

```text
Hidden Gem
No Hidden Fees
```

were preserved.

---

# 18. Super Bowl Text Investigation

The term:

```text
superbowl
```

appeared in 13 training listings.

These listings had approximately:

```text
Mean price = $1,601.92
Median price = $1,000
```

compared with the overall training values:

```text
Mean price ≈ $153
Median price ≈ $106
```

Therefore the correlation was real in the historical dataset.

However, it was highly event-specific and raised concerns about generalization.

TF-IDF minimum-document-frequency thresholds were investigated to reduce the influence of extremely rare title terms.

---

# 19. Final Decision on Text Features

Although title text contained useful individual signals, TF-IDF produced thousands of additional features and did not improve final validation performance enough to justify the added complexity.

Therefore:

```text
TF-IDF text features were removed from the final model.
```

This decision was based on validation performance rather than assumption.

---

# 20. Data Splitting Strategy

The cleaned data was initially divided into:

```text
80% development/training pool
20% final unseen test data
```

The 80% development portion was then split again:

```text
80% model training
20% validation
```

Relative to the full cleaned dataset, this produced approximately:

```text
64% model training
16% validation
20% final test
```

The final test set remained untouched during model and hyperparameter selection.

The roles were:

```text
Training   → model learning
Validation → model comparison and feature decisions
Test       → final unbiased evaluation
```

---

# 21. Initial Preprocessing Pipeline

The first modelling pipeline contained:

### Numerical features

Numerical features were:

- median-imputed when necessary
- standardized

### Categorical features

The following categorical variables were one-hot encoded:

```text
neighbourhood_group
neighbourhood
room_type
```

### Listing title

The `name` column was initially processed using TF-IDF.

The full encoded training matrix contained more than 3,000 features.

---

# 22. Baseline Model

A Dummy Regressor predicting the median training price was used as the baseline.

Performance was approximately:

| Metric | Baseline |
|---|---:|
| MAE | 83.24 |
| RMSE | 214.66 |
| R² | -0.0515 |

This baseline established the minimum performance that useful machine learning models needed to exceed.

---

# 23. Linear Regression

Linear Regression produced approximately:

| Metric | Result |
|---|---:|
| MAE | 82.52 |
| RMSE | 199.43 |
| R² | 0.0924 |

### Decision

**Rejected as final model.**

### Why?

Although it improved on the dummy baseline, it explained only a small portion of the price variation.

This indicated that Airbnb pricing relationships were too nonlinear and interaction-heavy for a simple linear model.

---

# 24. Ridge Regression

Ridge Regression introduced regularization to reduce instability caused by the large encoded feature space.

Performance was approximately:

| Metric | Result |
|---|---:|
| MAE | 76.79 |
| RMSE | 189.86 |
| R² | 0.1774 |

### Decision

**Rejected as final model.**

### Why?

Ridge improved substantially over standard Linear Regression, showing that regularization was useful.

However, its predictive performance remained too limited compared with later nonlinear models.

---

# 25. Decision Tree Regressor

The unrestricted Decision Tree produced:

### Training

```text
MAE = 0
RMSE = 0
R² = 1.0
```

### Validation

```text
MAE ≈ 80.77
RMSE ≈ 335.97
R² ≈ -1.58
```

### Decision

**Rejected.**

### Why?

This was a textbook example of severe overfitting.

The model perfectly memorized the training data but performed extremely poorly on unseen validation listings.

---

# 26. Random Forest – Raw Target

An untuned Random Forest was tested next.

Validation performance was approximately:

```text
MAE = 61.02
RMSE = 206.35
R² = 0.0284
```

Training performance was approximately:

```text
MAE = 23.02
RMSE = 95.42
R² = 0.8634
```

### Finding

Random Forest substantially improved MAE but still showed severe overfitting and poor raw-price R².

---

# 27. Log-Transformed Target

Because the price distribution was highly right-skewed, the target was transformed using:

```python
log1p(price)
```

Predictions were converted back to dollars using:

```python
expm1(prediction)
```

This produced a major improvement.

Random Forest with log target:

| Metric | Validation |
|---|---:|
| MAE | 52.29 |
| RMSE | 183.86 |
| Raw-price R² | 0.2286 |
| Log-space R² | 0.6346 |
| Log-space MAE | 0.2909 |

### Interpretation

The model captured relative price structure much better in logarithmic space.

The relatively low raw-price R² remained influenced by extremely expensive listings.

---

# 28. Random Forest Hyperparameter Tuning

`RandomizedSearchCV` was used to tune parameters including:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

One of the best configurations was approximately:

```text
n_estimators = 200
min_samples_split = 2
min_samples_leaf = 1
max_features = 0.7
max_depth = None
```

The tuned model achieved approximately:

### Training

```text
MAE = 24.79
RMSE = 167.05
R² = 0.5814
```

### Validation

```text
MAE = 52.00
RMSE = 183.15
R² = 0.2345
```

Log-space validation R²:

```text
0.6412
```

Cross-validation MAE was approximately:

```text
54.39
```

---

# 29. Why Random Forest Was Not Selected

Random Forest was an important improvement over the simpler models.

However:

- it continued to show a train-validation performance gap
- raw-price R² remained low
- additional Random Forest tuning produced diminishing returns
- boosting models offered stronger potential for sequential error correction

Therefore Random Forest was not selected as the final algorithm.

---

# 30. Feature Engineering Iteration for Boosting

A new feature-engineering strategy was developed for LightGBM.

The final retained engineered features included:

```text
geo_cluster
neighbourhood_target_enc
room_neigh_target_enc
neighbourhood_listing_count
```

---

# 31. Geographic Clustering

KMeans clustering was applied to:

```text
latitude
longitude
```

using:

```text
25 geographic clusters
```

This created:

```text
geo_cluster
```

The goal was to capture local geographic pricing regions that may not be represented well by latitude and longitude independently.

---

# 32. Smoothed Neighbourhood Target Encoding

A feature called:

```text
neighbourhood_target_enc
```

was created.

It represents historical neighbourhood price information.

To reduce overfitting for neighbourhoods with few observations, neighbourhood means were smoothed toward the global training mean.

Conceptually:

```text
many neighbourhood observations
        ↓
more trust in neighbourhood mean

few neighbourhood observations
        ↓
more shrinkage toward global mean
```

---

# 33. Room Type × Neighbourhood Group Target Encoding

Another engineered feature was:

```text
room_neigh_target_enc
```

based on:

```text
room_type × neighbourhood_group
```

This captures interactions such as:

```text
Entire home in Manhattan
Private room in Manhattan
Entire home in Brooklyn
Private room in Queens
```

Feature-importance analysis later showed that this was one of the strongest predictors in the model.

---

# 34. Neighbourhood Listing Count

The feature:

```text
neighbourhood_listing_count
```

was also created.

This represents the historical number of listings associated with each neighbourhood.

It provides the model with additional information about neighbourhood listing density and market activity.

---

# 35. LightGBM – Initial Model

LightGBM was evaluated because it can efficiently model nonlinear relationships and complex feature interactions.

The initial LightGBM model achieved approximately:

```text
Validation R² = 0.27
```

This was already competitive with the tuned Random Forest.

---

# 36. LightGBM with Engineered Features

After adding geographic clustering and target-encoded features:

```text
Validation R² ≈ 0.31
```

This demonstrated that the engineered location and interaction features contained useful predictive signal.

---

# 37. LightGBM Hyperparameter Tuning

`RandomizedSearchCV` was used with:

```text
30 candidate hyperparameter combinations
3-fold cross-validation
```

The tuned engineered LightGBM model achieved:

```text
Validation R² = 0.3185
```

and:

```text
Train R² = 0.4456
```

This became the strongest model before final extreme-price treatment.

---

# 38. Rejected Feature Engineering Experiments

Several additional features were tested.

They were removed because they reduced validation performance.

---

## 38.1 Distance to Times Square

A feature measuring distance to Times Square was tested.

Result:

```text
Validation performance decreased.
```

### Decision

Rejected.

The existing geographic features already represented location effectively, and this additional feature did not add useful independent information.

---

## 38.2 Finer Geographic Clustering

The original:

```text
25 geo clusters
```

was replaced experimentally with:

```text
60 clusters
```

Result:

```text
Validation performance decreased.
```

### Decision

Rejected.

The finer clustering appeared to over-segment geographic information rather than improve generalization.

---

## 38.3 Additional Log-Transformed Numerical Predictors

Log transformations were tested for several skewed numerical predictors.

Result:

```text
Validation R² fell to approximately 0.3088.
```

### Decision

Rejected.

LightGBM can naturally model nonlinear numerical relationships, so these transformations did not provide additional useful information.

---

## 38.4 Finer Target Encoding

A more detailed target encoding based on:

```text
neighbourhood × room_type
```

was tested instead of:

```text
neighbourhood_group × room_type
```

Result:

```text
Validation R² ≈ 0.2998
```

### Decision

Rejected.

This appeared to over-engineer an already strong signal and reduced generalization performance.

---

# 39. Feature Importance

Feature importance was examined using:

```text
gain importance
```

rather than only split count.

Gain importance measures how much a feature contributes to reducing model loss.

An important finding was that:

```text
room_neigh_target_enc
```

provided extremely strong predictive information.

This confirmed the importance of modelling interactions between room type and broader location.

---

# 40. CatBoost Experiment

CatBoost was tested as another gradient-boosting alternative.

CatBoost was given raw categorical variables directly:

```text
neighbourhood
neighbourhood_group
room_type
```

without manually target encoding them.

An incompatibility occurred between:

```text
TransformedTargetRegressor
```

and CatBoost's parameter cloning behavior.

This was handled by manually applying the logarithmic target transformation instead.

Final CatBoost performance was approximately:

```text
Train R² = 0.2736
Validation R² = 0.2705
```

---

# 41. Why CatBoost Was Rejected

CatBoost had a very small train-validation gap, indicating good resistance to overfitting.

However:

```text
Validation R² = 0.2705
```

was lower than the engineered LightGBM model:

```text
Validation R² = 0.3185
```

This indicated that the manually engineered neighbourhood and interaction features captured more useful signal for this dataset.

Therefore CatBoost was not selected.

---

# 42. Why LightGBM Was Selected

LightGBM provided the strongest overall combination of:

- nonlinear modelling
- interaction modelling
- compatibility with engineered features
- efficient training
- validation performance
- manageable overfitting
- deployment simplicity

The best full-price-range LightGBM model achieved:

```text
Validation R² = 0.3185
```

before final price-range treatment.

---

# 43. Extreme Price Tail Investigation

Raw-price R² remained limited even after substantial modelling improvements.

Further investigation showed that a very small number of extremely expensive listings produced disproportionately large squared prediction errors.

Because R² and RMSE depend on squared residuals:

```text
one very large prediction error
```

can have much more influence than several moderate errors.

Therefore percentile-based price cutoffs were tested experimentally.

---

# 44. Percentile Price-Range Experiments

The complete modelling workflow was rebuilt for different upper-price cutoffs.

Results:

| Percentile | Price Limit | Rows Retained | Validation R² |
|---|---:|---:|---:|
| 90th | $269 | 44,006 | 0.6004 |
| 95th | $355 | 46,443 | 0.5674 |
| 97th | $450 | 47,444 | 0.5421 |
| 99th | $799 | 48,410 | 0.4838 |

This demonstrated that the extreme upper-price tail had a major effect on raw-price R².

---

# 45. Important Interpretation of the Percentile Experiment

The improvement in R² after restricting the price range should not be interpreted as the algorithm suddenly becoming dramatically more intelligent.

Instead:

> The regression problem becomes more stable when the extremely high luxury-price tail is excluded.

The percentile experiment demonstrated the sensitivity of squared-error metrics to extreme targets.

---

# 46. 97th vs 99th Percentile Comparison

The two strongest practical candidates were compared directly.

| Metric | 97th Percentile | 99th Percentile |
|---|---:|---:|
| Price cutoff | $450 | $799 |
| Rows retained | 47,444 | 48,410 |
| Data retained | 97% | 99% |
| Train R² | 0.7512 | 0.7173 |
| Validation R² | 0.5421 | 0.4838 |
| Test R² | 0.5439 | 0.5145 |
| Validation MAE | $35.49 | $41.99 |
| Test MAE | $35.91 | $41.93 |
| Validation RMSE | $54.65 | $74.34 |
| Test RMSE | $55.50 | $72.69 |

---

# 47. Final Price-Range Decision

The **99th-percentile model** was selected.

The final supported historical price range is approximately:

```text
$1 – $799 per night
```

The 99th-percentile configuration retains approximately:

```text
99% of usable observations
```

Although the 97th-percentile model achieved higher R², it excluded a larger section of higher-priced listings.

The final decision prioritized:

```text
broader market coverage
```

over:

```text
maximizing R² through more aggressive price filtering
```

---

# 48. Final Model

The final model is:

```text
LightGBM Regressor
```

using:

```text
log1p(price)
```

target transformation.

The final model includes:

### Numerical Features

```text
latitude
longitude
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
has_review
days_since_last_review
```

### Engineered Features

```text
geo_cluster
neighbourhood_target_enc
room_neigh_target_enc
neighbourhood_listing_count
```

### Categorical Features

```text
neighbourhood_group
room_type
```

TF-IDF listing-title features are not included in the final model.

---

# 49. Final Model Performance

The final **99th-percentile LightGBM model** achieved:

## Validation Set

```text
R²   = 0.4838
MAE  = $41.99
RMSE = $74.34
```

## Final Unseen Test Set

```text
R²   = 0.5145
MAE  = $41.93
RMSE = $72.69
```

---

# 50. Final Metric Interpretation

## R²

Final unseen test:

```text
R² = 0.5145
```

The model explains approximately:

```text
51% of the variation in nightly prices
```

within the final supported modelling range.

R² should not be interpreted as:

```text
51% prediction accuracy
```

because R² measures explained variation rather than classification-style accuracy.

---

## MAE

Final test:

```text
MAE = $41.93
```

This means that the model's prediction differs from the true nightly price by approximately:

```text
$42 per night on average
```

---

## RMSE

Final test:

```text
RMSE = $72.69
```

RMSE is larger than MAE because it penalizes large errors more heavily.

This indicates that some listings still produce substantially larger prediction errors.

---

# 51. Model Comparison and Selection Decisions

| Model | Main Result | Decision | Reason |
|---|---|---|---|
| Dummy Baseline | R² = -0.0515 | Rejected | Explained no meaningful variation |
| Linear Regression | R² ≈ 0.0924 | Rejected | Underfit nonlinear pricing relationships |
| Ridge Regression | R² ≈ 0.1774 | Rejected | Better than Linear Regression but still limited |
| Decision Tree | Train R² = 1.0, Val R² ≈ -1.58 | Rejected | Severe overfitting |
| Random Forest – Raw Price | Val R² ≈ 0.028 | Rejected | Weak raw-price generalization |
| Random Forest – Log Price | Val R² ≈ 0.229 | Improved candidate | Log target significantly improved results |
| Tuned Random Forest | Val R² ≈ 0.2345 | Rejected as final | Hyperparameter tuning produced diminishing returns |
| CatBoost | Val R² ≈ 0.2705 | Rejected | Stable but captured less signal |
| Basic LightGBM | Val R² ≈ 0.27 | Promising | Stronger nonlinear modelling |
| Tuned Engineered LightGBM | Val R² = 0.3185 | Best full-range model | Strongest before price-range treatment |
| Final 99th-Percentile LightGBM | Val R² = 0.4838, Test R² = 0.5145 | **Selected** | Best trade-off between performance and market coverage |

---

# 52. Why the Baseline Models Were Important

The simpler models were not included because they were expected to become the final solution.

They established progressively stronger benchmarks.

The development path was:

```text
Dummy Baseline
      ↓
Linear Regression
      ↓
Ridge Regression
      ↓
Decision Tree
      ↓
Random Forest
      ↓
Log-Target Random Forest
      ↓
Tuned Random Forest
      ↓
LightGBM
      ↓
Engineered LightGBM
      ↓
Tuned LightGBM
      ↓
CatBoost Comparison
      ↓
Price-Range Experiments
      ↓
Final 99th-Percentile LightGBM
```

This made it possible to determine whether additional complexity produced genuine improvements rather than simply increasing training performance.

---

# 53. Why Random Forest Was Rejected

Random Forest initially showed strong training performance:

```text
Train R² = 0.8634
```

but very weak validation performance:

```text
Validation R² = 0.0284
```

This indicated substantial overfitting.

Log-transforming the target improved performance considerably.

After hyperparameter tuning:

```text
Train R² = 0.5814
Validation R² = 0.2345
Validation MAE = $52.00
Validation RMSE = $183.15
Log-space R² = 0.6412
```

Although this was a major improvement, further Random Forest tuning produced only small gains.

Therefore boosting models were explored.

---

# 54. Why CatBoost Was Rejected

CatBoost achieved:

```text
Train R² = 0.2736
Validation R² = 0.2705
```

This showed excellent train-validation stability.

However, the validation score remained below LightGBM.

The engineered LightGBM model was able to exploit manually created neighbourhood and interaction signals more effectively.

Therefore CatBoost was not selected.

---

# 55. Why LightGBM Was Selected

LightGBM ultimately produced the strongest combination of:

```text
predictive performance
nonlinear modelling
feature interactions
generalization
training efficiency
compatibility with engineered features
deployment simplicity
```

It also benefited strongly from:

```text
geo_cluster
neighbourhood_target_enc
room_neigh_target_enc
neighbourhood_listing_count
```

The final LightGBM model therefore became the production/deployment model.

---

# 56. Streamlit Application

A Streamlit web application was developed so that the trained model can be used interactively.

The app accepts inputs including:

```text
Neighbourhood Group
Neighbourhood
Latitude
Longitude
Room Type
Minimum Nights
Number of Reviews
Reviews per Month
Host Listing Count
Availability
Days Since Last Review
```

The application reconstructs the same engineered features used during training:

```text
geo_cluster
neighbourhood_target_enc
room_neigh_target_enc
neighbourhood_listing_count
```

It then sends the final feature row to the saved LightGBM model.

The output is:

```text
Estimated Nightly Price
```

---

# 57. Final Model Price Range

The Streamlit application clearly states that the final model was primarily trained for listings approximately within:

```text
$1 – $799 per night
```

Predictions outside this historical modelling range are displayed with an additional warning because high-end luxury listings were less reliably represented.

---

# 58. Saved Model Artifacts

The final deployed model uses the following saved artifacts:

```text
model_artifacts/
│
├── global_mean_99.pkl
├── kmeans_99.pkl
├── model_99.pkl
├── neigh_counts_99.pkl
├── neigh_map_99.pkl
└── room_neigh_map_99.pkl
```

These objects allow the Streamlit application to reproduce the same feature-engineering process used during model training.

---

# 59. Project Structure

```text
202618066_LAB_4/
│
├── Data/
│   ├── AB_NYC_2019.csv
│   └── AB_NYC_2019_cleaned.csv
│
├── model_artifacts/
│   ├── global_mean_99.pkl
│   ├── kmeans_99.pkl
│   ├── model_99.pkl
│   ├── neigh_counts_99.pkl
│   ├── neigh_map_99.pkl
│   └── room_neigh_map_99.pkl
│
├── app.py
├── Airbnb_Price_Prediction.ipynb
├── requirements.txt
└── README.md
```

---

# 60. Installation

Clone the GitHub repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project folder:

```bash
cd 202618066_LAB_4
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 61. Running the Streamlit Application

Run:

```bash
streamlit run app.py
```

Streamlit will display a local URL such as:

```text
http://localhost:8501
```

Open the URL in a browser to use the prediction application.

---

# 62. Screenshots

## Exploratory Data Analysis

_Add price distribution / boxplot screenshot here._

```text
screenshots/price_distribution.png
```

---

## Correlation Analysis

_Add correlation heatmap screenshot here._

```text
screenshots/correlation_heatmap.png
```

---

## Model Comparison

_Add final model comparison screenshot here._

```text
screenshots/model_comparison.png
```

---

## Streamlit Application

_Add main application screenshot here._

```text
screenshots/streamlit_app.png
```

---

## Prediction Example

_Add prediction-result screenshot here._

```text
screenshots/prediction_example.png
```

---

# 63. Key Findings

The major findings from this project were:

- Airbnb nightly price is strongly right-skewed.
- A very small number of extremely expensive listings have a major effect on RMSE and R².
- Room type is one of the strongest predictors of Airbnb price.
- Entire homes/apartments are substantially more expensive than private and shared rooms.
- Manhattan listings generally have higher prices than other neighbourhood groups.
- Neighbourhood-level information contains much more pricing signal than simple latitude/longitude correlations suggest.
- Geographic clustering improved the representation of local pricing patterns.
- Smoothed neighbourhood target encoding captured useful historical location-price information.
- `room_neigh_target_enc` became one of the strongest engineered predictors.
- Adding more features did not automatically improve performance.
- Distance to Times Square did not improve validation results.
- Increasing geographic clusters from 25 to 60 reduced validation performance.
- Additional log-transformed numerical predictors reduced performance.
- Finer neighbourhood × room-type target encoding reduced performance.
- TF-IDF listing-title features contained real signals but added additional noise and dimensionality and were excluded from the final model.
- Linear Regression underfit the pricing relationships.
- Decision Tree severely overfit.
- Random Forest improved substantially after log-transforming the target.
- Random Forest hyperparameter tuning reduced overfitting but eventually reached diminishing returns.
- CatBoost was stable but captured less predictive signal than engineered LightGBM.
- LightGBM produced the strongest full-range model.
- The extreme upper-price tail was one of the main reasons raw-price R² remained low.
- The 97th-percentile model produced higher R², but the 99th-percentile model retained broader market coverage.
- The final 99th-percentile model retained approximately 99% of usable observations while supporting historical prices up to approximately $799/night.

---

# 64. Limitations

The final project has several important limitations.

### Historical Dataset

The data represents NYC Airbnb listings from:

```text
2019
```

Current Airbnb market prices may differ substantially.

---

### Missing Property Characteristics

The dataset does not contain several variables that would likely improve price prediction, including:

- Number of bedrooms
- Number of bathrooms
- Maximum guests
- Property size
- Amenities
- Property quality
- View
- Floor
- Renovation status

This places an upper limit on achievable prediction accuracy.

---

### High-End Properties

The final model focuses primarily on listings within approximately:

```text
$1 – $799/night
```

Very expensive luxury listings may not be predicted accurately.

---

### Target Encoding

Target-encoded variables represent historical training-data price patterns.

These patterns may change over time.

---

### Market Changes

Airbnb regulations, demand, tourism, inflation, neighbourhood development, and market conditions can change.

Therefore this model should not be interpreted as a current real-estate pricing system.

---

### Prediction Interpretation

The model returns an:

```text
estimated nightly price
```

rather than a guaranteed Airbnb market price.

---

# 65. Final Conclusion

This project demonstrates a complete end-to-end machine learning workflow for Airbnb nightly-price prediction.

The project did not simply train a single model.

Instead, it systematically investigated:

```text
data cleaning
missing-value mechanisms
outliers
feature engineering
text features
geographic features
target encoding
linear models
tree models
Random Forest
target transformation
hyperparameter tuning
LightGBM
CatBoost
feature importance
price-range sensitivity
validation performance
unseen test performance
deployment
```

The final model is a tuned **LightGBM Regressor** using a logarithmic target transformation and carefully selected engineered features.

The final **99th-percentile model** retains approximately:

```text
99% of usable listings
```

and supports historical nightly prices approximately up to:

```text
$799
```

Final unseen test performance:

```text
R²   = 0.5145
MAE  = $41.93
RMSE = $72.69
```

The 99th-percentile version was deliberately selected instead of the higher-scoring 97th-percentile model because broader Airbnb market coverage was prioritized over maximizing R² alone.

The final trained model and preprocessing artifacts were integrated into a Streamlit application, making the machine learning workflow usable for new listing inputs.

---

# 66. Final Model Summary

```text
Algorithm:
LightGBM Regressor

Target:
Nightly Airbnb price

Target transformation:
log1p(price)

Final price cutoff:
99th percentile

Approximate maximum supported historical price:
$799/night

Data retained:
~99%

Validation R²:
0.4838

Final Test R²:
0.5145

Validation MAE:
$41.99

Final Test MAE:
$41.93

Validation RMSE:
$74.34

Final Test RMSE:
$72.69
```

---

**DS605 – Fundamentals of Machine Learning**  
**Lab 4 – End-to-End Machine Learning: Airbnb Price Prediction**