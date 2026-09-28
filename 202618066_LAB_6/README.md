# 202618066 - LAB 6
## Feature Extraction and Classification for Image and Text Data

This lab demonstrates how raw image and text data can be converted into numerical representations and classified using traditional machine learning algorithms.

The project contains two main tasks:

1. Asphalt Crack Image Classification
2. Email Spam Classification

The focus is on preprocessing, feature extraction, vector representation, model comparison, evaluation, and representation improvement.

---

# Project Structure

```text
202618066_LAB_6/
│
├── Data/
│   ├── Image/
│   │   ├── Cracks/
│   │   └── Non-Cracks/
│   │
│   └── Text/
│
├── 202618066_LAB_6.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Part A - Asphalt Crack Image Classification

## Dataset

The image dataset contains:

- Total images: **400**
- Crack images: **200**
- Non-crack images: **200**
- Original image size: **448 × 448 × 3**
- Resized image size: **256 × 256**

Class labels:

```text
0 = Non-Crack
1 = Crack
```

The dataset is balanced with 200 images in each class.

---

## Image Preprocessing Pipeline

```text
Raw Image
    ↓
Resize to 256 × 256
    ↓
Convert to Grayscale
    ↓
Gaussian Blur
    ↓
Canny Edge Detection
    ↓
Manual Feature Extraction
    ↓
Machine Learning Classification
```

### Resize

Images were resized from `448 × 448` to `256 × 256`.

Original number of pixels:

```text
448 × 448 = 200,704
```

After resizing:

```text
256 × 256 = 65,536
```

Resizing reduced computational cost while retaining enough structural information for crack detection.

### Grayscale

Color images contain three channels. OpenCV stores them in BGR order.

After grayscale conversion, each pixel contains only one intensity value:

```text
0   = Black
255 = White
```

Grayscale was used because crack detection mainly depends on brightness, darkness, contrast, and boundaries rather than color.

### Gaussian Blur

Gaussian blur was applied before Canny edge detection:

```python
cv2.GaussianBlur(gray, (5, 5), 0)
```

Its purpose is to reduce small image noise and prevent unnecessary texture variations from being detected as edges.

### Canny Edge Detection

Canny edge detection was applied using:

```python
cv2.Canny(blurred, 100, 200)
```

Cracks often create strong intensity boundaries against the asphalt surface, so edge information is useful for classification.

---

# Image Feature Extraction

Each image was represented using the following 8 numerical features:

| Feature | Description |
|---|---|
| Mean Intensity | Average brightness of the image |
| Contrast | Standard deviation of grayscale intensity |
| Minimum Intensity | Darkest pixel in the image |
| Maximum Intensity | Brightest pixel in the image |
| Dark Pixel Ratio | Proportion of pixels with intensity below 50 |
| Bright Pixel Ratio | Proportion of pixels with intensity above 200 |
| Edge Count | Number of detected Canny edge pixels |
| Edge Density | Edge count divided by total number of pixels |

For a resized image:

```math
N = 256 \times 256 = 65536
```

### Mean Intensity

```math
\mu = \frac{1}{N}\sum_{i=1}^{N}x_i
```

### Contrast

```math
\sigma =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}(x_i-\mu)^2
}
```

### Dark Pixel Ratio

```math
R_d =
\frac{\text{Number of pixels with intensity < 50}}{N}
```

### Bright Pixel Ratio

```math
R_b =
\frac{\text{Number of pixels with intensity > 200}}{N}
```

### Edge Density

```math
D_e =
\frac{\text{Number of edge pixels}}{N}
```

---

# Feature Analysis

Average feature values for the two classes were:

| Feature | Non-Crack | Crack |
|---|---:|---:|
| Mean Intensity | 157.4001 | 144.1359 |
| Contrast | 24.7643 | 31.5844 |
| Minimum Intensity | 59.07 | 25.04 |
| Maximum Intensity | 251.695 | 252.270 |
| Dark Pixel Ratio | 0.000724 | 0.009548 |
| Bright Pixel Ratio | 0.049785 | 0.075593 |
| Edge Count | 1875.985 | 6565.835 |
| Edge Density | 0.028625 | 0.100187 |

### Main Observations

Crack images generally had:

- lower mean intensity,
- higher contrast,
- lower minimum intensity,
- a higher dark-pixel ratio,
- many more edge pixels,
- a much higher edge density.

Maximum intensity was almost identical for both classes and therefore carried relatively little discriminatory information.

---

# Train-Test Split

An 80:20 stratified split was used.

```text
Training images: 320
Testing images:   80
```

Class distribution:

```text
Training:
160 Crack
160 Non-Crack

Testing:
40 Crack
40 Non-Crack
```

---

# Feature Scaling

`StandardScaler` was used for Logistic Regression.

```math
z = \frac{x-\mu}{\sigma}
```

After scaling, the training features had approximately:

```text
Mean = 0
Standard Deviation = 1
```

The scaler was fitted only on the training data to avoid data leakage.

---

# Image Classification Models

The following classifiers were compared:

- Logistic Regression
- Tuned Logistic Regression
- Decision Tree
- Random Forest

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression - Baseline | 0.9125 | 0.9231 | 0.9000 | 0.9114 |
| Logistic Regression - Tuned | 0.9250 | 0.9250 | 0.9250 | 0.9250 |
| Decision Tree | 0.9250 | 0.9250 | 0.9250 | 0.9250 |
| Random Forest | **0.9500** | **0.9500** | **0.9500** | **0.9500** |

Random Forest achieved the strongest predictive performance.

## Random Forest Confusion Matrix

```text
[[38, 2],
 [ 2,38]]
```

Interpretation:

```text
TN = 38
FP = 2
FN = 2
TP = 38
```

Out of 80 test images:

```text
Correct predictions   = 76
Incorrect predictions = 4
```

---

# Random Forest Feature Importance

| Feature | Importance |
|---|---:|
| Bright Pixel Ratio | 0.1891 |
| Mean Intensity | 0.1862 |
| Minimum Intensity | 0.1557 |
| Edge Density | 0.1370 |
| Edge Count | 0.1219 |
| Dark Pixel Ratio | 0.1113 |
| Contrast | 0.0828 |
| Maximum Intensity | 0.0159 |

The model relied on a combination of intensity and edge-related features rather than one feature alone.

---

# Image Representation Improvement

Since all images were resized to `256 × 256`, edge count and edge density were mathematically redundant:

```math
Edge\ Density = \frac{Edge\ Count}{65536}
```

Therefore, `edge_count` was removed.

```text
Original feature space: 8 features
Reduced feature space: 7 features
```

The reduced-feature Random Forest still achieved:

```text
Accuracy:  0.95
Precision: 0.95
Recall:    0.95
F1-score:  0.95
```

This reduced redundancy without reducing predictive performance.

---

# Part B - Email Spam Classification

## Dataset

The text dataset contains:

```text
5172 emails
```

The supplied dataset already contained numerical word-frequency features.

To demonstrate the vectorization workflow, email text was reconstructed from these word-count features and then vectorized again.

Because the source file already contained word counts, the reconstructed text does not preserve:

- original word order,
- punctuation,
- capitalization,
- sentence structure.

However, word-frequency information is preserved.

---

# Text Processing Pipeline

```text
Existing Word-Count Representation
        ↓
Reconstructed Text
        ↓
Train-Test Split
        ↓
CountVectorizer
        ↓
Numerical Word Vectors
        ↓
Classification
        ↓
Spam / Non-Spam
```

---

# CountVectorizer Representation

The baseline CountVectorizer produced:

```text
Training vector shape: (4137, 2974)
Testing vector shape:  (1035, 2974)
Vocabulary size:       2974
```

Each email is represented as a vector:

```math
x_i = [x_{i1}, x_{i2}, \ldots, x_{i2974}]
```

where each value represents the frequency of a vocabulary word in that email.

---

# Text Classification Models

The following classifiers were tested:

- Logistic Regression
- Multinomial Naive Bayes

## Logistic Regression Results

```text
Accuracy:        0.9807
Precision:       0.9545
Recall:          0.9800
F1-score:        0.9671
Training Time:   346.174 ms
Prediction Time: 9.166 ms
```

Confusion matrix:

```text
[[721, 14],
 [  6,294]]
```

## Multinomial Naive Bayes Results

```text
Accuracy:        0.9420
Precision:       0.8681
Recall:          0.9433
F1-score:        0.9042
Training Time:   10.348 ms
Prediction Time: 1.990 ms
```

Confusion matrix:

```text
[[692, 43],
 [ 17,283]]
```

Naive Bayes was much faster, but Logistic Regression achieved better predictive performance.

---

# Text Representation Improvement

A more complex CountVectorizer configuration was tested:

```python
CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000
)
```

This introduced:

- stop-word removal,
- unigram features,
- bigram features,
- rare-word filtering,
- vocabulary limitation.

## Representation Comparison

| Representation | Vocabulary Size | Accuracy | Precision | Recall | F1-score | Training Time (ms) | Prediction Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline CountVectorizer | 2974 | **0.9807** | **0.9545** | **0.9800** | **0.9671** | 346.174 | 9.166 |
| Improved CountVectorizer | 5000 | 0.9739 | 0.9446 | 0.9667 | 0.9555 | 310.213 | 1.620 |

The more complex representation increased dimensionality but slightly reduced predictive performance.

This shows that adding more features does not automatically improve a classifier.

---

# Final Results Summary

## Image Classification

The strongest image classifier was:

```text
Random Forest
Accuracy  = 95.00%
Precision = 95.00%
Recall    = 95.00%
F1-score  = 95.00%
```

Reducing the feature space from 8 features to 7 preserved the same predictive performance.

## Text Classification

The strongest text configuration was:

```text
Baseline CountVectorizer
+
Logistic Regression
```

with:

```text
Accuracy  = 98.07%
Precision = 95.45%
Recall    = 98.00%
F1-score  = 96.71%
```

---

# Key Observations

1. Manual image feature extraction successfully converted raw images into a compact numerical feature space.
2. Crack images generally had lower mean intensity, higher contrast, more dark pixels, and much higher edge density.
3. Random Forest performed better than Logistic Regression and Decision Tree for image classification.
4. Removing redundant `edge_count` reduced feature dimensionality without reducing image classification performance.
5. CountVectorizer successfully converted email text into a high-dimensional sparse numerical representation.
6. Logistic Regression outperformed Multinomial Naive Bayes for spam classification.
7. Multinomial Naive Bayes was faster but less accurate.
8. Adding stop-word removal and bigrams increased the text feature space but did not improve predictive performance.
9. More features do not necessarily produce a better machine learning model.
10. Feature representation quality is as important as the choice of classifier.

---

# Technologies Used

- Python
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Conclusion

This lab demonstrates how images and text can be transformed into meaningful numerical representations before applying traditional machine learning models.

For image classification, handcrafted intensity and edge features allowed Random Forest to achieve **95% test accuracy**.

For text classification, CountVectorizer combined with Logistic Regression achieved **98.07% test accuracy**.

The experiments also showed that increasing representation complexity does not always improve performance. A simpler, less redundant feature space can provide equal or better predictive performance while remaining easier to interpret.
