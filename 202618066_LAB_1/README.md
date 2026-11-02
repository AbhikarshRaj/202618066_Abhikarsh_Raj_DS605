# Books Data Scraping, Preprocessing and Exploratory Data Analysis

## Project Information

- **Project Title:** Books Data Scraping and Exploratory Data Analysis
- **Name:** Abhikarsh Raj
- **Student ID:** 202618066
- **Website:** https://books.toscrape.com/
- **Programming Language:** Python
- **Primary Libraries:** Scrapy, Pandas, NumPy, Matplotlib, Seaborn, WordCloud

---

## 1. Project Overview

The objective of this project is to develop a complete data pipeline for extracting, cleaning, transforming, and analysing book-related information from [Books to Scrape](https://books.toscrape.com/).

The project uses Python and Scrapy to collect book information from multiple catalogue pages and individual product pages. The extracted information is then processed using Pandas and NumPy to handle missing values, remove duplicate records, convert data types, and create additional analytical features.

Exploratory Data Analysis (EDA) is performed using visualizations and statistical summaries to identify patterns in book prices, ratings, categories, and stock availability.

The project also generates a word cloud using book descriptions to explore frequently occurring terms in the collected textual data.

### Project Objectives

1. Scrape at least 100 books from a minimum of five catalogue pages.
2. Extract relevant information from individual book pages.
3. Clean and preprocess the collected dataset.
4. Engineer additional features for analysis.
5. Generate meaningful statistical visualizations.
6. Identify patterns and relationships between book attributes.
7. Present data-driven observations and conclusions.

---

## 2. Dataset Description

The dataset is collected from Books to Scrape, an online website designed for web scraping practice.

The following attributes are extracted:

| Attribute | Description |
|---|---|
| Title | Name of the book |
| Category | Category to which the book belongs |
| Price | Listed price of the book |
| Rating | Customer rating from 1 to 5 |
| Availability | Stock availability information |
| Description | Product description |
| UPC | Unique Product Code |
| Number of Reviews | Number of customer reviews |
| Product URL | Individual product page URL |

The project collects **200 book records** from multiple catalogue pages.

---

## 3. Project Workflow

The complete project follows four major stages:

### Stage 1: Data Scraping

- Scrapy is used for automated web scraping.
- Catalogue pagination is followed to collect books from multiple pages.
- Individual product pages are visited to extract detailed information.
- Extracted records are exported into a raw CSV dataset.

### Stage 2: Data Preprocessing

The raw dataset is processed using Pandas and NumPy.

The following operations are performed:

- Removal of unnecessary spaces and inconsistent text.
- Identification and removal of duplicate books using UPC.
- Handling missing product descriptions.
- Conversion of book prices into numerical values.
- Conversion of ratings into numerical values.
- Extraction of stock quantities from availability information.
- Removal of invalid records.

### Feature Engineering

The following additional features are created:

| Feature | Description |
|---|---|
| description_word_count | Number of words in the book description |
| price_band | Classification of books into Low, Medium, High and Premium price categories |
| affordability_score | Difference between the maximum book price and the individual book price |
| value_score | Ratio of book rating to book price |
| recommended | Identifies books with a rating of at least 4 and a price below or equal to the median price |

### Stage 3: Data Visualization and Analysis

Matplotlib and Seaborn are used to generate visualizations.

The following visualizations are included:

1. Distribution of Book Prices
2. Distribution of Book Ratings
3. Average Book Price by Category
4. Relationship Between Book Price and Rating
5. Average Stock Count by Rating
6. Word Cloud of Book Descriptions

### Stage 4: Insights and Interpretation

Statistical summaries and visualizations are used to identify:

- Distribution of book prices.
- Distribution of customer ratings.
- Differences in average prices across categories.
- Relationship between book prices and ratings.
- Stock availability across rating levels.
- Frequently occurring words in book descriptions.
- Potential high-value and recommended books.

---

## 4. Key Observations and Findings

Based on the exploratory data analysis, the following observations were identified:

**1. Stock Availability Across Ratings**

The average stock count remains relatively similar across different rating levels. Books with ratings ranging from 1 to 5 have comparable inventory levels. This suggests that there is no obvious relationship between customer ratings and stock availability in the collected dataset.

**2. Price Differences Across Categories**

Book prices vary considerably across categories. Categories such as Christian Fiction, Business, and Spirituality have comparatively lower average prices in the analysed dataset.

**3. Relationship Between Price and Rating**

Books with higher prices do not necessarily receive higher ratings. Both inexpensive and expensive books appear across different rating levels. Therefore, book price alone does not appear to be a reliable indicator of customer ratings.

**4. Rating Distribution**

The rating distribution provides an overview of customer preferences and helps identify which rating levels are most frequently assigned to the collected books.

**5. Category Representation**

The category distribution highlights differences in the number of books available across various categories and identifies the more frequently represented categories.

**6. Book Value and Affordability**

The engineered value score and affordability score provide additional ways to compare books based on their prices and ratings. Books with higher ratings and relatively lower prices can be identified as potential value-for-money options.

**7. Description Analysis**

The word cloud provides a visual representation of frequently occurring words in book descriptions. It helps identify common themes and vocabulary used to describe the available books.

---

## 5. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Scrapy | Web scraping and data extraction |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| WordCloud | Textual data visualization |
| Jupyter Notebook | Data analysis and experimentation |

---

## 6. Repository Structure

The repository is organized into the following structure:

```text
Books-Data-Scraping/
│
├── data/
│   ├── raw/
│   │   └── books_raw.csv
│   │
│   └── processed/
│       └── books_cleaned.csv
│
├── scraper/
│   ├── scrapy.cfg
│   └── books_scraper/
│       ├── __init__.py
│       ├── items.py
│       ├── settings.py
│       └── spiders/
│           ├── __init__.py
│           └── books_spider.py
│
├── notebooks/
│   └── Data_Scrapping.ipynb
│
├── visualizations/
│   ├── price_distribution.png
│   ├── rating_distribution.png
│   ├── average_price_by_category.png
│   ├── price_vs_rating.png
│   ├── stock_by_rating.png
│   └── wordcloud.png
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

Navigate into the project directory:

```bash
cd Books-Data-Scraping
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

For Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Scrapy Spider

Navigate to the Scrapy project directory:

```bash
cd scraper
```

Execute the spider:

```bash
scrapy crawl books -O ../data/raw/books_raw.csv
```

### Step 5: Run the Data Analysis

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Execute the notebook cells sequentially to perform:

- Data inspection
- Data cleaning
- Feature engineering
- Statistical analysis
- Visualization generation
- Word cloud generation

---

## 8. Limitations

The analysis has certain limitations:

1. The dataset represents only the books available on the website at the time of scraping.
2. The collected dataset does not represent the entire book market.
3. The website does not provide written customer-review text. Therefore, book descriptions are used for word cloud generation instead of customer reviews.
4. Customer ratings alone do not explain the reasons behind customer satisfaction or dissatisfaction.
5. Stock availability is a snapshot and may change over time.
6. The value score is a simple rating-to-price ratio and does not account for factors such as book quality, length, or publishing information.
7. The analysis identifies statistical patterns but does not establish causal relationships.

---

## 9. Conclusion

This project demonstrates a complete data analysis workflow, beginning with web scraping and ending with statistical interpretation.

The collected book information is cleaned and transformed into a structured dataset. Additional features are engineered to support comparisons involving affordability, ratings, and book value.

The visualizations provide insights into price distributions, category-level pricing, customer ratings, and stock availability. The analysis also indicates that higher-priced books do not necessarily receive higher customer ratings.

Overall, the project demonstrates how web scraping, data preprocessing, feature engineering, exploratory data analysis, and visualization can be integrated into a single data pipeline to extract meaningful insights from publicly available web data.

---

## 10. Acknowledgement

Data for this project was collected from:

[Books to Scrape](https://books.toscrape.com/)

The website is intended for practising web scraping techniques and data extraction.

**Project developed as part of an academic data scraping and analysis assignment.**