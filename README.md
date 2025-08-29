
# Fake News Detection on Threads

This repository contains the complete implementation of our **Fake News Detection** pipeline using data scraped from Meta's **Threads** platform. The project encompasses end-to-end processing — from data collection and preprocessing to model training and evaluation, addressing the challenge of **fake news classification** using advanced NLP, machine learning and deep learning techniques.

## Project Overview

The rapid spread of misinformation on social media, especially on emerging platforms like Threads, necessitates robust detection mechanisms. This work explores fake news detection by:

- Collecting real-world data from Threads.
- Annotating news via cross-referencing with reliable fact-checking websites.
- The analysis revealed distinct patterns in engagement, posting times, sentiment and bot behavior analysis.
- Preprocessing text for machine learning.
- Handling **class imbalance** using **LLM-generated synthetic samples**.
- Employing traditional ML models and transformer based techniques.


## Installation

1. Clone the repository:

```bash
git clone https://github.com/Abhi-Ptdr/ThreadsFND.git
cd ThreadsFND
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

##  Data Collection (Threads Web Scraping)

We use Selenium and BeautifulSoup to extract Threads post data, including text, engagement, time, and user verification.

### Requirements

```bash
pip install selenium beautifulsoup4 pandas
```

### Setup ChromeDriver

1. Download [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/) that matches your browser version.
2. Update the path in the script:

```python
service = Service('C:\\path\\to\\chromedriver.exe')  # Windows
# or
service = Service('/usr/local/bin/chromedriver')         # macOS/Linux
```

### Running the Script

Run using Jupyter Notebook:

```bash
jupyter notebook Data_Collection.ipynb
```

Or as a Python script:

```bash
python Data_Collection.py
```

### Output

A CSV file named `FakeThreads.csv` will be saved with:

- Post URL
- Username
- Post Text
- Post Date
- Post Time (HH:MM)
- Likes
- Replies
- Reposts
- Shares
- Verified User
- Accessed Date
- Accessed Time
- Code

---

## Dataset

- **Raw Data Source**: Scraped Threads posts.
- **Labeling**: Verified via platforms like PolitiFact, LeadStories and USAToday.
- **Augmented Data**: Fake samples generated via LLM.

Files:
- `FakeThreads.csv`: Original scraped data from Threads.
- `Cleaned_NewThreadsData_Labeled.csv`: Cleaned Original scraped and annotated dataset.
- `Embedded_Cleaned_UnbalancedData.csv`: Dataset after converting post text to embeddings.
- `balanced_dataset_with_T5Large.csv`: Balanced dataset by inserting LLM-generated fake samples.
- `Cleaned_balanced_dataset.csv`: Cleaned balanced dataset.
- `Embedded_Data_after_Balancing.csv`: Embeddings generated on Balanced dataset.



## Implementation Steps
- To get started, simply run the fake_news_detection_on_threads.ipynb notebook using Jupyter or Kaggle to execute the entire pipeline. which consist of following steps.

### 1. Data Preprocessing
- Clean text and normalize formatting.
- Tokenize and embed text using `all-mpnet-base-v2`.

### 2. Exploratory Data Analysis (EDA)
- Explore class distribution, engagement, and sentiment.

### 3. Data Augmentation
- Use LLM (FLAN-T5) to generate realistic fake posts.

### 4. Model Training
- Train and evaluate ML and transformer based models (Random Forest, XGBoost, roberta-base etc.).

### 5. Evaluation
- Evaluate using accuracy, precision, recall, and F1-score.

## Models Used

- **Embeddings**: Sentence-BERT (`all-mpnet-base-v2`)
- **Data Augmentation**: FLAN-T5 for fake news generation
- **Classifiers**: Naïve Bayes, Random Forest, XGBoost, LightGBM, AdaBoost, roberta-base, vinai/bertweet-base, microsoft/deberta-v3-base

## Contributing

We welcome contributions! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).


## Contact

For questions, collaborations, or feedback regarding this project, feel free to reach out:

**Abhishek Patidar**  
M.Tech in Computer Science and Engineering  
Atal Bihari Vajpayee Indian Institute of Information Technology and Management, Gwalior

Email: abhipatidar253@gmail.com 
LinkedIn: [linkedin.com/in/abhiptdr](https://www.linkedin.com/in/abhiptdr/)
