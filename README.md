# Artificial Intelligence Laboratory Portfolio

**Author:** Muhammad Dawood  
**Course:** Introduction to Artificial Intelligence  
**Institution:** University AI Laboratory  

---

## Repository Structure

```text
ai-lab-portfolio/
│
├── .gitignore
├── requirements.txt
├── README.md
├── sample-data.csv
│
├── Lab_01_Environment_Setup.ipynb
├── Lab_02_Web_Scraping_and_EDA.ipynb
├── Lab_03_Text_and_Image_Features.ipynb
├── Lab 04.ipynb
├── Titanic Passenger Survival Prediction System using Train-Test Split Approach.ipynb
│
├── lab01/
│   ├── lab01_setup.ipynb
│   ├── inspect.py
│   └── report.md
│
├── lab02/
│   ├── scraping_eda.ipynb
│   ├── articles.csv
│   ├── figures/
│   │   ├── title_length_distribution.png
│   │   ├── sentiment_distribution.png
│   │   ├── pair_plot.png
│   │   ├── tfidf_top20.png
│   │   └── two_source_comparison.png
│   └── report.md
│
├── lab03/
│   ├── text_features.ipynb
│   ├── image_features.ipynb
│   ├── image_utils.py
│   ├── dataset.npz
│   └── report.md
│
├── lab04/
│   ├── sklearn_intro.ipynb
│   ├── linear_model.pkl
│   ├── sentiment_pipeline.pkl
│   ├── figures/
│   │   └── actual_vs_predicted.png
│   └── report.md
│
└── lab05/
    ├── titanic_classification.ipynb
    ├── titanic_best_model.joblib
    ├── pclass_encoder.pkl
    ├── gender_encoder.pkl
    ├── sibling_encoder.pkl
    ├── embarked_encoder.pkl
    ├── survived_encoder.pkl
    ├── figures/
    │   └── roc_curve.png
    └── report.md
```

---

## Laboratories Overview

### Laboratory 01: Environment Setup, First Notebook & Inspection Script
- **Topics:** Anaconda environment isolation, Jupyter kernel registration (`Python (ai-lab)`), pandas inspection protocols (`head()`, `info()`, `describe()`), coefficient of variation analysis.
- **In-Lab Exercises:** Missing value analysis on Titanic dataset ($N=891$), dispersion comparison ($\text{CV}_{\text{parch}} \approx 2.116$).
- **Home Assignment:** Reusable dataset inspection utility `inspect.py` tested across public benchmarks (Titanic and Iris datasets).
- **Report:** [lab01/report.md](lab01/report.md)

### Laboratory 02: Web Scraping, Feature Engineering and Exploratory Data Analysis
- **Topics:** HTML parsing with BeautifulSoup, NLP feature extraction with spaCy (`en_core_web_sm`), sentiment and subjectivity analysis with TextBlob, TF-IDF ranking.
- **In-Lab Exercises:** Cross-CMS selector mismatch diagnostics, extended verb and sentence-length extraction, critique of KDE on micro-samples ($N=5$), stop-word influence on TF-IDF.
- **Home Assignment:** Scraped 20-article corpus across two distinct genres (commercial tech reviews vs. encyclopedic AI references); comparative statistical and grouped visual analysis of sentiment polarity and noun density.
- **Report:** [lab02/report.md](lab02/report.md)

### Laboratory 03: Transforming Textual and Image Data into Machine-Understandable Formats
- **Topics:** Bag-of-Words tokenization (`CountVectorizer`), grayscale luminance conversion ($Y = 0.299R + 0.587G + 0.114B$), 2D image matrix flattening to 1D feature vectors.
- **In-Lab Exercises:** N-gram vocabulary expansion, document-length bias under raw term counts, image downsampling ($64\times 64$ and $32\times 32$) information loss, reusable `image_to_features` pipeline.
- **Home Assignment:** 20-sample labelled image dataset ($10\times$ circle geometry, $10\times$ rectangle geometry) stacked into $(20, 4096)$ matrix saved in `dataset.npz`; analytical discussion of the curse of dimensionality ($p \gg N$).
- **Report:** [lab03/report.md](lab03/report.md)

### Laboratory 04: Introduction to Scikit-learn and Traditional Machine Learning
- **Topics:** Supervised learning workflows, California Housing regression ($R^2 \approx 0.576$), Iris 3-class classification ($\text{Accuracy} = 1.000$), diagnostic scatter plots, Scikit-learn `Pipeline`, model persistence with `joblib`.
- **In-Lab Exercises:** Invariance of Ordinary Least Squares under feature standardization, coefficient interpretability across differing measurement units, confusion matrix error localization, polymorphic estimator API demonstration with `DecisionTreeClassifier`.
- **Home Assignment:** End-to-end `TfidfVectorizer` $\rightarrow$ `LogisticRegression` sentiment polarity classification pipeline trained on scraped articles; model persistence and live string prediction test.
- **Report:** [lab04/report.md](lab04/report.md)

### Laboratory 05: Treating a Classification Problem as a Machine Learning Problem
- **Topics:** Titanic passenger survival prediction, label encoding, train-test splitting, Support Vector Classification, interactive prediction inference.
- **In-Lab Exercises:** Feature engineering with `FamilySize` and `IsAlone` (+5% accuracy gain), ROC curve and AUC evaluation ($\text{AUC} = 0.850$), 5-fold cross-validation stability benchmarking, multi-model comparison (SVC, Logistic Regression, KNN, Decision Tree), production deployment justification.
- **Home Assignment:** Leakage-free preprocessing architecture utilizing `ColumnTransformer` inside a unified `Pipeline` with 5-fold `GridSearchCV` hyperparameter tuning; formal data leakage theoretical comparison.
- **Report:** [lab05/report.md](lab05/report.md)

---

## Installation & Reproduction

Clone the repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/ai-lab-portfolio.git
cd ai-lab-portfolio
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
