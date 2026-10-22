import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

def solve_lab04():
    os.makedirs("lab04/figures", exist_ok=True)
    notebook_path = "Lab 04.ipynb"
    print(f"Reading {notebook_path}...")
    nb = nbformat.read(notebook_path, as_version=4)

    # In Cell 18: ensure the scatterplot is saved to lab04/figures/actual_vs_predicted.png
    nb.cells[18].source = '''import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("lab04/figures", exist_ok=True)
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test_reg, y=y_pred_reg, alpha=0.5)
plt.xlabel("Actual Values (Median House Value)")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Actual vs Predicted")
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2, label="Perfect Fit (y=x)")
plt.legend()
plt.tight_layout()
plt.savefig("lab04/figures/actual_vs_predicted.png", dpi=150)
plt.show()'''

    # In Cell 16: ensure linear_model.pkl is saved in both root and lab04/
    nb.cells[16].source = '''import joblib
import os

# Save model
joblib.dump(reg_model, 'linear_model.pkl')
os.makedirs("lab04", exist_ok=True)
joblib.dump(reg_model, 'lab04/linear_model.pkl')

# Load model
loaded_model = joblib.load('linear_model.pkl')
print("Loaded Model R2 Score:", r2_score(y_test_reg, loaded_model.predict(X_test_reg)))'''

    # Exercise solutions code cell
    ex_cell_code = """# ============================================================================
# In-Lab Exercises: Solutions to Exercises 1, 2, 3, and 4
# ============================================================================
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Exercise 1: Compare R2 Scores and Scaling Invariance ---
print("=== EXERCISE 1: R2 Comparison and Feature Scaling Invariance ===")
r2_plain = r2_score(y_test_reg, y_pred_reg)
r2_pipe = r2_score(y_test_reg, y_pred_pipe)
print(f"Plain Linear Regression R2 : {r2_plain:.6f}")
print(f"Pipeline (StandardScaler) R2: {r2_pipe:.6f}")
print(f"Difference in R2           : {abs(r2_plain - r2_pipe):.10f}")
print("\\nExplanation:")
print("Ordinary Least Squares (OLS) minimizes sum of squared residuals via the normal equation (X^T X)^(-1) X^T y.")
print("Because standardization is an invertible affine transformation (x_scaled = (x - mu) / sigma),")
print("the column space spanned by X remains identical; the regression coefficients simply rescale inversely,")
print("leaving the fitted values y_hat, residuals, and R2 mathematically invariant.")
print("Model families where standardization DOES change performance:")
print("  - Regularized Linear Models (Ridge, Lasso, ElasticNet) where L1/L2 penalty penalizes unscaled coefficients unequally.")
print("  - Distance-based models (K-Nearest Neighbors, Support Vector Machines).")
print("  - Gradient-descent-trained models (Neural Networks / Multi-layer Perceptrons).")

# --- Exercise 2: Feature Coefficients Analysis ---
print("\\n=== EXERCISE 2: Coefficients Analysis ===")
feature_names = df_reg.columns[:-1]
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Raw_Coefficient': reg_model.coef_,
    'Abs_Coefficient': np.abs(reg_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)
display(coef_df)

top_two = coef_df.iloc[:2]['Feature'].tolist()
print(f"Two features with largest absolute coefficients in raw model: {top_two[0]} and {top_two[1]}")
print("Why raw coefficients cannot be compared directly:")
print("In raw unscaled regression, coefficient magnitude is inversely proportional to the feature's measurement unit.")
print("A feature measured in tiny units (large numerical range, e.g. population) will have an artificially tiny coefficient,")
print("while a feature with a small numerical range (e.g. average rooms) has a large coefficient.")
print("Standardization puts all features on unit variance (z-scores), making coefficient magnitudes directly comparable.")

# --- Exercise 3: Iris Confusion Matrix and Per-Class Recall ---
print("\\n=== EXERCISE 3: Iris Classification Error Analysis ===")
cm = confusion_matrix(y_test_cls, y_pred_cls)
iris_targets = ['setosa', 'versicolor', 'virginica']
cm_df = pd.DataFrame(cm, index=[f"True_{s}" for s in iris_targets], columns=[f"Pred_{s}" for s in iris_targets])
print("Confusion Matrix:")
display(cm_df)

print("Classification Report:")
print(classification_report(y_test_cls, y_pred_cls, target_names=iris_targets))
print("Reading:")
print("The off-diagonal errors occur between 'versicolor' (class 1) and 'virginica' (class 2).")
print("Setosa (class 0) is linearly separable and classified with 100% precision and recall.")
print("Versicolor and virginica overlap in petal and sepal feature space, yielding small misclassification exchange.")

# --- Exercise 4: Estimator API Consistency with DecisionTreeClassifier ---
print("\\n=== EXERCISE 4: Decision Tree Estimator API Demonstration ===")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_cls, y_train_cls)
y_pred_dt = dt_model.predict(X_test_cls)

print("DecisionTreeClassifier Accuracy:", accuracy_score(y_test_cls, y_pred_dt))
print("\\nConfusion Matrix (Decision Tree):\\n", confusion_matrix(y_test_cls, y_pred_dt))
print("\\nClassification Report (Decision Tree):\\n", classification_report(y_test_cls, y_pred_dt, target_names=iris_targets))
print("Takeaway:")
print("Scikit-learn implements a unified, polymorphic estimator interface across all algorithms:")
print("Every estimator provides .fit(X, y), .predict(X), and .score(X, y), allowing seamless algorithm substitution.")"""

    # Home assignment code cell
    home_cell_code = """# ============================================================================
# Home Assignment: Sentiment Classification Pipeline
# ============================================================================
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the article dataset built in Laboratory 2
df_articles = pd.read_csv("articles.csv")
print(f"Loaded articles.csv with {len(df_articles)} documents.")

# 2. Determine median sentiment polarity and build binary target
corpus_median = df_articles['sentiment_polarity'].median()
print(f"Corpus Median Sentiment Polarity: {corpus_median:.4f}")

# Target: 1 if above median, 0 if below or equal
df_articles['target'] = (df_articles['sentiment_polarity'] > corpus_median).astype(int)
print("Target class distribution:\\n", df_articles['target'].value_counts())

X_text = df_articles['text']
y_target = df_articles['target']

# 3. Train-Test Split (seeded)
X_tr, X_te, y_tr, y_te = train_test_split(X_text, y_target, test_size=0.3, random_state=42, stratify=y_target)

# 4. Build Pipeline: TfidfVectorizer -> LogisticRegression
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=100)),
    ('classifier', LogisticRegression(random_state=42))
])

# 5. Fit pipeline and evaluate
sentiment_pipeline.fit(X_tr, y_tr)
y_pred_sent = sentiment_pipeline.predict(X_te)

print("\\n--- Sentiment Pipeline Evaluation ---")
print(f"Test Accuracy: {accuracy_score(y_te, y_pred_sent):.4f}")
print("\\nConfusion Matrix:\\n", confusion_matrix(y_te, y_pred_sent))
print("\\nClassification Report:\\n", classification_report(y_te, y_pred_sent, zero_division=0))

# 6. Persist Pipeline
joblib.dump(sentiment_pipeline, "sentiment_pipeline.pkl")
joblib.dump(sentiment_pipeline, "lab04/sentiment_pipeline.pkl")
print("Saved sentiment_pipeline.pkl and lab04/sentiment_pipeline.pkl")

# 7. Demonstration in fresh session on raw strings with NO manual preprocessing
print("\\n--- Inference on Raw Unseen Strings ---")
reloaded_pipe = joblib.load("sentiment_pipeline.pkl")

test_raw_strings = [
    "This groundbreaking artificial intelligence platform provides extraordinary performance and exceptional ease of use.",
    "The defective tool failed completely, causing continuous errors, frustrating delays, and severe disappointment."
]

preds = reloaded_pipe.predict(test_raw_strings)
probs = reloaded_pipe.predict_proba(test_raw_strings)[:, 1]

for text, pred, prob in zip(test_raw_strings, preds, probs):
    label = "Above Median (Positive)" if pred == 1 else "Below/At Median (Negative)"
    print(f"\\nRaw Input: '{text}'")
    print(f"Prediction: {label} (P(positive) = {prob:.4f})")"""

    # Check and insert exercise and home cells
    found_ex = False
    for c in nb.cells:
        if "In-Lab Exercises: Solutions" in c.source:
            c.source = ex_cell_code
            found_ex = True
            break
    if not found_ex:
        nb.cells.append(nbformat.v4.new_code_cell(source=ex_cell_code))

    found_home = False
    for c in nb.cells:
        if "Home Assignment: Sentiment Classification Pipeline" in c.source:
            c.source = home_cell_code
            found_home = True
            break
    if not found_home:
        nb.cells.append(nbformat.v4.new_code_cell(source=home_cell_code))

    nbformat.write(nb, notebook_path)
    print("Lab 04 notebook updated. Executing all cells...")

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "."}})
    
    nbformat.write(nb, notebook_path)
    print("Lab 04 notebook successfully executed and saved with outputs!")

if __name__ == "__main__":
    solve_lab04()
