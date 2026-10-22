import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

def solve_lab01():
    notebook_path = "Lab_01_Environment_Setup.ipynb"
    print(f"Reading {notebook_path}...")
    nb = nbformat.read(notebook_path, as_version=4)

    # Cell 20: Git configuration check
    nb.cells[20].source = "!git --version"

    # Cell 23: Set Repository URL
    nb.cells[23].source = """# Record your repository URL here so the instructor can find it.
REPOSITORY_URL = "https://github.com/muhammaddawood123/ai-lab-portfolio"

print("Submitted repository:", REPOSITORY_URL)"""

    # Cell 28: Exercise 2 solution
    nb.cells[28].source = """# --- Exercise 2: your answer ---------------------------------------------
# Analysis from df.info(): Total entries = 891.
# - 'age' has 714 non-null entries -> 891 - 714 = 177 missing values
# - 'embarked' has 889 non-null entries -> 891 - 889 = 2 missing values
# - 'embark_town' has 889 non-null entries -> 891 - 889 = 2 missing values
# - 'deck' has 203 non-null entries -> 891 - 203 = 688 missing values

missing_columns = {
    "age": 177,
    "embarked": 2,
    "embark_town": 2,
    "deck": 688,
}

print("Columns with missing values (your answer):", missing_columns)
print()
print("Checked against pandas:")
print(df.isnull().sum()[df.isnull().sum() > 0])"""

    # Cell 29: Exercise 3 code
    nb.cells[29].source = """# --- Exercise 3: coefficient of variation --------------------------------
# std / mean tells you how spread a column is RELATIVE to its own size.
# A model without scaling treats a large-variance column as more important.

numeric = df.select_dtypes(include="number")
cv = (numeric.std() / numeric.mean()).abs().sort_values(ascending=False)

print("Coefficient of variation (std / mean), largest first:")
print(cv.round(3))
print()
print("Largest:", cv.index[0])"""

    # Cell 30: Exercise 3 written explanation
    nb.cells[30].source = """**Your two-sentence answer to Exercise 3:**

The numeric attribute exhibiting the highest coefficient of variation (std / mean) is **`parch`** (cv ≈ 2.116, closely followed by **`fare`** at cv ≈ 1.543), demonstrating extreme relative dispersion and skewness across passengers. Without input scaling, distance-based and gradient-descent algorithms treat features with large absolute variances as overwhelmingly dominant, severely biasing model convergence and distorting equal feature weighting."""

    # Cell 32: Home assignment inspect function tested on two public datasets
    nb.cells[32].source = '''def inspect(path_or_url):
    """Load a CSV and print a compact, standard first-look report.

    Parameters
    ----------
    path_or_url : str
        A local file path or an http(s) URL pointing at a CSV file.
    """
    import pandas as pd

    df = pd.read_csv(path_or_url)

    print("=" * 70)
    print("SOURCE :", path_or_url)
    print("SHAPE  :", df.shape[0], "rows x", df.shape[1], "columns")
    print("=" * 70)

    print("\\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\\n--- MISSING VALUES ---")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("None.")
    else:
        report = pd.DataFrame({
            "missing": miss,
            "percent": (miss / len(df) * 100).round(1),
        })
        print(report.sort_values("missing", ascending=False))

    print("\\n--- NUMERIC SUMMARY ---")
    print(df.describe())

    return df


# Test it on two public datasets
print("=== DATASET 1: Titanic ===")
df_titanic = inspect("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

print("\\n=== DATASET 2: Iris ===")
df_iris = inspect("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")'''

    # Save updated notebook structure
    nbformat.write(nb, notebook_path)
    print("Updated notebook saved. Now executing all cells...")

    # Execute the notebook using ExecutePreprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "."}})
    
    # Save executed notebook
    nbformat.write(nb, notebook_path)
    print("Notebook successfully executed and saved with outputs!")

if __name__ == "__main__":
    solve_lab01()
