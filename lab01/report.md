# Laboratory 01: Environment Setup, First Notebook, and Portfolio Repository

**Student:** Muhammad Dawood  
**Course:** Introduction to Artificial Intelligence  
**Repository:** [https://github.com/muhammaddawood123ddd-beep/ai-lab](https://github.com/muhammaddawood123ddd-beep/ai-lab)

---

## 1. Environment & Installation Evidence
- **Python Version:** Python 3.12 / Python 3.10 `ai-lab` virtual environment
- **Platform:** Windows 10/11 x64 architecture
- **Jupyter Kernel:** Registered and active as `Python (ai-lab)` (`kernel.json` registered in user profile)
- **Git Version:** `git version 2.47.3.windows.1`

---

## 2. In-Lab Exercises

### Exercise 1: Environment & Jupyter Kernel Setup
The dedicated virtual environment `ai-lab` was configured with core data science dependencies (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `ipykernel`). The kernel was registered using:
```bash
python -m ipykernel install --user --name ai-lab --display-name "Python (ai-lab)"
```

### Exercise 2: Missing Values Identification from `info()`
From executing `df.info()` on the Titanic dataset ($N = 891$ total entries):
- `age`: 714 non-null entries $\rightarrow$ **177 missing values** (19.9%)
- `deck`: 203 non-null entries $\rightarrow$ **688 missing values** (77.2%)
- `embarked`: 889 non-null entries $\rightarrow$ **2 missing values** (0.2%)
- `embark_town`: 889 non-null entries $\rightarrow$ **2 missing values** (0.2%)

### Exercise 3: Coefficient of Variation and Feature Scaling Analysis
The numeric attribute exhibiting the highest coefficient of variation ($\text{CV} = \sigma / |\mu|$) is **`parch`** ($\text{CV} \approx 2.116$), closely followed by **`fare`** ($\text{CV} \approx 1.543$). 

**Two-sentence explanation:**  
When an unscaled feature has an extreme coefficient of variation and wide dynamic range, distance-based and gradient-descent algorithms treat features with large numerical variances as overwhelmingly dominant, severely biasing model convergence and distorting equal feature weighting. Feature standardisation (e.g., $Z$-score scaling) ensures that each input dimension contributes proportionally based on statistical patterns rather than arbitrary measurement units.

### Exercise 4: Git Workflow and Branching Strategy
A separate branch `lab01/exercises` was created to perform the exploratory data analysis. After verification, it was merged into `main` to preserve a clean and auditable history.

---

## 3. Home Assignment: Compact Dataset Inspector
The reusable inspection tool `inspect.py` was implemented and evaluated against two public datasets:
1. **Titanic Dataset** (891 rows $\times$ 15 columns)
2. **Iris Dataset** (150 rows $\times$ 5 columns)

Both datasets were parsed without errors, reporting shape, exact data types, missing value percentages, and parametric summary statistics.
