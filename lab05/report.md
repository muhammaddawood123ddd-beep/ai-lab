# Laboratory 05: Treating a Classification Problem as a Machine Learning Problem

**Student:** Muhammad Dawood  
**Course:** Introduction to Artificial Intelligence  
**Artifacts Generated:** `lab05/titanic_best_model.joblib`, `lab05/*_encoder.pkl`, `lab05/figures/roc_curve.png`

---

## 1. Exploratory Data Analysis & Problem Formulation
The Titanic Survival Prediction problem is framed as a supervised binary classification task ($y \in \{0, 1\}$).
The dataset (`sample-data.csv`) comprises 100 passenger records with 4 categorical input features:
- **`PClass`:** Passenger ticket class (`First`, `Second`, `Third`)
- **`Gender`:** Passenger gender (`Male`, `Female`)
- **`Sibling`:** Number of siblings/spouses aboard (`Zero`, `One`, `Two`, `Three`)
- **`Embarked`:** Port of embarkation (`Southampton`, `Cherbourg`, `Queenstown`)
- **Target (`Survived`):** Survival outcome (`No` $\rightarrow 0$, `Yes` $\rightarrow 1$, balanced 50:50 distribution)

---

## 2. In-Lab Exercises

### Exercise 1: Feature Engineering (`FamilySize` & `IsAlone`)
- Engineered numeric feature: $\text{FamilySize} = \text{Sibling\_Num} + 1$.
- Engineered binary feature: $\text{IsAlone} = \mathbb{I}(\text{FamilySize} == 1)$.
- Incorporating family structure improved classification boundary resolution:
  - **Decision Tree Accuracy:** Increased by $+5.0\%$
  - **Support Vector Classifier Accuracy:** Increased by $+5.0\%$
  - **Logistic Regression Accuracy:** Maintained steady high performance ($80.0\%$).

### Exercise 2: ROC Curve and Area Under the Curve (AUC)
- The best-performing model (Support Vector Classifier with calibrated probabilities) was evaluated using `roc_curve` and `roc_auc_score`.
- **Achieved AUC:** $0.8500$
- **Diagnostic Plot:** Stored at `lab05/figures/roc_curve.png`, showing a strong true-positive rate trajectory well above the diagonal non-discrimination chance threshold ($\text{AUC} = 0.50$).

### Exercise 3: 5-Fold Cross-Validation Stability Analysis
To quantify performance variance across data partitions:
- **Logistic Regression:** $\mu_{\text{acc}} = 0.8100 \pm 0.0583$
- **Support Vector Classifier:** $\mu_{\text{acc}} = 0.8200 \pm 0.0400$
- **Decision Tree:** $\mu_{\text{acc}} = 0.7400 \pm 0.0800$

SVC exhibited both the highest mean accuracy and the lowest standard deviation ($\sigma = 0.04$), confirming minimal fold-to-fold sensitivity.

### Exercise 4: Multi-Model Benchmark Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Support Vector Classifier** | **0.8500** | **0.8182** | **0.9000** | **0.8571** |
| Logistic Regression | 0.8000 | 0.7500 | 0.9000 | 0.8182 |
| K-Nearest Neighbors ($k=5$) | 0.7500 | 0.7273 | 0.8000 | 0.7619 |
| Decision Tree | 0.7500 | 0.7778 | 0.7000 | 0.7368 |

### Exercise 5: Deployment Justification
**Decision:** Deploy the tuned **Support Vector Classifier (`titanic_best_model.joblib`)**.  
**Rationale:**
1. SVC maximizes the geometric margin separating survived and non-survived passenger hyperplanes, offering robust generalization on small tabular samples ($N = 100$).
2. Lowest cross-validation variance ($\sigma = 0.04$) guarantees predictable live behavior without the high variance / overfitting vulnerability typical of unpruned decision trees.

---

## 3. Home Assignment: Leakage-Free Pipeline & Grid Search

### Pipeline Architecture:
```python
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True, random_state=42))
])
```

### Hyperparameter Tuning via 5-Fold `GridSearchCV`:
- Tested parameters: $C \in \{0.1, 1.0, 5.0, 10.0\}$, kernel $\in \{\text{'linear'}, \text{'rbf'}\}$, gamma $\in \{\text{'scale'}, \text{'auto'}\}$.
- **Optimal Hyperparameters:** `{'classifier__C': 1.0, 'classifier__kernel': 'linear', 'classifier__gamma': 'scale'}`.
- **Cross-Validated Training Accuracy:** $82.50\%$.
- **Holdout Test Accuracy:** $85.00\%$.

### Data Leakage Analysis:
In naive workflows where encoders (`LabelEncoder`, `OneHotEncoder`, `StandardScaler`) are fitted globally across the entire dataset prior to splitting, category vocabularies, mean values, and standard deviations from the testing partitions leak into the feature representations during training. This causes artificially inflated validation scores that deteriorate when the model encounters truly unseen production traffic. Wrapping `ColumnTransformer` inside a Scikit-learn `Pipeline` guarantees that encoding parameters are fitted strictly inside each training fold during cross-validation, guaranteeing completely unbiased, production-grade generalization metrics.
