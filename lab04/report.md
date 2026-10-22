# Laboratory 04: Introduction to Scikit-learn and Traditional Machine Learning

**Student:** Muhammad Dawood  
**Course:** Introduction to Artificial Intelligence  
**Artifacts Generated:** `linear_model.pkl`, `sentiment_pipeline.pkl`, `lab04/figures/actual_vs_predicted.png`

---

## 1. Regression and Classification Benchmark Metrics

### California Housing Regression
- **Algorithm:** Ordinary Least Squares `LinearRegression`
- **Metric ($R^2$ Score):** $0.5758$ (Test set, $N = 4{,}128$)
- **Pipeline ($Z$-Score Scaled) $R^2$ Score:** $0.5758$
- **Scatter Plot Diagnostic (`actual_vs_predicted.png`):** Shows points centered along the diagonal $y = x$. A distinct horizontal line of points appears at $y = 5.0$, diagnosing that the California Housing median value target was capped at \$500,000.

### Iris Species Classification
- **Algorithm:** `LogisticRegression(max_iter=200)`
- **Accuracy Score:** $1.000$ (Test set, $N = 30$)
- **Algorithm:** `DecisionTreeClassifier(random_state=42)`
- **Accuracy Score:** $1.000$
- **API Consistency:** Demonstrates Scikit-learn's uniform estimator design pattern (`.fit()`, `.predict()`, `.score()`).

---

## 2. In-Lab Exercises

### Exercise 1: Invariance of OLS under Feature Standardization
- **Plain $R^2$:** $0.575788$
- **Pipeline $R^2$:** $0.575788$
- **Mathematical Explanation:** Ordinary Least Squares solves the normal equation $\hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$. Standardizing features via an invertible affine transformation $\tilde{\mathbf{X}} = (\mathbf{X} - \boldsymbol{\mu}) \boldsymbol{\Sigma}^{-1}$ preserves the linear subspace spanned by the column vectors of $\mathbf{X}$. The projection matrix $\mathbf{P} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T$ and predicted values $\hat{\mathbf{y}} = \mathbf{P}\mathbf{y}$ remain mathematically identical.
- **Models where scaling DOES matter:**
  1. Regularized regression (Ridge, Lasso, ElasticNet) where penalties $\lambda \|\mathbf{w}\|_2^2$ or $\lambda \|\mathbf{w}\|_1$ penalize unscaled coefficients inequitably.
  2. Distance-based estimators: $k$-Nearest Neighbors (KNN) and Support Vector Machines (SVM).
  3. Neural Networks / MLPs trained via gradient descent.

### Exercise 2: Feature Coefficients Comparison
- **Top 2 Raw Features:** `AveBedrms` ($\beta \approx +0.783$) and `MedInc` ($\beta \approx +0.449$).
- **Why raw coefficients cannot be directly compared:** Raw coefficients represent the expected change in target per unit change in that specific feature. Because features have drastically different units and scales (e.g., median income in tens of thousands vs population in thousands), coefficient magnitude reflects the arbitrary unit of measurement rather than relative feature importance. Standardizing features to unit variance resolves this disparity.

### Exercise 3: Iris Confusion Matrix Analysis
- Setosa (Class 0) is linearly separable with zero confusion.
- The two species susceptible to mutual misclassification in larger or noisy splits are **Versicolor** (Class 1) and **Virginica** (Class 2), as their petal width and length distributions partially overlap near the decision threshold.

### Exercise 4: Estimator API Polymorphism
Replacing `LogisticRegression` with `DecisionTreeClassifier` required changing only the instantiated class. The downstream evaluation calls (`.fit(X_train, y_train)`, `.predict(X_test)`, `confusion_matrix()`, `classification_report()`) remained completely unchanged, illustrating the power and consistency of Scikit-learn's estimator abstraction.

---

## 3. Home Assignment: Sentiment Classification Pipeline
- **Dataset:** Scraped news/blog corpus (`articles.csv`).
- **Target Formulation:** Binary split on the corpus median sentiment polarity ($1$ if above median, $0$ if below/equal).
- **Architecture:** Chained Scikit-learn `Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('classifier', LogisticRegression(random_state=42))])`.
- **Model Persistence:** Serialized using `joblib.dump(sentiment_pipeline, "sentiment_pipeline.pkl")`.
- **Inference Verification:** The loaded model was evaluated on unseen, raw natural language strings, demonstrating end-to-end tokenization, TF-IDF calculation, and probabilistic classification with zero manual preprocessing.
