# Laboratory 03: Transforming Textual and Image Data into Machine-Understandable Formats

**Student:** Muhammad Dawood  
**Course:** Introduction to Artificial Intelligence  
**Artifacts Generated:** `lab03/image_utils.py`, `dataset.npz`, `lab03/dataset.npz`

---

## 1. Discussion Questions (Part A: Text Vectorization)

1. **What does each column represent in the Bag-of-Words matrix?**  
   Each column corresponds to a distinct, unique token (unigram or n-gram) present in the fitted vocabulary across the corpus, sorted alphabetically.
   
2. **What is the machine-understandable format?**  
   A 2D numerical matrix (sparse document-term matrix or dense 2D NumPy array) of integers where rows represent individual documents and columns represent token frequencies.
   
3. **How is this format an oversimplification of original text?**  
   Bag-of-Words completely discards word order, grammatical tense, syntactic structure, and modifier context (treating *"not bad"* and *"bad, not"* identically). It also ignores word polysemy and semantic synonymy.

---

## 2. Discussion Questions (Part B: Image Feature Extraction)

1. **Grayscale conversion as feature engineering:**  
   Grayscale conversion collapses the three correlated RGB color channels ($R, G, B$) into a single luminance intensity channel ($Y = 0.299R + 0.587G + 0.114B$). This achieves an exact $3\times$ (66.7%) dimensionality reduction while preserving essential geometric contours, spatial edges, and contrast differences.
   
2. **Length of the flattened vector:**  
   For an image of resolution $H \times W$ pixels converted to single-channel grayscale, the flattened 1D feature vector has length exactly $H \cdot W$ scalar values.
   
3. **Why flattening is necessary:**  
   Standard classical estimators (e.g., Logistic Regression, SVMs, Decision Trees) operate on tabular matrices $\mathbf{X} \in \mathbb{R}^{N \times D}$. Two-dimensional spatial matrices cannot be directly multiplied by 1D weight vectors ($w^T x + b$) without unrolling into a 1D vector of scalar pixel features.
   
4. **Advantage of resizing (e.g., $64 \times 64$):**  
   Real-world images arrive in heterogeneous resolutions and aspect ratios. Resizing enforces fixed input dimensionality ($64 \times 64 = 4{,}096$ features) across all instances, which is required for batch matrix operations and fixed-weight model architectures.

---

## 3. Home-Lab Exercises Summary

- **Exercise 1 (N-gram range `(1, 2)`):**  
  Adding bigrams expanded the vocabulary from 13 unigram features to 22 n-gram features because all valid consecutive 2-word combinations were extracted.
  
- **Exercise 2 (Document Length Bias):**  
  A repetitive document containing the token *"sun"* five times yields an inflated count of 5 in that column. Raw frequency counts disproportionately weigh document length rather than informational significance, motivating length normalization (TF-IDF or L2 norm).
  
- **Exercise 3 (Multiscale Resolution & Information Loss):**  
  Downsampling from original resolution to $64\times 64$ ($4{,}096$ features) and $32\times 32$ ($1{,}024$ features) degrades high-frequency edge textures and fine anti-aliasing details.
  
- **Exercise 4 (`image_to_features` Pipeline):**  
  The function `image_to_features(path, size=(64, 64))` was verified against three differently sized input images, confirming that all outputs produce uniform 4,096-dimensional vectors.

---

## 4. Home Assignment: Learning in 4,096 Dimensions with 20 Samples ($p \gg N$)

Learning a decision boundary with only 20 training samples ($N = 20$) in a 4,096-dimensional feature space ($p = 4{,}096$) is fundamentally ill-posed due to the **curse of dimensionality** ($p \gg N$). In such high-dimensional spaces, samples become equidistant and isolated in vast, empty feature space, causing linear classifiers to overfit noise and achieve zero generalization. To train reliable models on image data, dimensionality reduction techniques (e.g., PCA in Laboratory 6) or parameter-efficient convolutional networks (Laboratory 8) with shared kernel weights and spatial inductive bias are strictly required.
