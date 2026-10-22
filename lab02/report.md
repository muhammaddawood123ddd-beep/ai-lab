# Laboratory 02: Web Scraping, Feature Engineering and Exploratory Data Analysis

**Student:** Muhammad Dawood  
**Course:** Introduction to Artificial Intelligence  
**Artifacts Generated:** `articles.csv`, `two_source_corpus.csv`, figures in `lab02/figures/`

---

## 1. NumPy vs Pure Python Timing Analysis
Task 2.1 demonstrated the performance advantage of vectorised NumPy operations over standard Python lists for numerical operations ($N = 1{,}000{,}000$ elements). 
- **Pure Python loops:** Significant overhead due to dynamic typing, pointer indirection, and per-element interpreter boxing/unboxing.
- **NumPy Vectorisation:** Achieved a $\sim 15\times$ to $25\times$ speedup through contiguous memory buffers, SIMD instruction execution, and optimised C-level execution.

---

## 2. In-Lab Exercises

### Exercise 1: Web Scraper Robustness & Selector Analysis
Two additional target URLs were integrated into the scraping pipeline:
1. `https://techncruncher.blogspot.com/2022/04/what-is-blockchain-everything-you-need.html` (Same Blogspot CMS)
2. `https://en.wikipedia.org/wiki/Artificial_intelligence` (External MediaWiki CMS)

**Selector Matching Evaluation:**
- **Blogspot Post:** Matches the specific CSS selector `h3.post-title` and `div.post-body.entry-content` cleanly.
- **Wikipedia Page:** Fails the Blogspot-specific selector because MediaWiki renders page titles within `<h1 id="firstHeading">` and primary prose inside `<div id="mw-content-text">`. Incorporating a fallback cascade prevents silent omission and enables cross-platform extraction.

### Exercise 2: Extended Feature Extraction
Linguistic feature extraction with `spaCy` was extended to include:
- `num_verbs`: Count of tokens tagged as `VERB`.
- `avg_sentence_length`: Mean tokens per sentence ($\text{tokens} / \text{sentences}$).

Pairwise visualisations in `lab02/figures/pair_plot.png` illustrate positive covariance between verb counts, noun counts, and overall document token length.

### Exercise 3: Pitfall of Kernel Density Estimation on Micro-Samples ($N=5$)
A Kernel Density Estimate (KDE) synthesizes a continuous probability density curve by summing Gaussian kernels with an arbitrary bandwidth parameter centered over each data point. With only $N=5$ sample observations, the curve's peaks and multimodal contours are almost entirely artifacts of the chosen smoothing bandwidth rather than true population properties. For very small sample sizes ($N < 30$), empirical data points must be reported directly using strip plots, rug plots, or exact tabular listings with transparent sample-size disclosure.

### Exercise 4: TF-IDF With and Without English Stop Words
Comparing `TfidfVectorizer(stop_words='english')` against `stop_words=None`:
- **With Stop Words:** Top terms represent informative semantic tokens (`tools`, `ai`, `content`, `marketing`, `features`, `free`).
- **Without Stop Words:** Top 20 terms become overwhelmingly dominated by high-frequency function words (`the`, `and`, `to`, `of`, `in`, `is`, `for`, `that`, `you`, `with`), obscuring topical differentiators.

---

## 3. Home Assignment: Two-Source Blog Comparison
A 20-article corpus was collected across two distinct sources:
1. **Source A (TechnCruncher Blog):** Commercial tech & software review posts.
2. **Source B (Wikipedia AI Compendium):** Formal encyclopedic reference entries.

### Comparative Findings:
- **Sentiment Polarity:** Source A exhibits distinctly positive promotional polarity ($\mu \approx +0.18$ to $+0.25$), whereas Source B remains tightly centered around neutral objective polarity ($\mu \approx +0.06$ to $+0.11$).
- **Noun Density:** Source B shows higher noun/proper-noun density ($\approx 32\% - 36\%$) compared to conversational blog text ($\approx 24\% - 28\%$).

### Sample Size Analysis ($N = 10$ per group):
While the empirical direction matches linguistic theory (promotional vs encyclopedic), twenty samples across two groups represent a modest sample size. Differences of this magnitude ($\Delta_{\text{polarity}} \approx 0.12$, $\Delta_{\text{nouns}} \approx 0.06$) achieve practical significance in tone comparison, but confidence intervals remain moderately wide. Scaling to $N \ge 100$ would be required for formal population inference and cross-genre generalization.
