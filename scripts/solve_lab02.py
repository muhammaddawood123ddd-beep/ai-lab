import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

def solve_lab02():
    os.makedirs("lab02/figures", exist_ok=True)
    notebook_path = "Lab_02_Web_Scraping_and_EDA.ipynb"
    print(f"Reading {notebook_path}...")
    nb = nbformat.read(notebook_path, as_version=4)

    # Fix Cell 3 line magic (%python -> !python or comment out download if already installed)
    nb.cells[3].source = '''# Installation of required libraries (run once per environment)
%pip install requests beautifulsoup4 lxml pandas spacy matplotlib seaborn textblob

# Ensure English language model for spaCy is available
try:
    import spacy
    spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' is already installed and loaded.")
except Exception:
    !python -m spacy download en_core_web_sm'''

    # Cell 8: Scraper with original 5 URLs + 2 additional URLs for Exercise 1
    nb.cells[8].source = '''import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

urls = [
    'https://techncruncher.blogspot.com/2025/01/top-10-ai-tools-that-will-transform.html',
    'https://techncruncher.blogspot.com/2023/12/limewire-ai-studio-review-2023-details.html',
    'https://techncruncher.blogspot.com/2023/01/top-10-ai-tools-in-2023-that-will-make.html',
    'https://techncruncher.blogspot.com/2022/11/top-10-ai-content-generator-writer.html',
    'https://techncruncher.blogspot.com/2022/09/cj-affiliate-ultimate-guide-to.html',
    # Exercise 1 additions:
    'https://techncruncher.blogspot.com/2022/04/what-is-blockchain-everything-you-need.html',
    'https://en.wikipedia.org/wiki/Artificial_intelligence'
]

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
articles = []

for url in urls:
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, 'lxml')

        # Selector for Blogspot
        title = soup.find('h3', class_='post-title')
        if not title:
            # Fallback for pages with different header structure
            title = soup.find('h1') or soup.title
        title_text = title.get_text(strip=True) if title else "No Title"

        content_div = soup.find('div', class_='post-body entry-content')
        if not content_div:
            # Non-blogspot page (Exercise 1 selector mismatch demonstration)
            print(f"[SELECTOR MISMATCH / FALLBACK] {url}: 'div.post-body.entry-content' not found.")
            content_div = soup.find('div', id='mw-content-text') or soup.find('article') or soup.find('body')

        if content_div:
            paragraphs = content_div.find_all('p')
            text = ' '.join(p.get_text(strip=True) for p in paragraphs)
        else:
            text = ''

        if len(text) > 100:
            articles.append({'url': url, 'title': title_text, 'text': text})
            print(f"[OK] Scraped: {title_text[:45]}... ({len(text)} chars)")
        else:
            print(f"[SKIP] Too short or empty: {url}")

        time.sleep(1)
    except Exception as e:
        print(f"[ERROR] {url} -> {e}")

df = pd.DataFrame(articles)
# Compute basic feature: title length
df['title_length'] = df['title'].apply(len)
print(f"\\nSuccessfully prepared corpus with {len(df)} articles.")
display(df[['title', 'title_length']].head())'''

    # Cell 18: Save figure outputs to lab02/figures
    nb.cells[18].source = '''import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("lab02/figures", exist_ok=True)
plt.style.use('ggplot')

# 1. Title Length Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['title_length'], kde=True, bins=5, color='skyblue')
plt.title('Distribution of Article Title Lengths')
plt.xlabel('Title Length (Characters)')
plt.ylabel('Frequency')
plt.savefig("lab02/figures/title_length_distribution.png", dpi=150)
plt.show()

# 2. Sentiment Polarity Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['sentiment_polarity'], kde=True, bins=5, color='lightcoral')
plt.title('Distribution of Article Sentiment Polarity')
plt.xlabel('Polarity Score (-1.0 to 1.0)')
plt.ylabel('Frequency')
plt.savefig("lab02/figures/sentiment_distribution.png", dpi=150)
plt.show()'''

    # Cell 20: Save pairplot
    nb.cells[20].source = '''numeric_cols = [
    'title_length',
    'num_tokens',
    'num_sentences',
    'num_entities',
    'sentiment_polarity'
]

g = sns.pairplot(df[numeric_cols])
plt.suptitle('Pair Plot of Derived Features', y=1.02)
g.savefig("lab02/figures/pair_plot.png", dpi=150)
plt.show()'''

    # Cell 23: Save TF-IDF figure
    nb.cells[23].source = '''from sklearn.feature_extraction.text import TfidfVectorizer

# stop_words='english' discards 'the', 'a', ...; max_features keeps the top 20
vectorizer = TfidfVectorizer(stop_words='english', max_features=20)

X = vectorizer.fit_transform(df['text'])
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Sum each term's score across all documents to rank global importance
tfidf_sums = tfidf_df.sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=tfidf_sums.index, y=tfidf_sums.values)
plt.title("Top 20 TF-IDF Terms Across Articles")
plt.ylabel("TF-IDF Score Sum")
plt.xlabel("Term")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig("lab02/figures/tfidf_top20.png", dpi=150)
plt.show()

print("Sparse matrix shape (documents x terms):", X.shape)'''

    # Cell 27: Exercise 3 written answer
    nb.cells[27].source = """**Your answer to Exercise 3 (why a KDE over five points is not a population distribution):**

A Kernel Density Estimate (KDE) synthesizes a continuous probability distribution by superimposing smooth Gaussian kernels with a chosen bandwidth parameter over individual data points. With only five sample observations ($N = 5$), the shape, spread, and modal peaks of the curve are almost entirely mathematical artifacts of the bandwidth selection rather than genuine representations of the underlying population distribution, conveying an illusion of smooth probability where statistical support is virtually absent. Instead of presenting a KDE on very small samples, researchers should report the empirical observations directly using a strip plot, rug plot, or dot plot alongside a summary table with transparent sample-size disclosure."""

    # Cell 29: Home Assignment Complete Implementation
    nb.cells[29].source = '''# --- Home assignment complete implementation --------------------------------------------
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import os

nlp = spacy.load("en_core_web_sm")

SOURCE_A = {
    "name": "TechnCruncher (Tech Blog)",
    "urls": [
        'https://techncruncher.blogspot.com/2025/12/how-i-get-free-traffic-from-chatgpt-in.html',
        'https://techncruncher.blogspot.com/2025/01/top-10-ai-tools-that-will-transform.html',
        'https://techncruncher.blogspot.com/2023/12/limewire-ai-studio-review-2023-details.html',
        'https://techncruncher.blogspot.com/2023/01/top-10-ai-tools-in-2023-that-will-make.html',
        'https://techncruncher.blogspot.com/2022/11/top-10-ai-content-generator-writer.html',
        'https://techncruncher.blogspot.com/2022/03/prowritingaid-vs-grammarly-which.html',
        'https://techncruncher.blogspot.com/2022/07/top-10-ai-marketing-tools-you-should-use.html',
        'https://techncruncher.blogspot.com/2022/09/cj-affiliate-ultimate-guide-to.html',
        'https://techncruncher.blogspot.com/2022/04/what-is-blockchain-everything-you-need.html',
        'https://techncruncher.blogspot.com/p/a-bout-us-hello-friends-welcome-to.html'
    ],
    "title_selector": ("h3", "post-title"),
    "content_selector": ("div", "post-body entry-content")
}

SOURCE_B = {
    "name": "Wikipedia AI Compendium (Reference)",
    "urls": [
        'https://en.wikipedia.org/wiki/Artificial_intelligence',
        'https://en.wikipedia.org/wiki/Machine_learning',
        'https://en.wikipedia.org/wiki/Deep_learning',
        'https://en.wikipedia.org/wiki/Natural_language_processing',
        'https://en.wikipedia.org/wiki/Computer_vision',
        'https://en.wikipedia.org/wiki/Artificial_neural_network',
        'https://en.wikipedia.org/wiki/Supervised_learning',
        'https://en.wikipedia.org/wiki/Reinforcement_learning',
        'https://en.wikipedia.org/wiki/Support_vector_machine',
        'https://en.wikipedia.org/wiki/Decision_tree'
    ],
    "title_selector": ("h1", None),
    "content_selector": ("div", "mw-parser-output")
}

def scrape_source(source):
    rows = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in source["urls"]:
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.content, "lxml")

            tag, cls = source["title_selector"]
            t = soup.find(tag, class_=cls) if cls else soup.find(tag)
            title = t.get_text(strip=True) if t else soup.title.get_text(strip=True)

            tag, cls = source["content_selector"]
            block = soup.find(tag, class_=cls) if cls else soup.find(tag)
            if not block:
                paragraphs = soup.find_all("p")
            else:
                paragraphs = block.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs[:15])

            if len(text) > 100:
                rows.append({"source": source["name"], "url": url, "title": title, "text": text})
            time.sleep(0.5)
        except Exception as e:
            print(f"[ERROR] {url} -> {e}")
    return pd.DataFrame(rows)

print("Scraping Source A and Source B...")
corpus = pd.concat([scrape_source(SOURCE_A), scrape_source(SOURCE_B)], ignore_index=True)
corpus.to_csv("two_source_corpus.csv", index=False)
print(f"Scraped {len(corpus)} total records into two_source_corpus.csv.")

# Feature extraction for sentiment polarity and noun density
def compute_source_metrics(text):
    doc = nlp(text[:2500])
    tokens = len(doc)
    nouns = len([t for t in doc if t.pos_ in ('NOUN', 'PROPN')])
    noun_density = (nouns / tokens) if tokens > 0 else 0.0
    blob = TextBlob(text)
    return pd.Series({
        'sentiment_polarity': blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity,
        'noun_density': noun_density
    })

metrics_df = corpus['text'].apply(compute_source_metrics)
corpus = pd.concat([corpus, metrics_df], axis=1)

# Grouped statistics
summary = corpus.groupby('source')[['sentiment_polarity', 'noun_density', 'sentiment_subjectivity']].agg(['mean', 'std']).round(4)
print("\\n--- Grouped Comparison Summary ---")
display(summary)

# Numerical difference calculation
mean_a = corpus[corpus['source'] == SOURCE_A['name']][['sentiment_polarity', 'noun_density']].mean()
mean_b = corpus[corpus['source'] == SOURCE_B['name']][['sentiment_polarity', 'noun_density']].mean()
diff = mean_a - mean_b
print(f"\\nDifference (Source A - Source B):")
print(f"  Sentiment Polarity Delta: {diff['sentiment_polarity']:+.4f}")
print(f"  Noun Density Delta      : {diff['noun_density']:+.4f}")

# Grouped bar chart
plot_data = corpus.melt(id_vars=['source'], value_vars=['sentiment_polarity', 'noun_density'], 
                        var_name='Metric', value_name='Score')

plt.figure(figsize=(9, 5))
sns.barplot(data=plot_data, x='Metric', y='Score', hue='source', ci=None, palette='Set2')
plt.title('Mean Sentiment Polarity and Noun Density by Article Source')
plt.ylabel('Mean Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
os.makedirs("lab02/figures", exist_ok=True)
plt.savefig("lab02/figures/two_source_comparison.png", dpi=150)
plt.show()

print("Grouped bar chart successfully generated and saved to lab02/figures/two_source_comparison.png")'''

    # Save updated notebook
    nbformat.write(nb, notebook_path)
    print("Notebook updated. Now executing all cells...")

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "."}})
    
    nbformat.write(nb, notebook_path)
    print("Lab 02 notebook successfully executed and saved with outputs!")

if __name__ == "__main__":
    solve_lab02()
