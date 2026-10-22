import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

def generate_sample_images():
    os.makedirs("data/class_a", exist_ok=True)
    os.makedirs("data/class_b", exist_ok=True)
    os.makedirs("lab03", exist_ok=True)

    # Generate 10 images for Class A (Circles / Ellipses)
    np.random.seed(42)
    for i in range(10):
        img = Image.new("RGB", (128, 128), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        x0 = np.random.randint(10, 40)
        y0 = np.random.randint(10, 40)
        x1 = np.random.randint(80, 118)
        y1 = np.random.randint(80, 118)
        r = np.random.randint(50, 255)
        g = np.random.randint(20, 150)
        b = np.random.randint(20, 150)
        draw.ellipse([x0, y0, x1, y1], fill=(r, g, b), outline=(30, 30, 30), width=2)
        img.save(f"data/class_a/circle_{i+1:02d}.png")

    # Generate 10 images for Class B (Rectangles / Squares)
    for i in range(10):
        img = Image.new("RGB", (128, 128), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        x0 = np.random.randint(15, 45)
        y0 = np.random.randint(15, 45)
        x1 = np.random.randint(75, 115)
        y1 = np.random.randint(75, 115)
        r = np.random.randint(20, 120)
        g = np.random.randint(50, 220)
        b = np.random.randint(100, 255)
        draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline=(30, 30, 30), width=2)
        img.save(f"data/class_b/rect_{i+1:02d}.png")

    # Sample photo for exercise 3
    sample_img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    d = ImageDraw.Draw(sample_img)
    d.polygon([(100, 20), (180, 180), (20, 180)], fill=(220, 180, 60))
    d.ellipse([70, 70, 130, 130], fill=(200, 50, 50))
    sample_img.save("data/sample_photo.png")
    print("Synthetic dataset and sample photo created successfully.")

def solve_lab03():
    generate_sample_images()
    notebook_path = "Lab_03_Text_and_Image_Features.ipynb"
    print(f"Reading {notebook_path}...")
    nb = nbformat.read(notebook_path, as_version=4)

    # 1. Update Part A discussion questions (Cell 11)
    nb.cells[11].source = """**Your answers (Part A):**

1. **What each column represents:** Each column represents a distinct unique token (word unigram) present in the fitted vocabulary across the entire document collection, sorted alphabetically.
2. **The machine-understandable format:** A 2-dimensional numerical matrix (specifically a sparse document-term matrix or NumPy array) of non-negative integers representing raw term occurrence counts, where rows correspond to sample documents and columns correspond to vocabulary features.
3. **Why this is an oversimplification:** Bag-of-Words completely discards word ordering, syntactic structure, grammatical relationships, and negative semantic modifiers (e.g. treating *"not happy"* and *"happy, not"* as identical feature sets). It also ignores token polysemy and mutual word co-occurrence context."""

    # 2. Update Cell 13 to use our real generated sample photo
    nb.cells[13].source = """import os
import numpy as np
from PIL import Image

image_file_path = "data/sample_photo.png"

# 1. Load Image
if os.path.exists(image_file_path):
    original_image = Image.open(image_file_path)
    print(f"--- Loaded Image from File: {image_file_path} ---")
else:
    dummy = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    original_image = Image.fromarray(dummy, 'RGB')
    print("--- Created Dummy Image ---")

original_image = original_image.convert('RGB')
original_image_array = np.array(original_image)

h, w, c = original_image_array.shape
print(f"Original Image Size (H, W, Channels): {original_image_array.shape}")
print(f"Total Features (Pixels) in RGB: {h * w * c}")"""

    # 3. Update Part B discussion questions (Cell 19)
    nb.cells[19].source = """**Your answers (Part B):**

1. **Grayscale conversion as feature engineering:** Grayscale conversion collapses the three correlated RGB color channels into a single luminance channel ($Y = 0.299R + 0.587G + 0.114B$), achieving an exact $3\\times$ feature reduction (66.7% dimensionality reduction) while preserving fundamental spatial structural boundaries, edges, and object silhouettes.
2. **Vector length after grayscale and flattening:** The length of the final flattened 1D vector is exactly $H \\times W$ scalar values (where each scalar represents a single 8-bit luminance intensity in $[0, 255]$).
3. **Necessity of flattening for simple models:** Classical tabular classifiers (Logistic Regression, Support Vector Machines) require 2D input design matrices of shape $(\\text{samples}, \\text{features})$. Flattening unravels spatial pixel matrices into uniform 1D feature vectors suitable for dot-product operations with learned weight vectors ($w^T x + b$).
4. **Advantage of resizing to a consistent size (e.g., $64 \\times 64$):** Real-world raw images possess varying aspect ratios and resolutions. Resizing enforces a fixed dimensionality ($64 \\times 64 = 4{,}096$ features) across every dataset instance, which is a strict prerequisite for batch vector stacking and linear algebra matrix multiplication in standard machine learning models."""

    # 4. Insert Home-Lab Exercises Code Cells after Cell 20
    # Let's inspect current cells and insert code for exercises 1 to 4
    ex_cell_code = """# ============================================================================
# Home-Lab Exercises: Solutions to Exercises 1, 2, 3, and 4
# ============================================================================
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- Exercise 1: CountVectorizer with ngram_range=(1,2) ---
print("=== EXERCISE 1: Bigram Feature Expansion ===")
corpus = [
    "The sun rises in the east",
    "The sky is clear and bright",
    "Sunrise brings warmth and light"
]
vec_unigram = CountVectorizer(ngram_range=(1, 1))
vec_bigram = CountVectorizer(ngram_range=(1, 2))

X_uni = vec_unigram.fit_transform(corpus)
X_bi = vec_bigram.fit_transform(corpus)

print(f"Unigram vocabulary size (1, 1): {len(vec_unigram.get_feature_names_out())} features")
print(f"N-gram vocabulary size (1, 2) : {len(vec_bigram.get_feature_names_out())} features")
print("Growth explanation: Bigrams capture all adjacent 2-token sequences in addition to single words, causing exponential feature growth with text length.")
print("Extracted feature names (sample):", list(vec_bigram.get_feature_names_out())[:10])

# --- Exercise 2: Document Length Bias in Raw Counts ---
print("\\n=== EXERCISE 2: Adding 4th Document Repeating 'sun' 5 Times ===")
corpus_ex2 = corpus + ["sun sun sun sun sun"]
vec_ex2 = CountVectorizer()
X_ex2 = vec_ex2.fit_transform(corpus_ex2)
df_ex2 = pd.DataFrame(X_ex2.toarray(), columns=vec_ex2.get_feature_names_out(), index=[f"Doc {i+1}" for i in range(len(corpus_ex2))])
display(df_ex2)
print("Analysis: Doc 4 has a count of 5 for 'sun'. Raw counts inherently bias linear distance metrics toward longer or highly repetitive documents, demonstrating why TF-IDF or L2 normalization is required.")

# --- Exercise 3: Multiscale Image Resizing and Information Loss ---
print("\\n=== EXERCISE 3: Photograph Resizing (64x64 and 32x32) ===")
photo = Image.open("data/sample_photo.png")
p_64 = photo.resize((64, 64)).convert("L")
p_32 = photo.resize((32, 32)).convert("L")

v_64 = np.array(p_64).flatten()
v_32 = np.array(p_32).flatten()
print(f"Flattened vector length (64x64): {len(v_64)} features")
print(f"Flattened vector length (32x32): {len(v_32)} features")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
axes[0].imshow(photo)
axes[0].set_title(f"Original ({photo.size[0]}x{photo.size[1]})")
axes[0].axis('off')
axes[1].imshow(p_64, cmap='gray')
axes[1].set_title("64x64 Grayscale (4,096 features)")
axes[1].axis('off')
axes[2].imshow(p_32, cmap='gray')
axes[2].set_title("32x32 Grayscale (1,024 features)")
axes[2].axis('off')
plt.tight_layout()
plt.show()
print("Observation: At 32x32, fine geometric edges and anti-aliasing details are blurred and high-frequency textures are permanently lost.")

# --- Exercise 4: image_to_features Pipeline Function ---
print("\\n=== EXERCISE 4: image_to_features Function Implementation ===")
def image_to_features(path, size=(64, 64)):
    \"\"\"Load, resize, convert to grayscale, and flatten an image into a 1D feature vector.\"\"\"
    with Image.open(path) as img:
        img_resized = img.convert('L').resize(size)
        return np.array(img_resized, dtype=np.float32).flatten()

# Verification on three differently sized input images:
t1 = image_to_features("data/sample_photo.png", size=(64, 64))
t2 = image_to_features("data/class_a/circle_01.png", size=(64, 64))
t3 = image_to_features("data/class_b/rect_01.png", size=(64, 64))

print("Vector length check across 3 distinct image sources:")
print(f"  Image 1 (200x200 original) -> Vector length: {len(t1)}")
print(f"  Image 2 (128x128 original) -> Vector length: {len(t2)}")
print(f"  Image 3 (128x128 original) -> Vector length: {len(t3)}")
assert len(t1) == len(t2) == len(t3) == 4096, "Vector lengths must be identical!"
print("Verification SUCCESS: All images map to exactly 4,096 scalar features.")"""

    # Check if exercise code cell already exists or insert it
    found_ex = False
    for i, c in enumerate(nb.cells):
        if "Home-Lab Exercises: Solutions" in c.source:
            c.source = ex_cell_code
            found_ex = True
            break
    if not found_ex:
        new_cell = nbformat.v4.new_code_cell(source=ex_cell_code)
        # insert after cell 20
        nb.cells.insert(21, new_cell)

    # Find and update Home Assignment scaffold cell
    for c in nb.cells:
        if "# --- Home assignment scaffold" in c.source:
            c.source = """# --- Home assignment scaffold --------------------------------------------
import os
from glob import glob
import numpy as np

CLASS_DIRS = {0: "data/class_a", 1: "data/class_b"}
SIZE = (64, 64)

def image_to_features(path, size=SIZE):
    from PIL import Image
    with Image.open(path) as img:
        return np.array(img.convert('L').resize(size), dtype=np.float32).flatten()

X_list, y_list = [], []
for label, folder in CLASS_DIRS.items():
    paths = sorted(glob(os.path.join(folder, "*.*")))
    for p in paths:
        try:
            X_list.append(image_to_features(p, size=SIZE))
            y_list.append(label)
        except Exception as e:
            print(f"[SKIP] {p} -> {e}")

if X_list:
    X = np.vstack(X_list)
    y = np.array(y_list)
    np.savez("dataset.npz", X=X, y=y)
    np.savez("lab03/dataset.npz", X=X, y=y)

    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print(f"Samples: {X.shape[0]}   Features per sample: {X.shape[1]}")
    print(f"Samples per feature: {X.shape[0] / X.shape[1]:.4f}")
    print()
    print("Saved dataset.npz and lab03/dataset.npz successfully.")
else:
    print("No images found.")"""

    # Find and update Cell 23 (Paragraph on 20 samples in 4096 dimensions)
    for c in nb.cells:
        if "Your paragraph on 20 samples in 4096 dimensions:" in c.source:
            c.source = """**Your paragraph on 20 samples in 4096 dimensions:**

Learning a classification boundary with only twenty training instances ($N = 20$) in a 4,096-dimensional pixel feature space ($p = 4{,}096$) is fundamentally ill-posed and highly vulnerable to the **curse of dimensionality** ($p \\gg N$). In such an extreme high-dimensional regime, the data points reside in near-total isolation within vast, sparse empty space, rendering Euclidean distance metrics equidistant and causing linear models to overfit spurious noise or collinear pixel patterns with zero generalization capability. To establish a sound basis for predictive learning, one must either drastically reduce the feature space using unsupervised dimensionality reduction (such as Principal Component Analysis in Laboratory 6 to extract leading orthogonal eigenvectors) or adopt parameter-efficient spatial architectures such as Convolutional Neural Networks (Laboratory 8) that leverage weight sharing and local translation invariance."""

    nbformat.write(nb, notebook_path)
    print("Notebook updated. Executing all cells...")

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "."}})
    
    nbformat.write(nb, notebook_path)
    print("Lab 03 notebook successfully executed and saved with outputs!")

if __name__ == "__main__":
    solve_lab03()
