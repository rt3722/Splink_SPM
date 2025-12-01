# What is BlockingPy?

## Overview

**BlockingPy** is a Python package for **Approximate Nearest Neighbour (ANN) blocking** in entity resolution (record linkage and deduplication) tasks. It was developed by researchers at Poznań University of Economics and Business and Statistics Poland.

Unlike traditional deterministic blocking methods that require exact matches on specific fields, BlockingPy uses modern ANN search algorithms and graph-based techniques to find similar records even when data contains errors, missing values, or inconsistencies.

**Paper:** [BlockingPy: approximate nearest neighbours for blocking of records for entity resolution](https://arxiv.org/html/2504.04266v3)

---

## The Problem BlockingPy Solves

### Why Blocking is Needed

In entity resolution, comparing every record with every other record is computationally infeasible for large datasets:
- 1 million records = ~500 billion comparisons
- 10 million records = ~50 trillion comparisons

**Blocking** reduces this by grouping potentially matching records into "blocks" and only comparing records within the same block.

### Limitations of Traditional Blocking

Traditional deterministic blocking methods (like those in Splink or recordlinkage) have significant limitations:

| Issue | Description |
|-------|-------------|
| **Exact match requirement** | `block_on("surname")` misses "Smith" vs "Smyth" |
| **Missing data** | Records with null values won't match |
| **Transliteration errors** | "Олександр" → "Aleksandr" vs "Ołeksandr" |
| **Typos** | "John" vs "Jhon" won't be blocked together |
| **Inconsistent formatting** | "123 Main St" vs "123 Main Street" |

### BlockingPy's Solution

BlockingPy uses **similarity-based blocking** through Approximate Nearest Neighbour algorithms:

1. **Convert records to vectors** (n-grams or embeddings)
2. **Find similar records** using ANN search
3. **Group into blocks** using graph connected components

This allows matching records even with data quality issues.

---

## Key Features

### 1. Multiple ANN Algorithms

BlockingPy supports a variety of state-of-the-art ANN algorithms:

| Algorithm | Library | Description |
|-----------|---------|-------------|
| **HNSW** | faiss, hnswlib, voyager | Hierarchical Navigable Small World graphs |
| **LSH** | mlpack, faiss | Locality Sensitive Hashing |
| **KNN** | mlpack, faiss | Exact k-Nearest Neighbours |
| **NND** | pynndescent | Nearest Neighbour Descent |
| **Annoy** | annoy | Random projections and NN trees |
| **IVF/IVFPQ** | faiss (GPU) | Inverted File with Product Quantization |
| **CAGRA** | faiss (GPU) | GPU-accelerated graph ANN |

### 2. CPU and GPU Support

```bash
# CPU installation
pip install blockingpy

# GPU installation (CUDA-enabled)
pip install blockingpy-gpu
```

### 3. Flexible Input Processing

- **Text data** → Automatically converted to n-gram representations
- **Pre-computed embeddings** → Direct vector input supported
- **Document-Term Matrices (DTMs)** → Sparse matrix input

### 4. Integration with Other Libraries

BlockingPy integrates seamlessly with:
- **Splink** - Use BlockingPy's blocks as input to Splink's probabilistic matching
- **recordlinkage** - Direct integration via block column

### 5. Built-in Evaluation Metrics

When ground truth is available, BlockingPy computes:
- **Recall** (pair completeness)
- **Precision**
- **F1 Score**
- **Reduction Ratio** (RR)
- **Confusion Matrix**

---

## How It Works

### The BlockingPy Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  1. INPUT DATA                                                   │
│     - Raw text fields (names, addresses, etc.)                  │
│     - Or pre-computed embeddings                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. TEXT ENCODING                                               │
│     - Character n-gram representation (default)                 │
│     - Or vector embeddings (model2vec, etc.)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. ANN SEARCH                                                  │
│     - Find k nearest neighbours for each record                 │
│     - Using HNSW, LSH, Annoy, or other algorithms              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. GRAPH CONSTRUCTION & CONNECTED COMPONENTS                    │
│     - Build undirected graph from neighbour relationships       │
│     - Find connected components (= blocks)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. OUTPUT                                                      │
│     - DataFrame with record indices and block assignments       │
│     - Reduction ratio and block size distribution               │
└─────────────────────────────────────────────────────────────────┘
```

### Reduction Ratio

The **Reduction Ratio (RR)** measures how many comparisons are eliminated:

**For deduplication:**
```
RR = 1 - (comparisons after blocking) / (total possible comparisons)
```

A reduction ratio of 0.999 means 99.9% of unnecessary comparisons are eliminated.

---

## Usage Examples

### Basic Deduplication

```python
from blockingpy import Blocker
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Create a blocking key by concatenating relevant fields
df['txt'] = df['first_name'] + df['last_name'] + df['dob'] + df['address']

# Initialize blocker and run
blocker = Blocker()
result = blocker.block(
    x=df['txt'],
    ann='hnsw',       # Algorithm choice
    random_seed=42
)

# View results
print(result)
print(result.result.head())
```

**Output:**
```
====================================================
Blocking based on the hnsw method.
Number of blocks: 500
Number of columns used for blocking: 1023
Reduction ratio: 0.998999
====================================================
Distribution of the size of the blocks:
Block Size | Number of Blocks
2          | 500
```

### Record Linkage (Two Datasets)

```python
from blockingpy import Blocker

# Prepare datasets
dataset_a['txt'] = dataset_a['name'] + dataset_a['address'] + dataset_a['dob']
dataset_b['txt'] = dataset_b['name'] + dataset_b['address'] + dataset_b['dob']

# Block across datasets
blocker = Blocker()
result = blocker.block(
    x=dataset_a['txt'],  # Reference dataset
    y=dataset_b['txt'],  # Query dataset
    ann='hnsw',
    random_seed=42
)
```

### Fine-Tuning Algorithm Parameters

```python
# Custom HNSW parameters
control_ann = {
    'random_seed': 2025,
    'hnsw': {
        'distance': 'cosine',
        'k_search': 30,
        'n_threads': 4,
        'M': 25,
        'ef_c': 200,
        'ef_s': 200,
    }
}

result = blocker.block(
    x=df['txt'],
    ann='hnsw',
    control_ann=control_ann
)
```

### Evaluation with Ground Truth

```python
# If you have ground truth labels
true_blocks = pd.DataFrame({
    'x': [...],      # Record indices
    'y': [...],      # Paired record indices  
    'block': [...]   # True block assignments
})

# Evaluate blocking quality
eval_result = blocker.eval(result, true_blocks)

print(eval_result.metrics)
# recall       0.997000
# precision    1.000000
# f1_score     0.998498

print(eval_result.confusion)
#                  Predicted Positive  Predicted Negative
# Actual Positive               997                    3
# Actual Negative                 0               999000
```

---

## Integration with Splink

BlockingPy can significantly improve Splink's blocking by capturing pairs that deterministic rules would miss:

```python
from blockingpy import Blocker
from splink import Linker, DuckDBAPI, SettingsCreator, block_on
import splink.comparison_library as cl

# Step 1: Use BlockingPy for ANN-based blocking
df['txt'] = df['first_name'] + df['surname'] + df['dob']
blocker = Blocker()
blocking_result = blocker.block(x=df['txt'], ann='hnsw')

# Step 2: Add block column to your data
df['ann_block'] = blocking_result.add_block_column(df, id_col_left='id')

# Step 3: Use the block in Splink
settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("ann_block"),  # Use ANN blocks
        block_on("first_name", "surname"),  # Plus traditional rules
    ],
    comparisons=[
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.DateOfBirthComparison("dob", input_is_string=True),
    ],
)

linker = Linker(df, settings, db_api=DuckDBAPI())
# ... continue with Splink workflow
```

This hybrid approach combines:
- **BlockingPy's fuzzy matching** for catching similar records with errors
- **Splink's deterministic rules** for high-confidence matches
- **Splink's probabilistic scoring** for final match decisions

---

## Performance Benchmarks

From the paper's experiments on synthetic datasets:

### CPU Performance (150,000 records)

| Algorithm | Time (s) | Recall | Reduction Ratio | Pairs (M) |
|-----------|----------|--------|-----------------|-----------|
| BlockingPy (faiss_lsh) | 56.5 | 0.818 | 1.000 | 0.40 |
| BlockingPy (voyager) | 110.3 | 0.715 | 1.000 | 0.68 |
| BlockingPy (faiss_hnsw) | 250.0 | 0.832 | 1.000 | 0.38 |
| P-Sig (blocklib) | 3.4 | 0.609 | 0.996 | 40.23 |
| λ-fold LSH (blocklib) | 19.8 | 0.450 | 0.992 | 95.06 |

**Key insight:** BlockingPy achieves much higher recall with far fewer candidate pairs compared to blocklib.

### GPU Performance (150,000 records)

| Algorithm | Time (s) | Recall | Pairs (M) |
|-----------|----------|--------|-----------|
| gpu_faiss flat | 22.5 | 0.839 | 0.37 |
| gpu_faiss cagra | 51.1 | 0.827 | 0.38 |
| gpu_faiss ivf | 71.5 | 0.801 | 0.49 |

---

## Installation

### CPU Version

```bash
pip install blockingpy
# or
poetry add blockingpy
```

### GPU Version (CUDA)

```bash
# Create environment
mamba create -n blockingpy-gpu python=3.10 -y
conda activate blockingpy-gpu

# Install FAISS GPU
mamba install -c pytorch/label/nightly faiss-gpu-cuvs -y

# Install BlockingPy GPU
pip install blockingpy-gpu
```

---

## When to Use BlockingPy

### ✅ Good Use Cases

| Scenario | Why BlockingPy Helps |
|----------|---------------------|
| **Dirty data** | Handles typos, missing values, transliteration |
| **No shared identifiers** | Works on text similarity, not exact keys |
| **Large datasets** | Efficient ANN algorithms scale well |
| **Supplementing Splink** | Catches matches that deterministic rules miss |
| **Embedding-based matching** | Supports vector inputs directly |

### ❌ Not Needed When

- Data is clean with consistent formatting
- You have reliable unique identifiers
- Simple deterministic rules achieve high recall
- Dataset is small enough for full comparison

---

## Software Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Blocker                               │
│  (Main coordinator - manages workflow)                       │
└──────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    Input     │ │     ANN      │ │   Controls   │ │   Blocking   │
│  Processing  │ │  Algorithms  │ │              │ │    Result    │
├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤
│ - DataHandler│ │ - HNSW       │ │ - control_txt│ │ - result df  │
│ - n-grams    │ │ - LSH        │ │ - control_ann│ │ - metrics    │
│ - embeddings │ │ - Annoy      │ │              │ │ - blocks     │
│              │ │ - KNN        │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

---

## Real-World Impact

BlockingPy is used in production at **Statistics Poland** for:
- Linking administrative datasets without identifiers
- Deduplicating records of Ukrainian refugees crossing into Poland after Feb 2022
- Handling transliteration issues (Ukrainian/Russian → Polish)

The package helped improve population flow estimates that were previously imprecise due to missed matches from deterministic blocking.

---

## Resources

| Resource | Link |
|----------|------|
| **GitHub Repository** | https://github.com/ncn-foreigners/BlockingPy |
| **Documentation** | https://blockingpy.readthedocs.io |
| **Academic Paper** | https://arxiv.org/html/2504.04266v3 |
| **Splink Integration Example** | https://blockingpy.readthedocs.io/en/stable/examples/splink_integration.html |
| **PyPI Package** | https://pypi.org/project/blockingpy/ |

---

## Summary

**BlockingPy** bridges the gap between traditional deterministic blocking and the need for fuzzy, similarity-based matching in entity resolution:

| Feature | Benefit |
|---------|---------|
| 🎯 **ANN-based blocking** | Finds similar records even with data errors |
| ⚡ **Multiple algorithms** | HNSW, LSH, Annoy, etc. with tunable parameters |
| 🖥️ **CPU + GPU support** | Scales from laptops to data centers |
| 🔗 **Splink integration** | Enhances probabilistic record linkage workflows |
| 📊 **Built-in evaluation** | Recall, precision, F1, reduction ratio |
| 🌍 **Production-proven** | Used by Statistics Poland for official statistics |

Whether you're deduplicating customer databases, linking administrative records, or preparing data for LLM-based entity resolution, BlockingPy provides a powerful alternative to rigid deterministic blocking rules.

