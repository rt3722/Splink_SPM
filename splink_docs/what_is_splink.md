# What is Splink?

## Overview

**Splink** is a free, open-source Python library developed by the UK Ministry of Justice for **probabilistic record linkage** (also known as **entity resolution** or **data matching**). It helps identify records that refer to the same real-world entity across one or more datasets, even when there is no unique identifier and data quality varies.

Splink is designed to scale to datasets containing millions of records, making it suitable for large-scale data integration projects. It implements the **Fellegi-Sunter probabilistic linkage model** with modern optimizations and provides a rich set of tools for model training, prediction, evaluation, and visualization.

---

## The Problem Splink Solves

In the real world, data about entities (people, organizations, transactions, etc.) is often:
- **Scattered across multiple databases** without shared unique identifiers
- **Inconsistent** due to typos, variations in spelling, missing data, or data entry errors
- **Duplicated** within the same dataset

**Record linkage** is the process of identifying which records from one or more data sources refer to the same entity. Traditional approaches include:

| Approach | Description | Limitations |
|----------|-------------|-------------|
| **Deterministic Matching** | Records match only if specific fields are exactly identical | Misses matches with typos, variations, or missing data |
| **Manual Review** | Human clerical review of potential matches | Doesn't scale to large datasets |

Splink addresses these limitations through **probabilistic matching**, which calculates the likelihood that two records are a match based on multiple pieces of evidence, tolerating data imperfections while scaling to millions of records.

---

## Theoretical Foundation: The Fellegi-Sunter Model

Splink implements the **Fellegi-Sunter probabilistic record linkage model** (1969), which provides a mathematical framework for deciding whether two records represent the same entity.

### Core Concepts

#### Match Weight
Every comparison between two records produces a **match weight** (log odds), which quantifies the evidence that the two records are a match. A higher match weight indicates stronger evidence of a match.

#### m and u Probabilities
For each comparison level (e.g., "exact match on first_name"), the model estimates two key probabilities:

| Parameter | Description | Example |
|-----------|-------------|---------|
| **m probability** | The probability that truly matching records fall into this comparison level | 95% of true matches have exact match on first_name |
| **u probability** | The probability that truly non-matching records fall into this comparison level | 5% of non-matches happen to have exact match on first_name (coincidence) |

#### Bayes Factors
The **Bayes Factor** for each comparison level is the ratio m/u. High Bayes factors (m >> u) provide strong evidence of a match; low Bayes factors (m << u) provide evidence against a match.

The overall match weight is the sum of the log2(Bayes Factors) across all comparisons.

#### Probability Two Random Records Match
This is the prior probability that any two randomly selected records from your data are a match. For large datasets, this is typically very small (e.g., 1 in 1,000,000).

---

## Key Features

### 1. **Scalability**
- Handles datasets with millions of records
- Uses **blocking rules** to avoid comparing every record with every other record (which would be computationally infeasible)
- Supports multiple high-performance SQL backends

### 2. **Flexible Comparisons**
- Rich library of pre-built comparison functions for common scenarios:
  - Names (with fuzzy matching)
  - Dates of birth
  - Addresses and postcodes
  - Email addresses
  - Geographic coordinates (distance in km)
- Support for multiple string similarity algorithms:
  - Jaro-Winkler
  - Levenshtein distance
  - Damerau-Levenshtein
  - Jaccard similarity
  - Cosine similarity
- Custom SQL comparisons for domain-specific logic

### 3. **Unsupervised and Semi-Supervised Training**
- **Expectation Maximisation (EM)** algorithm for unsupervised parameter estimation
- Ability to incorporate labeled data when available
- Term frequency adjustments to handle common values (e.g., "John Smith")

### 4. **Rich Visualizations**
- Match weight charts
- Waterfall charts showing how match scores are computed
- Comparison viewer dashboard
- Cluster visualization studio
- Interactive training history charts

### 5. **Model Persistence**
- Save and load trained models as JSON
- Enables model reuse and real-time linkage applications

---

## Supported SQL Backends

Splink is backend-agnostic and supports multiple SQL execution engines:

| Backend | Best For | Scale |
|---------|----------|-------|
| **DuckDB** | Local development, small to medium datasets | Up to ~10 million records |
| **Apache Spark** | Big data, distributed computing | Billions of records |
| **AWS Athena** | Cloud-native, serverless | Large datasets |
| **PostgreSQL** | Integration with existing database infrastructure | Medium datasets |
| **SQLite** | Lightweight, embedded applications | Small datasets |

---

## Link Types

Splink supports three types of linkage tasks:

| Link Type | Description | Use Case |
|-----------|-------------|----------|
| **`dedupe_only`** | Find duplicates within a single dataset | Cleaning a customer database |
| **`link_only`** | Find matches between two or more datasets (no within-dataset deduplication) | Linking two separate systems |
| **`link_and_dedupe`** | Find matches within and between datasets | Comprehensive entity resolution |

---

## How Splink Works: The Workflow

### Step 1: Data Preparation
- Ensure each record has a unique identifier
- Standardize data where possible (e.g., date formats, case normalization)
- Consider feature engineering (e.g., extracting name components)

### Step 2: Define the Linkage Model

```python
from splink import SettingsCreator, block_on
import splink.comparison_library as cl

settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.DateOfBirthComparison("dob", input_is_string=True),
        cl.ExactMatch("city"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname"),
        block_on("dob", "first_name"),
    ],
)
```

### Step 3: Create a Linker

```python
from splink import Linker, DuckDBAPI

db_api = DuckDBAPI()
linker = Linker(df, settings, db_api=db_api)
```

### Step 4: Train the Model

```python
# Estimate u probabilities from random sample
linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

# Estimate probability two random records match
linker.training.estimate_probability_two_random_records_match(
    deterministic_matching_rules=[block_on("first_name", "surname", "dob")],
    recall=0.7
)

# Estimate m probabilities using EM algorithm
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "dob")
)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("surname", "dob")
)
```

### Step 5: Generate Predictions

```python
# Generate pairwise predictions
df_predictions = linker.inference.predict(threshold_match_probability=0.9)
```

### Step 6: Cluster Records

```python
# Group records into clusters representing the same entity
df_clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predictions, 
    threshold_match_probability=0.95
)
```

### Step 7: Evaluate and Visualize

```python
# View match weights
linker.visualisations.match_weights_chart()

# Waterfall chart for specific comparisons
records = df_predictions.as_record_dict(limit=5)
linker.visualisations.waterfall_chart(records)

# Generate interactive dashboards
linker.visualisations.comparison_viewer_dashboard(df_predictions, "dashboard.html")
```

---

## Key Concepts Explained

### Blocking Rules
**Blocking** is a technique to reduce the number of comparisons by only comparing records that share certain attributes. Without blocking, comparing N records would require N×(N-1)/2 comparisons—which is computationally infeasible for large datasets.

```python
# Only compare records where first_name AND surname match exactly
block_on("first_name", "surname")

# Only compare records where date of birth matches
block_on("dob")
```

### Comparisons and Comparison Levels
A **comparison** defines how to assess similarity between two records for a particular attribute. Each comparison has multiple **levels** representing different degrees of similarity:

```python
# Example: Name comparison with multiple levels
# Level 1: Exact match (highest evidence for match)
# Level 2: Jaro-Winkler similarity > 0.92
# Level 3: Jaro-Winkler similarity > 0.88
# Level 4: Jaro-Winkler similarity > 0.70
# Level 5: Anything else (evidence against match)
```

### Term Frequency Adjustments
Common values (like "John" or "Smith") provide less evidence for a match than rare values (like "Zebedee"). Splink can adjust match weights based on how frequently values appear in the data.

### Match Probability vs Match Weight
- **Match Weight**: Log-odds score (can be positive or negative); sum of evidence
- **Match Probability**: Probability between 0 and 1 that two records are a match

---

## Common Use Cases

| Domain | Use Case |
|--------|----------|
| **Healthcare** | Patient matching across hospital systems |
| **Government** | Linking census records, benefits databases |
| **Finance** | Customer deduplication, fraud detection |
| **Research** | Linking survey responses to administrative data |
| **Marketing** | Customer data unification across channels |
| **HR/Staffing** | Candidate deduplication, resume matching |

---

## Comparison with Other Tools

| Feature | Splink | Traditional SQL | dedupe (Python) |
|---------|--------|-----------------|-----------------|
| Probabilistic matching | ✅ | ❌ | ✅ |
| Scalability | Millions of records | Limited | Moderate |
| Unsupervised training | ✅ | N/A | ✅ |
| Visual diagnostics | ✅ Rich | ❌ | Limited |
| Multiple backends | ✅ | Single DB | In-memory |
| Production-ready | ✅ | ✅ | ✅ |
| Free & open source | ✅ | ✅ | ✅ |

---

## Installation

```bash
# Basic installation
pip install splink

# With DuckDB backend (recommended for getting started)
pip install splink[duckdb]

# With Spark backend
pip install splink[spark]
```

---

## Resources

- **Official Documentation**: https://moj-analytical-services.github.io/splink/
- **GitHub Repository**: https://github.com/moj-analytical-services/splink
- **Tutorials**: https://moj-analytical-services.github.io/splink/demos/tutorials/00_Tutorial_Introduction.html
- **API Reference**: https://moj-analytical-services.github.io/splink/api_docs/
- **Discussion Forum**: https://github.com/moj-analytical-services/splink/discussions

---

## Summary

**Splink** is a powerful, production-ready Python library for probabilistic record linkage. It combines:

- 🎯 **Fellegi-Sunter probabilistic model** for accurate matching
- ⚡ **Blocking** for computational efficiency at scale  
- 🔧 **Rich comparison library** with fuzzy matching support
- 🤖 **Unsupervised learning** via Expectation Maximisation
- 📊 **Interactive visualizations** for model understanding
- 🔌 **Multiple SQL backends** for flexibility

Whether you're deduplicating a customer database or linking records across multiple systems, Splink provides the tools to do it accurately and at scale.

