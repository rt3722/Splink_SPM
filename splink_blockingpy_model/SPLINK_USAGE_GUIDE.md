# Splink + BlockingPy Usage Guide

After running the data preparation script, use this guide to build your matching model.

## Overview

The cleaned data has two key columns for blocking:
- **`text_blob`**: Combined text for BlockingPy ANN-based blocking
- **`description`**: SOV_DESCRIPTION for additional blocking

## Step 1: BlockingPy Blocking

Use BlockingPy to create initial blocks based on text similarity:

```python
import pandas as pd
from blockingpy import Blocker

# Load cleaned data
df = pd.read_parquet("output/train_cleaned.parquet")

# BlockingPy blocking using text_blob
blocker = Blocker()
res = blocker.block(x=df['text_blob'], ann='hnsw', random_seed=42)

# Add block column to dataframe
df = res.add_block_column(df)

print(res)
# Shows: Number of blocks, reduction ratio, block size distribution
```

## Step 2: Splink Configuration

### Blocking Rules

Combine BlockingPy blocks with traditional blocking rules:

```python
from splink import block_on

blocking_rules_to_generate_predictions=[
    block_on("block"),                                    # BlockingPy ANN blocks
    block_on("firstname_cleaned", "lastname_cleaned"),    # Exact name blocking
    block_on("firstname_soundex", "lastname_soundex"),    # Phonetic blocking (Smith/Smyth)
    block_on("firstname_cleaned"),                        # First name only
]
```

### Understanding Array Comparisons

`ArrayIntersectAtSizes` creates comparison levels based on how many array elements match:

```python
# Example: cl.ArrayIntersectAtSizes("names_array", [2, 1])
#
# This creates 3 levels:
# Level 1 (strongest): ≥2 elements match  → High m probability
# Level 2 (weaker):    ≥1 element matches → Medium m probability  
# Else level:          0 elements match   → Evidence against match
```

| Threshold | Meaning | Use Case |
|-----------|---------|----------|
| `[1]` | At least 1 match | Good for unique identifiers (phone, email) |
| `[2]` | At least 2 matches | Good for less unique fields (employers, degrees) |
| `[2, 1]` | Multiple levels | Captures both strong and weak matches |

### Understanding Numeric Comparisons

`AbsoluteDifferenceAtThresholds` compares numeric values:

```python
# Example: cl.AbsoluteDifferenceAtThresholds("months_experience", [1, 2, 3])
#
# This creates 4 levels (very tight for deduplication):
# Level 1: Difference ≤ 1 month  → Strong match
# Level 2: Difference ≤ 2 months → Good match
# Level 3: Difference ≤ 3 months → Weak match
# Else:    Difference > 3 months → Evidence against
```

## Step 3: Full Splink Example

```python
import pandas as pd
from blockingpy import Blocker
from splink import Linker, SettingsCreator, block_on, DuckDBAPI
import splink.comparison_library as cl

# Load cleaned data
df = pd.read_parquet("output/train_cleaned.parquet")

# BlockingPy blocking using text_blob
blocker = Blocker()
res = blocker.block(x=df['text_blob'], ann='hnsw', random_seed=42)
df = res.add_block_column(df)

# Splink settings with combined blocking strategies
settings = SettingsCreator(
    unique_id_column_name="unique_id",
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("block"),                                    # BlockingPy ANN blocks
        block_on("firstname_cleaned", "lastname_cleaned"),    # Exact name blocking
        block_on("firstname_soundex", "lastname_soundex"),    # Phonetic blocking (Smith/Smyth)
        block_on("firstname_cleaned"),                        # First name only
    ],
    comparisons=[
        # Name comparisons - exact match with fuzzy fallback
        cl.NameComparison("firstname_cleaned"),
        cl.NameComparison("lastname_cleaned"),
        
        # Phonetic comparisons - same soundex = similar sounding names
        cl.ExactMatch("firstname_soundex"),
        cl.ExactMatch("lastname_soundex"),
        
        # Name array intersection for all name variations
        # [2, 1] = Level 1: ≥2 matches (strong), Level 2: ≥1 match (weaker)
        cl.ArrayIntersectAtSizes("names_array", [2, 1]),
        
        # Contact comparisons - at least 1 match is significant
        cl.ArrayIntersectAtSizes("phones_array", [1]),     # ≥1 phone match
        cl.ArrayIntersectAtSizes("emails_array", [1]),     # ≥1 email match
        cl.ExactMatch("linkedin_cleaned"),
        
        # Employment comparisons - require at least 2 matches (more strict)
        cl.ArrayIntersectAtSizes("employers_array", [2]),  # ≥2 employer matches
        cl.ArrayIntersectAtSizes("titles_array", [2]),     # ≥2 title matches
        
        # Location comparisons - at least 1 match
        cl.ArrayIntersectAtSizes("countries_array", [1]),
        cl.ArrayIntersectAtSizes("regions_array", [1]),
        cl.ArrayIntersectAtSizes("municipalities_array", [1]),
        
        # Education - require at least 2 matches (more strict)
        cl.ArrayIntersectAtSizes("degrees_array", [2]),    # ≥2 degree matches
        
        # Numeric comparisons for experience
        # [1, 2, 3] = tight thresholds (≤1mo, ≤2mo, ≤3mo diff)
        cl.AbsoluteDifferenceAtThresholds("months_experience", [1, 2, 3]),
        cl.AbsoluteDifferenceAtThresholds("avg_months_per_employer", [1, 2, 3]),
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=DuckDBAPI())
```

## Step 4: Train the Model

```python
# Estimate u probabilities from random sample
linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

# Estimate probability two random records match
linker.training.estimate_probability_two_random_records_match(
    deterministic_rules=[
        block_on("firstname_cleaned", "lastname_cleaned"),
        "l.phones_array[1] = r.phones_array[1]",
    ],
    recall=0.7
)

# Estimate m probabilities using EM algorithm
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("firstname_cleaned")
)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("firstname_soundex", "lastname_soundex")
)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("block")  # BlockingPy blocks
)
```

## Step 5: Generate Predictions

```python
# Generate predictions at threshold
df_predictions = linker.inference.predict(threshold_match_probability=0.9)

# View results
df_predictions.as_pandas_dataframe().head()
```

## Step 6: Cluster Records

```python
# Group records into clusters (same entity)
df_clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predictions,
    threshold_match_probability=0.95
)

# Export clusters
df_clusters.as_pandas_dataframe().to_parquet("output/clusters.parquet")
```

## Step 7: Visualize Results

```python
# Match weights chart
linker.visualisations.match_weights_chart()

# Waterfall chart for specific comparisons
records = df_predictions.as_record_dict(limit=5)
linker.visualisations.waterfall_chart(records)

# Interactive dashboard
linker.visualisations.comparison_viewer_dashboard(
    df_predictions, 
    "output/comparison_dashboard.html"
)
```

## Column Reference

### For Blocking (Exact Match)
| Column | Description |
|--------|-------------|
| `firstname_cleaned` | Exact first name blocking |
| `lastname_cleaned` | Exact last name blocking |
| `firstname_soundex` | Phonetic first name blocking |
| `lastname_soundex` | Phonetic last name blocking |
| `block` | BlockingPy ANN blocks |

### For Comparisons (Array Intersection)
| Column | Threshold | Rationale |
|--------|-----------|-----------|
| `names_array` | `[2, 1]` | Names vary, allow weak matches |
| `phones_array` | `[1]` | Unique identifier, 1 match is strong |
| `emails_array` | `[1]` | Unique identifier, 1 match is strong |
| `employers_array` | `[2]` | Common employers, need 2 matches |
| `titles_array` | `[2]` | Titles can be similar, need 2 matches |
| `degrees_array` | `[2]` | Degrees are common, need 2 matches |
| `countries_array` | `[1]` | Location signal |
| `regions_array` | `[1]` | Location signal |
| `municipalities_array` | `[1]` | Location signal |

### For Comparisons (Numeric Difference)
| Column | Thresholds | Meaning |
|--------|------------|---------|
| `months_experience` | `[1, 2, 3]` | ≤1mo, ≤2mo, ≤3mo difference (tight) |
| `avg_months_per_employer` | `[1, 2, 3]` | ≤1mo, ≤2mo, ≤3mo difference (tight) |

## Tips

1. **Start with high thresholds** (e.g., 0.95) and lower if needed
2. **Use multiple blocking rules** to catch different types of matches
3. **Phonetic blocking** catches typos and spelling variations
4. **Array thresholds**:
   - Use `[1]` for unique identifiers (phone, email)
   - Use `[2]` for less unique fields (employer, degree)
   - Use `[2, 1]` when you want multiple comparison levels

