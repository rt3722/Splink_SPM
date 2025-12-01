# Splink Q&A Document

This document contains frequently asked questions about Splink, the probabilistic record linkage library.

---

## Q1: Can I use vector embeddings in blocking rules with cosine similarity (e.g., cosine similarity > 0.8)?

### Short Answer
**No, you cannot directly use cosine similarity thresholds in Splink blocking rules.** Blocking rules in Splink must be **equi-join conditions** (equality-based) for performance reasons. However, you have several alternative approaches.

### Detailed Explanation

#### Why Blocking Rules Must Be Equality-Based

Splink's `block_on()` function is documented as:
> "Generates blocking rules of **equality conditions** based on the columns or SQL expressions specified."

The documentation explicitly states:
> "Further information on **equi-join conditions** can be found [here](https://moj-analytical-services.github.io/splink/topic_guides/blocking/performance.html)"

Blocking rules translate to SQL JOIN conditions. Equality conditions like `l.first_name = r.first_name` allow the SQL engine to use efficient hash joins or index lookups. Non-equality conditions like `cosine_similarity(l.embedding, r.embedding) > 0.8` would require computing similarity for all possible pairs, defeating the purpose of blocking (which is to reduce comparisons).

#### What You CAN Do with Cosine Similarity

**Option 1: Use Cosine Similarity in Comparisons (NOT Blocking)**

Splink provides `CosineSimilarityAtThresholds` and `CosineSimilarityLevel` for use in **comparisons** (not blocking):

```python
import splink.comparison_library as cl

settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        # Cosine similarity CAN be used in comparisons
        cl.CosineSimilarityAtThresholds("embedding", [0.9, 0.7, 0.5]),
        cl.NameComparison("first_name"),
    ],
    blocking_rules_to_generate_predictions=[
        # Blocking rules must be equality-based
        block_on("first_name", "surname"),
    ],
)
```

**Note:** From the GitHub issues, `CosineSimilarityLevel` is fully supported in **DuckDB** but has limitations in **Spark**:
> "Splink does not currently support `CosineSimilarityLevel` in Spark... the linked documentation page is incorrect."

For DuckDB, cosine similarity requires **fixed-length arrays**. When passing Python lists to PyArrow, they default to variable-length arrays, so you may need to explicitly cast them.

**Option 2: Pre-compute Embedding Blocks Using External Tools**

Use **BlockingPy** - a Python library for Approximate Nearest Neighbor (ANN) blocking that integrates with Splink:

```python
# Example workflow (conceptual)
from blockingpy import BlockingPy

# Step 1: Use BlockingPy to find candidate pairs based on embedding similarity
blocker = BlockingPy(method="lsh")  # Locality Sensitive Hashing
candidate_pairs = blocker.block(df, embedding_column="embedding")

# Step 2: Use those candidate pairs in Splink
# ... continue with Splink using pre-computed pairs
```

See: https://blockingpy.readthedocs.io/en/stable/examples/splink_integration.html

**Option 3: Use a Derived "Bucket" Column**

Pre-compute an embedding cluster/bucket and use that for blocking:

```python
# Pre-processing: cluster embeddings and assign bucket IDs
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=100)
df['embedding_bucket'] = kmeans.fit_predict(df['embedding'].tolist())

# Then use the bucket for blocking
settings = SettingsCreator(
    blocking_rules_to_generate_predictions=[
        block_on("embedding_bucket"),  # Equality-based!
    ],
    comparisons=[
        cl.CosineSimilarityAtThresholds("embedding", [0.9, 0.7, 0.5]),
    ],
)
```

### Summary Table

| Approach | Where | Supported? | Notes |
|----------|-------|------------|-------|
| `cosine_similarity > 0.8` in blocking rules | Blocking | ❌ No | Must be equality-based |
| `CosineSimilarityAtThresholds("embedding", [0.9, 0.7])` | Comparisons | ✅ Yes (DuckDB) | Use for scoring, not filtering candidates |
| Pre-computed embedding buckets | Blocking | ✅ Yes | Cluster embeddings first, then block on cluster ID |
| External ANN blocking (BlockingPy) | Pre-processing | ✅ Yes | Use LSH or other ANN methods |

---

## Q2: What data types are allowed in Splink?

### Short Answer
Splink supports the following data types:
- **String** (text)
- **Integer** (int)
- **Float** (numeric/decimal)
- **Date/DateTime**
- **Arrays** (lists)

### Detailed Explanation

#### Explicitly Documented Data Types

The `LiteralMatchLevel` comparison level explicitly lists supported literal data types:

```python
from splink.comparison_level_library import LiteralMatchLevel

LiteralMatchLevel(
    col_name="column_name",
    literal_value="some_value",
    literal_datatype="string",  # Must be one of: "string", "int", "float", "date"
    side_of_comparison="both"
)
```

**Supported `literal_datatype` values:**
- `"string"` - Text values
- `"int"` - Integer values
- `"float"` - Floating point numbers
- `"date"` - Date values

#### Data Types by Comparison Function

| Comparison Function | Expected Data Type | Notes |
|---------------------|-------------------|-------|
| `ExactMatch` | Any (string, int, float, date) | Works with any comparable type |
| `NameComparison` | String | Names, text fields |
| `JaroWinklerAtThresholds` | String | Fuzzy string matching |
| `LevenshteinAtThresholds` | String | Edit distance on strings |
| `DateOfBirthComparison` | Date or String | Set `input_is_string=True` if stored as string |
| `AbsoluteDateDifferenceAtThresholds` | Date or String | Requires `datetime_format` if string |
| `DistanceInKMAtThresholds` | Float | Latitude/longitude coordinates |
| `ArrayIntersectAtSizes` | Array | Arrays of values |
| `CosineSimilarityAtThresholds` | Array (fixed-length) | Vector embeddings |

#### Working with Dates

Dates can be stored as either **date types** or **strings**. Use the `input_is_string` parameter:

```python
import splink.comparison_library as cl

# If your date column is stored as a string (e.g., "1990-05-15")
cl.DateOfBirthComparison(
    "dob", 
    input_is_string=True,
    datetime_format="%Y-%m-%d"  # Specify format if not ISO 8601
)

# If your date column is a native date/datetime type
cl.DateOfBirthComparison(
    "dob", 
    input_is_string=False
)
```

#### Working with Arrays

Splink supports array columns for:
- **Array intersection comparisons** (`ArrayIntersectAtSizes`)
- **Pairwise string distance functions** (`PairwiseStringDistanceFunctionAtThresholds`)
- **Cosine similarity on embeddings** (`CosineSimilarityAtThresholds`)

```python
import splink.comparison_library as cl

# Array intersection - e.g., comparing lists of skills, aliases, etc.
cl.ArrayIntersectAtSizes("skills", size_threshold_or_thresholds=[3, 1])

# Cosine similarity on embedding vectors
cl.CosineSimilarityAtThresholds("embedding", [0.9, 0.7, 0.5])

# Pairwise string matching on arrays (e.g., multiple names)
cl.PairwiseStringDistanceFunctionAtThresholds(
    "names_array", 
    distance_function_name="jaro_winkler",
    distance_threshold_or_thresholds=[0.9, 0.8]
)
```

**Important:** For `CosineSimilarityAtThresholds` in DuckDB, arrays must be **fixed-length**:
> "Using DuckDB, the cosine similarity function can only be used if the data is a fixed length array."

#### Working with Geographic Data

For latitude/longitude coordinates:

```python
import splink.comparison_library as cl

cl.DistanceInKMAtThresholds(
    lat_col="latitude",
    long_col="longitude",
    km_thresholds=[1, 5, 10]
)
```

### Backend-Specific Considerations

Different backends may have variations in type support:

| Feature | DuckDB | Spark | SQLite | PostgreSQL |
|---------|--------|-------|--------|------------|
| String comparisons | ✅ | ✅ | ✅ | ✅ |
| Date comparisons | ✅ | ✅ | ✅ | ✅ |
| Fuzzy string matching | ✅ Native | ✅ Via UDFs | ⚠️ Via Python UDFs (slow) | ✅ |
| Array operations | ✅ | ✅ | Limited | ✅ |
| Cosine similarity | ✅ (fixed arrays) | ❌ Not implemented | ❌ | Limited |

### Summary

Splink is flexible with data types because it generates SQL that runs on your chosen backend. The key points are:

1. **Strings, integers, floats, and dates** are fully supported across all backends
2. **Arrays** are supported for intersection and pairwise comparisons
3. **Date handling** - specify `input_is_string=True` and provide format if dates are stored as strings
4. **Embeddings** (float arrays) work with `CosineSimilarityAtThresholds` in DuckDB with fixed-length arrays
5. **Backend matters** - some functions like cosine similarity aren't available in all backends

---

## Additional Resources

- [Splink Documentation](https://moj-analytical-services.github.io/splink/)
- [Blocking Performance Guide](https://moj-analytical-services.github.io/splink/topic_guides/blocking/performance.html)
- [Comparison Library API](https://moj-analytical-services.github.io/splink/api_docs/comparison_library.html)
- [BlockingPy Integration](https://blockingpy.readthedocs.io/en/stable/examples/splink_integration.html)

