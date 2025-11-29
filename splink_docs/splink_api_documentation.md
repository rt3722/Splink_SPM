# Splink API Documentation

This document contains comprehensive API documentation extracted from all Splink API documentation pages.

**Source:** https://moj-analytical-services.github.io/splink/api_docs/

**Last Updated:** November 26, 2025

---

## Table of Contents

1. [Linker API](#linker-api)
   - [Training](#training)
   - [Visualisations](#visualisations)
   - [Inference](#inference)
   - [Clustering](#clustering)
   - [Evaluation](#evaluation)
   - [Table Management](#table-management)
   - [Miscellaneous Functions](#miscellaneous-functions)
2. [Comparisons Library](#comparisons-library)
3. [Other APIs](#other-apis)
4. [In-build Datasets](#in-build-datasets)
5. [Splink Settings](#splink-settings)

---

## Linker API

The Linker API provides the main interface for working with Splink. All methods are accessed via the `linker` object.

### Training

**URL:** https://moj-analytical-services.github.io/splink/api_docs/training.html

Estimate the parameters of the linkage model, accessed via `linker.training`.

#### Methods

##### `estimate_probability_two_random_records_match(deterministic_matching_rules, recall, max_rows_limit=int(1000000000.0))`

Estimate the model parameter `probability_two_random_records_match` using a direct estimation approach.

**Parameters:**
- `deterministic_matching_rules` (list, required): A list of deterministic matching rules designed to admit very few (preferably no) false positives.
- `recall` (float, required): An estimate of the recall the deterministic matching rules will achieve, i.e., the proportion of all true matches these rules will recover.
- `max_rows_limit` (int, default: 1e9): Maximum number of rows to consider during estimation.

**Example:**
```python
deterministic_rules = [
    block_on("forename", "dob"),
    "l.forename = r.forename and levenshtein(r.surname, l.surname) <= 2",
    block_on("email")
]
linker.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.8
)
```

##### `estimate_u_using_random_sampling(max_pairs=1000000.0, seed=None)`

Estimate the u parameters of the linkage model using random sampling. The u parameters estimate the proportion of record comparisons that fall into each comparison level amongst truly non-matching records.

**Parameters:**
- `max_pairs` (int, default: 1000000.0): The maximum number of pairwise record comparisons to sample. Larger will give more accurate estimates but lead to longer runtimes.
- `seed` (int, default: None): Seed for random sampling. Assign to get reproducible u probabilities. Note, seed for random sampling is only supported for DuckDB and Spark, for Athena and SQLite set to None.

**Example:**
```python
linker.training.estimate_u_using_random_sampling(max_pairs=1e8)
```

**Returns:** None (Updates the estimated u parameters within the linker object)

##### `estimate_parameters_using_expectation_maximisation(blocking_rule, estimate_without_term_frequencies=False, fix_probability_two_random_records_match=False, fix_m_probabilities=False, fix_u_probabilities=True, populate_probability_two_random_records_match_from_trained_values=False)`

Estimate the parameters of the linkage model using expectation maximisation.

**Parameters:**
- `blocking_rule` (BlockingRuleCreator | str, required): The blocking rule used to generate pairwise record comparisons.
- `estimate_without_term_frequencies` (bool, default: False): If True, the iterations of the EM algorithm ignore any term frequency adjustments.
- `fix_probability_two_random_records_match` (bool, default: False): If True, do not update the probability two random records match after each iteration.
- `fix_m_probabilities` (bool, default: False): If True, do not update the m probabilities after each iteration.
- `fix_u_probabilities` (bool, default: True): If True, do not update the u probabilities after each iteration.
- `populate_probability_two_random_records_match_from_trained_values` (bool, default: False): If True, derive this parameter from the blocked value.

**Example:**
```python
br_training = block_on("first_name", "dob")
linker.training.estimate_parameters_using_expectation_maximisation(
    br_training
)
```

**Returns:** EMTrainingSession - An object containing information about the training session such as how parameters changed during the iteration history

##### `estimate_m_from_pairwise_labels(labels_splinkdataframe_or_table_name)`

Estimate the m probabilities of the linkage model from a dataframe of pairwise labels.

**Parameters:**
- `labels_splinkdataframe_or_table_name` (str, required): Name of table containing labels in the database or SplinkDataframe

**Example:**
```python
pairwise_labels = pd.read_csv("./data/pairwise_labels_to_estimate_m.csv")

linker.table_management.register_table(
    pairwise_labels, "labels", overwrite=True
)

linker.training.estimate_m_from_pairwise_labels("labels")
```

##### `estimate_m_from_label_column(label_colname)`

Estimate the m parameters of the linkage model from a label (ground truth) column in the input dataframe(s).

**Parameters:**
- `label_colname` (str, required): The name of the column containing the ground truth label in the input data.

**Example:**
```python
linker.training.estimate_m_from_label_column("social_security_number")
```

**Returns:** None (Updates the estimated m parameters within the linker object)

---

### Visualisations

**URL:** https://moj-analytical-services.github.io/splink/api_docs/visualisations.html

Visualisations to help you understand and diagnose your linkage model. Accessed via `linker.visualisations`. Most of the visualisations return an `altair.Chart` object.

#### Methods

##### `match_weights_chart(as_dict=False)`

Display a chart of the (partial) match weights of the linkage model.

**Parameters:**
- `as_dict` (bool, default: False): If True, return the chart as a dictionary.

**Example:**
```python
altair_chart = linker.visualisations.match_weights_chart()
altair_chart.save("mychart.png")
```

##### `m_u_parameters_chart(as_dict=False)`

Display a chart of the m and u parameters of the linkage model.

**Parameters:**
- `as_dict` (bool, default: False): If True, return the chart as a dictionary.

**Returns:** ChartReturnType - An altair chart

##### `match_weights_histogram(df_predict, target_bins=30, width=600, height=250, as_dict=False)`

Generate a histogram that shows the distribution of match weights in `df_predict`.

**Parameters:**
- `df_predict` (SplinkDataFrame, required): Output of `linker.inference.predict()`
- `target_bins` (int, default: 30): Target number of bins in histogram.
- `width` (int, default: 600): Width of output.
- `height` (int, default: 250): Height of output chart.
- `as_dict` (bool, default: False): If True, return the chart as a dictionary.

**Example:**
```python
df_predict = linker.inference.predict(threshold_match_weight=-2)
linker.visualisations.match_weights_histogram(df_predict)
```

##### `parameter_estimate_comparisons_chart(include_m=True, include_u=False, as_dict=False)`

Show a chart that shows how parameter estimates have differed across the different estimation methods you have used.

**Parameters:**
- `include_m` (bool, default: True): Show different estimates of m values.
- `include_u` (bool, default: False): Show different estimates of u values.
- `as_dict` (bool, default: False): If True, return the chart as a dictionary.

**Returns:** ChartReturnType - An Altair chart

##### `tf_adjustment_chart(output_column_name, n_most_freq=10, n_least_freq=10, vals_to_include=None, as_dict=False)`

Display a chart showing the impact of term frequency adjustments on a specific comparison level.

**Parameters:**
- `output_column_name` (str, required): Name of an output column for which term frequency adjustment has been applied.
- `n_most_freq` (int, default: 10): Number of most frequent values to show.
- `n_least_freq` (int, default: 10): Number of least frequent values to show.
- `vals_to_include` (list, default: None): Specific values for which to show term frequency adjustments.
- `as_dict` (bool, default: False): If True, return the chart as a dictionary.

**Returns:** ChartReturnType - An Altair chart

##### `waterfall_chart(records, filter_nulls=True, remove_sensitive_data=False, as_dict=False)`

Visualise how the final match weight is computed for the provided pairwise record comparisons.

**Parameters:**
- `records` (List[dict], required): Usually be obtained from `df.as_record_dict(limit=n)` where df is a SplinkDataFrame.
- `filter_nulls` (bool, default: True): Whether the visualisation shows null comparisons, which have no effect on final match weight.
- `remove_sensitive_data` (bool, default: False): When True, The waterfall chart will contain match weights only, and all of the (potentially sensitive) data from the input tables will be removed prior to the chart being created.
- `as_dict` (bool, default: False): If True, return the chart as a dictionary.

**Returns:** ChartReturnType - An altair chart

##### `comparison_viewer_dashboard(df_predict, out_path, overwrite=False, num_example_rows=2, minimum_comparison_vector_count=0, return_html_as_string=False)`

Generate an interactive html visualization of the linker's predictions and save to out_path.

**Parameters:**
- `df_predict` (SplinkDataFrame, required): The outputs of `linker.inference.predict()`
- `out_path` (str, required): The path (including filename) to save the html file to.
- `overwrite` (bool, default: False): Overwrite the html file if it already exists?
- `num_example_rows` (int, default: 2): Number of example rows per comparison vector.
- `minimum_comparison_vector_count` (int, default: 0): The minimum number of times that a comparison vector has to occur for it to be included in the dashboard.
- `return_html_as_string` (bool, default: False): If True, return the html as a string

**Example:**
```python
df_predictions = linker.inference.predict()
linker.visualisations.comparison_viewer_dashboard(
    df_predictions, "scv.html", True, 2
)
```

##### `cluster_studio_dashboard(df_predict, df_clustered, out_path, sampling_method='random', sample_size=10, cluster_ids=None, cluster_names=None, overwrite=False, return_html_as_string=False, _df_cluster_metrics=None)`

Generate an interactive html visualization of the predicted cluster and save to out_path.

**Parameters:**
- `df_predict` (SplinkDataFrame, required): The outputs of `linker.inference.predict()`
- `df_clustered` (SplinkDataFrame, required): The outputs of `linker.clustering.cluster_pairwise_predictions_at_threshold()`
- `out_path` (str, required): The path (including filename) to save the html file to.
- `sampling_method` (str, default: 'random'): random, by_cluster_size or lowest_density_clusters_by_size.
- `sample_size` (int, default: 10): Number of clusters to show in the dashboard.
- `cluster_ids` (list, default: None): The IDs of the clusters that will be displayed in the dashboard.
- `cluster_names` (list, default: None): If provided, the dashboard will display these names in the selection box.
- `overwrite` (bool, default: False): Overwrite the html file if it already exists?
- `return_html_as_string` (bool, default: False): If True, return the html as a string

---

### Inference

**URL:** https://moj-analytical-services.github.io/splink/api_docs/inference.html

Methods for making predictions and inferences from the trained linkage model. Accessed via `linker.inference`.

**Note:** Full documentation available at the URL above. Key methods include:
- `predict()` - Generate predictions for record pairs
- Methods for threshold selection and match weight calculation

---

### Clustering

**URL:** https://moj-analytical-services.github.io/splink/api_docs/linker_clustering.html

Methods for clustering linked records. Accessed via `linker.clustering`.

**Note:** Full documentation available at the URL above. Key methods include:
- `cluster_pairwise_predictions_at_threshold()` - Cluster predictions at a given threshold

---

### Evaluation

**URL:** https://moj-analytical-services.github.io/splink/api_docs/evaluation.html

Methods for evaluating the performance of the linkage model. Accessed via `linker.evaluation`.

**Note:** Full documentation available at the URL above. Includes methods for:
- Model evaluation metrics
- Edge (link) evaluation
- Cluster evaluation
- Accuracy analysis

---

### Table Management

**URL:** https://moj-analytical-services.github.io/splink/api_docs/table_management.html

Methods for managing tables in the database. Accessed via `linker.table_management`.

**Note:** Full documentation available at the URL above. Includes methods for:
- Registering tables
- Dropping tables
- Managing database connections

---

### Miscellaneous Functions

**URL:** https://moj-analytical-services.github.io/splink/api_docs/misc.html

Miscellaneous methods on the linker that don't fit into other categories. Accessed via `linker.misc`.

#### Methods

##### `save_model_to_json(out_path=None, overwrite=False)`

Save the configuration and parameters of the linkage model to a .json file. The model can later be loaded into a new linker using `Linker(df, settings="path/to/model.json", db_api=db_api)`.

**Parameters:**
- `out_path` (str, optional): Path to save the JSON file
- `overwrite` (bool, default: False): Whether to overwrite existing file

**Example:**
```python
linker.misc.save_model_to_json("my_settings.json", overwrite=True)
```

**Returns:** dict[str, Any] - The settings as a dictionary

##### `query_sql(sql, output_type='pandas')`

Run a SQL query against your backend database and return the resulting output.

**Parameters:**
- `sql` (str, required): The SQL to be queried.
- `output_type` (str, default: 'pandas'): One of splink_df/splinkdf or pandas. This determines the type of table that your results are output in.

**Example:**
```python
linker = Linker(df, settings, db_api)
df_predict = linker.inference.predict()
linker.misc.query_sql(f"select * from {df_predict.physical_name} limit 10")
```

---

## Comparisons Library

### Comparison Library

**URL:** https://moj-analytical-services.github.io/splink/api_docs/comparison_library.html

Pre-built comparison functions for common matching scenarios. Accessed via `splink.comparison_library`.

**Note:** Full documentation available at the URL above. Includes comparisons for:
- Names
- Dates
- Addresses
- Emails
- Phone numbers
- And many more...

### Comparison Level Library

**URL:** https://moj-analytical-services.github.io/splink/api_docs/comparison_level_library.html

Pre-built comparison levels for use in comparisons. Accessed via `splink.comparison_level_library`.

**Note:** Full documentation available at the URL above. Includes comparison levels for:
- Exact matches
- Fuzzy matches
- Null handling
- And more...

---

## Other APIs

### Exploratory

**URL:** https://moj-analytical-services.github.io/splink/api_docs/exploratory.html

Functions for exploratory data analysis. Accessed via `splink.exploratory`.

**Note:** Full documentation available at the URL above.

### Blocking Analysis

**URL:** https://moj-analytical-services.github.io/splink/api_docs/blocking_analysis.html

Functions for analyzing blocking rules. Accessed via `splink.blocking_analysis`.

**Note:** Full documentation available at the URL above.

### Blocking

**URL:** https://moj-analytical-services.github.io/splink/api_docs/blocking.html

Functions for creating and managing blocking rules. Accessed via `splink.blocking`.

**Note:** Full documentation available at the URL above. Includes:
- `block_on()` - Create blocking rules
- Blocking rule library functions

### Clustering

**URL:** https://moj-analytical-services.github.io/splink/api_docs/clustering.html

Functions for clustering linked records. Accessed via `splink.clustering`.

**Note:** Full documentation available at the URL above.

### SplinkDataFrame

**URL:** https://moj-analytical-services.github.io/splink/api_docs/splink_dataframe.html

The SplinkDataFrame class provides a wrapper around database tables with additional functionality.

**Note:** Full documentation available at the URL above. Key methods include:
- `as_pandas_dataframe()` - Convert to pandas DataFrame
- `as_record_dict()` - Convert to list of dictionaries
- `to_parquet()` - Save to Parquet format
- And more...

### EM Training Session API

**URL:** https://moj-analytical-services.github.io/splink/api_docs/em_training_session.html

API for working with Expectation Maximisation training sessions.

**Note:** Full documentation available at the URL above.

### Column Expressions

**URL:** https://moj-analytical-services.github.io/splink/api_docs/column_expression.html

Functions for creating column expressions for use in comparisons and blocking rules.

**Note:** Full documentation available at the URL above.

---

## In-build Datasets

### SplinkDatasets

**URL:** https://moj-analytical-services.github.io/splink/api_docs/datasets.html

Access to built-in datasets for testing and examples. Accessed via `splink.splink_datasets`.

**Note:** Full documentation available at the URL above. Includes datasets such as:
- `fake_1000` - Synthetic dataset with 1000 records
- And more...

---

## Splink Settings

### Settings Dict

**URL:** https://moj-analytical-services.github.io/splink/api_docs/settings_dict_guide.html

Complete guide to all settings and configuration options available when developing your data linkage model.

#### Key Settings

##### `link_type`
The type of data linking task. Required.

**Options:**
- `'dedupe_only'` - Find duplicates within a single dataset
- `'link_and_dedupe'` - Find links within and between input datasets
- `'link_only'` - Find links between datasets only

##### `probability_two_random_records_match`
The probability that two records chosen at random (with no blocking) are a match.

**Default:** 0.0001

##### `unique_id_column_name`
The name of the column in the input dataset representing the unique id.

**Default:** 'unique_id'

##### `comparisons`
A list specifying how records should be compared for probabilistic matching. Each element is a dictionary with:
- `output_column_name` - Name used to refer to this comparison
- `comparison_levels` - List of comparison levels with SQL conditions
- `comparison_description` - Optional label for charting outputs

##### `blocking_rules_to_generate_predictions`
A list of one or more blocking rules to apply. Each rule is a SQL expression.

**Example:**
```python
blocking_rules_to_generate_predictions = [
    "l.first_name = r.first_name AND l.surname = r.surname",
    "l.dob = r.dob"
]
```

##### `em_convergence`
Convergence tolerance for the Expectation Maximisation algorithm.

**Default:** 0.0001

##### `max_iterations`
The maximum number of Expectation Maximisation iterations to run.

**Default:** 25

**Note:** For complete documentation of all settings, see the full guide at the URL above.

---

## Summary

This document provides a comprehensive overview of the Splink API. For detailed documentation on any specific API, please refer to the individual documentation pages linked above.

**Total API Documentation Pages:** 17

**Main Categories:**
- Linker API (7 sections)
- Comparisons Library (2 sections)
- Other APIs (7 sections)
- In-build Datasets (1 section)
- Settings (1 section)

For the most up-to-date information, always refer to the official Splink documentation at: https://moj-analytical-services.github.io/splink/

