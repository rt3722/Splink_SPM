## Linker Methods

Linker.inference.compare_two_records:
Use the linkage model to compare and score a pairwise record comparison
based on the two input records provided.

If your inputs contain multiple rows, scores for the cartesian product of
the two inputs will be returned.

If your inputs contain hardcoded term frequency columns (e.g.
a tf_first_name column), then these values will be used instead of any
provided term frequency lookup tables. or term frequency values derived
from the input data.

Args:
    record_1 (dict): dictionary representing the first record.  Columns names
        and data types must be the same as the columns in the settings object
    record_2 (dict): dictionary representing the second record.  Columns names
        and data types must be the same as the columns in the settings object
    include_found_by_blocking_rules (bool, optional): If True, outputs a column
        indicating whether the record pair would have been found by any of the
        blocking rules specified in
        settings.blocking_rules_to_generate_predictions. Defaults to False.

Examples:
    ```py
    linker = Linker(df, "saved_settings.json", db_api=db_api)

    # You should load or pre-compute tf tables for any tables with
    # term frequency adjustments
    linker.table_management.compute_tf_table("first_name")
    # OR
    linker.table_management.register_term_frequency_lookup(df, "first_name")

    record_1 = {'unique_id': 1,
        'first_name': "John",
        'surname': "Smith",
        'dob': "1971-05-24",
        'city': "London",
        'email': "john@smith.net"
        }

    record_2 = {'unique_id': 1,
        'first_name': "Jon",
        'surname': "Smith",
        'dob': "1971-05-23",
        'city': "London",
        'email': "john@smith.net"
        }
    df = linker.inference.compare_two_records(record_1, record_2)

    ```

Returns:
    SplinkDataFrame: Pairwise comparison with scored prediction

Linker.inference.deterministic_link:
Uses the blocking rules specified by
`blocking_rules_to_generate_predictions` in your settings to
generate pairwise record comparisons.

For deterministic linkage, this should be a list of blocking rules which
are strict enough to generate only true links.

Deterministic linkage, however, is likely to result in missed links
(false negatives).

Returns:
    SplinkDataFrame: A SplinkDataFrame of the pairwise comparisons.


Examples:

    ```py
    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[
            block_on("first_name", "surname"),
            block_on("dob", "first_name"),
        ],
    )

    linker = Linker(df, settings, db_api=db_api)
    splink_df = linker.inference.deterministic_link()
    ```

Linker.inference.find_matches_to_new_records:
Given one or more records, find records in the input dataset(s) which match
and return in order of the Splink prediction score.

This effectively provides a way of searching the input datasets
for given record(s)

Args:
    records_or_tablename (List[dict]): Input search record(s) as list of dict,
        or a table registered to the database.
    blocking_rules (list, optional): Blocking rules to select
        which records to find and score. If [], do not use a blocking
        rule - meaning the input records will be compared to all records
        provided to the linker when it was instantiated. Defaults to [].
    match_weight_threshold (int, optional): Return matches with a match weight
        above this threshold. Defaults to -4.

Examples:
    ```py
    linker = Linker(df, "saved_settings.json", db_api=db_api)

    # You should load or pre-compute tf tables for any tables with
    # term frequency adjustments
    linker.table_management.compute_tf_table("first_name")
    # OR
    linker.table_management.register_term_frequency_lookup(df, "first_name")

    record = {'unique_id': 1,
        'first_name': "John",
        'surname': "Smith",
        'dob': "1971-05-24",
        'city': "London",
        'email': "john@smith.net"
        }
    df = linker.inference.find_matches_to_new_records(
        [record], blocking_rules=[]
    )
    ```

Returns:
    SplinkDataFrame: The pairwise comparisons.

Linker.inference.predict:
Create a dataframe of scored pairwise comparisons using the parameters
of the linkage model.

Uses the blocking rules specified in the
`blocking_rules_to_generate_predictions` key of the settings to
generate the pairwise comparisons.

Args:
    threshold_match_probability (float, optional): If specified,
        filter the results to include only pairwise comparisons with a
        match_probability above this threshold. Defaults to None.
    threshold_match_weight (float, optional): If specified,
        filter the results to include only pairwise comparisons with a
        match_weight above this threshold. Defaults to None.
    materialise_after_computing_term_frequencies (bool): If true, Splink
        will materialise the table containing the input nodes (rows)
        joined to any term frequencies which have been asked
        for in the settings object.  If False, this will be
        computed as part of a large CTE pipeline.   Defaults to True
    materialise_blocked_pairs: In the blocking phase, materialise the table
        of pairs of records that will be scored

Examples:
    ```py
    linker = linker(df, "saved_settings.json", db_api=db_api)
    splink_df = linker.inference.predict(threshold_match_probability=0.95)
    splink_df.as_pandas_dataframe(limit=5)
    ```
Returns:
    SplinkDataFrame: A SplinkDataFrame of the scored pairwise comparisons.

Linker.training.estimate_m_from_label_column:
Estimate the m parameters of the linkage model from a label (ground truth)
column in the input dataframe(s).

The m parameters represent the proportion of record comparisons that fall
into each comparison level amongst truly matching records.

The ground truth column is used to generate pairwise record comparisons
which are then assumed to be matches.

For example, if the entity being matched is persons, and your input dataset(s)
contain social security number, this could be used to estimate the m values
for the model.

Note that this column does not need to be fully populated.  A common case is
where a unique identifier such as social security number is only partially
populated.

Args:
    label_colname (str): The name of the column containing the ground truth
        label in the input data.

Examples:
    ```py
    linker.training.estimate_m_from_label_column("social_security_number")
    ```

Returns:
    Nothing: Updates the estimated m parameters within the linker object.

Linker.training.estimate_m_from_pairwise_labels:
Estimate the m probabilities of the linkage model from a dataframe of
pairwise labels.

The table of labels should be in the following format, and should
be registered with your database:

|source_dataset_l|unique_id_l|source_dataset_r|unique_id_r|
|----------------|-----------|----------------|-----------|
|df_1            |1          |df_2            |2          |
|df_1            |1          |df_2            |3          |

Note that `source_dataset` and `unique_id` should correspond to the
values specified in the settings dict, and the `input_table_aliases`
passed to the `linker` object. Note that at the moment, this method does
not respect values in a `clerical_match_score` column.  If provided, these
are ignored and it is assumed that every row in the table of labels is a score
of 1, i.e. a perfect match.

Args:
  labels_splinkdataframe_or_table_name (str): Name of table containing labels
    in the database or SplinkDataframe

Examples:
    ```py
    pairwise_labels = pd.read_csv("./data/pairwise_labels_to_estimate_m.csv")

    linker.table_management.register_table(
        pairwise_labels, "labels", overwrite=True
    )

    linker.training.estimate_m_from_pairwise_labels("labels")
    ```

Linker.training.estimate_parameters_using_expectation_maximisation:
Estimate the parameters of the linkage model using expectation maximisation.

By default, the m probabilities are estimated, but not the u probabilities,
because good estimates for the u probabilities can be obtained from
`linker.training.estimate_u_using_random_sampling()`.  You can change this by
setting `fix_u_probabilities` to False.

The blocking rule provided is used to generate pairwise record comparisons.
Usually, this should be a blocking rule that results in a dataframe where
matches are between about 1% and 99% of the blocked comparisons.

By default, m parameters are estimated for all comparisons except those which
are included in the blocking rule.

For example, if the blocking rule is `block_on("first_name")`, then
parameter estimates will be made for all comparison except those which use
`first_name` in their sql_condition

By default, the probability two random records match is allowed to vary
during EM estimation, but is not saved back to the model.  See
[this PR](https://github.com/moj-analytical-services/splink/pull/734) for
the rationale.



Args:
    blocking_rule (BlockingRuleCreator | str): The blocking rule used to
        generate pairwise record comparisons.
    estimate_without_term_frequencies (bool, optional): If True, the iterations
        of the EM algorithm ignore any term frequency adjustments and only
        depend on the comparison vectors. This allows the EM algorithm to run
        much faster, but the estimation of the parameters will change slightly.
    fix_probability_two_random_records_match (bool, optional): If True, do not
        update the probability two random records match after each iteration.
        Defaults to False.
    fix_m_probabilities (bool, optional): If True, do not update the m
        probabilities after each iteration. Defaults to False.
    fix_u_probabilities (bool, optional): If True, do not update the u
        probabilities after each iteration. Defaults to True.
    populate_prob... (bool,optional): The full name of this parameter is
        populate_probability_two_random_records_match_from_trained_values. If
        True, derive this parameter from the blocked value. Defaults to False.

Examples:
    ```py
    br_training = block_on("first_name", "dob")
    linker.training.estimate_parameters_using_expectation_maximisation(
        br_training
    )
    ```

Returns:
    EMTrainingSession:  An object containing information about the training
        session such as how parameters changed during the iteration history

Linker.training.estimate_probability_two_random_records_match:
Estimate the model parameter `probability_two_random_records_match` using
a direct estimation approach.

This method counts the number of matches found using deterministic rules and
divides by the total number of possible record comparisons. The recall of the
deterministic rules is used to adjust this proportion up to reflect missed
matches, providing an estimate of the probability that two random records from
the input data are a match.

Note that if more than one deterministic rule is provided, any duplicate
pairs are automatically removed, so you do not need to worry about double
counting.

See [here](https://github.com/moj-analytical-services/splink/issues/462)
for discussion of methodology.

Args:
    deterministic_matching_rules (list): A list of deterministic matching
        rules designed to admit very few (preferably no) false positives.
    recall (float): An estimate of the recall the deterministic matching
        rules will achieve, i.e., the proportion of all true matches these
        rules will recover.
    max_rows_limit (int): Maximum number of rows to consider during estimation.
        Defaults to 1e9.

Examples:
    ```py
    deterministic_rules = [
        block_on("forename", "dob"),
        "l.forename = r.forename and levenshtein(r.surname, l.surname) <= 2",
        block_on("email")
    ]
    linker.training.estimate_probability_two_random_records_match(
        deterministic_rules, recall=0.8
    )
    ```
Returns:
    Nothing: Updates the estimated parameter within the linker object and
        returns nothing.

Linker.training.estimate_u_using_random_sampling:
Estimate the u parameters of the linkage model using random sampling.

The u parameters estimate the proportion of record comparisons that fall
into each comparison level amongst truly non-matching records.

This procedure takes a sample of the data and generates the cartesian
product of pairwise record comparisons amongst the sampled records.
The validity of the u values rests on the assumption that the resultant
pairwise comparisons are non-matches (or at least, they are very unlikely to be
matches). For large datasets, this is typically true.

The results of estimate_u_using_random_sampling, and therefore an entire splink
model, can be made reproducible by setting the seed parameter. Setting the seed
will have performance implications as additional processing is required.

Args:
    max_pairs (int): The maximum number of pairwise record comparisons to
        sample. Larger will give more accurate estimates but lead to longer
        runtimes.  In our experience at least 1e9 (one billion) gives best
        results but can take a long time to compute. 1e7 (ten million)
        is often adequate whilst testing different model specifications, before
        the final model is estimated.
    seed (int): Seed for random sampling. Assign to get reproducible u
        probabilities. Note, seed for random sampling is only supported for
        DuckDB and Spark, for Athena and SQLite set to None.

Examples:
    ```py
    linker.training.estimate_u_using_random_sampling(max_pairs=1e8)
    ```

Returns:
    Nothing: Updates the estimated u parameters within the linker object and
        returns nothing.

Linker.visualisations.cluster_studio_dashboard:
Generate an interactive html visualization of the predicted cluster and
save to `out_path`.

Args:
    df_predict (SplinkDataFrame): The outputs of `linker.predict()`
    df_clustered (SplinkDataFrame): The outputs of
        `linker.cluster_pairwise_predictions_at_threshold()`
    out_path (str): The path (including filename) to save the html file to.
    sampling_method (str, optional): `random`, `by_cluster_size` or
        `lowest_density_clusters_by_size`. Defaults to `random`.
    sample_size (int, optional): Number of clusters to show in the dahboard.
        Defaults to 10.
    cluster_ids (list): The IDs of the clusters that will be displayed in the
        dashboard.  If provided, ignore the `sampling_method` and `sample_size`
        arguments. Defaults to None.
    overwrite (bool, optional): Overwrite the html file if it already exists?
        Defaults to False.
    cluster_names (list, optional): If provided, the dashboard will display
        these names in the selection box. Ony works in conjunction with
        `cluster_ids`.  Defaults to None.
    return_html_as_string: If True, return the html as a string

Examples:
    ```py
    df_p = linker.inference.predict()
    df_c = linker.visualisations.cluster_pairwise_predictions_at_threshold(
        df_p, 0.5
    )

    linker.cluster_studio_dashboard(
        df_p, df_c, [0, 4, 7], "cluster_studio.html"
    )
    ```

    Optionally, in Jupyter, you can display the results inline
    Otherwise you can just load the html file in your browser

    ```py
    from IPython.display import IFrame
    IFrame(src="./cluster_studio.html", width="100%", height=1200)
    ```

Linker.visualisations.comparison_viewer_dashboard:
Generate an interactive html visualization of the linker's predictions and
save to `out_path`.  For more information see
[this video](https://www.youtube.com/watch?v=DNvCMqjipis)


Args:
    df_predict (SplinkDataFrame): The outputs of `linker.predict()`
    out_path (str): The path (including filename) to save the html file to.
    overwrite (bool, optional): Overwrite the html file if it already exists?
        Defaults to False.
    num_example_rows (int, optional): Number of example rows per comparison
        vector. Defaults to 2.
    return_html_as_string: If True, return the html as a string

Examples:
    ```py
    df_predictions = linker.predict()
    linker.visualisations.comparison_viewer_dashboard(
        df_predictions, "scv.html", True, 2
    )
    ```

    Optionally, in Jupyter, you can display the results inline
    Otherwise you can just load the html file in your browser

    ```py
    from IPython.display import IFrame
    IFrame(src="./scv.html", width="100%", height=1200)
    ```

Linker.visualisations.m_u_parameters_chart:
Display a chart of the m and u parameters of the linkage model

Args:
    as_dict (bool, optional): If True, return the chart as a dictionary.

Examples:
    ```py
    altair_chart = linker.visualisations.m_u_parameters_chart()
    altair_chart.save("mychart.png")
    ```

Returns:
    altair_chart: An altair chart

Linker.visualisations.match_weights_chart:
Display a chart of the (partial) match weights of the linkage model

Args:
    as_dict (bool, optional): If True, return the chart as a dictionary.

Examples:
    ```py
    altair_chart = linker.visualisations.match_weights_chart()
    altair_chart.save("mychart.png")
    ```
Returns:
    altair_chart: An Altair chart

Linker.visualisations.match_weights_histogram:
Generate a histogram that shows the distribution of match weights in
`df_predict`

Args:
    df_predict (SplinkDataFrame): Output of `linker.inference.predict()`
    target_bins (int, optional): Target number of bins in histogram. Defaults to
        30.
    width (int, optional): Width of output. Defaults to 600.
    height (int, optional): Height of output chart. Defaults to 250.
    as_dict (bool, optional): If True, return the chart as a dictionary.

Examples:
    ```py
    df_predict = linker.inference.predict(threshold_match_weight=-2)
    linker.visualisations.match_weights_histogram(df_predict)
    ```
Returns:
    altair_chart: An Altair chart

Linker.visualisations.parameter_estimate_comparisons_chart:
Show a chart that shows how parameter estimates have differed across
the different estimation methods you have used.

For example, if you have run two EM estimation sessions, blocking on
different variables, and both result in parameter estimates for
first_name, this chart will enable easy comparison of the different
estimates

Args:
    include_m (bool, optional): Show different estimates of m values. Defaults
        to True.
    include_u (bool, optional): Show different estimates of u values. Defaults
        to False.
    as_dict (bool, optional): If True, return the chart as a dictionary.

Examples:
    ```py
    linker.training.estimate_parameters_using_expectation_maximisation(
        blocking_rule=block_on("first_name"),
    )

    linker.training.estimate_parameters_using_expectation_maximisation(
        blocking_rule=block_on("surname"),
    )

    linker.visualisations.parameter_estimate_comparisons_chart()
    ```

Returns:
    altair_chart: An Altair chart

Linker.visualisations.tf_adjustment_chart:
Display a chart showing the impact of term frequency adjustments on a
specific comparison level.
Each value

Args:
    output_column_name (str): Name of an output column for which term frequency
         adjustment has been applied.
    n_most_freq (int, optional): Number of most frequent values to show. If this
         or `n_least_freq` set to None, all values will be shown.
        Default to 10.
    n_least_freq (int, optional): Number of least frequent values to show. If
        this or `n_most_freq` set to None, all values will be shown.
        Default to 10.
    vals_to_include (list, optional): Specific values for which to show term
        frequency adjustments.
        Defaults to None.
    as_dict (bool, optional): If True, return the chart as a dictionary.

Examples:
    ```py
    linker.visualisations.tf_adjustment_chart("first_name")
    ```

Returns:
    altair_chart: An Altair chart

Linker.visualisations.waterfall_chart:
Visualise how the final match weight is computed for the provided pairwise
record comparisons.

Records must be provided as a list of dictionaries. This would usually be
obtained from `df.as_record_dict(limit=n)` where `df` is a SplinkDataFrame.

Examples:
    ```py
    df = linker.inference.predict(threshold_match_weight=2)
    records = df.as_record_dict(limit=10)
    linker.visualisations.waterfall_chart(records)
    ```

Args:
    records (List[dict]): Usually be obtained from `df.as_record_dict(limit=n)`
        where `df` is a SplinkDataFrame.
    filter_nulls (bool, optional): Whether the visualisation shows null
        comparisons, which have no effect on final match weight. Defaults to
        True.
    remove_sensitive_data (bool, optional): When True, The waterfall chart will
        contain match weights only, and all of the (potentially sensitive) data
        from the input tables will be removed prior to the chart being created.
    as_dict (bool, optional): If True, return the chart as a dictionary.


Returns:
    altair_chart: An Altair chart

Linker.clustering.cluster_pairwise_predictions_at_threshold:
Clusters the pairwise match predictions that result from
`linker.inference.predict()` into groups of connected record using the connected
components graph clustering algorithm

Records with an estimated `match_probability` at or above
`threshold_match_probability` (or records with a `match_weight` at or above
`threshold_match_weight`) are considered to be a match (i.e. they represent
the same entity).

Args:
    df_predict (SplinkDataFrame): The results of `linker.predict()`
    threshold_match_probability (float, optional): Pairwise comparisons with a
        `match_probability` at or above this threshold are matched
    threshold_match_weight (float, optional): Pairwise comparisons with a
        `match_weight` at or above this threshold are matched. Only one of
        threshold_match_probability or threshold_match_weight should be provided

Returns:
    SplinkDataFrame: A SplinkDataFrame containing a list of all IDs, clustered
        into groups based on the desired match threshold.

Examples:
    ```python
    df_predict = linker.inference.predict(threshold_match_probability=0.5)
    df_clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predict, threshold_match_probability=0.95
    )
    ```

Linker.clustering.compute_graph_metrics:
Generates tables containing graph metrics (for nodes, edges and clusters),
and returns a data class of Splink dataframes

Args:
    df_predict (SplinkDataFrame): The results of `linker.inference.predict()`
    df_clustered (SplinkDataFrame): The outputs of
        `linker.clustering.cluster_pairwise_predictions_at_threshold()`
    threshold_match_probability (float, optional): Filter the pairwise match
        predictions to include only pairwise comparisons with a
        match_probability at or above this threshold. If not provided, the value
        will be taken from metadata on `df_clustered`. If no such metadata is
        available, this value _must_ be provided.

Returns:
    GraphMetricsResult: A data class containing SplinkDataFrames
    of cluster IDs and selected node, edge or cluster metrics.
        attribute "nodes" for nodes metrics table
        attribute "edges" for edge metrics table
        attribute "clusters" for cluster metrics table

Examples:
    ```python
    df_predict = linker.inference.predict(threshold_match_probability=0.5)
    df_clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predict, threshold_match_probability=0.95
    )
    graph_metrics = linker.clustering.compute_graph_metrics(
        df_predict, df_clustered, threshold_match_probability=0.95
    )

    node_metrics = graph_metrics.nodes.as_pandas_dataframe()
    edge_metrics = graph_metrics.edges.as_pandas_dataframe()
    cluster_metrics = graph_metrics.clusters.as_pandas_dataframe()
    ```

Linker.evaluation.accuracy_analysis_from_labels_column:
Generate an accuracy chart or table from ground truth data, where the ground
truth is in a column in the input dataset called `labels_column_name`

Args:
    labels_column_name (str): Column name containing labels in the input table
    threshold_match_probability (float, optional): Where the
        `clerical_match_score` provided by the user is a probability rather
        than binary, this value is used as the threshold to classify
        `clerical_match_score`s as binary matches or non matches.
        Defaults to 0.5.
    match_weight_round_to_nearest (float, optional): When provided, thresholds
        are rounded.  When large numbers of labels are provided, this is
        sometimes necessary to reduce the size of the ROC table, and therefore
        the number of points plotted on the chart. Defaults to None.
    add_metrics (list(str), optional): Precision and recall metrics are always
        included. Where provided, `add_metrics` specifies additional metrics
        to show, with the following options:

        - `"specificity"`: specificity, selectivity, true negative rate (TNR)
        - `"npv"`: negative predictive value (NPV)
        - `"accuracy"`: overall accuracy (TP+TN)/(P+N)
        - `"f1"`/`"f2"`/`"f0_5"`: F-scores for β=1 (balanced), β=2
        (emphasis on recall) and β=0.5 (emphasis on precision)
        - `"p4"` -  an extended F1 score with specificity and NPV included
        - `"phi"` - φ coefficient or Matthews correlation coefficient (MCC)

Examples:
    ```py
    linker.evaluation.accuracy_analysis_from_labels_column("ground_truth", add_metrics=["f1"])
    ```

Returns:
    chart: An altair chart

Linker.evaluation.accuracy_analysis_from_labels_table:
Generate an accuracy chart or table from labelled (ground truth) data.

The table of labels should be in the following format, and should be registered
as a table with your database using
`labels_table = linker.register_labels_table(my_df)`

|source_dataset_l|unique_id_l|source_dataset_r|unique_id_r|clerical_match_score|
|----------------|-----------|----------------|-----------|--------------------|
|df_1            |1          |df_2            |2          |0.99                |
|df_1            |1          |df_2            |3          |0.2                 |

Note that `source_dataset` and `unique_id` should correspond to the values
specified in the settings dict, and the `input_table_aliases` passed to the
`linker` object.

For `dedupe_only` links, the `source_dataset` columns can be ommitted.

Args:
    labels_splinkdataframe_or_table_name (str | SplinkDataFrame): Name of table
        containing labels in the database
    threshold_match_probability (float, optional): Where the
        `clerical_match_score` provided by the user is a probability rather
        than binary, this value is used as the threshold to classify
        `clerical_match_score`s as binary matches or non matches.
        Defaults to 0.5.
    match_weight_round_to_nearest (float, optional): When provided, thresholds
        are rounded.  When large numbers of labels are provided, this is
        sometimes necessary to reduce the size of the ROC table, and therefore
        the number of points plotted on the chart. Defaults to None.
    add_metrics (list(str), optional): Precision and recall metrics are always
        included. Where provided, `add_metrics` specifies additional metrics
        to show, with the following options:

        - `"specificity"`: specificity, selectivity, true negative rate (TNR)
        - `"npv"`: negative predictive value (NPV)
        - `"accuracy"`: overall accuracy (TP+TN)/(P+N)
        - `"f1"`/`"f2"`/`"f0_5"`: F-scores for β=1 (balanced), β=2
        (emphasis on recall) and β=0.5 (emphasis on precision)
        - `"p4"` -  an extended F1 score with specificity and NPV included
        - `"phi"` - φ coefficient or Matthews correlation coefficient (MCC)

Returns:
    altair.Chart: An altair chart

Examples:
    ```py
    linker.accuracy_analysis_from_labels_table("ground_truth", add_metrics=["f1"])
    ```

Linker.evaluation.labelling_tool_for_specific_record:
Create a standalone, offline labelling dashboard for a specific record
as identified by its unique id

Args:
    unique_id (str): The unique id of the record for which to create the
        labelling tool
    source_dataset (str, optional): If there are multiple datasets, to
        identify the record you must also specify the source_dataset. Defaults
        to None.
    out_path (str, optional): The output path for the labelling tool. Defaults
        to "labelling_tool.html".
    overwrite (bool, optional): If true, overwrite files at the output
        path if they exist. Defaults to False.
    match_weight_threshold (int, optional): Include possible matches in the
        output which score above this threshold. Defaults to -4.
    view_in_jupyter (bool, optional): If you're viewing in the Jupyter
        html viewer, set this to True to extract your labels. Defaults to False.
    show_splink_predictions_in_interface (bool, optional): Whether to
        show information about the Splink model's predictions that could
        potentially bias the decision of the clerical labeller. Defaults to
        True.

Linker.evaluation.prediction_errors_from_labels_column:
Generate a dataframe containing false positives and false negatives
based on the comparison between the splink match probability and the
labels column.  A label column is a column in the input dataset that contains
the 'ground truth' cluster to which the record belongs

Args:
    label_colname (str): Name of labels column in input data
    include_false_positives (bool, optional): Defaults to True.
    include_false_negatives (bool, optional): Defaults to True.
    threshold_match_probability (float, optional): Threshold above which a score
        is considered to be a match. Defaults to 0.5.

Returns:
    SplinkDataFrame:  Table containing false positives and negatives

Examples:
    ```py
    linker.evaluation.prediction_errors_from_labels_column(
        "ground_truth_cluster",
        include_false_negatives=True,
        include_false_positives=False
    ).as_pandas_dataframe()
    ```

Linker.evaluation.prediction_errors_from_labels_table:
Find false positives and false negatives based on the comparison between the
`clerical_match_score` in the labels table compared with the splink predicted
match probability

The table of labels should be in the following format, and should be registered
as a table with your database using

`labels_table = linker.table_management.register_labels_table(my_df)`

|source_dataset_l|unique_id_l|source_dataset_r|unique_id_r|clerical_match_score|
|----------------|-----------|----------------|-----------|--------------------|
|df_1            |1          |df_2            |2          |0.99                |
|df_1            |1          |df_2            |3          |0.2                 |

Args:
    labels_splinkdataframe_or_table_name (str | SplinkDataFrame): Name of table
        containing labels in the database
    include_false_positives (bool, optional): Defaults to True.
    include_false_negatives (bool, optional): Defaults to True.
    threshold_match_probability (float, optional): Threshold probability
        above which a prediction considered to be a match. Defaults to 0.5.

Examples:
    ```py
    labels_table = linker.table_management.register_labels_table(df_labels)

    linker.evaluation.prediction_errors_from_labels_table(
       labels_table, include_false_negatives=True, include_false_positives=False
    ).as_pandas_dataframe()
    ```

Returns:
    SplinkDataFrame:  Table containing false positives and negatives

Linker.evaluation.unlinkables_chart:
Generate an interactive chart displaying the proportion of records that
are "unlinkable" for a given splink score threshold and model parameters.

Unlinkable records are those that, even when compared with themselves, do not
contain enough information to confirm a match.

Args:
    x_col (str, optional): Column to use for the x-axis.
        Defaults to "match_weight".
    name_of_data_in_title (str, optional): Name of the source dataset to use for
        the title of the output chart.
    as_dict (bool, optional): If True, return a dict version of the chart.

Returns:
    altair.Chart: An altair chart

Examples:
    After estimating the parameters of the model, run:

    ```py
    linker.evaluation.unlinkables_chart()
    ```

Linker.table_management.compute_tf_table:
Compute a term frequency table for a given column and persist to the database

This method is useful if you want to pre-compute term frequency tables e.g.
so that real time linkage executes faster, or so that you can estimate
various models without having to recompute term frequency tables each time

Examples:

    Real time linkage
    ```py
    linker = Linker(df, settings="saved_settings.json", db_api=db_api)
    linker.table_management.compute_tf_table("surname")
    linker.compare_two_records(record_left, record_right)
    ```
    Pre-computed term frequency tables
    ```py
    linker = Linker(df, db_api)
    df_first_name_tf = linker.table_management.compute_tf_table("first_name")
    df_first_name_tf.write.parquet("folder/first_name_tf")
    >>>
    # On subsequent data linking job, read this table rather than recompute
    df_first_name_tf = pd.read_parquet("folder/first_name_tf")
    linker.table_management.register_term_frequency_lookup(
        df_first_name_tf, "first_name"
    )

    ```


Args:
    column_name (str): The column name in the input table

Returns:
    SplinkDataFrame: The resultant table as a splink data frame

Linker.table_management.delete_tables_created_by_splink_from_db:
No docstring available

Linker.table_management.invalidate_cache:
Invalidate the Splink cache.  Any previously-computed tables
will be recomputed.
This is useful, for example, if the input data tables have changed.

Linker.table_management.register_labels_table:
No docstring available

Linker.table_management.register_table:
Register a table to your backend database, to be used in one of the
splink methods, or simply to allow querying.

Tables can be of type: dictionary, record level dictionary,
pandas dataframe, pyarrow table and in the spark case, a spark df.

Examples:
    ```py
    test_dict = {"a": [666,777,888],"b": [4,5,6]}
    linker.table_management.register_table(test_dict, "test_dict")
    linker.query_sql("select * from test_dict")
    ```

Args:
    input_table: The data you wish to register. This can be either a dictionary,
        pandas dataframe, pyarrow table or a spark dataframe.
    table_name (str): The name you wish to assign to the table.
    overwrite (bool): Overwrite the table in the underlying database if it
        exists

Returns:
    SplinkDataFrame: An abstraction representing the table created by the sql
        pipeline

Linker.table_management.register_table_input_nodes_concat_with_tf:
Register a pre-computed version of the input_nodes_concat_with_tf table that
you want to re-use e.g. that you created in a previous run.

This method allows you to register this table in the Splink cache so it will be
used rather than Splink computing this table anew.

Args:
    input_data (AcceptableInputTableType): The data you wish to register. This
        can be either a dictionary, pandas dataframe, pyarrow table or a spark
        dataframe.
    overwrite (bool): Overwrite the table in the underlying database if it
        exists.

Returns:
    SplinkDataFrame: An abstraction representing the table created by the sql
        pipeline

Linker.table_management.register_table_predict:
Register a pre-computed version of the prediction table for use in Splink.

This method allows you to register a pre-computed prediction table in the Splink
cache so it will be used rather than Splink computing the table anew.

Examples:
    ```py
    predict_df = pd.read_parquet("path/to/predict_df.parquet")
    predict_as_splinkdataframe = linker.table_management.register_table_predict(predict_df)
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        predict_as_splinkdataframe, threshold_match_probability=0.75
    )
    ```

Args:
    input_data (AcceptableInputTableType): The data you wish to register. This
        can be either a dictionary, pandas dataframe, pyarrow table, or a spark
        dataframe.
    overwrite (bool, optional): Overwrite the table in the underlying database
        if it exists. Defaults to False.

Returns:
    SplinkDataFrame: An abstraction representing the table created by the SQL
        pipeline.

Linker.table_management.register_term_frequency_lookup:
Register a pre-computed term frequency lookup table for a given column.

This method allows you to register a term frequency table in the Splink
cache for a specific column. This table will then be used during linkage
rather than computing the term frequency table anew from your input data.

Args:
    input_data (AcceptableInputTableType): The data representing the term
        frequency table. This can be either a dictionary, pandas dataframe,
        pyarrow table, or a spark dataframe.
    col_name (str): The name of the column for which the term frequency
        lookup table is being registered.
    overwrite (bool, optional): Overwrite the table in the underlying
        database if it exists. Defaults to False.

Returns:
    SplinkDataFrame: An abstraction representing the registered term
    frequency table.

Examples:
    ```py
    tf_table = [
        {"first_name": "theodore", "tf_first_name": 0.012},
        {"first_name": "alfie", "tf_first_name": 0.013},
    ]
    tf_df = pd.DataFrame(tf_table)
    linker.table_management.register_term_frequency_lookup(
        tf_df,
        "first_name"
    )
    ```

Linker.misc.query_sql:
Run a SQL query against your backend database and return
the resulting output.

Examples:
    ```py
    linker = Linker(df, settings, db_api)
    df_predict = linker.inference.predict()
    linker.misc.query_sql(f"select * from {df_predict.physical_name} limit 10")
    ```

Args:
    sql (str): The SQL to be queried.
    output_type (str): One of splink_df/splinkdf or pandas.
        This determines the type of table that your results are output in.

Linker.misc.save_model_to_json:
Save the configuration and parameters of the linkage model to a `.json` file.

The model can later be loaded into a new linker using
`Linker(df, settings="path/to/model.json", db_api=db_api).

The settings dict is also returned in case you want to save it a different way.

Examples:
    ```py
    linker.misc.save_model_to_json("my_settings.json", overwrite=True)
    ```
Args:
    out_path (str, optional): File path for json file. If None, don't save to
        file. Defaults to None.
    overwrite (bool, optional): Overwrite if already exists? Defaults to False.

Returns:
    dict: The settings as a dictionary.


## Comparison Methods

AbsoluteDateDifferenceAtThresholds:
Represents a comparison of the data in `col_name` with multiple levels based on
absolute time differences:

- Exact match in `col_name`
- Absolute time difference levels at specified thresholds
- ...
- Anything else

For example, with metrics = ['day', 'month'] and thresholds = [1, 3] the levels
are:

- Exact match in `col_name`
- Absolute time difference in `col_name` <= 1 day
- Absolute time difference in `col_name` <= 3 months
- Anything else

This comparison uses the AbsoluteTimeDifferenceLevel, which computes the total
elapsed time between two dates, rather than counting calendar intervals.

Args:
    col_name (str): The name of the column to compare.
    input_is_string (bool): If True, the input dates are treated as strings
        and parsed according to `datetime_format`.
    metrics (Union[DateMetricType, List[DateMetricType]]): The unit(s) of time
        to use when comparing dates. Can be 'second', 'minute', 'hour', 'day',
        'month', or 'year'.
    thresholds (Union[int, float, List[Union[int, float]]]): The threshold(s)
        to use for the time difference level(s).
    datetime_format (str, optional): The format string for parsing dates if
        `input_is_string` is True. ISO 8601 format used if not provided.
    term_frequency_adjustments (bool, optional): Whether to apply term frequency
        adjustments. Defaults to False.
    invalid_dates_as_null (bool, optional): If True and `input_is_string` is
        True, treat invalid dates as null. Defaults to True.

AbsoluteTimeDifferenceAtThresholds:
Represents a comparison of the data in `col_name` with multiple levels based on
absolute time differences:

- Exact match in `col_name`
- Absolute time difference levels at specified thresholds
- ...
- Anything else

For example, with metrics = ['day', 'month'] and thresholds = [1, 3] the levels
are:

- Exact match in `col_name`
- Absolute time difference in `col_name` <= 1 day
- Absolute time difference in `col_name` <= 3 months
- Anything else

This comparison uses the AbsoluteTimeDifferenceLevel, which computes the total
elapsed time between two dates, rather than counting calendar intervals.

Args:
    col_name (str): The name of the column to compare.
    input_is_string (bool): If True, the input dates are treated as strings
        and parsed according to `datetime_format`.
    metrics (Union[DateMetricType, List[DateMetricType]]): The unit(s) of time
        to use when comparing dates. Can be 'second', 'minute', 'hour', 'day',
        'month', or 'year'.
    thresholds (Union[int, float, List[Union[int, float]]]): The threshold(s)
        to use for the time difference level(s).
    datetime_format (str, optional): The format string for parsing dates if
        `input_is_string` is True. ISO 8601 format used if not provided.
    term_frequency_adjustments (bool, optional): Whether to apply term frequency
        adjustments. Defaults to False.
    invalid_dates_as_null (bool, optional): If True and `input_is_string` is
        True, treat invalid dates as null. Defaults to True.

ArrayIntersectAtSizes:
Represents a comparison of the data in `col_name` with multiple levels based on
the intersection sizes of array elements:

- Intersection at specified size thresholds
- ...
- Anything else

For example, with size_threshold_or_thresholds = [3, 1], the levels are:

- Intersection of arrays in `col_name` has at least 3 elements
- Intersection of arrays in `col_name` has at least 1 element
- Anything else (e.g., empty intersection)

Args:
    col_name (str): The name of the column to compare.
    size_threshold_or_thresholds (Union[int, list[int]], optional): The
        size threshold(s) for the intersection levels.
        Defaults to [1].

CosineSimilarityAtThresholds:
Represents a comparison of the data in `col_name` with two or more levels:

- Cosine similarity levels at specified thresholds
- ...
- Anything else

For example, with score_threshold_or_thresholds = [0.9, 0.7] the levels are:

- Cosine similarity in `col_name` >= 0.9
- Cosine similarity in `col_name` >= 0.7
- Anything else

Args:
    col_name (str): The name of the column to compare.
    score_threshold_or_thresholds (Union[float, list], optional): The
        threshold(s) to use for the cosine similarity level(s).
        Defaults to [0.9, 0.7].

CustomComparison:
Represents a comparison of the data with custom supplied levels.

Args:
    output_column_name (str): The column name to use to refer to this comparison
    comparison_levels (list): A list of some combination of
        `ComparisonLevelCreator` objects, or dicts. These represent the
        similarity levels assessed by the comparison, in order of decreasing
        specificity
    comparison_description (str, optional): An optional description of the
        comparison

DamerauLevenshteinAtThresholds:
Represents a comparison of the data in `col_name` with three or more levels:

- Exact match in `col_name`
- Damerau-Levenshtein levels at specified distance thresholds
- ...
- Anything else

For example, with distance_threshold_or_thresholds = [1, 3] the levels are

- Exact match in `col_name`
- Damerau-Levenshtein distance in `col_name` <= 1
- Damerau-Levenshtein distance in `col_name` <= 3
- Anything else

Args:
    col_name (str): The name of the column to compare.
    distance_threshold_or_thresholds (Union[int, list], optional): The
        threshold(s) to use for the Damerau-Levenshtein similarity level(s).
        Defaults to [1, 2].

DateOfBirthComparison:
Generate an 'out of the box' comparison for a date of birth column
in the `col_name` provided.

Note that `input_is_string` is a required argument: you must denote whether the
`col_name` contains if of type date/dattime or string.

The default arguments will give a comparison with comparison levels:

- Exact match (all other dates)
- Damerau-Levenshtein distance <= 1
- Date difference <= 1 month
- Date difference <= 1 year
- Date difference <= 10 years
- Anything else

Args:
    col_name (Union[str, ColumnExpression]): The column name
    input_is_string (bool): If True, the provided `col_name` must be of type
        string.  If False, it must be a date or datetime.
    datetime_thresholds (Union[int, float, List[Union[int, float]]], optional):
        Numeric thresholds for date differences. Defaults to [1, 1, 10].
    datetime_metrics (Union[DateMetricType, List[DateMetricType]], optional):
        Metrics for date differences. Defaults to ["month", "year", "year"].
    datetime_format (str, optional): The datetime format used to cast strings
        to dates.  Only used if input is a string.
    invalid_dates_as_null (bool, optional): If True, treat invalid dates as null
        as opposed to allowing e.g. an exact or levenshtein match where one side
        or both are an invalid date.  Only used if input is a string.  Defaults
        to True.

DistanceFunctionAtThresholds:
Represents a comparison of the data in `col_name` with three or more levels:

- Exact match in `col_name`
- Custom distance function levels at specified thresholds
- ...
- Anything else

For example, with distance_threshold_or_thresholds = [1, 3]
and distance_function 'hamming', with higher_is_more_similar False
the levels are:

- Exact match in `col_name`
- Hamming distance of `col_name` <= 1
- Hamming distance of `col_name` <= 3
- Anything else

Args:
    col_name (str): The name of the column to compare.
    distance_function_name (str): the name of the SQL distance function
    distance_threshold_or_thresholds (Union[float, list], optional): The
        threshold(s) to use for the distance function level(s).
    higher_is_more_similar (bool): Are higher values of the distance function
        more similar? (e.g. True for Jaro-Winkler, False for Levenshtein)
        Default is True

DistanceInKMAtThresholds:
A comparison of the latitude, longitude coordinates defined in
'lat_col' and 'long col' giving the great circle distance between them in km.

An example of the output with km_thresholds = [1, 10] would be:

- The two coordinates are within 1 km of one another
- The two coordinates are within 10 km of one another
- Anything else (i.e. the distance between coordinates are > 10km apart)

Args:
    lat_col (str): The name of the latitude column to compare.
    long_col (str): The name of the longitude column to compare.
    km_thresholds (iterable[float] | float): The km threshold(s) for the
        distance levels.

EmailComparison:
Generate an 'out of the box' comparison for an email address column with the
in the `col_name` provided.

The default comparison levels are:

- Null comparison: e.g., one email is missing or invalid.
- Exact match on full email: e.g., `john@smith.com` vs. `john@smith.com`.
- Exact match on username part of email: e.g., `john@company.com` vs.
`john@other.com`.
- Jaro-Winkler similarity > 0.88 on full email: e.g., `john.smith@company.com`
vs. `john.smyth@company.com`.
- Jaro-Winkler similarity > 0.88 on username part of email: e.g.,
`john.smith@company.com` vs. `john.smyth@other.com`.
- Anything else: e.g., `john@company.com` vs. `rebecca@other.com`.

Args:
    col_name (Union[str, ColumnExpression]): The column name or expression for
        the email addresses to be compared.

ExactMatch:
Class to author Comparisons
Args:
    col_name_or_names (str, ColumnExpression): Input column name(s).
        Can be a single item or a dict.

ForenameSurnameComparison:
Generate an 'out of the box' comparison for forename and surname columns
in the `forename_col_name` and `surname_col_name` provided.

It's recommended to derive an additional column containing a concatenated
forename and surname column so that term frequencies can be applied to the
full name.  If you have derived a column, provide it at
`forename_surname_concat_col_name`.

The default comparison levels are:

- Null comparison on both forename and surname
- Exact match on both forename and surname
- Columns reversed comparison (forename and surname swapped)
- Jaro-Winkler similarity > 0.92 on both forename and surname
- Jaro-Winkler similarity > 0.88 on both forename and surname
- Exact match on surname
- Exact match on forename
- Anything else

Args:
    forename_col_name (Union[str, ColumnExpression]): The column name or
        expression for the forenames to be compared.
    surname_col_name (Union[str, ColumnExpression]): The column name or
        expression for the surnames to be compared.
    jaro_winkler_thresholds (Union[float, list[float]], optional): Thresholds
        for Jaro-Winkler similarity. Defaults to [0.92, 0.88].
    forename_surname_concat_col_name (str, optional): The column name for
        concatenated forename and surname values. If provided, term
        frequencies are applied on the exact match using this column

JaccardAtThresholds:
Represents a comparison of the data in `col_name` with three or more levels:

- Exact match in `col_name`
- Jaccard score levels at specified thresholds
- ...
- Anything else

For example, with score_threshold_or_thresholds = [0.9, 0.7] the levels are:

- Exact match in `col_name`
- Jaccard score in `col_name` >= 0.9
- Jaccard score in `col_name` >= 0.7
- Anything else

Args:
    col_name (str): The name of the column to compare.
    score_threshold_or_thresholds (Union[float, list], optional): The
        threshold(s) to use for the Jaccard similarity level(s).
        Defaults to [0.9, 0.7].

JaroAtThresholds:
Represents a comparison of the data in `col_name` with three or more levels:

- Exact match in `col_name`
- Jaro score levels at specified thresholds
- ...
- Anything else

For example, with score_threshold_or_thresholds = [0.9, 0.7] the levels are:

- Exact match in `col_name`
- Jaro score in `col_name` >= 0.9
- Jaro score in `col_name` >= 0.7
- Anything else

Args:
    col_name (str): The name of the column to compare.
    score_threshold_or_thresholds (Union[float, list], optional): The
        threshold(s) to use for the Jaro similarity level(s).
        Defaults to [0.9, 0.7].

JaroWinklerAtThresholds:
Represents a comparison of the data in `col_name` with three or more levels:

- Exact match in `col_name`
- Jaro-Winkler score levels at specified thresholds
- ...
- Anything else

For example, with score_threshold_or_thresholds = [0.9, 0.7] the levels are:

- Exact match in `col_name`
- Jaro-Winkler score in `col_name` >= 0.9
- Jaro-Winkler score in `col_name` >= 0.7
- Anything else

Args:
    col_name (str): The name of the column to compare.
    score_threshold_or_thresholds (Union[float, list], optional): The
        threshold(s) to use for the Jaro-Winkler similarity level(s).
        Defaults to [0.9, 0.7].

LevenshteinAtThresholds:
Represents a comparison of the data in `col_name` with three or more levels:

- Exact match in `col_name`
- Levenshtein levels at specified distance thresholds
- ...
- Anything else

For example, with distance_threshold_or_thresholds = [1, 3] the levels are

- Exact match in `col_name`
- Levenshtein distance in `col_name` <= 1
- Levenshtein distance in `col_name` <= 3
- Anything else

Args:
    col_name (str): The name of the column to compare
    distance_threshold_or_thresholds (Union[int, list], optional): The
        threshold(s) to use for the levenshtein similarity level(s).
        Defaults to [1, 2].

NameComparison:
Generate an 'out of the box' comparison for a name column in the `col_name`
provided.

It's also possible to include a level for a dmetaphone match, but this requires
you to derive a dmetaphone column prior to importing it into Splink. Note
this is expected to be a column containing arrays of dmetaphone values, which
are of length 1 or 2.

The default comparison levels are:

- Null comparison
- Exact match
- Jaro-Winkler similarity > 0.92
- Jaro-Winkler similarity > 0.88
- Jaro-Winkler similarity > 0.70
- Anything else

Args:
    col_name (Union[str, ColumnExpression]): The column name or expression for
        the names to be compared.
    jaro_winkler_thresholds (Union[float, list[float]], optional): Thresholds
        for Jaro-Winkler similarity. Defaults to [0.92, 0.88, 0.7].
    dmeta_col_name (str, optional): The column name for dmetaphone values.
        If provided, array intersection level is included. This column must
        contain arrays of dmetaphone values, which are of length 1 or 2.

PairwiseStringDistanceFunctionAtThresholds:
Represents a comparison of the *most similar pair* of values
where the first value is in the array data in `col_name` for the first record
and the second value is in the array data in `col_name` for the second record.
The comparison has three or more levels:

- Exact match between any pair of values
- User-selected string distance function levels at specified thresholds
- ...
- Anything else

For example, with distance_threshold_or_thresholds = [1, 3]
and distance_function 'levenshtein' the levels are:

- Exact match between any pair of values
- Levenshtein distance between the most similar pair of values <= 1
- Levenshtein distance between the most similar pair of values <= 3
- Anything else

Args:
    col_name (str): The name of the column to compare.
    distance_function_name (str): the name of the string distance function.
        Must be one of "levenshtein," "damera_levenshtein," "jaro_winkler,"
        or "jaro."
    distance_threshold_or_thresholds (Union[float, list], optional): The
        threshold(s) to use for the distance function level(s).

PostcodeComparison:
Generate an 'out of the box' comparison for a postcode column with the
in the `col_name` provided.

The default comparison levels are:

- Null comparison
- Exact match on full postcode
- Exact match on sector
- Exact match on district
- Exact match on area
- Distance in km (if lat_col and long_col are provided)

It's also possible to include levels for distance in km, but this requires
you to have geocoded your postcodes prior to importing them into Splink. Use
the `lat_col` and `long_col` arguments to tell Splink where to find the
latitude and longitude columns.

See https://ideal-postcodes.co.uk/guides/uk-postcode-format
for definitions

Args:
    col_name (Union[str, ColumnExpression]): The column name or expression for
        the postcodes to be compared.
    invalid_postcodes_as_null (bool, optional): If True, treat invalid postcodes
        as null. Defaults to False.
    lat_col (Union[str, ColumnExpression], optional): The column name or
        expression for latitude. Required if `km_thresholds` is provided.
    long_col (Union[str, ColumnExpression], optional): The column name or
        expression for longitude. Required if `km_thresholds` is provided.
    km_thresholds (Union[float, List[float]], optional): Thresholds for distance
        in kilometers. If provided, `lat_col` and `long_col` must also be
        provided.


## Comparison Level Methods

AbsoluteDateDifferenceLevel:
Computes the absolute elapsed time between two dates (total duration).

This function computes the amount of time that has passed between two dates,
in contrast to functions like `date_diff` found in some SQL backends,
which count the number of full calendar intervals (e.g., months, years) crossed.

For instance, the difference between January 29th and March 2nd would be less
than two months in terms of elapsed time, unlike a `date_diff` calculation that
would give an answer of 2 calendar intervals crossed.

That the thresold is inclusive e.g. a level with a 10 day threshold
will include difference in date of 10 days.

Args:
    col_name (str): The name of the input column containing the dates to compare
    input_is_string (bool): Indicates if the input date/times are in
        string format, requiring parsing according to `datetime_format`.
    threshold (int): The maximum allowed difference between the two dates,
        in units specified by `date_metric`.
    metric (str): The unit of time to use when comparing the dates.
        Can be 'second', 'minute', 'hour', 'day', 'month', or 'year'.
    datetime_format (str, optional): The format string for parsing dates.
        ISO 8601 format used if not provided.

AbsoluteDifferenceLevel:
Represents a comparison level where the absolute difference between two
numerical values is within a specified threshold.

Args:
    col_name (str | ColumnExpression): Input column name or ColumnExpression.
    difference_threshold (int | float): The maximum allowed absolute difference
        between the two values.

AbsoluteTimeDifferenceLevel:
Computes the absolute elapsed time between two dates (total duration).

This function computes the amount of time that has passed between two dates,
in contrast to functions like `date_diff` found in some SQL backends,
which count the number of full calendar intervals (e.g., months, years) crossed.

For instance, the difference between January 29th and March 2nd would be less
than two months in terms of elapsed time, unlike a `date_diff` calculation that
would give an answer of 2 calendar intervals crossed.

That the thresold is inclusive e.g. a level with a 10 day threshold
will include difference in date of 10 days.

Args:
    col_name (str): The name of the input column containing the dates to compare
    input_is_string (bool): Indicates if the input date/times are in
        string format, requiring parsing according to `datetime_format`.
    threshold (int): The maximum allowed difference between the two dates,
        in units specified by `date_metric`.
    metric (str): The unit of time to use when comparing the dates.
        Can be 'second', 'minute', 'hour', 'day', 'month', or 'year'.
    datetime_format (str, optional): The format string for parsing dates.
        ISO 8601 format used if not provided.

And:
Initialize self.  See help(type(self)) for accurate signature.

ArrayIntersectLevel:
Represents a comparison level based around the size of an intersection of
arrays

Args:
    col_name (str): Input column name
    min_intersection (int, optional): The minimum cardinality of the
        intersection of arrays for this comparison level. Defaults to 1

ArraySubsetLevel:
Represents a comparison level where the smaller array is an
exact subset of the larger array. If arrays are equal length, they
must have the same elements

The order of items in the arrays does not matter for this comparison.

Args:
    col_name (str | ColumnExpression): Input column name or ColumnExpression
    empty_is_subset (bool): If True, an empty array is considered a subset of
        any array (including another empty array). Default is False.

ColumnsReversedLevel:
Represents a comparison level where the columns are reversed. For example,
if surname is in the forename field and vice versa

By default, col_l = col_r.  If the symmetrical argument is True, then
col_l = col_r AND col_r = col_l.

Args:
    col_name_1 (str): First column, e.g. forename
    col_name_2 (str): Second column, e.g. surname
    symmetrical (bool): If True, equality is required in in both directions.
        Default is False.

CosineSimilarityLevel:
A comparison level using a cosine similarity function

e.g. array_cosine_similarity(val_l, val_r) >= similarity_threshold

Args:
    col_name (str): Input column name
    similarity_threshold (float): The threshold to use to assess
        similarity. Should be between 0 and 1.

CustomLevel:
Represents a comparison level with a custom sql expression

Must be in a form suitable for use in a SQL CASE WHEN expression
e.g. "substr(name_l, 1, 1) = substr(name_r, 1, 1)"

Args:
    sql_condition (str): SQL condition to assess similarity
    label_for_charts (str, optional): A label for this level to be used in
        charts. Default None, so that `sql_condition` is used
    base_dialect_str (str, optional): If specified, the SQL dialect that
        this expression will parsed as when attempting to translate to
        other backends

DamerauLevenshteinLevel:
A comparison level using a Damerau-Levenshtein distance function

e.g. damerau_levenshtein(val_l, val_r) <= distance_threshold

Args:
    col_name (str): Input column name
    distance_threshold (int): The threshold to use to assess
        similarity

DistanceFunctionLevel:
A comparison level using an arbitrary distance function

e.g. `custom_distance(val_l, val_r) >= (<=) distance_threshold`

The function given by `distance_function_name` must exist in the SQL
backend you use, and must take two parameters of the type in `col_name,
returning a numeric type

Args:
    col_name (str | ColumnExpression): Input column name
    distance_function_name (str): the name of the SQL distance function
    distance_threshold (Union[int, float]): The threshold to use to assess
        similarity
    higher_is_more_similar (bool): Are higher values of the distance function
        more similar? (e.g. True for Jaro-Winkler, False for Levenshtein)
        Default is True

DistanceInKMLevel:
Use the haversine formula to transform comparisons of lat,lngs
into distances measured in kilometers

Arguments:
    lat_col (str): The name of a latitude column or the respective array
        or struct column column containing the information
        For example: long_lat['lat'] or long_lat[0]
    long_col (str): The name of a longitudinal column or the respective array
        or struct column column containing the information, plus an index.
        For example: long_lat['long'] or long_lat[1]
    km_threshold (int): The total distance in kilometers to evaluate your
        comparisons against
    not_null (bool): If true, ensure no attempt is made to compute this if
        any inputs are null. This is only necessary if you are not
        capturing nulls elsewhere in your comparison level.

ElseLevel:
Initialize self.  See help(type(self)) for accurate signature.

ExactMatchLevel:
Represents a comparison level where there is an exact match

e.g. val_l = val_r

Args:
    col_name (str): Input column name
    term_frequency_adjustments (bool, optional): If True, apply term frequency
        adjustments to the exact match level. Defaults to False.

JaccardLevel:
A comparison level using a Jaccard distance function

e.g. `jaccard(val_l, val_r) >= distance_threshold`

Args:
    col_name (str): Input column name
    distance_threshold (Union[int, float]): The threshold to use to assess
        similarity

JaroLevel:
A comparison level using a Jaro distance function

e.g. `jaro(val_l, val_r) >= distance_threshold`

Args:
    col_name (str): Input column name
    distance_threshold (Union[int, float]): The threshold to use to assess
        similarity

JaroWinklerLevel:
A comparison level using a Jaro-Winkler distance function

e.g. `jaro_winkler(val_l, val_r) >= distance_threshold`

Args:
    col_name (str): Input column name
    distance_threshold (Union[int, float]): The threshold to use to assess
        similarity

LevenshteinLevel:
A comparison level using a sqlglot_dialect_name distance function

e.g. levenshtein(val_l, val_r) <= distance_threshold

Args:
    col_name (str): Input column name
    distance_threshold (int): The threshold to use to assess
        similarity

LiteralMatchLevel:
Represents a comparison level where a column matches a literal value

e.g. val_l = 'literal' AND/OR val_r = 'literal'

Args:
    col_name (Union[str, ColumnExpression]): Input column name or
        ColumnExpression
    literal_value (str): The literal value to compare against e.g. 'male'
    literal_datatype (str): The datatype of the literal value.
        Must be one of: "string", "int", "float", "date"
    side_of_comparison (str, optional): Which side(s) of the comparison to
        apply. Must be one of: "left", "right", "both". Defaults to "both".

Not:
Initialize self.  See help(type(self)) for accurate signature.

NullLevel:
Initialize self.  See help(type(self)) for accurate signature.

Or:
Initialize self.  See help(type(self)) for accurate signature.

PairwiseStringDistanceFunctionLevel:
A comparison level using the *most similar* string distance
between any pair of values between arrays in an array column.

The function given by `distance_function_name` must be one of
"levenshtein," "damera_levenshtein," "jaro_winkler," or "jaro."

Args:
    col_name (str | ColumnExpression): Input column name
    distance_function_name (str): the name of the string distance function
    distance_threshold (Union[int, float]): The threshold to use to assess
        similarity

PercentageDifferenceLevel:
Represents a comparison level where the difference between two numerical
values is within a specified percentage threshold.

The percentage difference is calculated as the absolute difference between the
two values divided by the greater of the two values.

Args:
    col_name (str): Input column name.
    percentage_threshold (float): The threshold percentage to use
        to assess similarity e.g. 0.1 for 10%.


## Exploratory Functions

completeness_chart:
Generate a summary chart of data completeness (proportion of non-nulls) of
columns in each of the input table or tables. By default, completeness is assessed
for all columns in the input data.

Args:
    table_or_tables: A single table or a list of tables of data
    db_api (DatabaseAPISubClass): The backend database API to use
    cols (List[str], optional): List of column names to calculate completeness. If
        none, all columns will be computed. Default to None.
    table_names_for_chart: A list of names.  Must be the same length as
        table_or_tables.

profile_columns:
Profiles the specified columns of the dataframe initiated with the linker.

This can be computationally expensive if the dataframe is large.

For the provided columns with column_expressions (or for all columns if left empty)
calculate:
- A distribution plot that shows the count of values at each percentile.
- A top n chart, that produces a chart showing the count of the top n values
within the column
- A bottom n chart, that produces a chart showing the count of the bottom
n values within the column

This should be used to explore the dataframe, determine if columns have
sufficient completeness for linking, analyse the cardinality of columns, and
identify the need for standardisation within a given column.

Args:

    column_expressions (list, optional): A list of strings containing the
        specified column names.
        If left empty this will default to all columns.
    top_n (int, optional): The number of top n values to plot.
    bottom_n (int, optional): The number of bottom n values to plot.

Returns:
    altair.Chart or dict: A visualization or JSON specification describing the
    profiling charts.

Note:
    - The `linker` object should be an instance of the initiated linker.
    - The provided `column_expressions` can be a list of column names to profile.
        If left empty, all columns will be profiled.
    - The `top_n` and `bottom_n` parameters determine the number of top and bottom
        values to display in the respective charts.


## Blocking Functions

block_on:
Generates blocking rules of equality conditions  based on the columns
or SQL expressions specified.

When multiple columns or SQL snippets are provided, the function generates a
compound blocking rule, connecting individual match conditions with
"AND" clauses.

Further information on equi-join conditions can be found
[here](https://moj-analytical-services.github.io/splink/topic_guides/blocking/performance.html)

Args:
    col_names_or_exprs: A list of input columns or SQL conditions
        you wish to create blocks on.
    salting_partitions (optional, int): Whether to add salting
        to the blocking rule. More information on salting can
        be found within the docs.
    arrays_to_explode (optional, List[str]): List of arrays to explode
        before applying the blocking rule.

Examples:
    ``` python
    from splink import block_on
    br_1 = block_on("first_name")
    br_2 = block_on("substr(surname,1,2)", "surname")
    ```

count_comparisons_from_blocking_rule:
Analyse a blocking rule to understand the number of comparisons it will generate.

Read more about the definition of pre and post filter conditions
[here]("https://moj-analytical-services.github.io/splink/topic_guides/blocking/performance.html?h=filter+cond#filter-conditions")

Args:
    table_or_tables (dataframe, str): Input data
    blocking_rule (Union[BlockingRuleCreator, str, Dict[str, Any]]): The blocking
        rule to analyse
    link_type (user_input_link_type_options): The link type - "link_only",
        "dedupe_only" or "link_and_dedupe"
    db_api (DatabaseAPISubClass): Database API
    unique_id_column_name (str, optional):  Defaults to "unique_id".
    source_dataset_column_name (Optional[str], optional):  Defaults to None.
    compute_post_filter_count (bool, optional): Whether to use a slower methodology
        to calculate how many comparisons will be generated post filter conditions.
        Defaults to True.
    max_rows_limit (int, optional): Calculation of post filter counts will only
        proceed if the fast method returns a value below this limit. Defaults
        to int(1e9).

Returns:
    dict[str, Union[int, str]]: A dictionary containing the results

cumulative_comparisons_to_be_scored_from_blocking_rules_chart:
TODO: Add docstring here

cumulative_comparisons_to_be_scored_from_blocking_rules_data:
TODO: Add docstring here

n_largest_blocks:
Find the values responsible for creating the largest blocks of records.

For example, when blocking on first name and surname, the 'John Smith' block
might be the largest block of records.  In cases where values are highly skewed
a few values may be resonsible for generating a large proportion of all comparisons.
This function helps you find the culprit values.

The analysis is performed pre filter conditions, read more about what this means
[here]("https://moj-analytical-services.github.io/splink/topic_guides/blocking/performance.html?h=filter+cond#filter-conditions")

Args:
    table_or_tables (dataframe, str): Input data
    blocking_rule (Union[BlockingRuleCreator, str, Dict[str, Any]]): The blocking
        rule to analyse
    link_type (user_input_link_type_options): The link type - "link_only",
        "dedupe_only" or "link_and_dedupe"
    db_api (DatabaseAPISubClass): Database API
    n_largest (int, optional): How many rows to return. Defaults to 5.

Returns:
    SplinkDataFrame: A dataframe containing the n_largest blocks


## SplinkDataFrame Methods

DuckDBDataFrame.as_duckdbpyrelation:
Return the dataframe as a duckdbpyrelation.  Only available when using the
DuckDB backend.

Args:
    limit (int, optional): If provided, return this number of rows (equivalent
        to a limit statement in SQL). Defaults to None, meaning return all rows

Returns:
    duckdb.DuckDBPyRelation: A DuckDBPyRelation object

DuckDBDataFrame.as_pandas_dataframe:
Return the dataframe as a pandas dataframe.

This can be computationally expensive if the dataframe is large.

Args:
    limit (int, optional): If provided, return this number of rows (equivalent
        to a limit statement in SQL). Defaults to None, meaning return all rows

Examples:
    ```py
    df_predict = linker.inference.predict()
    df_ten_edges = df_predict.as_pandas_dataframe(10)
    ```
Returns:
    pandas.DataFrame: pandas Dataframe

DuckDBDataFrame.as_record_dict:
Return the dataframe as a list of record dictionaries.

This can be computationally expensive if the dataframe is large.

Examples:
    ```py
    df_predict = linker.inference.predict()
    ten_edges = df_predict.as_record_dict(10)
    ```
Args:
    limit (int, optional): If provided, return this number of rows (equivalent
    to a limit statement in SQL). Defaults to None, meaning return all rows

Returns:
    list: a list of records, each of which is a dictionary

DuckDBDataFrame.as_spark_dataframe:
Return the dataframe as a spark dataframe.  Only available when using the
Spark backend.

Returns:
    spark.DataFrame: A Spark DataFrame

DuckDBDataFrame.check_file_exists:
No docstring available

DuckDBDataFrame.drop_table_from_database_and_remove_from_cache:
Drops the table from the underlying database, and removes it
from the (linker) cache.

By default this will fail if the table is not one created by Splink,
but this check can be overriden

Examples:
    ```py
    df_predict = linker.inference.predict()
    df_predict.drop_table_from_database_and_remove_from_cache()
    # predictions table no longer in the database / cache
    ```
Args:
    force_non_splink_table (bool, optional): If True, skip check if the
        table was created by Splink and always drop regardless. If False,
        only drop if table was created by Splink. Defaults to False.

DuckDBDataFrame.to_csv:
Save the dataframe in csv format.

Examples:
    ```py
    df_predict = linker.inference.predict()
    df_predict.to_csv("model_predictions.csv", overwrite=True)
    ```
Args:
    filepath (str): Filepath where csv will be saved.
    overwrite (bool, optional): If True, overwrites file if it already exists.
        Default is False.

DuckDBDataFrame.to_parquet:
Save the dataframe in parquet format.

Examples:
    ```py
    df_predict = linker.inference.predict()
    df_predict.to_parquet("model_predictions.parquet", overwrite=True)
    ```
Args:
    filepath (str): Filepath where csv will be saved.
    overwrite (bool, optional): If True, overwrites file if it already exists.
        Default is False.

DuckDBDataFrame.validate:
No docstring available


## EMTrainingSession Methods

EMTrainingSession.m_u_values_interactive_history_chart:
Display an interactive chart of the m and u values.

Returns:
    An interactive Altair chart.

EMTrainingSession.match_weights_interactive_history_chart:
Display an interactive chart of the match weights history.

Returns:
    An interactive Altair chart.

EMTrainingSession.probability_two_random_records_match_iteration_chart:
Display a chart showing the iteration history of the probability that two
random records match.

Returns:
    An interactive Altair chart.




---
tags:
  - settings
  - Dedupe
  - Link
  - Link and Dedupe
  - Expectation Maximisation
  - Comparisons
  - Blocking Rules
---

## Guide to Splink settings

This document enumerates all the settings and configuration options available when
developing your data linkage model.


<hr>

## `link_type`

The type of data linking task.  Required.

- When `dedupe_only`, `splink` find duplicates.  User expected to provide a single input dataset.

- When `link_and_dedupe`, `splink` finds links within and between input datasets.  User is expected to provide two or more input datasets.

- When `link_only`,  `splink` finds links between datasets, but does not attempt to deduplicate the datasets (it does not try and find links within each input dataset.) User is expected to provide two or more input datasets.

**Examples**: `['dedupe_only', 'link_only', 'link_and_dedupe']`

<hr>

## `probability_two_random_records_match`

The probability that two records chosen at random (with no blocking) are a match.  For example, if there are a million input records and each has on average one match, then this value should be 1/1,000,000.

If you estimate parameters using expectation maximisation (EM), this provides an initial value (prior) from which the EM algorithm will start iterating.  EM will then estimate the true value of this parameter.

**Default value**: `0.0001`

**Examples**: `[1e-05, 0.006]`

<hr>

## `em_convergence`

Convergence tolerance for the Expectation Maximisation algorithm

The algorithm will stop converging when the maximum of the change in model parameters between iterations is below this value

**Default value**: `0.0001`

**Examples**: `[0.0001, 1e-05, 1e-06]`

<hr>

## `max_iterations`

The maximum number of Expectation Maximisation iterations to run (even if convergence has not been reached)

**Default value**: `25`

**Examples**: `[20, 150]`

<hr>

## `unique_id_column_name`

Splink requires that the input dataset has a column that uniquely identifies each record.  `unique_id_column_name` is the name of the column in the input dataset representing this unique id

For linking tasks, ids must be unique within each dataset being linked, and do not need to be globally unique across input datasets

**Default value**: `unique_id`

**Examples**: `['unique_id', 'id', 'pk']`

<hr>

## `source_dataset_column_name`

The name of the column in the input dataset representing the source dataset

Where we are linking datasets, we can't guarantee that the unique id column is globally unique across datasets, so we combine it with a source_dataset column.  Usually, this is created by Splink for the user

**Default value**: `source_dataset`

**Examples**: `['source_dataset', 'dataset_name']`

<hr>

## `retain_matching_columns`

If set to true, each column used by the `comparisons` SQL expressions will be retained in output datasets

This is helpful so that the user can inspect matches, but once the comparison vector (gamma) columns are computed, this information is not actually needed by the algorithm.  The algorithm will run faster and use less resources if this is set to false.

**Default value**: `True`

**Examples**: `[False, True]`

<hr>

## `retain_intermediate_calculation_columns`

Retain intermediate calculation columns, such as the Bayes factors associated with each column in `comparisons`

The algorithm will run faster and use less resources if this is set to false.

**Default value**: `False`

**Examples**: `[False, True]`

<hr>

## comparisons

A list specifying how records should be compared for probabilistic matching.  Each element is a dictionary

???+ note "Settings keys nested within each member of `comparisons`"

    ### output_column_name

    The name used to refer to this comparison in the output dataset.  By default, Splink will set this to the name(s) of any input columns used in the comparison.  This key is most useful to give a clearer description to comparisons that use multiple input columns.  e.g. a location column that uses postcode and town may be named location

    For a comparison column that uses a single input column, e.g. first_name, this will be set first_name. For comparison columns that use multiple columns, if left blank, this will be set to the concatenation of columns used.

    **Examples**: `['first_name', 'surname']`

    <hr>

    ### comparison_description

    An optional label to describe this comparison, to be used in charting outputs.

    **Examples**: `['First name exact match', 'Surname with middle levenshtein level']`

    <hr>

    ### comparison_levels

    Comparison levels specify how input values should be compared.  Each level corresponds to an assessment of similarity, such as exact match, Jaro-Winkler match, one side of the match being null, etc

    Each comparison level represents a branch of a SQL case expression. They are specified in order of evaluation, each with a `sql_condition` that represents the branch of a case expression

    **Example**:
    ``` json
    [{
        "sql_condition": "first_name_l IS NULL OR first_name_r IS NULL",
        "label_for_charts": "null",
        "null_level": True
    },
    {
        "sql_condition": "first_name_l = first_name_r",
        "label_for_charts": "exact_match",
        "tf_adjustment_column": "first_name"
    },
    {
        "sql_condition": "ELSE",
        "label_for_charts": "else"
    }]
    ```

    <hr>

    ??? note "Settings keys nested within each member of `comparison_levels`"

        #### `sql_condition`

        A branch of a SQL case expression without WHEN and THEN e.g. `jaro_winkler_sim(surname_l, surname_r) > 0.88`

        **Examples**: `['forename_l = forename_r', 'jaro_winkler_sim(surname_l, surname_r) > 0.88']`

        <hr>

        #### label_for_charts

        A label for this comparison level, which will appear on charts as a reminder of what the level represents

        **Examples**: `['exact', 'postcode exact']`

        <hr>

        #### u_probability

        the u probability for this comparison level - i.e. the proportion of records that match this level amongst truly non-matching records

        **Examples**: `[0.9]`

        <hr>

        #### m_probability

        the m probability for this comparison level - i.e. the proportion of records that match this level amongst truly matching records

        **Examples**: `[0.1]`

        <hr>

        #### is_null_level

        If true, m and u values will not be estimated and instead the match weight will be zero for this column.  See treatment of nulls here on page 356, quote '. Under this MAR assumption, we can simply ignore missing data.': https://imai.fas.harvard.edu/research/files/linkage.pdf

        **Default value**: `False`

        <hr>

        #### tf_adjustment_column

        Make term frequency adjustments for this comparison level using this input column

        **Default value**: `None`

        **Examples**: `['first_name', 'postcode']`

        <hr>

        #### tf_adjustment_weight

        Make term frequency adjustments using this weight. A weight of 1.0 is a full adjustment.  A weight of 0.0 is no adjustment.  A weight of 0.5 is a half adjustment

        **Default value**: `1.0`

        **Examples**: `['first_name', 'postcode']`

        <hr>

        #### tf_minimum_u_value

        Where the term frequency adjustment implies a u value below this value, use this minimum value instead

        This prevents excessive weight being assigned to very unusual terms, such as a collision on a typo

        **Default value**: `0.0`

        **Examples**: `[0.001, 1e-09]`

        <hr>


## `blocking_rules_to_generate_predictions`

A list of one or more blocking rules to apply. A Cartesian join is applied if `blocking_rules_to_generate_predictions` is empty or not supplied.

Each rule is a SQL expression representing the blocking rule, which will be used to create a join.  The left table is aliased with `l` and the right table is aliased with `r`. For example, if you want to block on a `first_name` column, the blocking rule would be

`l.first_name = r.first_name`.

To block on first name and the first letter of surname, it would be

`l.first_name = r.first_name and substr(l.surname,1,1) = substr(r.surname,1,1)`.

Note that Splink deduplicates the comparisons generated by the blocking rules.

If empty or not supplied, all comparisons between the input dataset(s) will be generated and blocking will not be used. For large input datasets, this will generally be computationally intractable because it will generate comparisons equal to the number of rows squared.

**Default value**: `[]`

**Examples**: `[['l.first_name = r.first_name AND l.surname = r.surname', 'l.dob = r.dob']]`

<hr>

## `additional_columns_to_retain`

A list of columns not being used in the probabilistic matching comparisons that you want to include in your results.

By default, Splink drops columns which are not used by any comparisons.  This gives you the option to retain columns which are not used by the model.  A common example is if the user has labelled data (training data) and wishes to retain the labels in the outputs

**Default value**: `[]`

**Examples**: `[['cluster', 'col_2'], ['other_information']]`

<hr>

## `bayes_factor_column_prefix`

The prefix to use for the columns that will be created to store the Bayes factors

**Default value**: `bf_`

**Examples**: `['bf_', '__bf__']`

<hr>

## `term_frequency_adjustment_column_prefix`

The prefix to use for the columns that will be created to store the term frequency adjustments

**Default value**: `tf_`

**Examples**: `['tf_', '__tf__']`

<hr>

## `comparison_vector_value_column_prefix`

The prefix to use for the columns that will be created to store the comparison vector values

**Default value**: `gamma_`

**Examples**: `['gamma_', '__gamma__']`

<hr>

## `sql_dialect`

The SQL dialect in which `sql_conditions` are written.  Must be a valid SQLGlot dialect

**Default value**: `None`

**Examples**: `['spark', 'duckdb', 'presto', 'sqlite']`

<hr>


---
tags:
  - API
  - Datasets
  - Examples
---

# In-built datasets

Splink has some datasets available for use to help you get up and running, test ideas, or explore Splink features.
To use, simply import `splink_datasets`:
```py
from splink import splink_datasets

df = splink_datasets.fake_1000
```
which you can then use to set up a linker:
```py
from splink splink_datasets, Linker, DuckDBAPI, SettingsCreator

df = splink_datasets.fake_1000
linker = Linker(
    df,
    SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cl.exact_match("first_name"),
            cl.exact_match("surname"),
        ],
    ),
    db_api=DuckDBAPI()
)
```

??? tip "Troubleshooting"

    If you get a `SSLCertVerificationError` when trying to use the inbuilt datasets, this can be fixed with the `ssl` package by running:

    `ssl._create_default_https_context = ssl._create_unverified_context`.

## `splink_datasets`

Each attribute of `splink_datasets` is a dataset available for use, which exists as a pandas `DataFrame`.
These datasets are not packaged directly with Splink, but instead are downloaded only when they are requested.
Once requested they are cached for future use.
The cache can be cleared using `splink_dataset_utils` (see below),
which also contains information on available datasets, and which have already been cached.

### Available datasets

The datasets available are listed below:

{% include-markdown "./includes/generated_files/datasets_table.md" %}


## `splink_dataset_labels`

Some of the `splink_datasets` have corresponding clerical labels to help assess model performance. These are requested through the `splink_dataset_labels` module.

### Available datasets

The datasets available are listed below:

{% include-markdown "./includes/generated_files/dataset_labels_table.md" %}


## `splink_dataset_utils` API

In addition to `splink_datasets`, you can also import `splink_dataset_utils`,
which has a few functions to help managing `splink_datasets`.
This can be useful if you have limited internet connection and want to see what is already cached,
or if you need to clear cache items (e.g. if datasets were to be updated, or if space is an issue).

For example:
```py
from splink.datasets import splink_dataset_utils

splink_dataset_utils.show_downloaded_data()
splink_dataset_utils.clear_cache(['fake_1000'])
```

::: splink.internals.datasets.utils.SplinkDataUtils
    handler: python
    options:
      members:
        - list_downloaded_datasets
        - list_all_datasets
        - show_downloaded_data
        - clear_downloaded_data
      show_root_heading: false
      show_source: false
      heading_level: 3


Contents of ../docs/demos/examples/sqlite/deduplicate_50k_synthetic.ipynb:
## Linking a dataset of real historical persons

In this example, we deduplicate a more realistic dataset. The data is based on historical persons scraped from wikidata. Duplicate records are introduced with a variety of errors introduced.

Note, as explained in the [backends topic guide](https://moj-analytical-services.github.io/splink/topic_guides/backends.html#sqlite), SQLite does not natively support string fuzzy matching functions such as `damareau-levenshtein` and `jaro-winkler` (as used in this example). Instead, these have been imported as python User Defined Functions (UDFs). One drawback of python UDFs is that they are considerably slower than native-SQL comparisons. As such, if you are hitting issues with large run times, consider switching to DuckDB (or some other backend).


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/sqlite/deduplicate_50k_synthetic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink
# !pip install rapidfuzz

import pandas as pd

from splink import splink_datasets

pd.options.display.max_rows = 1000
# reduce size of dataset to make things run faster
df = splink_datasets.historical_50k.sample(5000)

from splink.backends.sqlite import SQLiteAPI
from splink.exploratory import profile_columns

db_api = SQLiteAPI()
profile_columns(
    df, db_api, column_expressions=["first_name", "postcode_fake", "substr(dob, 1,4)"]
)

from splink import block_on
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

blocking_rules =  [block_on("first_name", "surname"),
        block_on("surname", "dob"),
        block_on("first_name", "dob"),
        block_on("postcode_fake", "first_name")]

db_api = SQLiteAPI()

cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    blocking_rules=blocking_rules,
    db_api=db_api,
    link_type="dedupe_only"
)

import splink.comparison_library as cl
from splink import Linker

settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        block_on("first_name", "surname"),
        block_on("surname", "dob"),
        block_on("first_name", "dob"),
        block_on("postcode_fake", "first_name"),

    ],
    "comparisons": [
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.DamerauLevenshteinAtThresholds("dob", [1, 2]).configure(
            term_frequency_adjustments=True
        ),
        cl.DamerauLevenshteinAtThresholds("postcode_fake", [1, 2]),
        cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
        cl.ExactMatch(
            "occupation",
        ).configure(term_frequency_adjustments=True),
    ],
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
    "max_iterations": 10,
    "em_convergence": 0.01,
}

linker = Linker(df, settings, db_api=db_api)

linker.training.estimate_probability_two_random_records_match(
    [
        "l.first_name = r.first_name and l.surname = r.surname and l.dob = r.dob",
        "substr(l.first_name,1,2) = substr(r.first_name,1,2) and l.surname = r.surname and substr(l.postcode_fake,1,2) = substr(r.postcode_fake,1,2)",
        "l.dob = r.dob and l.postcode_fake = r.postcode_fake",
    ],
    recall=0.6,
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

training_blocking_rule = "l.first_name = r.first_name and l.surname = r.surname"
training_session_names = linker.training.estimate_parameters_using_expectation_maximisation(
    training_blocking_rule, estimate_without_term_frequencies=True
)

training_blocking_rule = "l.dob = r.dob"
training_session_dob = linker.training.estimate_parameters_using_expectation_maximisation(
    training_blocking_rule, estimate_without_term_frequencies=True
)

The final match weights can be viewed in the match weights chart:


linker.visualisations.match_weights_chart()

linker.evaluation.unlinkables_chart()

df_predict = linker.inference.predict()
df_e = df_predict.as_pandas_dataframe(limit=5)
df_e

You can also view rows in this dataset as a waterfall chart as follows:



records_to_plot = df_e.to_dict(orient="records")
linker.visualisations.waterfall_chart(records_to_plot, filter_nulls=False)

clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predict, threshold_match_probability=0.95
)

linker.visualisations.cluster_studio_dashboard(
    df_predict,
    clusters,
    "dashboards/50k_cluster.html",
    sampling_method="by_cluster_size",
    overwrite=True,
)

from IPython.display import IFrame

IFrame(src="./dashboards/50k_cluster.html", width="100%", height=1200)

linker.evaluation.accuracy_analysis_from_labels_column(
    "cluster", output_type="roc", match_weight_round_to_nearest=0.02
)

records = linker.evaluation.prediction_errors_from_labels_column(
    "cluster",
    threshold_match_probability=0.999,
    include_false_negatives=False,
    include_false_positives=True,
).as_record_dict()
linker.visualisations.waterfall_chart(records)

# Some of the false negatives will be because they weren't detected by the blocking rules
records = linker.evaluation.prediction_errors_from_labels_column(
    "cluster",
    threshold_match_probability=0.5,
    include_false_negatives=True,
    include_false_positives=False,
).as_record_dict(limit=50)

linker.visualisations.waterfall_chart(records)



Contents of ../docs/demos/examples/duckdb/accuracy_analysis_from_labels_column.ipynb:
## Evaluation when you have fully labelled data

In this example, our data contains a fully-populated ground-truth column called `cluster` that enables us to perform accuracy analysis of the final model


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/duckdb/accuracy_analysis_from_labels_column.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import splink_datasets

df = splink_datasets.fake_1000
df.head(2)

from splink import SettingsCreator, Linker, block_on, DuckDBAPI

import splink.comparison_library as cl

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
        block_on("dob"),
        block_on("email"),
    ],
    comparisons=[
        cl.ForenameSurnameComparison("first_name", "surname"),
        cl.DateOfBirthComparison(
            "dob",
            input_is_string=True,
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
    retain_intermediate_calculation_columns=True,
)

db_api = DuckDBAPI()
linker = Linker(df, settings, db_api=db_api)
deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    "l.email = r.email",
]

linker.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.7
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6, seed=5)

session_dob = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("dob"), estimate_without_term_frequencies=True
)
session_email = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("email"), estimate_without_term_frequencies=True
)
session_dob = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname"), estimate_without_term_frequencies=True
)

linker.evaluation.accuracy_analysis_from_labels_column(
    "cluster", output_type="table"
).as_pandas_dataframe(limit=5)

linker.evaluation.accuracy_analysis_from_labels_column("cluster", output_type="roc")

linker.evaluation.accuracy_analysis_from_labels_column(
    "cluster",
    output_type="threshold_selection",
    threshold_match_probability=0.5,
    add_metrics=["f1"],
)

# Plot some false positives
linker.evaluation.prediction_errors_from_labels_column(
    "cluster", include_false_negatives=True, include_false_positives=True
).as_pandas_dataframe(limit=5)

records = linker.evaluation.prediction_errors_from_labels_column(
    "cluster", include_false_negatives=True, include_false_positives=True
).as_record_dict(limit=5)

linker.visualisations.waterfall_chart(records)



Contents of ../docs/demos/examples/duckdb/bias_eval.ipynb:
!!! note

    This notebook is intended for usage alongside the second [Bias in Data Linking blog.](../../../blog/posts/2024-08-15-sd.md)

This notebook will guide you through a 5-step process for evaluating bias in a data linking pipeline. It offers an isolated approach to **bias detection**, exploring potential mitigation options, before drawing conclusions about the existence of bias in the pipeline.

This approach depends on users having pre-developed hypotheses about bias, based on a thorough understanding of the input data and model design. We’ll illustrate the approach with an example hypothesis, using Splink to show how each step can be practically applied.

Consider this hypothesis:

_I’m working with a 50,000-row dataset in which people appear multiple times over their lives. Because their information changes over time, records relating to the same individuals may not always be identical. I suspect that these changes could negatively impact linkage performance for those who update their details. Since women in this dataset are more likely to change their surname and postcode than men, this negative impact would disproportionately affect women, leading to a gender bias._

---

### **1. Generate synthetic data**

Since the hypothesis deals with changes in surnames and addresses, the synthetic data should include multiple records for the same person with these variations:

The synthetic data should also match the structure and format of the production data used to train the model.

import pandas as pd

synthetic_base_raw = [
    {"unique_id": 1, "person_id": 1, "first_name": "sarah", "surname": "brown", "dob": "1862-07-11", "birth_place": "london", "postcode_fake": "ba12 0ay", "gender": "female", "occupation": "politician"}
]

synthetic_comparison_raw = [
    {"unique_id": 2, "person_id": 1, "first_name": "sarah", "surname": "brown", "dob": "1862-07-11", "birth_place": "london", "postcode_fake": "ba12 0ay", "gender": "female", "occupation": "politician"},
    {"unique_id": 3, "person_id": 1, "first_name": "sarah", "surname": "doyle", "dob": "1862-07-11", "birth_place": "london", "postcode_fake": "ba12 0ay", "gender": "female", "occupation": "politician"},
    {"unique_id": 4, "person_id": 1, "first_name": "sarah", "surname": "brown", "dob": "1862-07-11", "birth_place": "london", "postcode_fake": "tf3 2ng", "gender": "female", "occupation": "politician"},
    {"unique_id": 5, "person_id": 1, "first_name": "sarah", "surname": "doyle", "dob": "1862-07-11", "birth_place": "london", "postcode_fake": "tf3 2ng", "gender": "female", "occupation": "politician"},
    {"unique_id": 6, "person_id": 2, "first_name": "jane", "surname": "brown", "dob": "1860-01-01", "birth_place": "london", "postcode_fake": "ba12 0ay", "gender": "female", "occupation": "artist"}
]

synthetic_base_df = pd.DataFrame(synthetic_base_raw)
synthetic_comparison_df = pd.DataFrame(synthetic_comparison_raw)

synthetic_base_df

synthetic_comparison_df

The first row is the base individual, with each subsequent row representing a different comparison related to the hypothesis. The first comparison is a self-link to serve as a baseline. Next, rows represent isolated surname changes and postcode changes, and finally a combination of both. 

Since this hypothesis relates to the linkage of records that _look different but are for the same person_, it’s helpful to include a record that _looks similar but is for a different person_ (like a sibling or partner). This will help account for the downstream effects of any mitigation option (Step 4).

---

### **2. Train and investigate model**

Train the model on the real production data (this Splink model is trained in 'model_create_h50k.ipynb' and saved as a json file).  
You can refer to the [Splink tutorial](https://moj-analytical-services.github.io/splink/demos/tutorials/00_Tutorial_Introduction.html) if you need any further explanation on how to prepare the data and define the model settings. 

from splink import DuckDBAPI
from splink import Linker, SettingsCreator
from splink import splink_datasets

db_api = DuckDBAPI()

production_df = splink_datasets.historical_50k

linker = Linker(production_df, settings='../../demo_settings/model_h50k.json', db_api=db_api)

It's useful to visualise the model parameters to learn the relative importance of different parts of your data for linking.

You can do this in Splink by using a match weights chart:

linker.visualisations.match_weights_chart()

When developing this model, it's essential to evaluate the parameter weights to ensure they align with the pipeline's goals. However, this approach requires considering the bias hypothesis.

At first glance, the model appears normal, but a few key points stand out regarding the hypothesis (focussed on surname and postcode changes):

- A non-match on the surname is more predictive than a non-match on the first name.
- A postcode match has the highest predictive weight.

The cumulative effect of these weights makes it difficult to fully understand their impact without generating comparisons.

---

### **3. Perform and evaluate linkage**

In a standard linkage with Splink, all records which meet the criteria of the blocking rules would be compared against each other. However, in bias detection we may not be interested in a lot of the comparisons this process will generate match probabilities for. It can be easier to manually generate the comparisons that are relevant to the hypothesis. 

def compare_records(base_records, comparison_records, linker):
    results = []
    for record_1 in base_records:
        results.extend(
            linker.inference.compare_two_records(record_1, record_2).as_pandas_dataframe()
            for record_2 in comparison_records
        )
    all_comparisons_df = pd.concat(results, ignore_index=True)
    return all_comparisons_df

comparisons = compare_records(synthetic_base_raw, synthetic_comparison_raw, linker)

Look at the resulting match probabilities in terms of the threshold of your pipeline.  
This pipeline has a high match threshold of 0.999, medium of 0.99, and low of 0.95.

def highlight_cells(val):
    if val >= 0.999:
        color = '#c4f5bf'  # High threshold, green
    elif val >= 0.99:
        color = '#faf9c0'  # Medium threshold, yellow
    elif val >= 0.95:
        color = '#f5e1bf'  # Low threshold, orange
    else:
        color = '#f5c8bf'  # Below threshold, red
    return f'background-color: {color}'

columns_of_interest = ['match_weight', 'match_probability', 'unique_id_l', 'unique_id_r', 'first_name_l', 'first_name_r', 'surname_l','surname_r', 'dob_l', 'dob_r','postcode_fake_l', 'postcode_fake_r','birth_place_l', 'birth_place_r', 'occupation_l', 'occupation_r']
comparisons[columns_of_interest].style.map(highlight_cells, subset=['match_probability'])

The first three comparisons show high match probabilities. A surname change or postcode change alone allow for a linkage at a high match threshold.  
However, when combined they lower the match probability to 0.9781 - which could only be linked at a low threshold in the example pipeline. 

Additionally, it is useful to note that the sibling scenario has a a higher match probability than the surname and postcode change scenario, and it can be matched at a medium threshold. 

It’s helpful to break these results down further to understand how the individual model parameters are combining to result in these final probabilities. This can be done in Splink using the waterfall chart:

records_to_plot = comparisons.to_dict(orient="records")
linker.visualisations.waterfall_chart(records_to_plot)

Taking a closer look at each comparison helps reveal which features have the most impact in terms of the hypothesis. Some key takeaways are:

- Matching on postcode and surname are the strongest indicators of a link. Therefore, when both surname and postcode change, you lose that strong link advantage. It's not that a non-match on surname or postcode is overly detrimental to the records being linked, but without the boost from a surname or postcode match, they only meet the low linkage threshold. 

- In the sibling scenario, you can see the impact of one or two strong predictors in a model can have. Even with three non-matches (first name, DOB, and occupation), matching on both postcode and surname links the records at a medium threshold. 

- As first name matches are less predictive than surname matches the sibling scenario has a more predictive name match than the same person changing their information. 

- Due to the way individual factors are weighted, the combination of non-matches in this scenario means that people who change both their postcode and surname won't be linked at higher thresholds. Given this affects women more than men in our data, it leads to a **gender bias** in the pipeline.

---

### **4. Identify mitigations**

Now that bias has been detected, it's time to explore potential mitigation strategies:

<u>Are there viable technical solutions?</u>

In our example, a change in surname and postcode allows records to be linked at a low threshold. We should consider what can be done in the pipeline to ensure these comparisons are linked.

Any technical solution should align with the hypothesis, as opposed to just pushing comparisons randomly over/under a threshold. Different elements of the pipeline can be altered to address the bias. 

- Altering the _input data_ - there's likely no potential solution here, as the bias stems from legitimate data qualities, not errors. 
- Updating the _model design_ - a TF adjustment might alter results, but this would be random so wouldn't address the hypothesis. However, **adjusting comparison levels** might help. 
- Adjusting the _output data_ - since the records link at a low threshold, **lowering the threshold** across the model would result in a link.

<u>Could they negatively impact overall performance?</u>

**Lowering the threshold**

This could link records with surname and postcode changes but might also increase false positives, impacting overall performance. This trade-off needs careful consideration. In synthetic data, for instance, siblings would link at low thresholds. This is just one scenario, and there are most likely others. Given these potential issues, we’re not pursuing a lower threshold.

**Adjusting comparison levels**

We'd need to decide that a factor’s predictive power doesn't fit our hypothesis - but also make sure this applies across the whole model. There are lots of options, like making surname or postcode less predictive, or boosting factors like occupation, first name, DOB, or birthplace. We'll focus on one example to demonstrate the necessary considerations. 

Say we decide to manually make first name as predictive as surname, and we find that this change won’t impact overall performance. Therefore, we can take this technical solution to the next consideration.

<u>Could they introduce further bias?</u>

It's important to think about other groups that this decision could impact. 

In this dataset, there's records relating to a community of Vietnamese people. They share similar postcodes and often list the same birthplace (because the dataset defaults to the country of birth if they were born outside the UK). Many families in this community have the same surname, as do many unrelated individuals (due to less variation in Vietnamese surnames). Additionally, because Vietnamese naming conventions differ from Western ones, surnames are often incorrectly recorded as first names in this dataset. 

Therefore, increasing the predictive power of first name matches will likely lead to more false positives for this community, introducing bias against the Vietnamese population in the dataset. The decision is made to not attempt bias mitigation.

---

### **5. Make a statement about bias** 

When a bias is detected but not mitigated, it’s crucial to explore it further to gain more insight. This helps refine the understanding of the bias and lays the groundwork for impact assessment.

In our example, we can dig deeper into how the link fails with postcode and surname changes. To do this, we’ll create some more synthetic data with partial changes in surname and postcode:

synthetic_comparison_partial_raw = [
    {"unique_id": 2, "person_id": 1, "first_name": "sarah", "surname": "brown-doyle", "dob": "1862-07-11", "birth_place": "london", "postcode_fake": "tf3 2ng", "gender": "female", "occupation": "politician"},
    {"unique_id": 3, "person_id": 1, "first_name": "sarah", "surname": "doyle", "dob": "1862-07-11", "birth_place": "london", "postcode_fake": "ba13 2ng", "gender": "female", "occupation": "politician"},
    {"unique_id": 4, "person_id": 1, "first_name": "sarah", "surname": "brown-doyle", "dob": "1862-07-11", "birth_place": "london", "postcode_fake": "ba13 2ng", "gender": "female", "occupation": "politician"}
]

synthetic_comparison_partial_df = pd.DataFrame(synthetic_comparison_partial_raw)

In this scenario, we generate new records to be compared to the base. These represent a partial surname change (double-barrel) and a full postcode change, a partial postcode change (same area, different district) with a full surname change, and a combination of partial postcode and partial surname changes.

We can then use the same model to link these records and examine the resulting match probabilities. 

comparisons_partial = compare_records(synthetic_base_raw, synthetic_comparison_partial_raw, linker)

comparisons_partial[columns_of_interest].style.map(highlight_cells, subset=['match_probability'])

In all three cases, the records will remain linked at a high threshold. This insight can help make the final bias statement:

_Given how the model weighs the different parameters to produce match probabilities, a full change in both surname and postcode will lead to a non-link. As women are more likely than men to make these changes in the input data, this will introduce a gender bias in the pipeline._

Now that this bias has been detected, it's important to note that we can't yet determine its exact impact on the linked data. This is because the final results are influenced by many other factors in the pipeline, which were intentionally excluded from this process to isolate the bias. 

Further investigation is needed to understand impact — such as understanding how many records in the input data undergo these changes, or how many records with a full surname and postcode change end up linked or not linked in the final data. 



Contents of ../docs/demos/examples/duckdb/quick_and_dirty_persons.ipynb:
## Historical people: Quick and dirty

This example shows how to get some initial record linkage results as quickly as possible.

There are many ways to improve the accuracy of this model. But this may be a good place to start if you just want to give Splink a try and see what it's capable of.


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/duckdb/quick_and_dirty_persons.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink.datasets import splink_datasets

df = splink_datasets.historical_50k
df.head(5)

from splink import block_on, SettingsCreator
import splink.comparison_library as cl


settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("full_name"),
        block_on("substr(full_name,1,6)", "dob", "birth_place"),
        block_on("dob", "birth_place"),
        block_on("postcode_fake"),
    ],
    comparisons=[
        cl.ForenameSurnameComparison(
            "first_name",
            "surname",
            forename_surname_concat_col_name="first_and_surname",
        ),
        cl.DateOfBirthComparison(
            "dob",
            input_is_string=True,
        ),
        cl.LevenshteinAtThresholds("postcode_fake", 2),
        cl.JaroWinklerAtThresholds("birth_place", 0.9).configure(
            term_frequency_adjustments=True
        ),
        cl.ExactMatch("occupation").configure(term_frequency_adjustments=True),
    ],
)

from splink import Linker, DuckDBAPI


linker = Linker(df, settings, db_api=DuckDBAPI(), set_up_basic_logging=False)
deterministic_rules = [
    "l.full_name = r.full_name",
    "l.postcode_fake = r.postcode_fake and l.dob = r.dob",
]

linker.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.6
)

linker.training.estimate_u_using_random_sampling(max_pairs=2e6)

results = linker.inference.predict(threshold_match_probability=0.9)

results.as_pandas_dataframe(limit=5)



Contents of ../docs/demos/examples/duckdb/pairwise_labels.ipynb:
<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/duckdb/pairwise_labels.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Estimating m from a sample of pairwise labels



In this example, we estimate the m probabilities of the model from a table containing pairwise record comparisons which we know are 'true' matches. For example, these may be the result of work by a clerical team who have manually labelled a sample of matches.

The table must be in the following format:

| source_dataset_l | unique_id_l | source_dataset_r | unique_id_r |
| ---------------- | ----------- | ---------------- | ----------- |
| df_1             | 1           | df_2             | 2           |
| df_1             | 1           | df_2             | 3           |

It is assumed that every record in the table represents a certain match.

Note that the column names above are the defaults. They should correspond to the values you've set for [`unique_id_column_name`](https://moj-analytical-services.github.io/splink/settings_dict_guide.html#unique_id_column_name) and [`source_dataset_column_name`](https://moj-analytical-services.github.io/splink/settings_dict_guide.html#source_dataset_column_name), if you've chosen custom values.


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink.datasets import splink_dataset_labels

pairwise_labels = splink_dataset_labels.fake_1000_labels

# Choose labels indicating a match
pairwise_labels = pairwise_labels[pairwise_labels["clerical_match_score"] == 1]
pairwise_labels

We now proceed to estimate the Fellegi Sunter model:


from splink import splink_datasets

df = splink_datasets.fake_1000
df.head(2)

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
    ],
    comparisons=[
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.DateOfBirthComparison(
            "dob",
            input_is_string=True,
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=DuckDBAPI(), set_up_basic_logging=False)
deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    "l.email = r.email",
]

linker.training.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

# Register the pairwise labels table with the database, and then use it to estimate the m values
labels_df = linker.table_management.register_labels_table(pairwise_labels, overwrite=True)
linker.training.estimate_m_from_pairwise_labels(labels_df)


# If the labels table already existing in the dataset you could run
# linker.training.estimate_m_from_pairwise_labels("labels_tablename_here")

training_blocking_rule = block_on("first_name")
linker.training.estimate_parameters_using_expectation_maximisation(training_blocking_rule)

linker.visualisations.parameter_estimate_comparisons_chart()

linker.visualisations.match_weights_chart()



Contents of ../docs/demos/examples/duckdb/link_only.ipynb:
## Linking without deduplication

A simple record linkage model using the `link_only` [link type](https://moj-analytical-services.github.io/splink/settings_dict_guide.html#link_type).

With `link_only`, only between-dataset record comparisons are generated. No within-dataset record comparisons are created, meaning that the model does not attempt to find within-dataset duplicates.


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/duckdb/link_only.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import splink_datasets

df = splink_datasets.fake_1000

# Split a simple dataset into two, separate datasets which can be linked together.
df_l = df.sample(frac=0.5)
df_r = df.drop(df_l.index)

df_l.head(2)

import splink.comparison_library as cl

from splink import DuckDBAPI, Linker, SettingsCreator, block_on

settings = SettingsCreator(
    link_type="link_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
    ],
    comparisons=[
        cl.NameComparison(
            "first_name",
        ),
        cl.NameComparison("surname"),
        cl.DateOfBirthComparison(
            "dob",
            input_is_string=True,
            invalid_dates_as_null=True,
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
)

linker = Linker(
    [df_l, df_r],
    settings,
    db_api=DuckDBAPI(),
    input_table_aliases=["df_left", "df_right"],
)

from splink.exploratory import completeness_chart

completeness_chart(
    [df_l, df_r],
    cols=["first_name", "surname", "dob", "city", "email"],
    db_api=DuckDBAPI(),
    table_names_for_chart=["df_left", "df_right"],
)


deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    block_on("email"),
]


linker.training.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6, seed=1)

session_dob = linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))
session_email = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("email")
)
session_first_name = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name")
)

results = linker.inference.predict(threshold_match_probability=0.9)

results.as_pandas_dataframe(limit=5)



Contents of ../docs/demos/examples/duckdb/transactions.ipynb:
## Linking banking transactions

This example shows how to perform a one-to-one link on banking transactions.

The data is fake data, and was generated has the following features:

- Money shows up in the destination account with some time delay
- The amount sent and the amount received are not always the same - there are hidden fees and foreign exchange effects
- The memo is sometimes truncated and content is sometimes missing

Since each origin payment should end up in the destination account, the `probability_two_random_records_match` of the model is known.


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/duckdb/transactions.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets

df_origin = splink_datasets.transactions_origin
df_destination = splink_datasets.transactions_destination

display(df_origin.head(2))
display(df_destination.head(2))

In the following chart, we can see this is a challenging dataset to link:

- There are only 151 distinct transaction dates, with strong skew
- Some 'memos' are used multiple times (up to 48 times)
- There is strong skew in the 'amount' column, with 1,400 transactions of around 60.00


from splink.exploratory import profile_columns

db_api = DuckDBAPI()
profile_columns(
    [df_origin, df_destination],
    db_api=db_api,
    column_expressions=[
        "memo",
        "transaction_date",
        "amount",
    ],
)

from splink import DuckDBAPI, block_on
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

# Design blocking rules that allow for differences in transaction date and amounts
blocking_rule_date_1 = """
    strftime(l.transaction_date, '%Y%m') = strftime(r.transaction_date, '%Y%m')
    and substr(l.memo, 1,3) = substr(r.memo,1,3)
    and l.amount/r.amount > 0.7   and l.amount/r.amount < 1.3
"""

# Offset by half a month to ensure we capture case when the dates are e.g. 31st Jan and 1st Feb
blocking_rule_date_2 = """
    strftime(l.transaction_date+15, '%Y%m') = strftime(r.transaction_date, '%Y%m')
    and substr(l.memo, 1,3) = substr(r.memo,1,3)
    and l.amount/r.amount > 0.7   and l.amount/r.amount < 1.3
"""

blocking_rule_memo = block_on("substr(memo,1,9)")

blocking_rule_amount_1 = """
round(l.amount/2,0)*2 = round(r.amount/2,0)*2 and yearweek(r.transaction_date) = yearweek(l.transaction_date)
"""

blocking_rule_amount_2 = """
round(l.amount/2,0)*2 = round((r.amount+1)/2,0)*2 and yearweek(r.transaction_date) = yearweek(l.transaction_date + 4)
"""

blocking_rule_cheat = block_on("unique_id")


brs = [
    blocking_rule_date_1,
    blocking_rule_date_2,
    blocking_rule_memo,
    blocking_rule_amount_1,
    blocking_rule_amount_2,
    blocking_rule_cheat,
]


db_api = DuckDBAPI()

cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=[df_origin, df_destination],
    blocking_rules=brs,
    db_api=db_api,
    link_type="link_only"
)

# Full settings for linking model
import splink.comparison_level_library as cll
import splink.comparison_library as cl

comparison_amount = {
    "output_column_name": "amount",
    "comparison_levels": [
        cll.NullLevel("amount"),
        cll.ExactMatchLevel("amount"),
        cll.PercentageDifferenceLevel("amount", 0.01),
        cll.PercentageDifferenceLevel("amount", 0.03),
        cll.PercentageDifferenceLevel("amount", 0.1),
        cll.PercentageDifferenceLevel("amount", 0.3),
        cll.ElseLevel(),
    ],
    "comparison_description": "Amount percentage difference",
}

# The date distance is one sided becaause transactions should only arrive after they've left
# As a result, the comparison_template_library date difference functions are not appropriate
within_n_days_template = "transaction_date_r - transaction_date_l <= {n} and transaction_date_r >= transaction_date_l"

comparison_date = {
    "output_column_name": "transaction_date",
    "comparison_levels": [
        cll.NullLevel("transaction_date"),
        {
            "sql_condition": within_n_days_template.format(n=1),
            "label_for_charts": "1 day",
        },
        {
            "sql_condition": within_n_days_template.format(n=4),
            "label_for_charts": "<=4 days",
        },
        {
            "sql_condition": within_n_days_template.format(n=10),
            "label_for_charts": "<=10 days",
        },
        {
            "sql_condition": within_n_days_template.format(n=30),
            "label_for_charts": "<=30 days",
        },
        cll.ElseLevel(),
    ],
    "comparison_description": "Transaction date days apart",
}


settings = SettingsCreator(
    link_type="link_only",
    probability_two_random_records_match=1 / len(df_origin),
    blocking_rules_to_generate_predictions=[
        blocking_rule_date_1,
        blocking_rule_date_2,
        blocking_rule_memo,
        blocking_rule_amount_1,
        blocking_rule_amount_2,
        blocking_rule_cheat,
    ],
    comparisons=[
        comparison_amount,
        cl.LevenshteinAtThresholds("memo", [2, 6, 10]),
        comparison_date,
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(
    [df_origin, df_destination],
    settings,
    input_table_aliases=["__ori", "_dest"],
    db_api=db_api,
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

linker.training.estimate_parameters_using_expectation_maximisation(block_on("memo"))

session = linker.training.estimate_parameters_using_expectation_maximisation(block_on("amount"))

linker.visualisations.match_weights_chart()

df_predict = linker.inference.predict(threshold_match_probability=0.001)

linker.visualisations.comparison_viewer_dashboard(
    df_predict, "dashboards/comparison_viewer_transactions.html", overwrite=True
)
from IPython.display import IFrame

IFrame(
    src="./dashboards/comparison_viewer_transactions.html", width="100%", height=1200
)

pred_errors = linker.evaluation.prediction_errors_from_labels_column(
    "ground_truth", include_false_positives=True, include_false_negatives=False
)
linker.visualisations.waterfall_chart(pred_errors.as_record_dict(limit=5))

pred_errors = linker.evaluation.prediction_errors_from_labels_column(
    "ground_truth", include_false_positives=False, include_false_negatives=True
)
linker.visualisations.waterfall_chart(pred_errors.as_record_dict(limit=5))



Contents of ../docs/demos/examples/duckdb/deterministic_dedupe.ipynb:
## Linking a dataset of real historical persons with Deterrministic Rules

While Splink is primarily a tool for probabilistic records linkage, it includes functionality to perform deterministic (i.e. rules based) linkage.

Significant work has gone into optimising the performance of rules based matching, so Splink is likely to be significantly faster than writing the basic SQL by hand.

In this example, we deduplicate a 50k row dataset based on historical persons scraped from wikidata. Duplicate records are introduced with a variety of errors introduced. The probabilistic dedupe of the same dataset can be found at `Deduplicate 50k rows historical persons`.


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/duckdb/deterministic_dedupe.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

import pandas as pd

from splink import splink_datasets

pd.options.display.max_rows = 1000
df = splink_datasets.historical_50k
df.head()

When defining the settings object, specity your deterministic rules in the `blocking_rules_to_generate_predictions` key.

For a deterministic linkage, the linkage methodology is based solely on these rules, so there is no need to define `comparisons` nor any other parameters required for model training in a probabilistic model.


Prior to running the linkage, it's usually a good idea to check how many record comparisons will be generated by your deterministic rules:


from splink import DuckDBAPI, block_on
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

db_api = DuckDBAPI()
cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    blocking_rules=[
        block_on("first_name", "surname", "dob"),
        block_on("surname", "dob", "postcode_fake"),
        block_on("first_name", "dob", "occupation"),
    ],
    db_api=db_api,
    link_type="dedupe_only",
)

from splink import Linker, SettingsCreator

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname", "dob"),
        block_on("surname", "dob", "postcode_fake"),
        block_on("first_name", "dob", "occupation"),
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=db_api)


The results of the linkage can be viewed with the `deterministic_link` function.


df_predict = linker.inference.deterministic_link()
df_predict.as_pandas_dataframe().head()

Which can be used to generate clusters.

Note, for deterministic linkage, each comparison has been assigned a match probability of 1, so to generate clusters, set `threshold_match_probability=1` in the `cluster_pairwise_predictions_at_threshold` function.


clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predict
)

clusters.as_pandas_dataframe(limit=5)

These results can then be passed into the `Cluster Studio Dashboard`.


linker.visualisations.cluster_studio_dashboard(
    df_predict,
    clusters,
    "dashboards/50k_deterministic_cluster.html",
    sampling_method="by_cluster_size",
    overwrite=True,
)

from IPython.display import IFrame

IFrame(src="./dashboards/50k_deterministic_cluster.html", width="100%", height=1200)



Contents of ../docs/demos/examples/duckdb/deduplicate_50k_synthetic.ipynb:
## Linking a dataset of real historical persons

In this example, we deduplicate a more realistic dataset. The data is based on historical persons scraped from wikidata. Duplicate records are introduced with a variety of errors introduced.


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/duckdb/deduplicate_50k_synthetic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import splink_datasets

df = splink_datasets.historical_50k

df.head()

from splink import DuckDBAPI
from splink.exploratory import profile_columns

db_api = DuckDBAPI()
profile_columns(df, db_api, column_expressions=["first_name", "substr(surname,1,2)"])

from splink import DuckDBAPI, block_on
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

blocking_rules = [
    block_on("substr(first_name,1,3)", "substr(surname,1,4)"),
    block_on("surname", "dob"),
    block_on("first_name", "dob"),
    block_on("postcode_fake", "first_name"),
    block_on("postcode_fake", "surname"),
    block_on("dob", "birth_place"),
    block_on("substr(postcode_fake,1,3)", "dob"),
    block_on("substr(postcode_fake,1,3)", "first_name"),
    block_on("substr(postcode_fake,1,3)", "surname"),
    block_on("substr(first_name,1,2)", "substr(surname,1,2)", "substr(dob,1,4)"),
]

db_api = DuckDBAPI()

cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    blocking_rules=blocking_rules,
    db_api=db_api,
    link_type="dedupe_only",
)

import splink.comparison_library as cl

from splink import Linker, SettingsCreator

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=blocking_rules,
    comparisons=[
        cl.ForenameSurnameComparison(
            "first_name",
            "surname",
            forename_surname_concat_col_name="first_name_surname_concat",
        ),
        cl.DateOfBirthComparison(
            "dob", input_is_string=True
        ),
        cl.PostcodeComparison("postcode_fake"),
        cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
        cl.ExactMatch("occupation").configure(term_frequency_adjustments=True),
    ],
    retain_intermediate_calculation_columns=True,
)
# Needed to apply term frequencies to first+surname comparison
df["first_name_surname_concat"] = df["first_name"] + " " + df["surname"]
linker = Linker(df, settings, db_api=db_api)

linker.training.estimate_probability_two_random_records_match(
    [
        "l.first_name = r.first_name and l.surname = r.surname and l.dob = r.dob",
        "substr(l.first_name,1,2) = substr(r.first_name,1,2) and l.surname = r.surname and substr(l.postcode_fake,1,2) = substr(r.postcode_fake,1,2)",
        "l.dob = r.dob and l.postcode_fake = r.postcode_fake",
    ],
    recall=0.6,
)

linker.training.estimate_u_using_random_sampling(max_pairs=5e6)

training_blocking_rule = block_on("first_name", "surname")
training_session_names = (
    linker.training.estimate_parameters_using_expectation_maximisation(
        training_blocking_rule, estimate_without_term_frequencies=True
    )
)

training_blocking_rule = block_on("dob")
training_session_dob = (
    linker.training.estimate_parameters_using_expectation_maximisation(
        training_blocking_rule, estimate_without_term_frequencies=True
    )
)

The final match weights can be viewed in the match weights chart:


linker.visualisations.match_weights_chart()

linker.evaluation.unlinkables_chart()

df_predict = linker.inference.predict()
df_e = df_predict.as_pandas_dataframe(limit=5)
df_e

You can also view rows in this dataset as a waterfall chart as follows:


records_to_plot = df_e.to_dict(orient="records")
linker.visualisations.waterfall_chart(records_to_plot, filter_nulls=False)

clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predict, threshold_match_probability=0.95
)

from IPython.display import IFrame

linker.visualisations.cluster_studio_dashboard(
    df_predict,
    clusters,
    "dashboards/50k_cluster.html",
    sampling_method="by_cluster_size",
    overwrite=True,
)


IFrame(src="./dashboards/50k_cluster.html", width="100%", height=1200)

linker.evaluation.accuracy_analysis_from_labels_column(
    "cluster", output_type="accuracy", match_weight_round_to_nearest=0.02
)

records = linker.evaluation.prediction_errors_from_labels_column(
    "cluster",
    threshold_match_probability=0.999,
    include_false_negatives=False,
    include_false_positives=True,
).as_record_dict()
linker.visualisations.waterfall_chart(records)

# Some of the false negatives will be because they weren't detected by the blocking rules
records = linker.evaluation.prediction_errors_from_labels_column(
    "cluster",
    threshold_match_probability=0.5,
    include_false_negatives=True,
    include_false_positives=False,
).as_record_dict(limit=50)

linker.visualisations.waterfall_chart(records)



Contents of ../docs/demos/examples/duckdb/febrl4.ipynb:
## Linking the febrl4 datasets

See A.2 [here](https://arxiv.org/pdf/2008.04443.pdf) and [here](https://recordlinkage.readthedocs.io/en/latest/ref-datasets.html) for the source of this data.

It consists of two datasets, A and B, of 5000 records each, with each record in dataset A having a corresponding record in dataset B. The aim will be to capture as many of those 5000 true links as possible, with minimal false linkages.

It is worth noting that we should not necessarily expect to capture _all_ links. There are some links that although we know they _do_ correspond to the same person, the data is so mismatched between them that we would not reasonably expect a model to link them, and indeed should a model do so may indicate that we have overengineered things using our knowledge of true links, which will not be a helpful reference in situations where we attempt to link unlabelled data, as will usually be the case.


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/duckdb/febrl4.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

### Exploring data and defining model


Firstly let's read in the data and have a little look at it


from splink import splink_datasets

df_a = splink_datasets.febrl4a
df_b = splink_datasets.febrl4b


def prepare_data(data):
    data = data.rename(columns=lambda x: x.strip())
    data["cluster"] = data["rec_id"].apply(lambda x: "-".join(x.split("-")[:2]))
    data["date_of_birth"] = data["date_of_birth"].astype(str).str.strip()
    data["soc_sec_id"] = data["soc_sec_id"].astype(str).str.strip()
    data["postcode"] = data["postcode"].astype(str).str.strip()
    return data


dfs = [prepare_data(dataset) for dataset in [df_a, df_b]]

display(dfs[0].head(2))
display(dfs[1].head(2))

Next, to better understand which variables will prove useful in linking, we have a look at how populated each column is, as well as the distribution of unique values within each


from splink import DuckDBAPI, Linker, SettingsCreator

basic_settings = SettingsCreator(
    unique_id_column_name="rec_id",
    link_type="link_only",
    # NB as we are linking one-one, we know the probability that a random pair will be a match
    # hence we could set:
    # "probability_two_random_records_match": 1/5000,
    # however we will not specify this here, as we will use this as a check that
    # our estimation procedure returns something sensible
)

linker = Linker(dfs, basic_settings, db_api=DuckDBAPI())

It's usually a good idea to perform exploratory analysis on your data so you understand what's in each column and how often it's missing


from splink.exploratory import completeness_chart

completeness_chart(dfs, db_api=DuckDBAPI())

from splink.exploratory import profile_columns

profile_columns(dfs, db_api=DuckDBAPI(), column_expressions=["given_name", "surname"])

Next let's come up with some candidate blocking rules, which define which record comparisons are generated, and have a look at how many comparisons each will generate.

For blocking rules that we use in prediction, our aim is to have the union of all rules cover all true matches, whilst avoiding generating so many comparisons that it becomes computationally intractable - i.e. each true match should have at least _one_ of the following conditions holding.


from splink import DuckDBAPI, block_on
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

blocking_rules = [
    block_on("given_name", "surname"),
    # A blocking rule can also be an aribtrary SQL expression
    "l.given_name = r.surname and l.surname = r.given_name",
    block_on("date_of_birth"),
    block_on("soc_sec_id"),
    block_on("state", "address_1"),
    block_on("street_number", "address_1"),
    block_on("postcode"),
]


db_api = DuckDBAPI()
cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=dfs,
    blocking_rules=blocking_rules,
    db_api=db_api,
    link_type="link_only",
    unique_id_column_name="rec_id",
    source_dataset_column_name="source_dataset",
)

The broadest rule, having a matching postcode, unsurpisingly gives the largest number of comparisons.
For this small dataset we still have a very manageable number, but if it was larger we might have needed to include a further `AND` condition with it to break the number of comparisons further.


Now we get the full settings by including the blocking rules, as well as deciding the actual comparisons we will be including in our model.

We will define two models, each with a separate linker with different settings, so that we can compare performance. One will be a very basic model, whilst the other will include a lot more detail.


import splink.comparison_level_library as cll
import splink.comparison_library as cl


# the simple model only considers a few columns, and only two comparison levels for each
simple_model_settings = SettingsCreator(
    unique_id_column_name="rec_id",
    link_type="link_only",
    blocking_rules_to_generate_predictions=blocking_rules,
    comparisons=[
        cl.ExactMatch("given_name").configure(term_frequency_adjustments=True),
        cl.ExactMatch("surname").configure(term_frequency_adjustments=True),
        cl.ExactMatch("street_number").configure(term_frequency_adjustments=True),
    ],
    retain_intermediate_calculation_columns=True,
)

# the detailed model considers more columns, using the information we saw in the exploratory phase
# we also include further comparison levels to account for typos and other differences
detailed_model_settings = SettingsCreator(
    unique_id_column_name="rec_id",
    link_type="link_only",
    blocking_rules_to_generate_predictions=blocking_rules,
    comparisons=[
        cl.NameComparison("given_name").configure(term_frequency_adjustments=True),
        cl.NameComparison("surname").configure(term_frequency_adjustments=True),
        cl.DateOfBirthComparison(
            "date_of_birth",
            input_is_string=True,
            datetime_format="%Y%m%d",
            invalid_dates_as_null=True,
        ),
        cl.DamerauLevenshteinAtThresholds("soc_sec_id", [1, 2]),
        cl.ExactMatch("street_number").configure(term_frequency_adjustments=True),
        cl.DamerauLevenshteinAtThresholds("postcode", [1, 2]).configure(
            term_frequency_adjustments=True
        ),
        # we don't consider further location columns as they will be strongly correlated with postcode
    ],
    retain_intermediate_calculation_columns=True,
)


linker_simple = Linker(dfs, simple_model_settings, db_api=DuckDBAPI())
linker_detailed = Linker(dfs, detailed_model_settings, db_api=DuckDBAPI())

### Estimating model parameters


We need to furnish our models with parameter estimates so that we can generate results. We will focus on the detailed model, generating the values for the simple model at the end


We can instead estimate the probability two random records match, and compare with the known value of 1/5000 = 0.0002, to see how well our estimation procedure works.

To do this we come up with some deterministic rules - the aim here is that we generate very few false positives (i.e. we expect that the majority of records with at least one of these conditions holding are true matches), whilst also capturing the majority of matches - our guess here is that these two rules should capture 80% of all matches.


deterministic_rules = [
    block_on("soc_sec_id"),
    block_on("given_name", "surname", "date_of_birth"),
]

linker_detailed.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.8
)

Even playing around with changing these deterministic rules, or the nominal recall leaves us with an answer which is pretty close to our known value


Next we estimate `u` and `m` values for each comparison, so that we can move to generating predictions


# We generally recommend setting max pairs higher (e.g. 1e7 or more)
# But this will run faster for the purpose of this demo
linker_detailed.training.estimate_u_using_random_sampling(max_pairs=1e6)

When training the `m` values using expectation maximisation, we need somre more blocking rules to reduce the total number of comparisons. For each rule, we want to ensure that we have neither proportionally too many matches, or too few.

We must run this multiple times using different rules so that we can obtain estimates for all comparisons - if we block on e.g. `date_of_birth`, then we cannot compute the `m` values for the `date_of_birth` comparison, as we have only looked at records where these match.


session_dob = (
    linker_detailed.training.estimate_parameters_using_expectation_maximisation(
        block_on("date_of_birth"), estimate_without_term_frequencies=True
    )
)
session_pc = (
    linker_detailed.training.estimate_parameters_using_expectation_maximisation(
        block_on("postcode"), estimate_without_term_frequencies=True
    )
)

If we wish we can have a look at how our parameter estimates changes over these training sessions


session_dob.m_u_values_interactive_history_chart()

For variables that aren't used in the `m`-training blocking rules, we have two estimates --- one from each of the training sessions (see for example `street_number`). We can have a look at how the values compare between them, to ensure that we don't have drastically different values, which may be indicative of an issue.


linker_detailed.visualisations.parameter_estimate_comparisons_chart()

We repeat our parameter estimations for the simple model in much the same fashion


linker_simple.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.8
)
linker_simple.training.estimate_u_using_random_sampling(max_pairs=1e7)
session_ssid = (
    linker_simple.training.estimate_parameters_using_expectation_maximisation(
        block_on("given_name"), estimate_without_term_frequencies=True
    )
)
session_pc = linker_simple.training.estimate_parameters_using_expectation_maximisation(
    block_on("street_number"), estimate_without_term_frequencies=True
)
linker_simple.visualisations.parameter_estimate_comparisons_chart()

# import json
# we can have a look at the full settings if we wish, including the values of our estimated parameters:
# print(json.dumps(linker_detailed._settings_obj.as_dict(), indent=2))
# we can also get a handy summary of of the model in an easily readable format if we wish:
# print(linker_detailed._settings_obj.human_readable_description)
# (we suppress output here for brevity)

We can now visualise some of the details of our models. We can look at the match weights, which tell us the relative importance for/against a match for each of our comparsion levels.

Comparing the two models will show the added benefit we get in the more detailed model --- what in the simple model is classed as 'all other comparisons' is instead broken down further, and we can see that the detail of how this is broken down in fact gives us quite a bit of useful information about the likelihood of a match.


linker_simple.visualisations.match_weights_chart()

linker_detailed.visualisations.match_weights_chart()

As well as the match weights, which give us an idea of the overall effect of each comparison level, we can also look at the individual `u` and `m` parameter estimates, which tells us about the prevalence of coincidences and mistakes (for further details/explanation about this see [this article](https://www.robinlinacre.com/maths_of_fellegi_sunter/)). We might want to revise aspects of our model based on the information we ascertain here.

Note however that some of these values are very small, which is why the match weight chart is often more useful for getting a decent picture of things.


# linker_simple.m_u_parameters_chart()
linker_detailed.visualisations.m_u_parameters_chart()

It is also useful to have a look at unlinkable records - these are records which do not contain enough information to be linked at some match probability threshold. We can figure this out be seeing whether records are able to be matched with themselves.

This is of course relative to the information we have put into the model - we see that in our simple model, at a 99% match threshold nearly 10% of records are unlinkable, as we have not included enough information in the model for distinct records to be adequately distinguished; this is not an issue in our more detailed model.


linker_simple.evaluation.unlinkables_chart()

linker_detailed.evaluation.unlinkables_chart()

Our simple model doesn't do _terribly_, but suffers if we want to have a high match probability --- to be 99% (match weight ~7) certain of matches we have ~10% of records that we will be unable to link.

Our detailed model, however, has enough nuance that we can at least self-link records.


### Predictions

Now that we have had a look into the details of the models, we will focus on only our more detailed model, which should be able to capture more of the genuine links in our data


predictions = linker_detailed.inference.predict(threshold_match_probability=0.2)
df_predictions = predictions.as_pandas_dataframe()
df_predictions.head(5)

We can see how our model performs at different probability thresholds, with a couple of options depending on the space we wish to view things


linker_detailed.evaluation.accuracy_analysis_from_labels_column(
    "cluster", output_type="accuracy"
)

and we can easily see how many individuals we identify and link by looking at clusters generated at some threshold match probability of interest - in this example 99%


clusters = linker_detailed.clustering.cluster_pairwise_predictions_at_threshold(
    predictions, threshold_match_probability=0.99
)
df_clusters = clusters.as_pandas_dataframe().sort_values("cluster_id")
df_clusters.groupby("cluster_id").size().value_counts()

In this case, we happen to know what the true links are, so we can manually inspect the ones that are doing worst to see what our model is not capturing - i.e. where we have false negatives.

Similarly, we can look at the non-links which are performing the best, to see whether we have an issue with false positives.

Ordinarily we would not have this luxury, and so would need to dig a bit deeper for clues as to how to improve our model, such as manually inspecting records across threshold probabilities,


df_predictions["cluster_l"] = df_predictions["rec_id_l"].apply(
    lambda x: "-".join(x.split("-")[:2])
)
df_predictions["cluster_r"] = df_predictions["rec_id_r"].apply(
    lambda x: "-".join(x.split("-")[:2])
)
df_true_links = df_predictions[
    df_predictions["cluster_l"] == df_predictions["cluster_r"]
].sort_values("match_probability")

records_to_view = 3
linker_detailed.visualisations.waterfall_chart(
    df_true_links.head(records_to_view).to_dict(orient="records")
)

df_non_links = df_predictions[
    df_predictions["cluster_l"] != df_predictions["cluster_r"]
].sort_values("match_probability", ascending=False)
linker_detailed.visualisations.waterfall_chart(
    df_non_links.head(records_to_view).to_dict(orient="records")
)

## Further refinements

Looking at the non-links we have done well in having no false positives at any substantial match probability --- however looking at some of the true links we can see that there are a few that we are not capturing with sufficient match probability.

We can see that there are a few features that we are not capturing/weighting appropriately

- single-character transpostions, particularly in postcode (which is being lumped in with more 'severe typos'/probable non-matches)
- given/sur-names being swapped with typos
- given/sur-names being cross-matches on one only, with no match on the other cross

We will quickly see if we can incorporate these features into a new model. As we are now going into more detail with the inter-relationship between given name and surname, it is probably no longer sensible to model them as independent comparisons, and so we will need to switch to a combined comparison on full name.


# we need to append a full name column to our source data frames
# so that we can use it for term frequency adjustments
dfs[0]["full_name"] = dfs[0]["given_name"] + "_" + dfs[0]["surname"]
dfs[1]["full_name"] = dfs[1]["given_name"] + "_" + dfs[1]["surname"]


extended_model_settings = {
    "unique_id_column_name": "rec_id",
    "link_type": "link_only",
    "blocking_rules_to_generate_predictions": blocking_rules,
    "comparisons": [
        {
            "output_column_name": "Full name",
            "comparison_levels": [
                {
                    "sql_condition": "(given_name_l IS NULL OR given_name_r IS NULL) and (surname_l IS NULL OR surname_r IS NULL)",
                    "label_for_charts": "Null",
                    "is_null_level": True,
                },
                # full name match
                cll.ExactMatchLevel("full_name", term_frequency_adjustments=True),
                # typos - keep levels across full name rather than scoring separately
                cll.JaroWinklerLevel("full_name", 0.9),
                cll.JaroWinklerLevel("full_name", 0.7),
                # name switched
                cll.ColumnsReversedLevel("given_name", "surname"),
                # name switched + typo
                {
                    "sql_condition": "jaro_winkler_similarity(given_name_l, surname_r) + jaro_winkler_similarity(surname_l, given_name_r) >= 1.8",
                    "label_for_charts": "switched + jaro_winkler_similarity >= 1.8",
                },
                {
                    "sql_condition": "jaro_winkler_similarity(given_name_l, surname_r) + jaro_winkler_similarity(surname_l, given_name_r) >= 1.4",
                    "label_for_charts": "switched + jaro_winkler_similarity >= 1.4",
                },
                # single name match
                cll.ExactMatchLevel("given_name", term_frequency_adjustments=True),
                cll.ExactMatchLevel("surname", term_frequency_adjustments=True),
                # single name cross-match
                {
                    "sql_condition": "given_name_l = surname_r OR surname_l = given_name_r",
                    "label_for_charts": "single name cross-matches",
                },  # single name typos
                cll.JaroWinklerLevel("given_name", 0.9),
                cll.JaroWinklerLevel("surname", 0.9),
                # the rest
                cll.ElseLevel(),
            ],
        },
        cl.DateOfBirthComparison(
            "date_of_birth",
            input_is_string=True,
            datetime_format="%Y%m%d",
            invalid_dates_as_null=True,
        ),
        {
            "output_column_name": "Social security ID",
            "comparison_levels": [
                cll.NullLevel("soc_sec_id"),
                cll.ExactMatchLevel("soc_sec_id", term_frequency_adjustments=True),
                cll.DamerauLevenshteinLevel("soc_sec_id", 1),
                cll.DamerauLevenshteinLevel("soc_sec_id", 2),
                cll.ElseLevel(),
            ],
        },
        {
            "output_column_name": "Street number",
            "comparison_levels": [
                cll.NullLevel("street_number"),
                cll.ExactMatchLevel("street_number", term_frequency_adjustments=True),
                cll.DamerauLevenshteinLevel("street_number", 1),
                cll.ElseLevel(),
            ],
        },
        {
            "output_column_name": "Postcode",
            "comparison_levels": [
                cll.NullLevel("postcode"),
                cll.ExactMatchLevel("postcode", term_frequency_adjustments=True),
                cll.DamerauLevenshteinLevel("postcode", 1),
                cll.DamerauLevenshteinLevel("postcode", 2),
                cll.ElseLevel(),
            ],
        },
        # we don't consider further location columns as they will be strongly correlated with postcode
    ],
    "retain_intermediate_calculation_columns": True,
}

# train
linker_advanced = Linker(dfs, extended_model_settings, db_api=DuckDBAPI())
linker_advanced.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.8
)
# We recommend increasing target rows to 1e8 improve accuracy for u
# values in full name comparison, as we have subdivided the data more finely

# Here, 1e7 for speed
linker_advanced.training.estimate_u_using_random_sampling(max_pairs=1e7)

session_dob = (
    linker_advanced.training.estimate_parameters_using_expectation_maximisation(
        "l.date_of_birth = r.date_of_birth", estimate_without_term_frequencies=True
    )
)

session_pc = (
    linker_advanced.training.estimate_parameters_using_expectation_maximisation(
        "l.postcode = r.postcode", estimate_without_term_frequencies=True
    )
)

linker_advanced.visualisations.parameter_estimate_comparisons_chart()

linker_advanced.visualisations.match_weights_chart()

predictions_adv = linker_advanced.inference.predict()
df_predictions_adv = predictions_adv.as_pandas_dataframe()
clusters_adv = linker_advanced.clustering.cluster_pairwise_predictions_at_threshold(
    predictions_adv, threshold_match_probability=0.99
)
df_clusters_adv = clusters_adv.as_pandas_dataframe().sort_values("cluster_id")
df_clusters_adv.groupby("cluster_id").size().value_counts()

This is a pretty modest improvement on our previous model - however it is worth re-iterating that we should not necessarily expect to recover _all_ matches, as in several cases it may be unreasonable for a model to have reasonable confidence that two records refer to the same entity.

If we wished to improve matters we could iterate on this process - investigating where our model is not performing as we would hope, and seeing how we can adjust these areas to address these shortcomings.




Contents of ../docs/demos/examples/duckdb/real_time_record_linkage.ipynb:
## Real time linkage


In this notebook, we demonstrate splink's incremental and real time linkage capabilities - specifically:

- the `linker.compare_two_records` function, that allows you to interactively explore the results of a linkage model; and
- the `linker.find_matches_to_new_records` that allows you to incrementally find matches to a small number of new records


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/duckdb/real_time_record_linkage.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Uncomment and run this cell if you're running in Google Colab.
# !pip install ipywidgets
# !pip install splink
# !jupyter nbextension enable --py widgetsnbextension

### Step 1: Load a pre-trained linkage model


import urllib.request
import json
from pathlib import Path
from splink import Linker, DuckDBAPI, block_on, SettingsCreator, splink_datasets

df = splink_datasets.fake_1000

url = "https://raw.githubusercontent.com/moj-analytical-services/splink_demos/master/demo_settings/real_time_settings.json"

with urllib.request.urlopen(url) as u:
    settings = json.loads(u.read().decode())


linker = Linker(df, settings, db_api=DuckDBAPI())

linker.visualisations.waterfall_chart(
    linker.inference.predict().as_record_dict(limit=2)
)

### Step Comparing two records

It's now possible to compute a match weight for any two records using `linker.compare_two_records()`


record_1 = {
    "unique_id": 1,
    "first_name": "Lucas",
    "surname": "Smith",
    "dob": "1984-01-02",
    "city": "London",
    "email": "lucas.smith@hotmail.com",
}

record_2 = {
    "unique_id": 2,
    "first_name": "Lucas",
    "surname": "Smith",
    "dob": "1983-02-12",
    "city": "Machester",
    "email": "lucas.smith@hotmail.com",
}

linker._settings_obj._retain_intermediate_calculation_columns = True


# To `compare_two_records` the linker needs to compute term frequency tables
# If you have precomputed tables, you can linker.register_term_frequency_lookup()
linker.table_management.compute_tf_table("first_name")
linker.table_management.compute_tf_table("surname")
linker.table_management.compute_tf_table("dob")
linker.table_management.compute_tf_table("city")
linker.table_management.compute_tf_table("email")


df_two = linker.inference.compare_two_records(record_1, record_2)
df_two.as_pandas_dataframe()

### Step 3: Interactive comparisons

One interesting applicatin of `compare_two_records` is to create a simple interface that allows the user to input two records interactively, and get real time feedback.

In the following cell we use `ipywidets` for this purpose. ✨✨ Change the values in the text boxes to see the waterfall chart update in real time. ✨✨


import ipywidgets as widgets
from IPython.display import display


fields = ["unique_id", "first_name", "surname", "dob", "email", "city"]

left_text_boxes = []
right_text_boxes = []

inputs_to_interactive_output = {}

for f in fields:
    wl = widgets.Text(description=f, value=str(record_1[f]))
    left_text_boxes.append(wl)
    inputs_to_interactive_output[f"{f}_l"] = wl
    wr = widgets.Text(description=f, value=str(record_2[f]))
    right_text_boxes.append(wr)
    inputs_to_interactive_output[f"{f}_r"] = wr

b1 = widgets.VBox(left_text_boxes)
b2 = widgets.VBox(right_text_boxes)
ui = widgets.HBox([b1, b2])


def myfn(**kwargs):
    my_args = dict(kwargs)

    record_left = {}
    record_right = {}

    for key, value in my_args.items():
        if value == "":
            value = None
        if key.endswith("_l"):
            record_left[key[:-2]] = value
        elif key.endswith("_r"):
            record_right[key[:-2]] = value

    # Assuming 'linker' is defined earlier in your code
    linker._settings_obj._retain_intermediate_calculation_columns = True

    df_two = linker.inference.compare_two_records(record_left, record_right)

    recs = df_two.as_pandas_dataframe().to_dict(orient="records")

    display(linker.visualisations.waterfall_chart(recs, filter_nulls=False))


out = widgets.interactive_output(myfn, inputs_to_interactive_output)

display(ui, out)

## Finding matching records interactively

It is also possible to search the records in the input dataset rapidly using the `linker.find_matches_to_new_records()` function


record = {
    "unique_id": 123987,
    "first_name": "Robert",
    "surname": "Alan",
    "dob": "1971-05-24",
    "city": "London",
    "email": "robert255@smith.net",
}


df_inc = linker.inference.find_matches_to_new_records(
    [record], blocking_rules=[]
).as_pandas_dataframe()
df_inc.sort_values("match_weight", ascending=False)

## Interactive interface for finding records

Again, we can use `ipywidgets` to build an interactive interface for the `linker.find_matches_to_new_records` function


@widgets.interact(
    first_name="Robert",
    surname="Alan",
    dob="1971-05-24",
    city="London",
    email="robert255@smith.net",
)
def interactive_link(first_name, surname, dob, city, email):
    record = {
        "unique_id": 123987,
        "first_name": first_name,
        "surname": surname,
        "dob": dob,
        "city": city,
        "email": email,
        "group": 0,
    }

    for key in record.keys():
        if type(record[key]) == str:
            if record[key].strip() == "":
                record[key] = None

    df_inc = linker.inference.find_matches_to_new_records(
        [record], blocking_rules=[f"(true)"]
    ).as_pandas_dataframe()
    df_inc = df_inc.sort_values("match_weight", ascending=False)
    recs = df_inc.to_dict(orient="records")

    display(linker.visualisations.waterfall_chart(recs, filter_nulls=False))

linker.visualisations.match_weights_chart()



Contents of ../docs/demos/examples/duckdb/comparison_playground.ipynb:
# Uncomment and run this cell if you're running in Google Colab.
# !pip install ipywidgets
# !pip install splink
# !jupyter nbextension enable --py widgetsnbextension

### Comparison values

Run the following cell to get an interactive interface for trying out comparisons from the `comparison_library`.

The interface uses the default arguments for each comparison, and can only be used for comparisons that refer to a single column

In the following cells, you can get similar results for custom comparisons with an arbitrary number of input columns.

import splink.comparison_library as cl
from splink.internals.testing import comparison_vector_value, is_in_level
from splink import DuckDBAPI
import ipywidgets as widgets
from IPython.display import Markdown


def report_comparison_levels(comparison, values_dict, column_name):
    db_api = DuckDBAPI()
    levels = comparison.create_comparison_levels()
    comparison_dict = comparison.get_comparison("duckdb").as_dict()

    table_rows = []
    table_rows.append("| Match | Level | Description | SQL Condition |")
    table_rows.append("|-------|-------|-------------|---------------|")

    total_levels = len(levels)
    matched_level = None
    for i, level in enumerate(levels):
        level_number = total_levels - i - 1
        label = level.create_label_for_charts()
        sql_condition = comparison_dict['comparison_levels'][i]['sql_condition']

        is_match = is_in_level(level, values_dict, db_api)

        if is_match and matched_level is None:
            matched_level = level_number
            match_indicator = "✓"
            row = f"| {match_indicator} | **{level_number}** | **{label}** | `{sql_condition}` |"
        else:
            match_indicator = ""
            row = f"| {match_indicator} | {level_number} | {label} | `{sql_condition}` |"

        table_rows.append(row)

    markdown_output = "\n".join(table_rows)
    return markdown_output

def create_comparison_playground(column_name):

    comparison_types = [
        'ExactMatch', 'LevenshteinAtThresholds', 'JaroAtThresholds',
        'JaroWinklerAtThresholds', 'DamerauLevenshteinAtThresholds',
        'JaccardAtThresholds',
        'AbsoluteDateDifferenceAtThresholds',
        'ArrayIntersectAtSizes', 'DateOfBirthComparison',
         'EmailComparison',
        'NameComparison', 'PostcodeComparison'
    ]

    default_values = {
        'ExactMatch': ('john', 'jon'),
        'LevenshteinAtThresholds': ('smith', 'smyth'),
        'JaroAtThresholds': ('martha', 'matha'),
        'JaroWinklerAtThresholds': ('williams', 'willaims'),
        'DamerauLevenshteinAtThresholds': ('receive', 'recieve'),
        'CosineSimilarityAtThresholds': ('data science', 'science data'),
        'JaccardAtThresholds': ('0123456789', '012345678'),
        'AbsoluteDateDifferenceAtThresholds': ('2023-01-01', '2023-01-15'),
        'ArrayIntersectAtSizes': ('apple,banana,cherry', 'banana,cherry,date'),
        'DateOfBirthComparison': ('1990-05-15', '1990-05-16'),
        'EmailComparison': ('john.doe@example.com', 'john.doe@other.com'),
        'NameComparison': ('Elizabeth', 'Elisabeth'),
        'PostcodeComparison': ('SW1A 1AA', 'SW1A 1AB')
    }

    db_api = DuckDBAPI()

    docstrings = {}
    for comp_type in comparison_types:
        class_obj = getattr(cl, comp_type)
        init_doc = getattr(class_obj.__init__, '__doc__', None)
        docstrings[comp_type] = init_doc if init_doc else class_obj.__doc__


    def get_comparison(comp_type):
        if comp_type in ['DateOfBirthComparison']:

            return getattr(cl, comp_type)(column_name, input_is_string=True)
        if comp_type == 'AbsoluteDateDifferenceAtThresholds':
            return getattr(cl, comp_type)(column_name, input_is_string=True, metrics=["day", "month"], thresholds=[1, 1])
        elif comp_type in ['EmailComparison', 'ForenameSurnameComparison', 'NameComparison', 'PostcodeComparison', 'ArrayIntersectAtSizes']:
            return getattr(cl, comp_type)(column_name)
        else:
            return getattr(cl, comp_type)(column_name)

    def run_comparison(change):
        left_value = left_input.value if left_input.value != "" else None
        right_value = right_input.value if right_input.value != "" else None
        comparison = get_comparison(comparison_select.value)

        if comparison_select.value == 'ArrayIntersectAtSizes':
            left_value = left_value.split(',') if left_value else None
            right_value = right_value.split(',') if right_value else None

        values_dict = {f"{column_name}_l": left_value, f"{column_name}_r": right_value}

        output.clear_output()
        markdown_output = report_comparison_levels(comparison, values_dict, column_name)
        with output:
            display(Markdown("### Comparison levels:"))
            display(Markdown(markdown_output))

            docstring = docstrings.get(comparison_select.value, "No docstring available")
            processed_docstring = "\n".join(line.strip() for line in docstring.split("\n"))
            display(Markdown("### Comparison Function Docstring:"))
            display(Markdown(processed_docstring))


        # Store the markdown output for later use
        playground.markdown_output = markdown_output



    def on_comparison_change(change):
        new_value = change['new']
        left_value, right_value = default_values.get(new_value, ('', ''))

        # Temporarily unobserve the input widgets
        left_input.unobserve(run_comparison, names='value')
        right_input.unobserve(run_comparison, names='value')

        # Update the values
        left_input.value = left_value
        right_input.value = right_value

        # Re-observe the input widgets
        left_input.observe(run_comparison, names='value')
        right_input.observe(run_comparison, names='value')

        # Run the comparison once after updating both inputs
        run_comparison(None)


    comparison_select = widgets.Dropdown(
        options=comparison_types,
        value='ExactMatch',
        description='Comparison:',
    )
    left_input = widgets.Text(description=f"{column_name} Left:", value=default_values['ExactMatch'][0])
    right_input = widgets.Text(description=f"{column_name} Right:", value=default_values['ExactMatch'][1])
    output = widgets.Output()

    comparison_select.observe(on_comparison_change, names='value')
    for widget in (comparison_select, left_input, right_input):
        widget.observe(run_comparison, names='value')

    # Call run_comparison immediately to compute initial output
    playground = widgets.VBox([comparison_select, left_input, right_input, output])

    run_comparison(None)

    return playground

playground = create_comparison_playground("column")
display(playground)

import splink.comparison_level_library as cll

first_name_comparison = cl.CustomComparison(
    comparison_levels=[
        cll.NullLevel("first_name"),
        cll.ExactMatchLevel("first_name"),
        {
            "sql_condition": "first_name_l = surname_r",
            "label_for_charts": "Match on reversed cols: first_name and surname",
        },
        cll.JaroWinklerLevel("first_name", 0.8),
        cll.ElseLevel(),
    ]
)
# Need to be able to pass values in as a dict {"first_name_l": "Robin", "first_name_r": "Robyn", "surname_l": "Linacre", "surname_r": "Linacre"}
values = {
    "first_name_l": "Robin",
    "first_name_r": "Linacre",
    "surname_l": "Linacre",
    "surname_r": "Robin",
}
display(Markdown(report_comparison_levels(first_name_comparison, values, "first_name")))



Contents of ../docs/demos/examples/duckdb/cookbook.ipynb:
# Cookbook

This notebook contains a miscellaneous collection of runnable examples illustrating various Splink techniques.

## Array columns

### Comparing array columns

This example shows how we can use use `ArrayIntersectAtSizes` to assess the similarity of columns containing arrays.

# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

import logging
logging.getLogger("splink").setLevel(logging.ERROR)


import pandas as pd

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on


data = [
    {"unique_id": 1, "first_name": "John", "postcode": ["A", "B"]},
    {"unique_id": 2, "first_name": "John", "postcode": ["B"]},
    {"unique_id": 3, "first_name": "John", "postcode": ["A"]},
    {"unique_id": 4, "first_name": "John", "postcode": ["A", "B"]},
    {"unique_id": 5, "first_name": "John", "postcode": ["C"]},
]

df = pd.DataFrame(data)

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
    ],
    comparisons=[
        cl.ArrayIntersectAtSizes("postcode", [2, 1]),
        cl.ExactMatch("first_name"),
    ]
)


linker = Linker(df, settings, DuckDBAPI(), set_up_basic_logging=False)

linker.inference.predict().as_pandas_dataframe()

### Blocking on array columns

This example shows how we can use `block_on` to block on the individual elements of an array column - that is, pairwise comaprisons are created for pairs or records where any of the elements in the array columns match.

import pandas as pd

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on


data = [
    {"unique_id": 1, "first_name": "John", "postcode": ["A", "B"]},
    {"unique_id": 2, "first_name": "John", "postcode": ["B"]},
    {"unique_id": 3, "first_name": "John", "postcode": ["C"]},

]

df = pd.DataFrame(data)

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("postcode", arrays_to_explode=["postcode"]),
    ],
    comparisons=[
        cl.ArrayIntersectAtSizes("postcode", [2, 1]),
        cl.ExactMatch("first_name"),
    ]
)


linker = Linker(df, settings, DuckDBAPI(), set_up_basic_logging=False)

linker.inference.predict().as_pandas_dataframe()


## Other


### Using DuckDB without pandas

In this example, we read data directly using DuckDB and obtain results in native DuckDB `DuckDBPyRelation` format.


import duckdb
import tempfile
import os

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets

# Create a parquet file on disk to demontrate native DuckDB parquet reading
df = splink_datasets.fake_1000
temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".parquet")
temp_file_path = temp_file.name
df.to_parquet(temp_file_path)

# Example would start here if you already had a parquet file
duckdb_df = duckdb.read_parquet(temp_file_path)

db_api = DuckDBAPI(":default:")
settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        cl.NameComparison("first_name"),
        cl.JaroAtThresholds("surname"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "dob"),
        block_on("surname"),
    ],
)

linker = Linker(df, settings, db_api, set_up_basic_logging=False)

result = linker.inference.predict().as_duckdbpyrelation()

# Since result is a DuckDBPyRelation, we can use all the usual DuckDB API
# functions on it.

# For example, we can use the `sort` function to sort the results,
# or could use result.to_parquet() to write to a parquet file.
result.sort("match_weight")


### Fixing `m` or `u` probabilities during training


import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets


db_api = DuckDBAPI()

first_name_comparison = cl.CustomComparison(
    comparison_levels=[
        cll.NullLevel("first_name"),
        cll.ExactMatchLevel("first_name").configure(
            m_probability=0.9999,
            fix_m_probability=True,
            u_probability=0.7,
            fix_u_probability=True,
        ),
        cll.ElseLevel(),
    ]
)
settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        first_name_comparison,
        cl.ExactMatch("surname"),
        cl.ExactMatch("dob"),
        cl.ExactMatch("city"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("dob"),
    ],
    additional_columns_to_retain=["cluster"],
)

df = splink_datasets.fake_1000
linker = Linker(df, settings, db_api, set_up_basic_logging=False)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))

linker.visualisations.m_u_parameters_chart()

### Manually altering `m` and `u` probabilities post-training

This is not officially supported, but can be useful for ad-hoc alterations to trained models.

import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets
from splink.datasets import splink_dataset_labels

labels = splink_dataset_labels.fake_1000_labels

db_api = DuckDBAPI()


settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        cl.ExactMatch("first_name"),
        cl.ExactMatch("surname"),
        cl.ExactMatch("dob"),
        cl.ExactMatch("city"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("dob"),
    ],
)
df = splink_datasets.fake_1000
linker = Linker(df, settings, db_api, set_up_basic_logging=False)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))


surname_comparison = linker._settings_obj._get_comparison_by_output_column_name(
    "surname"
)
else_comparison_level = (
    surname_comparison._get_comparison_level_by_comparison_vector_value(0)
)
else_comparison_level._m_probability = 0.1


linker.visualisations.m_u_parameters_chart()

### Generate the (beta) labelling tool

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets

db_api = DuckDBAPI()

df = splink_datasets.fake_1000

settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        cl.ExactMatch("first_name"),
        cl.ExactMatch("surname"),
        cl.ExactMatch("dob"),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.ExactMatch("email"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
    ],
    max_iterations=2,
)

linker = Linker(df, settings, db_api, set_up_basic_logging=False)

linker.training.estimate_probability_two_random_records_match(
    [block_on("first_name", "surname")], recall=0.7
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))

pairwise_predictions = linker.inference.predict(threshold_match_weight=-10)

first_unique_id = df.iloc[0].unique_id
linker.evaluation.labelling_tool_for_specific_record(unique_id=first_unique_id, overwrite=True)




Contents of ../docs/demos/examples/duckdb/febrl3.ipynb:
## Deduplicating the febrl3 dataset

See A.2 [here](https://arxiv.org/pdf/2008.04443.pdf) and [here](https://recordlinkage.readthedocs.io/en/latest/ref-datasets.html) for the source of this data


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/duckdb/febrl3.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink.datasets import splink_datasets

df = splink_datasets.febrl3

df = df.rename(columns=lambda x: x.strip())

df["cluster"] = df["rec_id"].apply(lambda x: "-".join(x.split("-")[:2]))

df["date_of_birth"] = df["date_of_birth"].astype(str).str.strip()
df["soc_sec_id"] = df["soc_sec_id"].astype(str).str.strip()

df.head(2)

df["date_of_birth"] = df["date_of_birth"].astype(str).str.strip()
df["soc_sec_id"] = df["soc_sec_id"].astype(str).str.strip()

df["date_of_birth"] = df["date_of_birth"].astype(str).str.strip()
df["soc_sec_id"] = df["soc_sec_id"].astype(str).str.strip()

from splink import DuckDBAPI, Linker, SettingsCreator

# TODO:  Allow missingness to be analysed without a linker
settings = SettingsCreator(
    unique_id_column_name="rec_id",
    link_type="dedupe_only",
)

linker = Linker(df, settings, db_api=DuckDBAPI())

It's usually a good idea to perform exploratory analysis on your data so you understand what's in each column and how often it's missing:


from splink.exploratory import completeness_chart

completeness_chart(df, db_api=DuckDBAPI())

from splink.exploratory import profile_columns

profile_columns(df, db_api=DuckDBAPI(), column_expressions=["given_name", "surname"])

from splink import DuckDBAPI, block_on
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

blocking_rules = [
    block_on("soc_sec_id"),
    block_on("given_name"),
    block_on("surname"),
    block_on("date_of_birth"),
    block_on("postcode"),
]

db_api = DuckDBAPI()
cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    blocking_rules=blocking_rules,
    db_api=db_api,
    link_type="dedupe_only",
    unique_id_column_name="rec_id",
)

import splink.comparison_library as cl

from splink import Linker

settings = SettingsCreator(
    unique_id_column_name="rec_id",
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=blocking_rules,
    comparisons=[
        cl.NameComparison("given_name"),
        cl.NameComparison("surname"),
        cl.DateOfBirthComparison(
            "date_of_birth",
            input_is_string=True,
            datetime_format="%Y%m%d",
        ),
        cl.DamerauLevenshteinAtThresholds("soc_sec_id", [2]),
        cl.ExactMatch("street_number").configure(term_frequency_adjustments=True),
        cl.ExactMatch("postcode").configure(term_frequency_adjustments=True),
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=DuckDBAPI())

from splink import block_on

deterministic_rules = [
    block_on("soc_sec_id"),
    block_on("given_name", "surname", "date_of_birth"),
    "l.given_name = r.surname and l.surname = r.given_name and l.date_of_birth = r.date_of_birth",
]

linker.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.9
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

em_blocking_rule_1 = block_on("date_of_birth")
session_dob = linker.training.estimate_parameters_using_expectation_maximisation(
    em_blocking_rule_1
)

em_blocking_rule_2 = block_on("postcode")
session_postcode = linker.training.estimate_parameters_using_expectation_maximisation(
    em_blocking_rule_2
)

linker.visualisations.match_weights_chart()

results = linker.inference.predict(threshold_match_probability=0.2)

linker.evaluation.accuracy_analysis_from_labels_column(
    "cluster", match_weight_round_to_nearest=0.1, output_type="accuracy"
)

pred_errors_df = linker.evaluation.prediction_errors_from_labels_column(
    "cluster"
).as_pandas_dataframe()
len(pred_errors_df)
pred_errors_df.head()

The following chart seems to suggest that, where the model is making errors, it's because the data is corrupted beyond recognition and no reasonable linkage model could find these matches

records = linker.evaluation.prediction_errors_from_labels_column(
    "cluster"
).as_record_dict(limit=10)
linker.visualisations.waterfall_chart(records)



Contents of ../docs/demos/examples/spark/deduplicate_1k_synthetic.ipynb:
## Linking in Spark


<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/examples/spark/deduplicate_1k_synthetic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink
# !pip install pyspark

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from splink.backends.spark import similarity_jar_location

conf = SparkConf()
# This parallelism setting is only suitable for a small toy example
conf.set("spark.driver.memory", "12g")
conf.set("spark.default.parallelism", "8")
conf.set("spark.sql.codegen.wholeStage", "false")


# Add custom similarity functions, which are bundled with Splink
# documented here: https://github.com/moj-analytical-services/splink_scalaudfs
path = similarity_jar_location()
conf.set("spark.jars", path)

sc = SparkContext.getOrCreate(conf=conf)

spark = SparkSession(sc)
spark.sparkContext.setCheckpointDir("./tmp_checkpoints")

# Disable warnings for pyspark - you don't need to include this
import warnings

spark.sparkContext.setLogLevel("ERROR")
warnings.simplefilter("ignore", UserWarning)

from splink import splink_datasets

pandas_df = splink_datasets.fake_1000

df = spark.createDataFrame(pandas_df)

import splink.comparison_library as cl
from splink import Linker, SettingsCreator, SparkAPI, block_on

settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.LevenshteinAtThresholds(
            "dob"
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        "l.surname = r.surname",  # alternatively, you can write BRs in their SQL form
    ],
    retain_intermediate_calculation_columns=True,
    em_convergence=0.01,
)

linker = Linker(df, settings, db_api=SparkAPI(spark_session=spark))
deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    "l.email = r.email",
]

linker.training.estimate_probability_two_random_records_match(deterministic_rules, recall=0.6)

linker.training.estimate_u_using_random_sampling(max_pairs=5e5)

training_blocking_rule = "l.first_name = r.first_name and l.surname = r.surname"
training_session_fname_sname = (
    linker.training.estimate_parameters_using_expectation_maximisation(training_blocking_rule)
)

training_blocking_rule = "l.dob = r.dob"
training_session_dob = linker.training.estimate_parameters_using_expectation_maximisation(
    training_blocking_rule
)

results = linker.inference.predict(threshold_match_probability=0.9)

spark_df = results.as_spark_dataframe().show()



Contents of ../docs/demos/examples/athena/deduplicate_50k_synthetic.ipynb:
## Linking a dataset of real historical persons
In this example, we deduplicate a more realistic dataset. The data is based on historical persons scraped from wikidata. Duplicate records are introduced with a variety of errors introduced.

Create a boto3 session to be used within the linker

import boto3

boto3_session = boto3.Session(region_name="eu-west-1")

## AthenaLinker Setup

To work nicely with Athena, you need to outline various filepaths, buckets and the database(s) you wish to interact with.
<hr>

**The AthenaLinker has three required inputs:**
* input_table_or_tables - the input table to use for linking. This can either be a table in a database or a pandas dataframe
* output_database - the database to output all of your splink tables to.
* output_bucket - the s3 bucket you wish any parquet files produced by splink to be output to.

**and two optional inputs:**
* output_filepath - the s3 filepath to output files to. This is an extension of output_bucket and dictate the full filepath your files will be output to.
* input_table_aliases - the name of your table within your database, should you choose to use a pandas df as an input.

# Set the output bucket and the additional filepath to write outputs to
############################################
# EDIT THESE BEFORE ATTEMPTING TO RUN THIS #
############################################

from splink.backends.athena import AthenaAPI


bucket = "MYTESTBUCKET"
database = "MYTESTDATABASE"
filepath = "MYTESTFILEPATH"  # file path inside of your bucket

aws_filepath = f"s3://{bucket}/{filepath}"
db_api = AthenaAPI(
    boto3_session,
    output_bucket=bucket,
    output_database=database,
    output_filepath=filepath,
)

import splink.comparison_library as cl
from splink import block_on

from splink import Linker, SettingsCreator, splink_datasets

df = splink_datasets.historical_50k

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname"),
        block_on("surname", "dob"),
    ],
    comparisons=[
        cl.ExactMatch("first_name").configure(term_frequency_adjustments=True),
        cl.LevenshteinAtThresholds("surname", [1, 3]),
        cl.LevenshteinAtThresholds("dob", [1, 2]),
        cl.LevenshteinAtThresholds("postcode_fake", [1, 2]),
        cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
        cl.ExactMatch("occupation").configure(term_frequency_adjustments=True),
    ],
    retain_intermediate_calculation_columns=True,
)

from splink.exploratory import profile_columns

profile_columns(df, db_api, column_expressions=["first_name", "substr(surname,1,2)"])

from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)
from splink import block_on

cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    db_api=db_api,
    blocking_rules=[block_on("first_name", "surname"), block_on("surname", "dob")],
    link_type="dedupe_only",
)

import splink.comparison_library as cl


from splink import Linker, SettingsCreator

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname"),
        block_on("surname", "dob"),
    ],
    comparisons=[
        cl.ExactMatch("first_name").configure(term_frequency_adjustments=True),
        cl.LevenshteinAtThresholds("surname", [1, 3]),
        cl.LevenshteinAtThresholds("dob", [1, 2]),
        cl.LevenshteinAtThresholds("postcode_fake", [1, 2]),
        cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
        cl.ExactMatch("occupation").configure(term_frequency_adjustments=True),
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=db_api)

linker.training.estimate_probability_two_random_records_match(
    [
        block_on("first_name", "surname", "dob"),
        block_on("substr(first_name,1,2)", "surname", "substr(postcode_fake, 1,2)"),
        block_on("dob", "postcode_fake"),
    ],
    recall=0.6,
)

linker.training.estimate_u_using_random_sampling(max_pairs=5e6)

blocking_rule = block_on("first_name", "surname")
training_session_names = (
    linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule)
)

blocking_rule = block_on("dob")
training_session_dob = (
    linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule)
)

linker.visualisations.match_weights_chart()

linker.evaluation.unlinkables_chart()

df_predict = linker.inference.predict()
df_e = df_predict.as_pandas_dataframe(limit=5)
df_e

You can also view rows in this dataset as a waterfall chart as follows:

records_to_plot = df_e.to_dict(orient="records")
linker.visualisations.waterfall_chart(records_to_plot, filter_nulls=False)

clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predict, threshold_match_probability=0.95
)

linker.visualisations.cluster_studio_dashboard(
    df_predict,
    clusters,
    "dashboards/50k_cluster.html",
    sampling_method="by_cluster_size",
    overwrite=True,
)

from IPython.display import IFrame

IFrame(src="./dashboards/50k_cluster.html", width="100%", height=1200)



Contents of ../docs/demos/tutorials/03_Blocking.ipynb:
# Choosing blocking rules to optimise runtime

<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/03_Blocking.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

To link records, we need to compare pairs of records and decide which pairs are matches.

For example consider the following two records:

| first_name | surname | dob        | city   | email               |
| ---------- | ------- | ---------- | ------ | ------------------- |
| Robert     | Allen   | 1971-05-24 | nan    | roberta25@smith.net |
| Rob        | Allen   | 1971-06-24 | London | roberta25@smith.net |

These can be represented as a pairwise comparison as follows:

| first_name_l | first_name_r | surname_l | surname_r | dob_l      | dob_r      | city_l | city_r | email_l             | email_r             |
| ------------ | ------------ | --------- | --------- | ---------- | ---------- | ------ | ------ | ------------------- | ------------------- |
| Robert       | Rob          | Allen     | Allen     | 1971-05-24 | 1971-06-24 | nan    | London | roberta25@smith.net | roberta25@smith.net |

For most large datasets, it is computationally intractable to compare every row with every other row, since the number of comparisons rises quadratically with the number of records.

Instead we rely on blocking rules, which specify which pairwise comparisons to generate. For example, we could generate the subset of pairwise comparisons where either first name or surname matches.

This is part of a two step process to link data:

1.  Use blocking rules to generate candidate pairwise record comparisons

2.  Use a probabilistic linkage model to score these candidate pairs, to determine which ones should be linked

**Blocking rules are the most important determinant of the performance of your linkage job**.

When deciding on your blocking rules, you're trading off accuracy for performance:

- If your rules are too loose, your linkage job may fail.
- If they're too tight, you may miss some valid links.

This tutorial clarifies what blocking rules are, and how to choose good rules.

## Blocking rules in Splink

In Splink, blocking rules are specified as SQL expressions.

For example, to generate the subset of record comparisons where the first name and surname matches, we can specify the following blocking rule:

```python
from splink import block_on
block_on("first_name", "surname")
```

When executed, this blocking rule will be converted to a SQL statement with the following form:

```sql
SELECT ...
FROM input_tables as l
INNER JOIN input_tables as r
ON l.first_name = r.first_name AND l.surname = r.surname
```

Since blocking rules are SQL expressions, they can be arbitrarily complex. For example, you could create record comparisons where the initial of the first name and the surname match with the following rule:

```python
from splink import block_on
block_on("substr(first_name, 1, 2)", "surname")
```


## Devising effective blocking rules for prediction

The aims of your blocking rules are twofold:

1. Eliminate enough non-matching comparison pairs so your record linkage job is small enough to compute
2. Eliminate as few truly matching pairs as possible (ideally none)

It is usually impossible to find a single blocking rule which achieves both aims, so we recommend using multiple blocking rules.

When we specify multiple blocking rules, Splink will generate all comparison pairs that meet any one of the rules.

For example, consider the following blocking rule:

`block_on("first_name", "dob")`

This rule is likely to be effective in reducing the number of comparison pairs. It will retain all truly matching pairs, except those with errors or nulls in either the `first_name` or `dob` fields.

Now consider a second blocking rule:

`block_on("email")`.

This will retain all truly matching pairs, except those with errors or nulls in the `email` column.

Individually, these blocking rules are problematic because they exclude true matches where the records contain typos of certain types. But between them, they might do quite a good job.

For a true match to be eliminated by the use of these two blocking rules, it would have to have an error in _both_ `email` AND (`first_name` or `dob`).

This is not completely implausible, but it is significantly less likely than if we'd used a single rule.

More generally, we can often specify multiple blocking rules such that it becomes highly implausible that a true match would not meet at least one of these blocking criteria. This is the recommended approach in Splink. Generally we would recommend between about 3 and 10, though even more is possible.

The question then becomes how to choose what to put in this list.


## Splink tools to help choose your blocking rules

Splink contains a number of tools to help you choose effective blocking rules. Let's try them out, using our small test dataset:


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import DuckDBAPI, block_on, splink_datasets

df = splink_datasets.fake_1000

### Counting the number of comparisons created by a single blocking rule

On large datasets, some blocking rules imply the creation of trillions of record comparisons, which would cause a linkage job to fail.

Before using a blocking rule in a linkage job, it's therefore a good idea to count the number of records it generates to ensure it is not too loose:


from splink.blocking_analysis import count_comparisons_from_blocking_rule

db_api = DuckDBAPI()

br = block_on("substr(first_name, 1,1)", "surname")

counts = count_comparisons_from_blocking_rule(
    table_or_tables=df,
    blocking_rule=br,
    link_type="dedupe_only",
    db_api=db_api,
)

counts

br = "l.first_name = r.first_name and levenshtein(l.surname, r.surname) < 2"

counts = count_comparisons_from_blocking_rule(
    table_or_tables=df,
    blocking_rule= br,
    link_type="dedupe_only",
    db_api=db_api,
)
counts

The maximum number of comparisons that you can compute will be affected by your choice of SQL backend, and how powerful your computer is.

For linkages in DuckDB on a standard laptop, we suggest using blocking rules that create no more than about 20 million comparisons. For Spark and Athena, try starting with fewer than 100 million comparisons, before scaling up.


### Finding 'worst offending' values for your blocking rule

Blocking rules can be affected by skew:  some values of a field may be much more common than others, and this can lead to a disproportionate number of comparisons being generated.

It can be useful to identify whether your data is afflicted by this problem. 

from splink.blocking_analysis import n_largest_blocks

result = n_largest_blocks(    table_or_tables=df,
    blocking_rule= block_on("city", "first_name"),
    link_type="dedupe_only",
    db_api=db_api,
    n_largest=3
    )

result.as_pandas_dataframe()

In this case, we can see that `Oliver`s in `London` will result in 49 comparisons being generated.  This is acceptable on this small dataset, but on a larger dataset, `Oliver`s in `London` could be responsible for many million comparisons.

### Counting the number of comparisons created by a list of blocking rules

As noted above, it's usually a good idea to use multiple blocking rules. It's therefore useful to know how many record comparisons will be generated when these rules are applied.

Since the same record comparison may be created by several blocking rules, and Splink automatically deduplicates these comparisons, we cannot simply total the number of comparisons generated by each rule individually.

Splink provides a chart that shows the marginal (additional) comparisons generated by each blocking rule, after deduplication:


from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

blocking_rules_for_analysis = [
    block_on("substr(first_name, 1,1)", "surname"),
    block_on("surname"),
    block_on("email"),
    block_on("city", "first_name"),
    "l.first_name = r.first_name and levenshtein(l.surname, r.surname) < 2",
]


cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    blocking_rules=blocking_rules_for_analysis,
    db_api=db_api,
    link_type="dedupe_only",
)

### Digging deeper: Understanding why certain blocking rules create large numbers of comparisons

Finally, we can use the `profile_columns` function we saw in the previous tutorial to understand a specific blocking rule in more depth.

Suppose we're interested in blocking on city and first initial.

Within each distinct value of `(city, first initial)`, all possible pairwise comparisons will be generated.

So for instance, if there are 15 distinct records with `London,J` then these records will result in `n(n-1)/2 = 105` pairwise comparisons being generated.

In a larger dataset, we might observe 10,000 `London,J` records, which would then be responsible for `49,995,000` comparisons.

These high-frequency values therefore have a disproportionate influence on the overall number of pairwise comparisons, and so it can be useful to analyse skew, as follows:


from splink.exploratory import profile_columns

profile_columns(df, column_expressions=["city || left(first_name,1)"], db_api=db_api)

!!! note "Further Reading"
    :simple-readme: For a deeper dive on blocking, please refer to the [Blocking Topic Guides](../../topic_guides/blocking/blocking_rules.md).

    :material-tools: For more on the blocking tools in Splink, please refer to the [Blocking API documentation](../../api_docs/blocking.md).

    :bar_chart: For more on the charts used in this tutorial, please refer to the [Charts Gallery](../../charts/index.md#blocking).


## Next steps

Now we have chosen which records to compare, we can use those records to train a linkage model.




Contents of ../docs/demos/tutorials/07_Evaluation.ipynb:
## Evaluation of prediction results

 <a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/07_Quality_assurance.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

In the previous tutorial, we looked at various ways to visualise the results of our model.
These are useful for evaluating a linkage pipeline because they allow us to understand how our model works and verify that it is doing something sensible. They can also be useful to identify examples where the model is not performing as expected.

In addition to these spot checks, Splink also has functions to perform more formal accuracy analysis. These functions allow you to understand the likely prevalence of false positives and false negatives in your linkage models.

They rely on the existence of a sample of labelled (ground truth) matches, which may have been produced (for example) by human beings. For the accuracy analysis to be unbiased, the sample should be representative of the overall dataset.


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

# Rerun our predictions to we're ready to view the charts
import pandas as pd

from splink import DuckDBAPI, Linker, splink_datasets

pd.options.display.max_columns = 1000

db_api = DuckDBAPI()
df = splink_datasets.fake_1000

import json
import urllib

from splink import block_on

url = "https://raw.githubusercontent.com/moj-analytical-services/splink/847e32508b1a9cdd7bcd2ca6c0a74e547fb69865/docs/demos/demo_settings/saved_model_from_demo.json"

with urllib.request.urlopen(url) as u:
    settings = json.loads(u.read().decode())

# The data quality is very poor in this dataset, so we need looser blocking rules
# to achieve decent recall
settings["blocking_rules_to_generate_predictions"] = [
    block_on("first_name"),
    block_on("city"),
    block_on("email"),
    block_on("dob"),
]

linker = Linker(df, settings, db_api=DuckDBAPI())
df_predictions = linker.inference.predict(threshold_match_probability=0.01)

## Load in labels

The labels file contains a list of pairwise comparisons which represent matches and non-matches.

The required format of the labels file is described [here](https://moj-analytical-services.github.io/splink/linkerqa.html#splink.linker.Linker.roc_chart_from_labels).


from splink.datasets import splink_dataset_labels

df_labels = splink_dataset_labels.fake_1000_labels
labels_table = linker.table_management.register_labels_table(df_labels)
df_labels.head(5)

## View examples of false positives and false negatives

splink_df = linker.evaluation.prediction_errors_from_labels_table(
    labels_table, include_false_negatives=True, include_false_positives=False
)
false_negatives = splink_df.as_record_dict(limit=5)
linker.visualisations.waterfall_chart(false_negatives)

### False positives

# Note I've picked a threshold match probability of 0.01 here because otherwise
# in this simple example there are no false positives
splink_df = linker.evaluation.prediction_errors_from_labels_table(
    labels_table, include_false_negatives=False, include_false_positives=True, threshold_match_probability=0.01
)
false_postives = splink_df.as_record_dict(limit=5)
linker.visualisations.waterfall_chart(false_postives)

## Threshold Selection chart

Splink includes an interactive dashboard that shows key accuracy statistics:


linker.evaluation.accuracy_analysis_from_labels_table(
    labels_table, output_type="threshold_selection", add_metrics=["f1"]
)

## Receiver operating characteristic curve

A [ROC chart](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) shows how the number of false positives and false negatives varies depending on the match threshold chosen. The match threshold is the match weight chosen as a cutoff for which pairwise comparisons to accept as matches.


linker.evaluation.accuracy_analysis_from_labels_table(labels_table, output_type="roc")

## Truth table

Finally, Splink can also report the underlying table used to construct the ROC and precision recall curves.


roc_table = linker.evaluation.accuracy_analysis_from_labels_table(
    labels_table, output_type="table"
)
roc_table.as_pandas_dataframe(limit=5)

## Unlinkables chart

Finally, it can be interesting to analyse whether your dataset contains any 'unlinkable' records.

'Unlinkable records' are records with such poor data quality they don't even link to themselves at a high enough probability to be accepted as matches

For example, in a typical linkage problem, a 'John Smith' record with nulls for their address and postcode may be unlinkable.  By 'unlinkable' we don't mean there are no matches; rather, we mean it is not possible to determine whether there are matches.UnicodeTranslateError

A high proportion of unlinkable records is an indication of poor quality in the input dataset

linker.evaluation.unlinkables_chart()

For this dataset and this trained model, we can see that most records are (theoretically) linkable:  At a match weight 6, around around 99% of records could be linked to themselves.

!!! note "Further Reading"

    :material-tools: For more on the quality assurance tools in Splink, please refer to the [Evaluation API documentation](../../api_docs/evaluation.md).

    :bar_chart: For more on the charts used in this tutorial, please refer to the [Charts Gallery](../../charts/index.md#model-evaluation).

    :material-thumbs-up-down: For more on the Evaluation Metrics used in this tutorial, please refer to the [Edge Metrics guide.](../../topic_guides/evaluation/edge_metrics.md)


## :material-flag-checkered: That's it!

That wraps up the Splink tutorial! Don't worry, there are still plenty of resources to help on the next steps of your Splink journey:

:octicons-link-16: For some end-to-end notebooks of Splink pipelines, check out our [Examples](../examples/examples_index.md)

:simple-readme: For more deepdives into the different aspects of Splink, and record linkage more generally, check out our [Topic Guides](../../topic_guides/topic_guides_index.md)

:material-tools: For a reference on all the functionality avalable in Splink, see our [Documentation](../../api_docs/api_docs_index.md)




Contents of ../docs/demos/tutorials/00_Tutorial_Introduction.ipynb:
# Introductory tutorial

This is the introduction to a seven part tutorial which demonstrates how to de-duplicate a small dataset using simple settings.

The aim of the tutorial is to demonstrate core Splink functionality succinctly, rather that comprehensively document all configuration options.

The seven parts are:

- [1. Data prep pre-requisites](./01_Prerequisites.ipynb)

- [2. Exploratory analysis](./02_Exploratory_analysis.ipynb) <a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/02_Exploratory_analysis.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- [3. Choosing blocking rules to optimise runtimes](./03_Blocking.ipynb) <a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/03_Blocking.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- [4. Estimating model parameters](./04_Estimating_model_parameters.ipynb) <a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/04_Estimating_model_parameters.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- [5. Predicting results](./05_Predicting_results.ipynb) <a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/05_Predicting_results.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- [6. Visualising predictions](./06_Visualising_predictions.ipynb) <a> <a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/06_Visualising_predictions.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- [7. Evaluation](./07_Evaluation.ipynb) <a> <a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/07_Evaluation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- [8. Building your own model](./08_building_your_own_model.md) 


Throughout the tutorial, we use the duckdb backend, which is the recommended option for smaller datasets of up to around 1 million records on a normal laptop.

You can find these tutorial notebooks in the `docs/demos/tutorials/` folder of the  [splink repo](https://github.com/moj-analytical-services/splink/tree/master/docs/demos/tutorials), or click the Colab links to run in your browser.

## End-to-end demos

After following the steps of the tutorial, it might prove useful to have a look at some of the [example notebooks](https://moj-analytical-services.github.io/splink/demos/examples/examples_index.html) that show various use-case scenarios of Splink from start to finish.

## Interactive Introduction to Record Linkage Theory

If you'd like to learn more about record linkage theory, an interactive introduction is available [here](https://www.robinlinacre.com/intro_to_probabilistic_linkage/).

## LLM prompts

If you're using an LLM to suggest Splink code, see [here](./topic_guides/llms/prompting_llms.md) for suggested prompts and context.



Contents of ../docs/demos/tutorials/02_Exploratory_analysis.ipynb:
# Exploratory analysis

<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/02_Exploratory_analysis.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Exploratory analysis helps you understand features of your data which are relevant linking or deduplicating your data.

Splink includes a variety of charts to help with this, which are demonstrated in this notebook.


### Read in the data

For the purpose of this tutorial we will use a 1,000 row synthetic dataset that contains duplicates.

The first five rows of this dataset are printed below.

Note that the cluster column represents the 'ground truth' - a column which tells us with which rows refer to the same person. In most real linkage scenarios, we wouldn't have this column (this is what Splink is trying to estimate.)


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import  splink_datasets

df = splink_datasets.fake_1000
df = df.drop(columns=["cluster"])
df.head(5)

## Analyse missingness


It's important to understand the level of missingness in your data, because columns with higher levels of missingness are less useful for data linking.


from splink.exploratory import completeness_chart
from splink import DuckDBAPI
db_api = DuckDBAPI()
completeness_chart(df, db_api=db_api)

The above summary chart shows that in this dataset, the `email`, `city`, `surname` and `forename` columns contain nulls, but the level of missingness is relatively low (less than 22%).


## Analyse the distribution of values in your data


The distribution of values in your data is important for two main reasons:

1. Columns with higher cardinality (number of distinct values) are usually more useful for data linking. For instance, date of birth is a much stronger linkage variable than gender.

2. The skew of values is important. If you have a `city` column that has 1,000 distinct values, but 75% of them are `London`, this is much less useful for linkage than if the 1,000 values were equally distributed

The `linker.profile_columns()` method creates summary charts to help you understand these aspects of your data.

To profile all columns, leave the column_expressions argument empty.


from splink.exploratory import profile_columns

profile_columns(df, db_api=DuckDBAPI(), top_n=10, bottom_n=5)

This chart is very information-dense, but here are some key takehomes relevant to our linkage:

- There is strong skew in the `city` field with around 20% of the values being `London`. We therefore will probably want to use `term_frequency_adjustments` in our linkage model, so that it can weight a match on London differently to a match on, say, `Norwich`.

- Looking at the "Bottom 5 values by value count", we can see typos in the data in most fields. This tells us this information was possibly entered by hand, or using Optical Character Recognition, giving us an insight into the type of data entry errors we may see.

- Email is a much more uniquely-identifying field than any others, with a maximum value count of 6. It's likely to be a strong linking variable.


!!! note "Further Reading"

    :material-tools: For more on exploratory analysis tools in Splink, please refer to the [Exploratory Analysis API documentation](../../api_docs/exploratory.md).

    :bar_chart: For more on the charts used in this tutorial, please refer to the [Charts Gallery](../../charts/index.md#exploratory-analysis).


## Next steps

At this point, we have begun to develop a strong understanding of our data. It's time to move on to estimating a linkage model




Contents of ../docs/demos/tutorials/01_Prerequisites.ipynb:
# Data Prerequisites


Splink requires that you clean your data and assign unique IDs to rows before linking. 

This section outlines the additional data cleaning steps needed before loading data into Splink.


### Unique IDs

- Each input dataset must have a unique ID column, which is unique within the dataset.  By default, Splink assumes this column will be called `unique_id`, but this can be changed with the [`unique_id_column_name`](https://moj-analytical-services.github.io/splink/api_docs/settings_dict_guide.html#unique_id_column_name) key in your Splink settings.  The unique id is essential because it enables Splink to keep track each row correctly. 

### Conformant input datasets

- Input datasets must be conformant, meaning they share the same column names and data formats. For instance, if one dataset has a "date of birth" column and another has a "dob" column, rename them to match. Ensure data type and number formatting are consistent across both columns. The order of columns in input dataframes is not important.

### Cleaning

- Ensure data consistency by cleaning your data. This process includes standardizing date formats, matching text case, and handling invalid data. For example, if one dataset uses "yyyy-mm-dd" date format and another uses "mm/dd/yyyy," convert them to the same format before using Splink.  Try also to identify and rectify any obvious data entry errors, such as removing values such as 'Mr' or 'Mrs' from a 'first name' column.

### Ensure nulls are consistently and correctly represented

- Ensure null values (or other 'not known' indicators) are represented as true nulls, not empty strings. Splink treats null values differently from empty strings, so using true nulls guarantees proper matching across datasets.

## Further details on data cleaning and standardisation

Splink performs optimally with cleaned and standardized data. Here is a non-exhaustive list of suggestions for data cleaning rules to enhance matching accuracy:

- Trim leading and trailing whitespace from string values (e.g., " john smith " becomes "john smith").
- Remove special characters from string values (e.g., "O'Hara" becomes "Ohara").
- Standardise date formats as strings in "yyyy-mm-dd" format.
- Replace abbreviations with full words (e.g., standardize "St." and "Street" to "Street").





Contents of ../docs/demos/tutorials/04_Estimating_model_parameters.ipynb:
## Specifying a linkage model

To build a linkage model, the user defines the partial match weights that `splink` needs to estimate. This is done by defining how the information in the input records should be compared.

To be concrete, here is an example comparison:

| first_name_l | first_name_r | surname_l | surname_r | dob_l      | dob_r      | city_l | city_r | email_l             | email_r             |
| ------------ | ------------ | --------- | --------- | ---------- | ---------- | ------ | ------ | ------------------- | ------------------- |
| Robert       | Rob          | Allen     | Allen     | 1971-05-24 | 1971-06-24 | nan    | London | roberta25@smith.net | roberta25@smith.net |

What functions should we use to assess the similarity of `Rob` vs. `Robert` in the the `first_name` field?

Should similarity in the `dob` field be computed in the same way, or a different way?

Your job as the developer of a linkage model is to decide what comparisons are most appropriate for the types of data you have.

Splink can then estimate how much weight to place on a fuzzy match of `Rob` vs. `Robert`, relative to an exact match on `Robert`, or a non-match.

Defining these scenarios is done using `Comparison`s.


### Comparisons

The concept of a `Comparison` has a specific definition within Splink: it defines how data from one or more input columns is compared.

For example, one `Comparison` may represent how similarity is assessed for a person's date of birth.

Another `Comparison` may represent the comparison of a person's name or location.

A model is composed of many `Comparison`s, which between them assess the similarity of all of the columns being used for data linking.

Each `Comparison` contains two or more `ComparisonLevels` which define _n_ discrete gradations of similarity between the input columns within the Comparison.

As such `ComparisonLevels`are nested within `Comparisons` as follows:

```
Data Linking Model
├─-- Comparison: Date of birth
│    ├─-- ComparisonLevel: Exact match
│    ├─-- ComparisonLevel: One character difference
│    ├─-- ComparisonLevel: All other
├─-- Comparison: Surname
│    ├─-- ComparisonLevel: Exact match on surname
│    ├─-- ComparisonLevel: All other
│    etc.
```

Our example data would therefore result in the following comparisons, for `dob` and `surname`:

| dob_l      | dob_r      | comparison_level         | interpretation |
| ---------- | ---------- | ------------------------ | -------------- |
| 1971-05-24 | 1971-05-24 | Exact match              | great match    |
| 1971-05-24 | 1971-06-24 | One character difference | fuzzy match    |
| 1971-05-24 | 2000-01-02 | All other                | bad match      |

<br/>

| surname_l | surname_r | comparison_level | interpretation                                        |
| --------- | --------- | ---------------- | ----------------------------------------------------- |
| Rob       | Rob       | Exact match      | great match                                           |
| Rob       | Jane      | All other        | bad match                                             |
| Rob       | Robert    | All other        | bad match, this comparison has no notion of nicknames |

More information about specifying comparisons can be found [here](../../topic_guides/comparisons/customising_comparisons.ipynb) and [here](../../topic_guides/comparisons/comparisons_and_comparison_levels.md).

We will now use these concepts to build a data linking model.


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

# Begin by reading in the tutorial data again
from splink import splink_datasets

df = splink_datasets.fake_1000

### Specifying the model using comparisons

Splink includes a library of comparison functions at `splink.comparison_library` to make it simple to get started. These are split into two categories:

1. Generic `Comparison` functions which apply a particular fuzzy matching function. For example, levenshtein distance.


import splink.comparison_library as cl

city_comparison = cl.LevenshteinAtThresholds("city", 2)
print(city_comparison.get_comparison("duckdb").human_readable_description)

2. `Comparison` functions tailored for specific data types. For example, email.

email_comparison = cl.EmailComparison("email")
print(email_comparison.get_comparison("duckdb").human_readable_description)

## Specifying the full settings dictionary

`Comparisons` are specified as part of the Splink `settings`, a Python dictionary which controls all of the configuration of a Splink model:


from splink import Linker, SettingsCreator, block_on, DuckDBAPI

settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.LevenshteinAtThresholds("dob", 1),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "city"),
        block_on("surname"),

    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=DuckDBAPI())

In words, this setting dictionary says:

- We are performing a `dedupe_only` (the other options are `link_only`, or `link_and_dedupe`, which may be used if there are multiple input datasets).
- When comparing records, we will use information from the `first_name`, `surname`, `dob`, `city` and `email` columns to compute a match score.
- The `blocking_rules_to_generate_predictions` states that we will only check for duplicates amongst records where either the `first_name AND city` or `surname` is identical.
- We have enabled [term frequency adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html) for the 'city' column, because some values (e.g. `London`) appear much more frequently than others.
- We have set `retain_intermediate_calculation_columns` and `additional_columns_to_retain` to `True` so that Splink outputs additional information that helps the user understand the calculations. If they were `False`, the computations would run faster.


## Estimate the parameters of the model

Now that we have specified our linkage model, we need to estimate the [`probability_two_random_records_match`](../../api_docs//settings_dict_guide.md#probability_two_random_records_match), `u`, and `m` parameters.

- The `probability_two_random_records_match` parameter is the probability that two records taken at random from your input data represent a match (typically a very small number).

- The `u` values are the proportion of records falling into each `ComparisonLevel` amongst truly _non-matching_ records.

- The `m` values are the proportion of records falling into each `ComparisonLevel` amongst truly _matching_ records

You can read more about [the theory of what these mean](https://www.robinlinacre.com/m_and_u_values/).

We can estimate these parameters using unlabeled data. If we have labels, then we can estimate them even more accurately.

The rationale for the approach recommended in this tutorial is documented [here](../../topic_guides/training/training_rationale.md).


### Estimation of `probability_two_random_records_match`

In some cases, the `probability_two_random_records_match` will be known. For example, if you are linking two tables of 10,000 records and expect a one-to-one match, then you should set this value to `1/10_000` in your settings instead of estimating it.

More generally, this parameter is unknown and needs to be estimated.

It can be estimated accurately enough for most purposes by combining a series of deterministic matching rules and a guess of the recall corresponding to those rules. For further details of the rationale behind this appraoch see [here](https://github.com/moj-analytical-services/splink/issues/462#issuecomment-1227027995).

In this example, I guess that the following deterministic matching rules have a recall of about 70%. That means, between them, the rules recover 70% of all true matches.


deterministic_rules = [
    block_on("first_name", "dob"),
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    block_on("email")
]

linker.training.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)

### Estimation of `u` probabilities

Once we have the `probability_two_random_records_match` parameter, we can estimate the `u` probabilities.

We estimate `u` using the `estimate_u_using_random_sampling` method, which doesn't require any labels.

It works by sampling random pairs of records, since most of these pairs are going to be non-matches. Over these non-matches we compute the distribution of `ComparisonLevel`s for each `Comparison`.

For instance, for `gender`, we would find that the the gender matches 50% of the time, and mismatches 50% of the time.

For `dob` on the other hand, we would find that the `dob` matches 1% of the time, has a "one character difference" 3% of the time, and everything else happens 96% of the time.

The larger the random sample, the more accurate the predictions. You control this using the `max_pairs` parameter. For large datasets, we recommend using at least 10 million - but the higher the better and 1 billion is often appropriate for larger datasets.


linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

### Estimation of `m` probabilities

`m` is the trickiest of the parameters to estimate, because we have to have some idea of what the true matches are.

If we have labels, we can directly estimate it.

If we do not have labelled data, the `m` parameters can be estimated using an iterative maximum likelihood approach called Expectation Maximisation.

#### Estimating directly

If we have labels, we can estimate `m` directly using the `estimate_m_from_label_column` method of the linker.

For example, if the entity being matched is persons, and your input dataset(s) contain social security number, this could be used to estimate the m values for the model.

Note that this column does not need to be fully populated. A common case is where a unique identifier such as social security number is only partially populated.

For example (in this tutorial we don't have labels, so we're not actually going to use this):

```python
linker.estimate_m_from_label_column("social_security_number")
```

#### Estimating with Expectation Maximisation

This algorithm estimates the `m` values by generating pairwise record comparisons, and using them to maximise a likelihood function.

Each estimation pass requires the user to configure an estimation blocking rule to reduce the number of record comparisons generated to a manageable level.

In our first estimation pass, we block on `first_name` and `surname`, meaning we will generate all record comparisons that have `first_name` and `surname` exactly equal.

Recall we are trying to estimate the `m` values of the model, i.e. proportion of records falling into each `ComparisonLevel` amongst truly matching records.

This means that, in this training session, we cannot estimate parameter estimates for the `first_name` or `surname` columns, since they will be equal for all the comparisons we do.

We can, however, estimate parameter estimates for all of the other columns. The output messages produced by Splink confirm this.


training_blocking_rule = block_on("first_name", "surname")
training_session_fname_sname = (
    linker.training.estimate_parameters_using_expectation_maximisation(training_blocking_rule)
)

In a second estimation pass, we block on dob. This allows us to estimate parameters for the `first_name` and `surname` comparisons.

Between the two estimation passes, we now have parameter estimates for all comparisons.


training_blocking_rule = block_on("dob")
training_session_dob = linker.training.estimate_parameters_using_expectation_maximisation(
    training_blocking_rule
)

Note that Splink includes other algorithms for estimating m and u values, which are documented [here](https://moj-analytical-services.github.io/splink/api_docs/training.html).


## Visualising model parameters

Splink can generate a number of charts to help you understand your model. For an introduction to these charts and how to interpret them, please see [this](https://www.youtube.com/watch?v=msz3T741KQI&t=507s) video.

The final estimated match weights can be viewed in the match weights chart:


linker.visualisations.match_weights_chart()

linker.visualisations.m_u_parameters_chart()

We can also compare the estimates that were produced by the different EM training sessions

linker.visualisations.parameter_estimate_comparisons_chart()

### Saving the model

We can save the model, including our estimated parameters, to a `.json` file, so we can use it in the next tutorial.


settings = linker.misc.save_model_to_json(
    "../demo_settings/saved_model_from_demo.json", overwrite=True
)

## Detecting unlinkable records

An interesting application of our trained model that is useful to explore before making any predictions is to detect 'unlinkable' records.

Unlinkable records are those which do not contain enough information to be linked. A simple example would be a record containing only 'John Smith', and null in all other fields. This record may link to other records, but we'll never know because there's not enough information to disambiguate any potential links. Unlinkable records can be found by linking records to themselves - if, even when matched to themselves, they don't meet the match threshold score, we can be sure they will never link to anything.


linker.evaluation.unlinkables_chart()

In the above chart, we can see that about 1.3% of records in the input dataset are unlinkable at a threshold match weight of 6.11 (correponding to a match probability of around 98.6%)


!!! note "Further Reading"

    :material-tools: For more on the model estimation tools in Splink, please refer to the [Model Training API documentation](../../api_docs/training.md).

    :simple-readme: For a deeper dive on:

    * choosing comparisons, please refer to the [Comparisons Topic Guides](../../topic_guides/comparisons/customising_comparisons.ipynb)
    * the underlying model theory, please refer to the [Fellegi Sunter Topic Guide](../../topic_guides/theory/fellegi_sunter.md)
    * model training, please refer to the Model Training Topic Guides (Coming Soon).

    :bar_chart: For more on the charts used in this tutorial, please refer to the [Charts Gallery](../../charts/index.md).


## Next steps

Now we have trained a model, we can move on to using it predict matching records.




Contents of ../docs/demos/tutorials/05_Predicting_results.ipynb:
# Predicting which records match

<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/05_Predicting_results.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

In the previous tutorial, we built and estimated a linkage model.

In this tutorial, we will load the estimated model and use it to make predictions of which pairwise record comparisons match.


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import Linker, DuckDBAPI, splink_datasets

import pandas as pd

pd.options.display.max_columns = 1000

db_api = DuckDBAPI()
df = splink_datasets.fake_1000

## Load estimated model from previous tutorial


import json
import urllib

url = "https://raw.githubusercontent.com/moj-analytical-services/splink/847e32508b1a9cdd7bcd2ca6c0a74e547fb69865/docs/demos/demo_settings/saved_model_from_demo.json"

with urllib.request.urlopen(url) as u:
    settings = json.loads(u.read().decode())


linker = Linker(df, settings, db_api=DuckDBAPI())

# Predicting match weights using the trained model

We use `linker.predict()` to run the model.

Under the hood this will:

- Generate all pairwise record comparisons that match at least one of the `blocking_rules_to_generate_predictions`

- Use the rules specified in the `Comparisons` to evaluate the similarity of the input data

- Use the estimated match weights, applying term frequency adjustments where requested to produce the final `match_weight` and `match_probability` scores

Optionally, a `threshold_match_probability` or `threshold_match_weight` can be provided, which will drop any row where the predicted score is below the threshold.


df_predictions = linker.inference.predict(threshold_match_probability=0.2)
df_predictions.as_pandas_dataframe(limit=5)

## Clustering

The result of `linker.predict()` is a list of pairwise record comparisons and their associated scores. For instance, if we have input records A, B, C and D, it could be represented conceptually as:

```
A -> B with score 0.9
B -> C with score 0.95
C -> D with score 0.1
D -> E with score 0.99
```

Often, an alternative representation of this result is more useful, where each row is an input record, and where records link, they are assigned to the same cluster.

With a score threshold of 0.5, the above data could be represented conceptually as:

```
ID, Cluster ID
A,  1
B,  1
C,  1
D,  2
E,  2
```

The algorithm that converts between the pairwise results and the clusters is called connected components, and it is included in Splink. You can use it as follows:


clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predictions, threshold_match_probability=0.5
)
clusters.as_pandas_dataframe(limit=10)

sql = f"""
select *
from {df_predictions.physical_name}
limit 2
"""
linker.misc.query_sql(sql)

!!! note "Further Reading"
:material-tools: For more on the prediction tools in Splink, please refer to the [Prediction API documentation](../../api_docs/inference.md).


## Next steps

Now we have made predictions with a model, we can move on to visualising it to understand how it is working.




Contents of ../docs/demos/tutorials/06_Visualising_predictions.ipynb:
# Visualising predictions

<a target="_blank" href="https://colab.research.google.com/github/moj-analytical-services/splink/blob/master/docs/demos/tutorials/06_Visualising_predictions.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Splink contains a variety of tools to help you visualise your predictions.

The idea is that, by developing an understanding of how your model works, you can gain confidence that the predictions it makes are sensible, or alternatively find examples of where your model isn't working, which may help you improve the model specification and fix these problems.


# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

# Rerun our predictions to we're ready to view the charts
from splink import Linker, DuckDBAPI, splink_datasets

import pandas as pd

pd.options.display.max_columns = 1000

db_api = DuckDBAPI()
df = splink_datasets.fake_1000

import json
import urllib

url = "https://raw.githubusercontent.com/moj-analytical-services/splink/847e32508b1a9cdd7bcd2ca6c0a74e547fb69865/docs/demos/demo_settings/saved_model_from_demo.json"

with urllib.request.urlopen(url) as u:
    settings = json.loads(u.read().decode())


linker = Linker(df, settings, db_api=DuckDBAPI())
df_predictions = linker.inference.predict(threshold_match_probability=0.2)

## Waterfall chart

The waterfall chart provides a means of visualising individual predictions to understand how Splink computed the final matchweight for a particular pairwise record comparison.

To plot a waterfall chart, the user chooses one or more records from the results of `linker.inference.predict()`, and provides these records to the [`linker.visualisations.waterfall_chart()`](https://moj-analytical-services.github.io/splink/api_docs/visualisations.html#splink.internals.linker_components.visualisations.LinkerVisualisations.waterfall_chart) function.

For an introduction to waterfall charts and how to interpret them, please see [this](https://www.youtube.com/watch?v=msz3T741KQI&t=507s) video.


records_to_view = df_predictions.as_record_dict(limit=5)
linker.visualisations.waterfall_chart(records_to_view, filter_nulls=False)

## Comparison viewer dashboard

The [comparison viewer dashboard](https://moj-analytical-services.github.io/splink/api_docs/visualisations.html#splink.internals.linker_components.visualisations.LinkerVisualisations.comparison_viewer_dashboard) takes this one step further by producing an interactive dashboard that contains example predictions from across the spectrum of match scores.

An in-depth video describing how to interpret the dashboard can be found [here](https://www.youtube.com/watch?v=DNvCMqjipis).


linker.visualisations.comparison_viewer_dashboard(df_predictions, "scv.html", overwrite=True)

# You can view the scv.html file in your browser, or inline in a notbook as follows
from IPython.display import IFrame

IFrame(src="./scv.html", width="100%", height=1200)

## Cluster studio dashboard

Cluster studio is an interactive dashboards that visualises the results of clustering your predictions.

It provides examples of clusters of different sizes. The shape and size of clusters can be indicative of problems with record linkage, so it provides a tool to help you find potential false positive and negative links.


df_clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predictions, threshold_match_probability=0.5
)

linker.visualisations.cluster_studio_dashboard(
    df_predictions,
    df_clusters,
    "cluster_studio.html",
    sampling_method="by_cluster_size",
    overwrite=True,
)

# You can view the scv.html file in your browser, or inline in a notbook as follows
from IPython.display import IFrame

IFrame(src="./cluster_studio.html", width="100%", height=1000)

!!! note "Further Reading"

    :material-tools: For more on the visualisation tools in Splink, please refer to the [Visualisation API documentation](../../api_docs/visualisations.md).

    :bar_chart: For more on the charts used in this tutorial, please refer to the [Charts Gallery](../../charts/index.md#model-training.)


## Next steps

Now we have visualised the results of a model, we can move on to some more formal Quality Assurance procedures using labelled data.






Contents of /docs/index.md:
---
hide:
  - navigation
  - toc
---

<p align="center">
<img src="https://user-images.githubusercontent.com/7570107/85285114-3969ac00-b488-11ea-88ff-5fca1b34af1f.png" alt="Splink: data linkage at scale. (Splink logo)." style="max-width: 500px;">
</p>

# Fast, accurate and scalable probabilistic data linkage

Splink is a Python package for probabilistic record linkage (entity resolution) that allows you to deduplicate and link records from datasets without unique identifiers.

[Get Started with Splink](./getting_started.md){ .md-button .md-button--primary }

<hr>

## Key Features

⚡ **Speed:** Capable of linking a million records on a laptop in approximately one minute.<br>
🎯 **Accuracy:** Full support for term frequency adjustments and user-defined fuzzy matching logic.<br>
🌐 **Scalability:** Execute linkage jobs in Python (using DuckDB) or big-data backends like AWS Athena or Spark for 100+ million records.<br>
🎓 **Unsupervised Learning:** No training data is required, as models can be trained using an unsupervised approach.<br>
📊 **Interactive Outputs:** Provides a wide range of interactive outputs to help users understand their model and diagnose linkage problems.<br>

Splink's core linkage algorithm is based on Fellegi-Sunter's model of record linkage, with various customizations to improve accuracy.

## What does Splink do?

Consider the following records that lack a unique person identifier:

![tables showing what Splink does](https://raw.githubusercontent.com/moj-analytical-services/splink/master/docs/img/README/what_does_splink_do_1.drawio.png)

Splink predicts which rows link together:

![tables showing what Splink does](https://raw.githubusercontent.com/moj-analytical-services/splink/master/docs/img/README/what_does_splink_do_2.drawio.png)

and clusters these links to produce an estimated person ID:

![tables showing what Splink does](https://raw.githubusercontent.com/moj-analytical-services/splink/master/docs/img/README/what_does_splink_do_3.drawio.png)

## What data does Splink work best with?

Before using Splink, input data should be standardised, with consistent column names and formatting (e.g., lowercased, punctuation cleaned up, etc.).

Splink performs best with input data containing **multiple** columns that are **not highly correlated**. For instance, if the entity type is persons, you may have columns for full name, date of birth, and city. If the entity type is companies, you could have columns for name, turnover, sector, and telephone number.

High correlation occurs when the value of a column is highly constrained (predictable) from the value of another column. For example, a 'city' field is almost perfectly correlated with 'postcode'. Gender is highly correlated with 'first name'. Correlation is particularly problematic if **all** of your input columns are highly correlated.

Splink is not designed for linking a single column containing a 'bag of words'. For example, a table with a single 'company name' column, and no other details.

## Support

If after reading the documentatation you still have questions, please feel free to post on our [discussion forum](https://github.com/moj-analytical-services/splink/discussions).

## Use Cases

Here is a list of some of our known users and their use cases:

=== "Public Sector (UK)"

	- [Ministry of Justice](https://www.gov.uk/government/organisations/ministry-of-justice) created [linked datasets (combining courts, prisons and probation data)](https://www.adruk.org/our-work/browse-all-projects/data-first-harnessing-the-potential-of-linked-administrative-data-for-the-justice-system-169/) for use by researchers as part of the [Data First programme](https://www.gov.uk/guidance/ministry-of-justice-data-first)
	- [Office for National Statistics](https://www.ons.gov.uk/)'s [Business Index](https://unece.org/sites/default/files/2023-04/ML2023_S1_UK_Breton_A.pdf) (formerly the Inter Departmental Business Register), [Demographic Index](https://uksa.statisticsauthority.gov.uk/wp-content/uploads/2023/02/EAP182-Quality-work-for-Demographic-Index-MDQA.pdf) and the [2021 Census](https://github.com/Data-Linkage/Splink-census-linkage/blob/main/SplinkCaseStudy.pdf).  See also [this article](https://www.government-transformation.com/data/interview-modernizing-public-sector-insight-through-automated-linkage).
	- [Lewisham Council](https://lewisham.gov.uk/) (London) [identified and auto-enrolled over 500 additional eligible families](https://lewisham.gov.uk/articles/news/extra-funding-for-lewisham-schools-in-pilot-data-project) to receive Free School Meals
	- [London Office of Technology and Innovation](https://loti.london/) created a dashboard to help [better measure and reduce rough sleeping](https://loti.london/projects/rough-sleeping-insights-project/) across London
	- [Competition and Markets Authority](https://www.gov.uk/government/organisations/competition-and-markets-authority) identified ['Persons with Significant Control' and estimated ownership groups](https://assets.publishing.service.gov.uk/media/626ab6c4d3bf7f0e7f9d5a9b/220426_Annex_-State_of_Competition_Appendices_FINAL.pdf) across companies
	- [Office for Health Improvement and Disparities](https://www.gov.uk/government/organisations/office-for-health-improvement-and-disparities) linked Health and Justice data to [assess the pathways between probation and specialist alcohol and drug treatment services](https://www.gov.uk/government/statistics/pathways-between-probation-and-addiction-treatment-in-england#:~:text=Details,of%20Health%20and%20Social%20Care) as part of the [Better Outcomes through Linked Data programme](https://www.gov.uk/government/publications/ministry-of-justice-better-outcomes-through-linked-data-bold)
	- [Ministry of Defence](https://www.gov.uk/government/organisations/ministry-of-defence) recently launched their [Veteran's Card system](https://www.gov.uk/government/news/hm-armed-forces-veteran-cards-will-officially-launch-in-the-new-year-following-a-successful-assessment-from-the-central-digital-and-data-office) which uses Splink to verify applicants against historic records. This project was shortlisted for the [Civil Service Awards](https://www.civilserviceawards.com/creative-solutions-award/)
 	- [Gateshead Council](https://www.gateshead.gov.uk/), in partnership with the [National Innovation Centre for Data](https://www.nicd.org.uk/) are creating a [single view of debt](https://nicd.org.uk/knowledge-hub/an-end-to-end-guide-to-overcoming-unique-identifier-challenges-with-splink)
	- The Defense Health Agency (US Department of Defense) used Splink to identify duplicated hospital records across over 200 million data points in the military hospital data system


=== "Public Sector (International)"

	- The German Federal Statistical Office ([Destatis](https://www.destatis.de/EN/Home/_node.html)) uses Splink to conduct projects in linking register-based census data.
	- The [European Medicines Agency](https://www.ema.europa.eu/en/homepage) uses Splink to detect duplicate adverse event reports for veterinary medicines
	- [Chilean Ministry of Health](https://www.gob.cl/en/ministries/ministry-of-health/) and [University College London](https://www.ucl.ac.uk/) have [assessed the access to immunisation programs among the migrant population](https://ijpds.org/article/view/2348)
	- [Florida Cancer Registry](https://www.floridahealth.gov/diseases-and-conditions/cancer/cancer-registry/index.html), published a [feasibility study](https://scholar.googleusercontent.com/scholar?q=cache:sADwxy-D75IJ:scholar.google.com/+splink+florida&hl=en&as_sdt=0,5) which showed Splink was faster and more accurate than alternatives
	- [Catalyst Cooperative](https://catalyst.coop)'s [Public Utility Data Liberation Project](https://github.com/catalyst-cooperative/pudl) links public financial and operational data from electric utilities for use by US climate advocates, policymakers, and researchers seeking to accelerate the transition away from fossil fuels.

=== "Academia"

	- [Stanford University](https://www.stanford.edu/) investigated the impact of [receiving government assistance has on political attitudes](https://www.cambridge.org/core/journals/american-political-science-review/article/abs/does-receiving-government-assistance-shape-political-attitudes-evidence-from-agricultural-producers/39552BC5A496EAB6CB484FCA51C6AF21)
	- [Bern University](https://arbor.bfh.ch/) researched how [Active Learning can be applied to Biomedical Record Linkage](https://ebooks.iospress.nl/doi/10.3233/SHTI230545)

=== "Other"
	- [Marie Curie](https://podcasts.apple.com/gb/podcast/unlocking-data-at-marie-curie/id1724979056?i=1000649964922) have used Splink to build a single customer view on fundraising data which has been a "huge success [...] the tooling is just so much better. [...] The power of being able to select, plug in, configure and train a tool versus writing code. It's just mind boggling actually."  Amongst other benefits, the system is expected to "dramatically reduce manual reporting efforts previously required". See also the blog post [here](https://esynergy.co.uk/our-work/marie-curie/).
 	- [Club Brugge](https://www.clubbrugge.be/en) uses Splink to link football players from different data providers to their own database, simplifying and reducing the need for manual linkage labor.
	- [GN Group](https://www.gn.com/) use Splink to deduplicate large volumes of customer records

Sadly, we don't hear about the majority of our users or what they are working on. If you have a use case and it is not shown here please [add it to the list](https://github.com/moj-analytical-services/splink/edit/master/docs/index.md)!

## Awards

🥈 Civil Service Awards 2023: Best Use of Data, Science, and Technology - [Runner up](https://www.civilserviceawards.com/best-use-of-data-science-and-technology-award-2/)

🥇 Analysis in Government Awards 2022: People's Choice Award - [Winner](https://analysisfunction.civilservice.gov.uk/news/announcing-the-winner-of-the-first-analysis-in-government-peoples-choice-award/)

🥈 Analysis in Government Awards 2022: Innovative Methods - [Runner up](https://twitter.com/gov_analysis/status/1616073633692274689?s=20&t=6TQyNLJRjnhsfJy28Zd6UQ)

🥇 Analysis in Government Awards 2020: Innovative Methods - [Winner](https://www.gov.uk/government/news/launch-of-the-analysis-in-government-awards)

🥇 Ministry of Justice Data and Analytical Services Directorate (DASD) Awards 2020: Innovation and Impact - Winner


## Citation

If you use Splink in your research, we'd be grateful for a citation as follows:

```BibTeX
@article{Linacre_Lindsay_Manassis_Slade_Hepworth_2022,
	title        = {Splink: Free software for probabilistic record linkage at scale.},
	author       = {Linacre, Robin and Lindsay, Sam and Manassis, Theodore and Slade, Zoe and Hepworth, Tom and Kennedy, Ross and Bond, Andrew},
	year         = 2022,
	month        = {Aug.},
	journal      = {International Journal of Population Data Science},
	volume       = 7,
	number       = 3,
	doi          = {10.23889/ijpds.v7i3.1794},
	url          = {https://ijpds.org/article/view/1794},
}
```

## Acknowledgements

We are very grateful to [ADR UK](https://www.adruk.org/) (Administrative Data Research UK) for providing the initial funding for this work as part of the [Data First](https://www.adruk.org/our-work/browse-all-projects/data-first-harnessing-the-potential-of-linked-administrative-data-for-the-justice-system-169/) project.

We are extremely grateful to professors Katie Harron, James Doidge and Peter Christen for their expert advice and guidance in the development of Splink. We are also very grateful to colleagues at the UK's Office for National Statistics for their expert advice and peer review of this work. Any errors remain our own.




Contents of /docs/getting_started.md:
---
hide:
  - navigation
---

# Getting Started

## :material-download: Install
Splink supports python 3.8+.

To obtain the latest released version of Splink you can install from PyPI using pip:
```shell
pip install splink
```

or if you prefer, you can instead install Splink using conda:
```shell
conda install -c conda-forge splink
```

??? "Backend Specific Installs"
    ### Backend Specific Installs
    From Splink v3.9.7, packages required by specific Splink backends can be optionally installed by adding the `[<backend>]` suffix to the end of your pip install.

    **Note** that SQLite and DuckDB come packaged with Splink and do not need to be optionally installed.

    The following backends are supported:

    === ":simple-apachespark: Spark"
        ```sh
        pip install 'splink[spark]'
        ```

    === ":simple-amazonaws: Athena"
        ```sh
        pip install 'splink[athena]'
        ```

    === ":simple-postgresql: PostgreSQL"
        ```sh
        pip install 'splink[postgres]'
        ```



## :rocket: Quickstart

To get a basic Splink model up and running, use the following code. It demonstrates how to:

1. Estimate the parameters of a deduplication model
2. Use the parameter estimates to identify duplicate records
3. Use clustering to generate an estimated unique person ID.

???+ note "Simple Splink Model Example"
    ```py
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets

    db_api = DuckDBAPI()

    df = splink_datasets.fake_1000

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cl.NameComparison("first_name"),
            cl.JaroAtThresholds("surname"),
            cl.DateOfBirthComparison(
                "dob",
                input_is_string=True,
            ),
            cl.ExactMatch("city").configure(term_frequency_adjustments=True),
            cl.EmailComparison("email"),
        ],
        blocking_rules_to_generate_predictions=[
            block_on("first_name", "dob"),
            block_on("surname"),
        ]
    )

    linker = Linker(df, settings, db_api)

    linker.training.estimate_probability_two_random_records_match(
        [block_on("first_name", "surname")],
        recall=0.7,
    )

    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("first_name", "surname")
    )

    linker.training.estimate_parameters_using_expectation_maximisation(block_on("email"))

    pairwise_predictions = linker.inference.predict(threshold_match_weight=-5)

    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        pairwise_predictions, 0.95
    )

    df_clusters = clusters.as_pandas_dataframe(limit=5)
    ```

If you're using an LLM to suggest Splink code, see [here](./topic_guides/llms/prompting_llms.md) for suggested prompts and context.

## Tutorials

You can learn more about Splink in the step-by-step [tutorial](./demos/tutorials/00_Tutorial_Introduction.ipynb). Each has a corresponding Google Colab link to run the notebook in your browser.

## Example Notebooks

You can see end-to-end example of several use cases in the [example notebooks](./demos/examples/examples_index.md). Each has a corresponding Google Colab link to run the notebook in your browser.

## Getting help

If after reading the documentatation you still have questions, please feel free to post on our [discussion forum](https://github.com/moj-analytical-services/splink/discussions).



Contents of /docs/topic_guides/splink_fundamentals/settings.md:
---
tags:
  - settings
  - Dedupe
  - Link
  - Link and Dedupe
  - Comparisons
  - Blocking Rules
---

# Defining a Splink Model

## What makes a Splink Model?

When building any linkage model in Splink, there are 3 key things which need to be defined:

1. What **type of linkage** you want (defined by the [link type](link_type.md))
2. What **pairs of records** to consider (defined by [blocking rules](../blocking/blocking_rules.md))
3. What **features** to consider, and how they should be **compared** (defined by [comparisons](../comparisons/customising_comparisons.ipynb))


## Defining a Splink model with a settings dictionary

All aspects of a Splink model are defined via the `SettingsCreator` object.

For example, consider a simple model:

```py linenums="1"
import splink.comparison_library as cl
import splink.comparison_template_library as ctl

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
    ],
    comparisons=[
        ctl.NameComparison("first_name"),
        ctl.NameComparison("surname"),
        ctl.DateComparison(
            "dob",
            input_is_string=True,
            datetime_metrics=["month", "year"],
            datetime_thresholds=[
                1,
                1,
            ],
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        ctl.EmailComparison("email"),
    ],
)
```

Where:

**1. Type of linkage**

The `"link_type"` is defined as a deduplication for a single dataset.

```py linenums="5"
    link_type="dedupe_only",
```

**2. Pairs of records to consider**

The `"blocking_rules_to_generate_predictions"` define a subset of pairs of records for the model to be considered when making predictions. In this case, where there is a match on:

- `first_name`
- OR (`surname` AND `dob`).

```py linenums="6"
    blocking_rules_to_generate_predictions=[
            block_on("first_name"),
            block_on("surname", "dob"),
        ],
```

For more information on how blocking is used in Splink, see the [dedicated topic guide](../blocking/blocking_rules.md).

**3. Features to consider, and how they should be compared**

The `"comparisons"` define the features to be compared between records: `"first_name"`, `"surname"`, `"dob"`, `"city"` and `"email"`.

```py linenums="10"
    comparisons=[
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.DateComparison(
            "dob",
            input_is_string=True,
            datetime_metrics=["month", "year"],
            datetime_thresholds=[
                1,
                1,
            ],
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
```

Using functions from the [comparison library](../comparisons/customising_comparisons.ipynb#method-1-using-the-comparisonlibrary) to define **how** these features should be compared.



For more information on how comparisons are defined, see the [dedicated topic guide](../comparisons/customising_comparisons.ipynb).






With our finalised settings object, we can train a Splink model using the following code:

??? example "Example model using the settings dictionary"

    ```py
    import splink.comparison_library as cl
    import splink.comparison_template_library as ctl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets

    db_api = DuckDBAPI()
    df = splink_datasets.fake_1000

    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[
            block_on("first_name"),
            block_on("surname"),
        ],
        comparisons=[
            ctl.NameComparison("first_name"),
            ctl.NameComparison("surname"),
            ctl.DateComparison(
                "dob",
                input_is_string=True,
                datetime_metrics=["month", "year"],
                datetime_thresholds=[
                    1,
                    1,
                ],
            ),
            cl.ExactMatch("city").configure(term_frequency_adjustments=True),
            ctl.EmailComparison("email"),
        ],
    )

    linker = Linker(df, settings, db_api=db_api)
    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

    blocking_rule_for_training = block_on("first_name", "surname")
    linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule_for_training)

    blocking_rule_for_training = block_on("dob")
    linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule_for_training)

    pairwise_predictions = linker.inference.predict()

    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(pairwise_predictions, 0.95)
    clusters.as_pandas_dataframe(limit=5)

    ```



## Advanced usage of the settings dictionary

The section above refers to the three key aspects of the Splink settings dictionary. There are a variety of other lesser used settings, which can be found as the arguments to the `SettingsCreator`



## Saving a trained model

Once you have have a trained Splink model, it is often helpful to save out the model. The `save_model_to_json` function allows the user to save out the specifications of their trained model.

```py
linker.misc.save_model_to_json("model.json")
```

which, using the example settings and model training from above, gives the following output:

??? note "Model JSON"

    When the splink model is saved to disk using `linker.misc.save_model_to_json("model.json")` these settings become:


    ```json
    {
        "link_type": "dedupe_only",
        "probability_two_random_records_match": 0.0008208208208208208,
        "retain_matching_columns": true,
        "retain_intermediate_calculation_columns": false,
        "additional_columns_to_retain": [],
        "sql_dialect": "duckdb",
        "linker_uid": "29phy7op",
        "em_convergence": 0.0001,
        "max_iterations": 25,
        "bayes_factor_column_prefix": "bf_",
        "term_frequency_adjustment_column_prefix": "tf_",
        "comparison_vector_value_column_prefix": "gamma_",
        "unique_id_column_name": "unique_id",
        "source_dataset_column_name": "source_dataset",
        "blocking_rules_to_generate_predictions": [
            {
                "blocking_rule": "l.\"first_name\" = r.\"first_name\"",
                "sql_dialect": "duckdb"
            },
            {
                "blocking_rule": "l.\"surname\" = r.\"surname\"",
                "sql_dialect": "duckdb"
            }
        ],
        "comparisons": [
            {
                "output_column_name": "first_name",
                "comparison_levels": [
                    {
                        "sql_condition": "\"first_name_l\" IS NULL OR \"first_name_r\" IS NULL",
                        "label_for_charts": "first_name is NULL",
                        "is_null_level": true
                    },
                    {
                        "sql_condition": "\"first_name_l\" = \"first_name_r\"",
                        "label_for_charts": "Exact match on first_name",
                        "m_probability": 0.48854806009621365,
                        "u_probability": 0.0056770619302010565
                    },
                    {
                        "sql_condition": "jaro_winkler_similarity(\"first_name_l\", \"first_name_r\") >= 0.9",
                        "label_for_charts": "Jaro-Winkler distance of first_name >= 0.9",
                        "m_probability": 0.1903763096120358,
                        "u_probability": 0.003424501164330396
                    },
                    {
                        "sql_condition": "jaro_winkler_similarity(\"first_name_l\", \"first_name_r\") >= 0.8",
                        "label_for_charts": "Jaro-Winkler distance of first_name >= 0.8",
                        "m_probability": 0.08609678978546921,
                        "u_probability": 0.006620702251038765
                    },
                    {
                        "sql_condition": "ELSE",
                        "label_for_charts": "All other comparisons",
                        "m_probability": 0.23497884050628137,
                        "u_probability": 0.9842777346544298
                    }
                ],
                "comparison_description": "jaro_winkler at thresholds 0.9, 0.8 vs. anything else"
            },
            {
                "output_column_name": "surname",
                "comparison_levels": [
                    {
                        "sql_condition": "\"surname_l\" IS NULL OR \"surname_r\" IS NULL",
                        "label_for_charts": "surname is NULL",
                        "is_null_level": true
                    },
                    {
                        "sql_condition": "\"surname_l\" = \"surname_r\"",
                        "label_for_charts": "Exact match on surname",
                        "m_probability": 0.43210610613512185,
                        "u_probability": 0.004322481469643699
                    },
                    {
                        "sql_condition": "jaro_winkler_similarity(\"surname_l\", \"surname_r\") >= 0.9",
                        "label_for_charts": "Jaro-Winkler distance of surname >= 0.9",
                        "m_probability": 0.2514700606335103,
                        "u_probability": 0.002907020988387136
                    },
                    {
                        "sql_condition": "jaro_winkler_similarity(\"surname_l\", \"surname_r\") >= 0.8",
                        "label_for_charts": "Jaro-Winkler distance of surname >= 0.8",
                        "m_probability": 0.0757748206402343,
                        "u_probability": 0.0033636211436311888
                    },
                    {
                        "sql_condition": "ELSE",
                        "label_for_charts": "All other comparisons",
                        "m_probability": 0.2406490125911336,
                        "u_probability": 0.989406876398338
                    }
                ],
                "comparison_description": "jaro_winkler at thresholds 0.9, 0.8 vs. anything else"
            },
            {
                "output_column_name": "dob",
                "comparison_levels": [
                    {
                        "sql_condition": "\"dob_l\" IS NULL OR \"dob_r\" IS NULL",
                        "label_for_charts": "dob is NULL",
                        "is_null_level": true
                    },
                    {
                        "sql_condition": "\"dob_l\" = \"dob_r\"",
                        "label_for_charts": "Exact match on dob",
                        "m_probability": 0.39025358731716286,
                        "u_probability": 0.0016036280808555408
                    },
                    {
                        "sql_condition": "damerau_levenshtein(\"dob_l\", \"dob_r\") <= 1",
                        "label_for_charts": "Damerau-Levenshtein distance of dob <= 1",
                        "m_probability": 0.1489444378965258,
                        "u_probability": 0.0016546990388445707
                    },
                    {
                        "sql_condition": "ABS(EPOCH(try_strptime(\"dob_l\", '%Y-%m-%d')) - EPOCH(try_strptime(\"dob_r\", '%Y-%m-%d'))) <= 2629800.0",
                        "label_for_charts": "Abs difference of 'transformed dob <= 1 month'",
                        "m_probability": 0.08866691175438302,
                        "u_probability": 0.002594404665842722
                    },
                    {
                        "sql_condition": "ABS(EPOCH(try_strptime(\"dob_l\", '%Y-%m-%d')) - EPOCH(try_strptime(\"dob_r\", '%Y-%m-%d'))) <= 31557600.0",
                        "label_for_charts": "Abs difference of 'transformed dob <= 1 year'",
                        "m_probability": 0.10518866178811104,
                        "u_probability": 0.030622146410222362
                    },
                    {
                        "sql_condition": "ELSE",
                        "label_for_charts": "All other comparisons",
                        "m_probability": 0.26694640124381713,
                        "u_probability": 0.9635251218042348
                    }
                ],
                "comparison_description": "Exact match vs. Damerau-Levenshtein distance <= 1 vs. month difference <= 1 vs. year difference <= 1 vs. anything else"
            },
            {
                "output_column_name": "city",
                "comparison_levels": [
                    {
                        "sql_condition": "\"city_l\" IS NULL OR \"city_r\" IS NULL",
                        "label_for_charts": "city is NULL",
                        "is_null_level": true
                    },
                    {
                        "sql_condition": "\"city_l\" = \"city_r\"",
                        "label_for_charts": "Exact match on city",
                        "m_probability": 0.561103053663773,
                        "u_probability": 0.052019405886043986,
                        "tf_adjustment_column": "city",
                        "tf_adjustment_weight": 1.0
                    },
                    {
                        "sql_condition": "ELSE",
                        "label_for_charts": "All other comparisons",
                        "m_probability": 0.438896946336227,
                        "u_probability": 0.947980594113956
                    }
                ],
                "comparison_description": "Exact match 'city' vs. anything else"
            },
            {
                "output_column_name": "email",
                "comparison_levels": [
                    {
                        "sql_condition": "\"email_l\" IS NULL OR \"email_r\" IS NULL",
                        "label_for_charts": "email is NULL",
                        "is_null_level": true
                    },
                    {
                        "sql_condition": "\"email_l\" = \"email_r\"",
                        "label_for_charts": "Exact match on email",
                        "m_probability": 0.5521904988218763,
                        "u_probability": 0.0023577568563241916
                    },
                    {
                        "sql_condition": "NULLIF(regexp_extract(\"email_l\", '^[^@]+', 0), '') = NULLIF(regexp_extract(\"email_r\", '^[^@]+', 0), '')",
                        "label_for_charts": "Exact match on transformed email",
                        "m_probability": 0.22046667643566936,
                        "u_probability": 0.0010970118706508391
                    },
                    {
                        "sql_condition": "jaro_winkler_similarity(\"email_l\", \"email_r\") >= 0.88",
                        "label_for_charts": "Jaro-Winkler distance of email >= 0.88",
                        "m_probability": 0.21374764835824084,
                        "u_probability": 0.0007367990176013098
                    },
                    {
                        "sql_condition": "jaro_winkler_similarity(NULLIF(regexp_extract(\"email_l\", '^[^@]+', 0), ''), NULLIF(regexp_extract(\"email_r\", '^[^@]+', 0), '')) >= 0.88",
                        "label_for_charts": "Jaro-Winkler distance of transformed email >= 0.88",
                        "u_probability": 0.00027834629553827263
                    },
                    {
                        "sql_condition": "ELSE",
                        "label_for_charts": "All other comparisons",
                        "m_probability": 0.013595176384213488,
                        "u_probability": 0.9955300859598853
                    }
                ],
                "comparison_description": "jaro_winkler on username at threshold 0.88 vs. anything else"
            }
        ]
    }

    ```


This is simply the settings dictionary with additional entries for `"m_probability"` and `"u_probability"` in each of the `"comparison_levels"`, which have estimated during model training.

For example in the first name exact match level:

```json linenums="16", hl_lines="4 5"
{
    "sql_condition": "\"first_name_l\" = \"first_name_r\"",
    "label_for_charts": "Exact match on first_name",
    "m_probability": 0.48854806009621365,
    "u_probability": 0.0056770619302010565
},

```

where the `m_probability` and `u_probability` values here are then used to generate the match weight for an exact match on `"first_name"` between two records (i.e. the amount of evidence provided by records having the same first name) in model predictions.

## Loading a pre-trained model

When using a pre-trained model, you can read in the model from a json and recreate the linker object to make new pairwise predictions. For example:

```py
linker = Linker(
    new_df,
    settings="./path/to/model.json",
    db_api=db_api
)

```





Contents of /docs/topic_guides/splink_fundamentals/backends/backends.md:
---
tags:
  - Spark
  - DuckDB
  - Athena
  - SQLite
  - Postgres
  - Backends
---

# Splink's SQL backends: Spark, DuckDB, etc

Splink is a Python library. However, it implements all data linking computations by generating SQL, and submitting the SQL statements to a backend of the user's choosing for execution.

The Splink code you write is almost identical between backends, so it's straightforward to migrate between backends. Often, it's a good idea to start working using DuckDB on a sample of data, because it will produce results very quickly. When you're comfortable with your model, you may wish to migrate to a big data backend to estimate/predict on the full dataset.

## Choosing a backend

### Considerations when choosing a SQL backend for Splink

When choosing which backend to use when getting started with Splink, there are a number of factors to consider:

- the size of the dataset(s)
- the amount of boilerplate code/configuration required
- access to specific (sometimes proprietary) platforms
- the backend-specific features offered by Splink
- the level of support and active development offered by Splink

Below is a short summary of each of the backends available in Splink.

### :simple-duckdb: DuckDB

DuckDB is recommended for most users for all but the largest linkages.

It is the fastest backend, and is capable of linking large datasets, especially if you have access to high-spec machines.

As a rough guide it can:

- Link up to around 5 million records on a modern laptop (4 core/16GB RAM)
- Link tens of millions of records on high spec cloud computers very fast.

For further details, see the results of formal benchmarking [here](https://www.robinlinacre.com/fast_deduplication/).

DuckDB is also recommended because for many users its simplest to set up.

It can be run on any device with python installed and it is installed automatically with Splink via `pip install splink`. DuckDB has complete coverage for the functions in the Splink [comparison libraries](../../../api_docs/comparison_level_library.md).  Alongside the Spark linker, it receives most attention from the development team.

See the DuckDB [deduplication example notebook](../../../demos/examples/duckdb/deduplicate_50k_synthetic.ipynb) to get a better idea of how Splink works with DuckDB.


### :simple-apachespark: Spark

Spark is recommended for:

- Very large linkages, especially where DuckDB is performing poorly or running out of memory, or
- Or have easier access to a Spark cluster than a single high-spec instance to run DuckDB

It is not our default recommendation for most users because:

- It involves more configuration than users, such as registering UDFs and setting up a Spark cluster
- It is slower than DuckDB for many

The Spark linker has complete coverage for the functions in the Splink [comparison libraries](../../../api_docs/comparison_level_library.md).

If working with Databricks note that the Splink development team does not have access to a Databricks environment so we can struggle help DataBricks-specific issues.

See the Spark [deduplication example notebook](../../../demos/examples/spark/deduplicate_1k_synthetic.ipynb) for an example of how Splink works with Spark.

### :simple-amazonaws: Athena

Athena is a big data SQL backend provided on AWS which is great for large datasets (10+ million records). It requires access to a live AWS account and as a persistent database, requires some additional management of the tables created by Splink. Athena has reasonable, but not complete, coverage for fuzzy matching functions, see [Presto](https://prestodb.io/docs/current/functions/string.html). At this time, the Athena backend is being used sparingly by the Splink development team so receives minimal levels of support.

In addition, from a development perspective, the necessity for an AWS connection makes testing Athena code more difficult, so there may be occasional bugs that would normally be caught by our testing framework.

See the Athena [deduplication example notebook](../../../demos/examples/athena/deduplicate_50k_synthetic.ipynb) to get a better idea of how Splink works with Athena.

### :simple-sqlite: SQLite

SQLite is similar to DuckDB in that it is, generally, more suited to smaller datasets. SQLite is simple to setup and can be run directly in a Jupyter notebook, but is not as performant as DuckDB. SQLite has reasonable, but not complete, coverage for the functions in the Splink [comparison libraries](../../../api_docs/comparison_level_library.md), with gaps in array and date comparisons. String fuzzy matching, while not native to SQLite is available via python UDFs which has some [performance implications](#additional-information-for-specific-backends). SQLite is not actively been used by the Splink team so receives minimal levels of support.

### :simple-postgresql: PostgreSQL

PostgreSQL is a relatively new linker, so we have not fully tested performance or what size of datasets can processed with Splink. The Postgres backend requires a Postgres database, so it is recommend to use this backend only if you are working with a pre-existing Postgres database. Postgres has reasonable, but not complete, coverage for the functions in the Splink [comparison libraries](../../../api_docs/comparison_level_library.md), with gaps in string fuzzy matching functionality due to the lack of some string functions in Postgres. At this time, the Postgres backend is not being actively used by the Splink development team so receives minimal levels of support.

More details on using Postgres as a Splink backend can be found on the [Postgres page](./postgres.md).

## Using your chosen backend

Choose the relevant DBAPI:

Once you have initialised the `linker` object, there is no difference in the subsequent code between backends.

=== ":simple-duckdb: DuckDB"

    ```python
    from splink import Linker, DuckDBAPI

    linker = Linker(df, settings, db_api=DuckDBAPI(...))
    ```

=== ":simple-apachespark: Spark"

    ```python
    from splink import Linker, SparkAPI

    linker = Linker(df, settings, db_api=SparkAPI(...))
    ```

=== ":simple-amazonaws: Athena"

    ```python
    from splink import Linker, AthenaAPI

    linker = Linker(df, settings, db_api=AthenaAPI(...))
    ```

=== ":simple-sqlite: SQLite"

    ```python
    from splink import Linker, SQLiteAPI

    linker = Linker(df, settings, db_api=SQLiteAPI(...))

    ```

=== ":simple-postgresql: PostgreSQL"

    ```python
    from splink import Linker, PostgresAPI

    linker = Linker(df, settings, db_api=PostgresAPI(...))

    ```

## Additional Information for specific backends

### :simple-sqlite: SQLite

[**SQLite**](https://www.sqlite.org/index.html) does not have native support for [fuzzy string-matching](../../comparisons/comparators.md) functions.
However, the following are available for Splink users as python [user-defined functions (UDFs)](../../../dev_guides/udfs.md#sqlite)  which are automatically registered when calling `SQLiteAPI()`

* `levenshtein`
* `damerau_levenshtein`
* `jaro`
* `jaro_winkler`

However, there are a couple of points to note:

* These functions are implemented using the [RapidFuzz](https://maxbachmann.github.io/RapidFuzz/) package, which must be installed if you wish to make use of them, via e.g. `pip install rapidfuzz`. If you do not wish to do so you can disable the use of these functions when creating your linker:
```py
SQLiteAPI(register_udfs=False)
```
* As these functions are implemented in python they will be considerably slower than any native-SQL comparisons. If you find that your model-training or predictions are taking a large time to run, you may wish to consider instead switching to DuckDB (or some other backend).




Contents of /docs/topic_guides/blocking/blocking_rules.md:
---
tags:
  - Blocking
  - Performance
---

# What are Blocking Rules?

The primary driver the run time of Splink is the number of record pairs that the Splink model has to process.  This is controlled by the blocking rules.

This guide explains what blocking rules are, and how they can be used.

## Introduction

One of the main challenges to overcome in record linkage is the **scale** of the problem.

The number of pairs of records to compare grows using the formula $\frac{n\left(n-1\right)}2$, i.e. with (approximately) the square of the number of records, as shown in the following chart:

![](../../img/blocking/pairwise_comparisons.png)

For example, a dataset of 1 million input records would generate around 500 billion pairwise record comparisons.

So, when datasets get bigger the computation could get infeasibly large. We use **blocking** to reduce the scale of the computation to something more tractible.

## Blocking

Blocking is a technique for reducing the number of record pairs that are considered by a model.

Considering a dataset of 1 million records, comparing each record against all of the other records in the dataset generates ~500 billion pairwise comparisons. However, we know the vast majority of these record comparisons won't be matches, so processing the full ~500 billion comparisons would be largely pointless (as well as costly and time-consuming).

Instead, we can define a subset of potential comparisons using **Blocking Rules**. These are rules that define "blocks" of comparisons that should be considered. For example, the blocking rule:

`"block_on("first_name", "surname")`

will generate only those pairwise record comparisons where first name and surname match.  That is, is equivalent to joining input records the SQL condition  `l.first_name = r.first_name and l.surname = r.surname`

Within a Splink model, you can specify multiple Blocking Rules to ensure all potential matches are considered.  These are provided as a list.  Splink will then produce all record comparisons that satisfy at least one of your blocking rules.

???+ "Further Reading"

    For more information on blocking, please refer to [this article](https://toolkit.data.gov.au/data-integration/data-integration-projects/probabilistic-linking.html#key-steps-in-probabilistic-linking)

## Blocking in Splink

There are two areas in Splink where blocking is used:

- The first is to generate pairwise comparisons when finding links (running `predict()`). This is the sense in which 'blocking' is usually understood in the context of record linkage.  These blocking rules are provided in the model settings using `blocking_rules_to_generate_predictions`.

- The second is a less familiar application of blocking: using it for model training. This is a more advanced topic, and is covered in the [model training](./model_training.md) topic guide.


### Choosing `blocking_rules_to_generate_predictions`

The blocking rules specified in your settings at `blocking_rules_to_generate_predictions` are the single most important determinant of how quickly your linkage runs.  This is because the number of comparisons generated is usually many multiple times higher than the number of input records.

How can we choose a good set of blocking rules? **It's usually better to use a longer list of strict blocking rules, than a short list of loose blocking rules.**  Let's see why:

The aim of our blocking rules are to:

- Capture as many true matches as possible
- Reduce the total number of comparisons being generated

There is a tension between these aims, because by choosing loose blocking rules which generate more comparisons, you have a greater chance of capturing all true matches.

A single rule is unlikely to be able to achieve both aims.

For example, consider:

```
SettingsCreator(
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname")
    ]
)

```
This will generate comparisons for all true matches where names match. But it would miss a true match where there was a typo in the name.

This is why `blocking_rules_to_generate_predictions` is a list.

Suppose we also block on `postcode`:

```py
SettingsCreator(
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname"),
        block_on("postcode")
    ]
)
```

Now it doesn't matter if there's a typo in the name so long as postcode matches (and vice versa).

We could take this further and block on, say, `date_of_birth` as well.

By specifying a variety of `blocking_rules_to_generate_predictions`, even if each rule on its own is relatively tight, it becomes implausible that a truly matching record would not be captured by at least one of the rules.

### Tightening blocking rules for linking larger datasets

As the size of your input data grows, tighter blocking rules may be needed.  Blocking on, say `first_name` and `surname` may be insufficiently tight to reduce the number of comparisons down to a computationally tractable number.

In this situation, it's often best to use an even larger list of tighter blocking rules.

An example could be something like:
```py
SettingsCreator(
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname", "substr(postcode,1,3)"),
        block_on("surname", "dob"),
        block_on("first_name", "dob"),
        block_on("dob", "postcode")
        block_on("first_name", "postcode")
        block_on("surname", "postcode")
    ]
)
```

### Analysing `blocking_rules_to_generate_predictions`

It's generally a good idea to analyse the number of comparisons generated by your blocking rules **before** trying to use them to make predictions, to make sure you don't accidentally generate trillions of pairs.  You can use the following function to do this:

```py
from splink.blocking_analysis import count_comparisons_from_blocking_rule

br = block_on("substr(first_name, 1,1)", "surname")

count_comparisons_from_blocking_rule(
        table_or_tables=df,
        blocking_rule=br,
        link_type="dedupe_only",
        db_api=db_api,
    )
```

### More compelex blocking rules

It is possible to use more complex blocking rules that use non-equijoin conditions.  For example, you could use a blocking rule that uses a fuzzy matching function:

```sql
l.first_name and r.first_name and levenshtein(l.surname, r.surname) < 3
```

However, this will not be executed very efficiently, for reasons described in [this](performance.md) page.



Contents of /docs/topic_guides/blocking/model_training.md:
# Blocking for Model Training

Model Training Blocking Rules choose which record pairs from a dataset get considered when training a Splink model. These are used during Expectation Maximisation (EM), where we estimate the [m probability](../theory/fellegi_sunter.md#m-probability) (in most cases).

The aim of Model Training Blocking Rules is to reduce the number of record pairs considered when training a Splink model in order to reduce the computational resource required. Each Training Blocking Rule define a training "block" of records which have a combination of matches and non-matches that are considered by Splink's Expectation Maximisation algorithm.

The Expectation Maximisation algorithm seems to work best when the pairwise record comparisons are a mix of anywhere between around 0.1% and 99.9% true matches. It works less efficiently if there is a huge imbalance between the two (e.g. a billion non matches and only a hundred matches).

!!! note
    Unlike [blocking rules for prediction](./blocking_rules.md), it does not matter if Training Rules excludes some true matches - it just needs to generate examples of matches and non-matches.


## Using Training Rules in Splink


Blocking Rules for Model Training are used as a parameter in the `estimate_parameters_using_expectation_maximisation` function. After a `linker` object has been instantiated, you can estimate `m probability` with training sessions such as:

```python
from splink.duckdb.blocking_rule_library import block_on

blocking_rule_for_training = block_on("first_name")
linker.estimate_parameters_using_expectation_maximisation(
    blocking_rule_for_training
)

```

Here, we have defined a "block" of records where `first_name` are the same. As names are not unique, we can be pretty sure that there will be a combination of matches and non-matches in this "block" which is what is required for the EM algorithm.

Matching only on `first_name` will likely generate a large "block" of pairwise comparisons which will take longer to run. In this case it may be worthwhile applying a stricter blocking rule to reduce runtime. For example, a match on `first_name` and `surname`:

```python

from splink.duckdb.blocking_rule_library import block_on
blocking_rule = block_on(["first_name", "surname"])
linker.estimate_parameters_using_expectation_maximisation(
    blocking_rule_for_training
    )

```

which will still have a combination of matches and non-matches, but fewer record pairs to consider.


## Choosing Training Rules

The idea behind Training Rules is to consider "blocks" of record pairs with a mixture of matches and non-matches. In practice, most blocking rules have a mixture of matches and non-matches so the primary consideration should be to reduce the runtime of model training by choosing Training Rules that reduce the number of record pairs in the training set.

There are some tools within Splink to help choosing these rules. For example, the `count_num_comparisons_from_blocking_rule` gives the number of records pairs generated by a blocking rule:

```py
from splink.duckdb.blocking_rule_library import block_on
blocking_rule = block_on(["first_name", "surname"])
linker.count_num_comparisons_from_blocking_rule(blocking_rule)
```
> 1056

It is recommended that you run this function to check how many comparisons are generated before training a model so that you do not needlessly run a training session on billions of comparisons.

!!! note
    Unlike [blocking rules for prediction](./blocking_rules.md), Training Rules are treated separately for each EM training session therefore the total number of comparisons for Model Training is simply the sum of `count_num_comparisons_from_blocking_rule` across all Blocking Rules (as opposed to the result of `cumulative_comparisons_from_blocking_rules_records`).



Contents of /docs/topic_guides/blocking/performance.md:
# Blocking Rule Performance

When considering computational performance of blocking rules, there are two main drivers to address:

- How may pairwise comparisons are generated
- How quickly each pairwise comparison takes to run

Below we run through an example of how to address each of these drivers.

## Strict vs lenient Blocking Rules

One way to reduce the number of comparisons being considered within a model is to apply strict blocking rules. However, this can have a significant impact on the how well the Splink model works.

In reality, we recommend getting a model up and running with strict Blocking Rules and incrementally loosening them to see the impact on the runtime and quality of the results. By starting with strict blocking rules, the linking process will run faster which means you can iterate through model versions more quickly.

??? example "Example - Incrementally loosening Prediction Blocking Rules"

    When choosing Prediction Blocking Rules, consider how `blocking_rules_to_generate_predictions` may be made incrementally less strict. We may start with the following rule:

    `l.first_name = r.first_name and l.surname = r.surname and l.dob = r.dob`.

    This is a very strict rule, and will only create comparisons where full name and date of birth match. This has the advantage of creating few record comparisons, but the disadvantage that the rule will miss true matches where there are typos or nulls in any of these three fields.

    This blocking rule could be loosened to:

    `substr(l.first_name,1,1) = substr(r.first_name,1,1) and l.surname = r.surname and l.year_of_birth = r.year_of_birth`

    Now it allows for typos or aliases in the first name, so long as the first letter is the same, and errors in month or day of birth.

    Depending on the side of your input data, the rule could be further loosened to

    `substr(l.first_name,1,1) = substr(r.first_name,1,1) and l.surname = r.surname`

    or even

    `l.surname = r.surname`

    The user could use the `linker.count_num_comparisons_from_blocking_rule()` function to select which rule is appropriate for their data.

## Efficient Blocking Rules

While the number of pairwise comparisons is important for reducing the computation, it is also helpful to consider the efficiency of the Blocking Rules. There are a number of ways to define subsets of records (i.e. "blocks"), but they are not all computationally efficient.

From a performance perspective, here we consider two classes of blocking rule:

- Equi-join conditions
- Filter conditions

### Equi-join Conditions

Equi-joins are simply equality conditions between records, e.g.

`l.first_name = r.first_name`

Equality-based blocking rules can be executed efficiently by SQL engines in the sense that the engine is able to create only the record pairs that satisfy the blocking rule. The engine does **not** have to create all possible record pairs and then filter out the pairs that do not satisfy the blocking rule.  This is in contrast to filter conditions (see below), where the engine has to create a larger set of comparisons and then filter it down.

Due to this efficiency advantage, equality-based blocking rules should be considered the default method for defining blocking rules. For example, the above example can be written as:

```
from splink import block_on
block_on("first_name")
```


### Filter Conditions

Filter conditions refer to any Blocking Rule that isn't a simple equality between columns. E.g.

`levenshtein(l.surname, r.surname) < 3`

Blocking rules which use similarity or distance functions, such as the example above, are inefficient as the `levenshtein` function needs to be evaluated for all possible record comparisons before filtering out the pairs that do not satisfy the filter condition.


### Combining Blocking Rules Efficiently

Just as how Blocking Rules can impact on performance, so can how they are combined. The most efficient Blocking Rules combinations are "AND" statements. E.g.

`block_on("first_name", "surname")`

which is equivalent to

`l.first_name = r.first_name AND l.surname = r.surname`

"OR" statements are extremely inefficient and should almost never be used. E.g.

`l.first_name = r.first_name OR l.surname = r.surname`

In most SQL engines, an `OR` condition within a blocking rule will result in all possible record comparisons being generated.  That is, the whole blocking rule becomes a filter condition rather than an equi-join condition, so these should be avoided.  For further information, see [here](https://github.com/moj-analytical-services/splink/discussions/1417#discussioncomment-6420575).

Instead of the `OR` condition being included in the blocking rule, instead, provide two blocking rules to Splink.  This will achieve the desired outcome of generating all comparisons where either the first name or surname match.

```py
SettingsCreator(
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname")
    ]
)
```



??? note "Spark-specific Further Reading"

    Given the ability to parallelise operations in Spark, there are some additional configuration options which can improve performance of blocking. Please refer to the Spark Performance Topic Guides for more information.

    Note: In Spark Equi-joins are implemented using hash partitioning, which facilitates splitting the workload across multiple machines.




Contents of /docs/topic_guides/comparisons/comparisons_and_comparison_levels.md:
# Comparison and ComparisonLevels



## Comparing information

To find matching records, Splink creates pairwise record comparisons from the input records, and scores these comparisons.

Suppose for instance your data contains `first_name` and `surname` and `dob`:

|id |first_name|surname|dob       |
|---|----------|-------|----------|
|1  |john      |smith  |1991-04-11|
|2  |jon       |smith  |1991-04-17|
|3  |john      |smyth  |1991-04-11|

To compare these records, at the blocking stage, Splink will set these records against each other in a table of pairwise record comparisons:

|id_l|id_r|first_name_l|first_name_r|surname_l|surname_r|dob_l     |dob_r     |
|----|----|------------|------------|---------|---------|----------|----------|
|1   |2   |john        |jon         |smith    |smith    |1991-04-11|1991-04-17|
|1   |3   |john        |john        |smith    |smyth    |1991-04-11|1991-04-11|
|2   |3   |jon         |john        |smith    |smyth    |1991-04-17|1991-04-11|


When defining comparisons, we are defining rules that operate on each row of this latter table of pairwise comparisons

## Defining similarity


How how should we assess similarity between the records?

In Splink, we will use different measures of similarity for different columns in the data, and then combine these measures to get an overall similarity score.  But the most appropriate definition of similarity will differ between columns.

For example, two surnames that differ by a single character would usually be considered to be similar.  But a one character difference in a 'gender' field encoded as `M` or `F` is not similar at all!

To allow for this, Splink uses the concepts of `Comparison`s and `ComparisonLevel`s.  Each `Comparison` usually measures the similarity of a single column in the data, and each `Comparison` is made up of one or more `ComparisonLevel`s.

Within each `Comparison` are _n_ discrete `ComparisonLevel`s.  Each `ComparisonLevel` defines a discrete gradation (category) of similarity within a Comparison.  There can be as many `ComparisonLevels` as you want. For example:

```
Data Linking Model
├─-- Comparison: Gender
│    ├─-- ComparisonLevel: Exact match
│    ├─-- ComparisonLevel: All other
├─-- Comparison: First name
│    ├─-- ComparisonLevel: Exact match on surname
│    ├─-- ComparisonLevel: surnames have JaroWinklerSimilarity > 0.95
│    ├─-- ComparisonLevel: All other
```

The categories are discrete rather than continuous for performance reasons - so for instance, a `ComparisonLevel` may be defined as `jaro winkler similarity between > 0.95`, as opposed to using the Jaro-Winkler score as a continuous measure directly.

It is up to the user to decide how best to define similarity for the different columns (fields) in their data, and this is a key part of modelling a record linkage problem.

A much more detailed of how this works can be found in [this series of interactive tutorials](https://www.robinlinacre.com/probabilistic_linkage/) - refer in particular to [computing the Fellegi Sunter model](https://www.robinlinacre.com/computing_fellegi_sunter/).

## An example:


The concepts of `Comparison`s and `ComparisonLevel`s are best explained using an example.

Consider the following simple data linkage model with only two columns (in a real example there would usually be more):

```
Data Linking Model
├─-- Comparison: Date of birth
│    ├─-- ComparisonLevel: Exact match
│    ├─-- ComparisonLevel: One character difference
│    ├─-- ComparisonLevel: All other
├─-- Comparison: First name
│    ├─-- ComparisonLevel: Exact match on first_name
│    ├─-- ComparisonLevel: first_names have JaroWinklerSimilarity > 0.95
│    ├─-- ComparisonLevel: first_names have JaroWinklerSimilarity > 0.8
│    ├─-- ComparisonLevel: All other
```


In this model we have two `Comparison`s: one for date of birth and one for first name:

For data of birth, we have chosen three discrete `ComparisonLevel`s to measure similarity.  Either the dates of birth are an exact match, they differ by one character, or they are different in some other way.

For first name, we have chosen four discrete `ComparisonLevel`s to measure similarity.  Either the first names are an exact match, they have a JaroWinkler similarity of greater than 0.95, they have a JaroWinkler similarity of greater than 0.8, or they are different in some other way.

Note that these definitions are mutually exclusive, because they're implemented by Splink like an if statement.  For example, for first name, the `Comparison` is equivalent to the following pseudocode:

```python
if first_name_l_ == first_name_r:
    return "Assign to category: Exact match"
elif JaroWinklerSimilarity(first_name_l_, first_name_r) > 0.95:
    return "Assign to category: JaroWinklerSimilarity > 0.95"
elif JaroWinklerSimilarity(first_name_l_, first_name_r) > 0.8:
    return "Assign to category: JaroWinklerSimilarity > 0.8"
else:
    return "Assign to category: All other"
```

In the [next section](./customising_comparisons.ipynb), we will see how to define these `Comparison`s and `ComparisonLevel`s in Splink.



Contents of /docs/topic_guides/performance/drivers_of_performance.md:
---
tags:
  - Performance
---


This topic guide covers the fundamental drivers of the run time of Splink jobs.

## Blocking

The primary driver of run time is **the number of record pairs that the Splink model has to process**. In Splink, the number of pairs to consider is reduced using **Blocking Rules** which are covered in depth in their own set of [topic guides](../blocking/blocking_rules.md).

## Complexity of comparisons

The second most important driver of runtime is the complexity of comparisons, and the computional intensity of the fuzzy matching functions used.

Complexity is added to comparisons in a number of ways, including:

- Increasing the number of comparison levels
- Using more computationally expensive comparison functions
- Adding Term Frequency Adjustments

See [performance of comparison functions](../performance/performance_of_comparison_functions.ipynb) for benchmarking results.

## Retaining columns through the linkage process

The size your dataset has an impact on the performance of Splink. This is also applicable to the tables that Splink creates and uses under the hood. Some Splink functionality requires additional calculated columns to be stored. For example:

- The `comparison_viewer_dashboard` requires `retain_matching_columns` and `retain_intermediate_calculation_columns` to be set to `True` in the settings dictionary, but this makes some processes less performant.

## Filtering out pairwise comparisons in the `predict()` step

Reducing the number of pairwise comparisons that need to be returned will make Splink perform faster. One way of doing this is to filter comparisons with a match score below a given threshold (using a `threshold_match_probability` or `threshold_match_weight`) when you call `predict()`.

## Model training without term frequency adjustments

Model training with Term Frequency adjustments can be made more performant by setting `estimate_without_term_frequencies` parameter to `True` in `estimate_parameters_using_expectation_maximisation`.








Contents of /docs/api_docs/settings_dict_guide.md:
---
tags:
  - settings
  - Dedupe
  - Link
  - Link and Dedupe
  - Expectation Maximisation
  - Comparisons
  - Blocking Rules
---

## Guide to Splink settings

This document enumerates all the settings and configuration options available when
developing your data linkage model.


<hr>

## `link_type`

The type of data linking task.  Required.

- When `dedupe_only`, `splink` find duplicates.  User expected to provide a single input dataset.

- When `link_and_dedupe`, `splink` finds links within and between input datasets.  User is expected to provide two or more input datasets.

- When `link_only`,  `splink` finds links between datasets, but does not attempt to deduplicate the datasets (it does not try and find links within each input dataset.) User is expected to provide two or more input datasets.

**Examples**: `['dedupe_only', 'link_only', 'link_and_dedupe']`

<hr>

## `probability_two_random_records_match`

The probability that two records chosen at random (with no blocking) are a match.  For example, if there are a million input records and each has on average one match, then this value should be 1/1,000,000.

If you estimate parameters using expectation maximisation (EM), this provides an initial value (prior) from which the EM algorithm will start iterating.  EM will then estimate the true value of this parameter.

**Default value**: `0.0001`

**Examples**: `[1e-05, 0.006]`

<hr>

## `em_convergence`

Convergence tolerance for the Expectation Maximisation algorithm

The algorithm will stop converging when the maximum of the change in model parameters between iterations is below this value

**Default value**: `0.0001`

**Examples**: `[0.0001, 1e-05, 1e-06]`

<hr>

## `max_iterations`

The maximum number of Expectation Maximisation iterations to run (even if convergence has not been reached)

**Default value**: `25`

**Examples**: `[20, 150]`

<hr>

## `unique_id_column_name`

Splink requires that the input dataset has a column that uniquely identifies each record.  `unique_id_column_name` is the name of the column in the input dataset representing this unique id

For linking tasks, ids must be unique within each dataset being linked, and do not need to be globally unique across input datasets

**Default value**: `unique_id`

**Examples**: `['unique_id', 'id', 'pk']`

<hr>

## `source_dataset_column_name`

The name of the column in the input dataset representing the source dataset

Where we are linking datasets, we can't guarantee that the unique id column is globally unique across datasets, so we combine it with a source_dataset column.  Usually, this is created by Splink for the user

**Default value**: `source_dataset`

**Examples**: `['source_dataset', 'dataset_name']`

<hr>

## `retain_matching_columns`

If set to true, each column used by the `comparisons` SQL expressions will be retained in output datasets

This is helpful so that the user can inspect matches, but once the comparison vector (gamma) columns are computed, this information is not actually needed by the algorithm.  The algorithm will run faster and use less resources if this is set to false.

**Default value**: `True`

**Examples**: `[False, True]`

<hr>

## `retain_intermediate_calculation_columns`

Retain intermediate calculation columns, such as the Bayes factors associated with each column in `comparisons`

The algorithm will run faster and use less resources if this is set to false.

**Default value**: `False`

**Examples**: `[False, True]`

<hr>

## comparisons

A list specifying how records should be compared for probabilistic matching.  Each element is a dictionary

???+ note "Settings keys nested within each member of `comparisons`"

    ### output_column_name

    The name used to refer to this comparison in the output dataset.  By default, Splink will set this to the name(s) of any input columns used in the comparison.  This key is most useful to give a clearer description to comparisons that use multiple input columns.  e.g. a location column that uses postcode and town may be named location

    For a comparison column that uses a single input column, e.g. first_name, this will be set first_name. For comparison columns that use multiple columns, if left blank, this will be set to the concatenation of columns used.

    **Examples**: `['first_name', 'surname']`

    <hr>

    ### comparison_description

    An optional label to describe this comparison, to be used in charting outputs.

    **Examples**: `['First name exact match', 'Surname with middle levenshtein level']`

    <hr>

    ### comparison_levels

    Comparison levels specify how input values should be compared.  Each level corresponds to an assessment of similarity, such as exact match, Jaro-Winkler match, one side of the match being null, etc

    Each comparison level represents a branch of a SQL case expression. They are specified in order of evaluation, each with a `sql_condition` that represents the branch of a case expression

    **Example**:
    ``` json
    [{
        "sql_condition": "first_name_l IS NULL OR first_name_r IS NULL",
        "label_for_charts": "null",
        "null_level": True
    },
    {
        "sql_condition": "first_name_l = first_name_r",
        "label_for_charts": "exact_match",
        "tf_adjustment_column": "first_name"
    },
    {
        "sql_condition": "ELSE",
        "label_for_charts": "else"
    }]
    ```

    <hr>

    ??? note "Settings keys nested within each member of `comparison_levels`"

        #### `sql_condition`

        A branch of a SQL case expression without WHEN and THEN e.g. `jaro_winkler_sim(surname_l, surname_r) > 0.88`

        **Examples**: `['forename_l = forename_r', 'jaro_winkler_sim(surname_l, surname_r) > 0.88']`

        <hr>

        #### label_for_charts

        A label for this comparison level, which will appear on charts as a reminder of what the level represents

        **Examples**: `['exact', 'postcode exact']`

        <hr>

        #### u_probability

        the u probability for this comparison level - i.e. the proportion of records that match this level amongst truly non-matching records

        **Examples**: `[0.9]`

        <hr>

        #### m_probability

        the m probability for this comparison level - i.e. the proportion of records that match this level amongst truly matching records

        **Examples**: `[0.1]`

        <hr>

        #### is_null_level

        If true, m and u values will not be estimated and instead the match weight will be zero for this column.  See treatment of nulls here on page 356, quote '. Under this MAR assumption, we can simply ignore missing data.': https://imai.fas.harvard.edu/research/files/linkage.pdf

        **Default value**: `False`

        <hr>

        #### tf_adjustment_column

        Make term frequency adjustments for this comparison level using this input column

        **Default value**: `None`

        **Examples**: `['first_name', 'postcode']`

        <hr>

        #### tf_adjustment_weight

        Make term frequency adjustments using this weight. A weight of 1.0 is a full adjustment.  A weight of 0.0 is no adjustment.  A weight of 0.5 is a half adjustment

        **Default value**: `1.0`

        **Examples**: `['first_name', 'postcode']`

        <hr>

        #### tf_minimum_u_value

        Where the term frequency adjustment implies a u value below this value, use this minimum value instead

        This prevents excessive weight being assigned to very unusual terms, such as a collision on a typo

        **Default value**: `0.0`

        **Examples**: `[0.001, 1e-09]`

        <hr>


## `blocking_rules_to_generate_predictions`

A list of one or more blocking rules to apply. A Cartesian join is applied if `blocking_rules_to_generate_predictions` is empty or not supplied.

Each rule is a SQL expression representing the blocking rule, which will be used to create a join.  The left table is aliased with `l` and the right table is aliased with `r`. For example, if you want to block on a `first_name` column, the blocking rule would be

`l.first_name = r.first_name`.

To block on first name and the first letter of surname, it would be

`l.first_name = r.first_name and substr(l.surname,1,1) = substr(r.surname,1,1)`.

Note that Splink deduplicates the comparisons generated by the blocking rules.

If empty or not supplied, all comparisons between the input dataset(s) will be generated and blocking will not be used. For large input datasets, this will generally be computationally intractable because it will generate comparisons equal to the number of rows squared.

**Default value**: `[]`

**Examples**: `[['l.first_name = r.first_name AND l.surname = r.surname', 'l.dob = r.dob']]`

<hr>

## `additional_columns_to_retain`

A list of columns not being used in the probabilistic matching comparisons that you want to include in your results.

By default, Splink drops columns which are not used by any comparisons.  This gives you the option to retain columns which are not used by the model.  A common example is if the user has labelled data (training data) and wishes to retain the labels in the outputs

**Default value**: `[]`

**Examples**: `[['cluster', 'col_2'], ['other_information']]`

<hr>

## `bayes_factor_column_prefix`

The prefix to use for the columns that will be created to store the Bayes factors

**Default value**: `bf_`

**Examples**: `['bf_', '__bf__']`

<hr>

## `term_frequency_adjustment_column_prefix`

The prefix to use for the columns that will be created to store the term frequency adjustments

**Default value**: `tf_`

**Examples**: `['tf_', '__tf__']`

<hr>

## `comparison_vector_value_column_prefix`

The prefix to use for the columns that will be created to store the comparison vector values

**Default value**: `gamma_`

**Examples**: `['gamma_', '__gamma__']`

<hr>

## `sql_dialect`

The SQL dialect in which `sql_conditions` are written.  Must be a valid SQLGlot dialect

**Default value**: `None`

**Examples**: `['spark', 'duckdb', 'presto', 'sqlite']`

<hr>




Contents of /docs/topic_guides/training/training_rationale.md:
---
tags:
  - Training
---

In Splink, in most scenarios, we recommend a hybrid approach to training model parameters, whereby we use direct estimation techniques for the `probability_two_random_records_match` (λ) parameter and the `u` probabilities, and then use EM training for the `m` probabilities.

The overall rationale is that we found that whilst it's possible to train all parameters using EM, empirically we've found you get better parameter estimates, and fewer convergence problems using direct estimation of some parameters.

In particular:

- You can precisely estimate the `u` probabilities in most cases, so there's no reason to use a less reliable unsupervised technique.
- With `probability_two_random_records_match`, we [found that](https://github.com/moj-analytical-services/splink/issues/462) Expectation Maximisation often resulted in inaccurate results due to our 'blocking' methodology for training `m` values. In practice, the direct estimation technique gave better results, despite being somewhat imprecise.

The recommended sequence for model training and associated rationale is as follows:

### 1. Use [linker.training.estimate_probability_two_random_records_match](https://moj-analytical-services.github.io/splink/api_docs/training.html#splink.internals.linker_components.training.LinkerTraining.estimate_probability_two_random_records_match) to estimate the proportion of records.

The `probability_two_random_records_match` is one of the harder parameters to estimate because there's a catch-22: to know its value, we need to know which records match, but that's the whole problem we're trying to solve.

Luckily, in most cases it's relatively easy to come up with a good guess, within (say) half or double its true value. It turns out that this is good enough to get good estimates of the other parameters, and ultimatey to get good predictions.

In our methodology,  the user specifies a list of deterministic matching rules that they believe represent true matches. These will be strict, and therefore will miss some fuzzy matches. For all but the most messy datasets, they should capture the majority of matches. A recall parameter is then provided by the user, which is the user's guess of how many matches are missed by these rules.

In a typical case, the deterministic rules may capture (say) 80% of matches. If the user gets this wrong and provides recall of say 60% or 95%, the effect on `probability_two_random_records_match` is not huge.

For example, a typical parameter estimate for `probability_two_random_records_match` when expressed as a match weight may be -14.  Assuming true recall of 80%, if the user guessed recall wrong at 60% or 95%, the parameter would be estimated at -13.75 or -14.42 respectively.

In turn, so long as this parameter is roughly right, it serves as an 'anchor' to EM parameter estimation later, which prevents it iterating/converging to the 'wrong' place (there's no guarantee it converges to a global minimum, only a local one).

Example:
```
deterministic_rules = [
    block_on("first_name", "surname", "dob"),
    block_on("email")"
]

linker.training.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)
```

### 2. Use [linker.training.estimate_u_using_random_sampling](https://moj-analytical-services.github.io/splink/api_docs/training.html#splink.internals.linker_components.training.LinkerTraining.estimate_u_using_random_sampling) to train `u` probabilities.

This one is easy to justify: On a sufficiently large dataset, if you take two random records, they will almost certainly not be a match. The `u` probabilities are calculated on the basis of truly non-matching records. So we can directly calculate them from random record comparisons. The only errors will be:

- Sampling error—in which case we can increase our sample size.
- The small errors introduced by the fact that, occasionally, our sampled records will match. In practice, this rarely has a big effect.

Example:
```
linker.training.estimate_u_using_random_sampling(max_pairs=1e7)
```
Increase `max_pairs` if you need more precision. This step is usually straightforward and reliable.

### 3. Use [linker.training.estimate_parameters_using_expectation_maximisation](https://moj-analytical-services.github.io/splink/api_docs/training.html#splink.internals.linker_components.training.LinkerTraining.estimate_parameters_using_expectation_maximisation) to estimate `m` probabilities.

The `m` probabilities have the same 'catch-22' problem as the `probability_two_random_records_match`. Luckily, the magic of EM is that it [solves this problem](https://www.robinlinacre.com/em_intuition/).

In the context of record linkage on large datsets, one problem is that we cannot easily run EM on random pairs of records, because they are almost all non-matches. It would work if we had infinite computing resources and could create unlimited numbers of comparisons. But in a typical case, even if we create 1 billion random comparisons, perhaps only 1000 are matches. Now we only have a sample size of 1,000 to estimate our `m` values, and their estimates may be imprecise. This contrasts to `u` estimation, where we have a sample size of `max_pair`s , often millions or even billions.

To speed this up, we can use a trick. By blocking on some columns (e.g. first name and surname), we now restrict our record comparisons to a subset with a far higher proportion of matches than random comparisons.

This trick is vulnerable to the criticism that we may get a biased selection of matching records. The Fellegi-Sunter model assumes [columns are independent conditional on match status](https://www.robinlinacre.com/maths_of_fellegi_sunter/), which is rarely true in practice.

But if this assumption holds then the selection of records is unbiased, and the parameter estimates are correct.

#### The 'round robin'

One complexity of this approach is that when we block on first name and surname, we can't get parameter estimates for these two columns because we've forced all comparison to be equal.

That's why we need multiple EM training passes: we need a second pass blocking on e.g. date_of_birth to get parameter estimates for first_name and surname.

```
linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))
linker.training.estimate_parameters_using_expectation_maximisation(block_on("first_name", "surname"))
```

At this point, we have:

- One estimate for the `m` values for first_name and surname
- One estimate for the `m` values for date_of_birth
- Two estimates for each `m` value for any other columns

If we're worried about our estimates, we can run more training rounds to get more estimates, e.g. blocking on postcode. We can then use [linker.visualisations.parameter_estimate_comparisons_chart](https://moj-analytical-services.github.io/splink/api_docs/visualisations.html#splink.internals.linker_components.visualisations.LinkerVisualisations.parameter_estimate_comparisons_chart) to check that the various estimates are similar. Under the hood, Splink will take an average.

```
linker.visualisations.parameter_estimate_comparisons_chart()
```

Empirically, one of the nice things here is that, because we've fixed the `u` probabilities and the `probability_two_random_records_match` at sensible values, they anchor the EM training process and it turns out you don't get too many convergence problems.

Finally, if the user is still having convergence problems, there are two options:
1. In practice, we have found these can be often due to data quality issues e.g. `mr` and `mrs` in `first_name`, so it's a good idea to go back to exploratory data analysis and understand if there's a root cause
2. If there's no obvious cause, the user can fix the m values, see [here](https://github.com/moj-analytical-services/splink/pull/2379)  and [here](https://github.com/moj-analytical-services/splink/discussions/2512#discussioncomment-11303080)



Contents of /docs/demos/tutorials/08_building_your_own_model.md:

# Next steps: Tips for Building your own model

Now that you've completed the tutorial, this page summarises some recommendations for how to approach building your own Splink models.

These recommendations should help you create an accurate model as quickly as possible.  They're particularly applicable if you're working with large datasets, where you can get slowed down by long processing times.

In a nutshell, we recommend beginning with a small sample and a basic model, then iteratively adding complexity to resolve issues and improve performance.

## General workflow

- **For large datasets, start by linking a small non-random sample of records**. Building a model is an iterative process of writing data cleaning code, training models, finding issues, and circling back to fix them. You don't want long processing times slowing down this iteration cycle.

    Most of your code can be developed against a small sample of records, and only once that's working, re-run everything on the full dataset.

    You need a **non-random** sample of perhaps about 10,000 records. The same must be  non-random because it must retain lots of matches - for instance, retain all people aged over 70, or all people with a first name starting with the characters `pa`.  You should aim to be able to run your full training and prediction script in less than a minute.

    Remember to set a lower value (say `1e6`) of the `target_rows` when calling `estimate_u_using_random_sampling()` during this iteration process, but then increase in the final full-dataset run to a much higher value, maybe `1e8`, since large value of `target_rows` can cause long processing times even on relatively small datasets.

- **Start with a simple model**.  It's often tempting to start by designing a complex model, with many granular comparison levels in an attempt to reflect the real world closely.

    Instead, we recommend starting with with a simple, rough and ready model where most comparisons have 2-3 levels (exact match, possibly a fuzzy level, and everything else).  The idea is to get to the point of looking at prediction results as quickly as possible using e.g. the comparison viewer.  You can then start to look for where your simple model is getting it wrong, and use that as the basis for improving your model, and iterating until you're seeing good results.

## Blocking rules for prediction

- **Analyse the number of comparisons before running predict**.  Use the tools in `splink.blocking_analysis` to validate that your rules aren't going to create a vast number of comparisons before asking Splink to create those comparisons.

- **Many strict `blocking_rules_for_prediction` are generally better than few loose rules.**  Whilst individually, strict blocking rules are likely to exclude many true matches, between them it should be implausible that a truly matching record 'falls through' all the rules.  Many strict rules often result in far fewer overall comparisons and a small number of loose rules.  In practice, many of our real-life models have between about 10-15 `blocking_rules_for_prediction`.


## EM trainining

- **Predictions usually aren't very sensitive to `m` probabilities being a bit wrong**.  The hardest model parameters to estimate are the `m` probabilities.  It's fairly common for Expectation Maximisation to yield 'bad' (implausble) values.  Luckily, the accuracy of your model is usually not particularly sensitive to the `m` probabilities - it usually the `u` probabilities driving the biggest match weights.  If you're having problems, consider fixing some `m` probabilities by expert judgement - see [here](https://github.com/moj-analytical-services/splink/pull/2379) for how.

- **Convergence problems are often indicative of the need for further data cleaning**.  Whilst predictions often aren't terribly sensitive to `m` probabilities, question why the estimation procedue is producing bad parameter estimates.  To do this, it's often enough to look at a variety of predictions to see if you can spot edge cases where the model is not doing what's expected.  For instance, we may find matches where the first name is `Mr`.  By fixing this and reestimating, the parameter estimates often make more sense.

- **Blocking rules for EM training do not need high recall**.  The purpose of blocking rules for EM training is to find a subset of records which include a reasonably balanced mix of matches and non matches.  There is no requirement that these records contain all, or even most of the matches.  For more see [here](https://moj-analytical-services.github.io/splink/topic_guides/blocking/model_training.html)  To double check that parameter estimates are a result of a biased sample of matches, you can use `linker.visualisations.parameter_estimate_comparisons_chart`.

## Working with large datasets

To optimise memory usage and performance:

- **Avoid pandas for input/output** Whilst Splink supports inputs as pandas dataframes, and you can convert results to pandas using `.as_pandas_dataframe()`, we recommend against this for large datasets.  For large datasets, use the concept of a dataframe that's native to your database backend.  For example, if you're using Spark, it's best to read your files using Spark and pass Spark dataframes into Splink, and save any outputs using `splink_dataframe.as_spark_dataframe`.  With duckdb use the inbuilt duckdb csv/parquet reader, and output via `splinkdataframe.as_duckdbpyrelation`.

- **Avoid pandas for data cleaning**.  You will generally get substantially better performance by performing data cleaning in SQL using your chosen backend rather than using pandas.

- **Turn off intermediate columns when calling `predict()`**.  Whilst during the model development phase, it is useful to set `retain_intermediate_calculation_columns=True` and
    `retain_intermediate_calculation_columns_for_prediction=True` in your settings, you should generally turn these off when calling `predict()`.  This will result in a much smaller table as your result set.  If you want waterfall charts for individual pairs, you can use [`linker.inference.compare_two_records`](../../api_docs/inference.md)





Blog Articles:


Article from https://www.robinlinacre.com/intro_to_probabilistic_linkage/:
An Interactive Introduction to Record Linkage (Data Deduplication) in the Fellegi-Sunter framework>robinlinacreOriginally posted:2021-05-20.Last updated: 2023-09-12.Live edit this notebookhere.This is part1of thetutorialNext article →#An Interactive Introduction to the Fellegi-Sunter Model for Data Linkage/Deduplication#AimsThis is part one of a series of interactive articles that aim to provide an introduction to the theory of probabilistic record linkage and deduplication.In this article I provide a high-level introduction to theFellegi-Sunterframework and an interactive example of a linkage model.Subsequent articles explore the theory in more depth.These materials align closely to the probabilistic model used bySplink, a free software package for record linkage at scale.These articles cover the theory only.  For practical model building using Splink, seethe tutorial in the Splink docs.#What is probabilistic record linkage?Probablistic record linkage is a technique used to link together records that lack unique identifiers.In the absence of a unique identifier such as a National Insurance number, we can use a combination of individually non-unique variables such as name, gender and date of birth to identify individuals.Record linkage can be done within datasets (deduplication), between datasets (linkage), or both1.Linkage is 'probabilistic' in the sense that it subject to uncertainty and relies on the balance of evidence. For instance, in a large dataset, observing that two records match on the full nameJohn Smithprovides some evidence that these two records may refer to the same person, but this evidence is inconclusive because it's possible there are two differentJohn Smiths.More broadly,  it is often impossible to classify pairs of records as matches or non-matches beyond any doubt. Instead, the aim of probabilisitic record linkage is to quantify the probability that a pair of records refer to the same entity by considering evidence in favour and against a match and weighting it appropriately.The most common type of probabilistic record linkage model is called the Fellegi-Sunter model.We start with aprior, which represents the probability that two records drawn at random are a match. We then compare the two records, increasing the match probability where information in the record matches, and decreasing it when information differs.The amount we increase and decrease the match probability is determined by the 'partial_match_weights' of the model.For example, a match on postcode gives us more evidence in favour of a match on gender, since the latter is much more likely to occur by chance.The final prediction is a simple calculation:  we sum uppartial_match_weights to compute a final match weight, which is then converted into a probability.#ExampleLet's take a look at an example of a simple Fellegi-Sunter model to calculate match probability interactively. This model will compare the two records in the table, and assess whether they refer to the same person, or different people.You may edit the values in the table to see how the match probability changes.We can decompose this calculation into the sum of thepartial_match_weights using a waterfall chart, which is read from left to right. We start with the prior, and take each column into account in turn.  The size of the bar corresponds to thepartial_match_weight.You can hover over the bars to see how the probability changes as each subsequent field is taken into account.The final estimated match probability is shown in the rightmost bar.  Note that the y axis on the right converts match weight into probability.In the next article, we will look at partial match weights in great depth.#FootnotesRecord linkage and deduplication are equivalent problems.  The only difference is that linkage involves finding matching entities across datasets and deduplication involves finding matches within datasets.↩Probabilistic Linkage Tutorial Navigation:An Interactive Introduction to Record Linkage (Data Deduplication) in the Fellegi-Sunter frameworkPartial match weightsm and u values in the Fellegi-Sunter modelThe mathematics of the Fellegi Sunter modelComputing the Fellegi Sunter modelWhy Probabilistic Linkage is More Accurate than Fuzzy Matching For Data DeduplicationThe Intuition Behind the Use of Expectation Maximisation to Train Record Linkage ModelsBack homeThis site is built usingObservable HQandGatsby.js. Source codehere. Saythanks!



Article from https://www.robinlinacre.com/partial_match_weights/:
Partial match weights>robinlinacreOriginally posted:2023-09-20.Live edit this notebookhere.← Previous articleThis is part2of thetutorialNext article →#Partial Match WeightsIn the previous article we saw that the Fellegi-Sunter model makes its predictions with a simple calculation: the sum of thepartial_match_weights.But what are partial match weights and where do they come from?1#What are they?Some columns are more important than others to the calculation of overall match probability.2This is quantified by the partial match weights.For example:A match on date of birth provides stronger evidence in favour of a match than a match on gender.A mismatch on date of birth may provide more evidence against a match than a mismatch on address (because people move house).Positive values indicate evidence in favour of a match whilst negative values mean evidence against a match.The concept applies more generally than just to matches and non-matches: partial match weights can be estimated for more subtlescenarios like 'first name not an exact match, but similar'.It is up to the modeller to define these scenarios and the resultant partial match weights they wish to estimate.#Example:  First name columnFor first name, we could define the following three scenarios:Exact match: First names are identicalFuzzy match:  First names are similar according to theJaro WinklermeasureAll other casesThe estimated partial match weights may look like this:3Observe that:An exact match on first name provides stronger evidence in favour of a match than a fuzzy matchIf the first name is completely different (neither and exact nor fuzzy match), this is evidence against the records being a match.These scenarios are mutually exclusive, and are implemented like a sequence ofifstatements:if first_name_l == first_name_r:first_name_partial_match_weight = 7.2elif jaro_winkler_sim(first_name_l, first_name_r) >= 0.9:first_name_partial_match_weight = 5.9else:first_name_partial_match_weight = -3.3#Remaining columnsWe can also estimate partial match weights for the remaining columns.The number of scenarios modelled and the definition of these scenarios can vary depending on the type of information.For example, since gender is a categorical variable we'd likely have just two partial weights: match, and non-match.For date of birth, we may define three scenarios, butLevenshteinwould be more suitable thanJaro Winklerfor assessing similarity.We can plot all the partial match weights in a single chart like this:This provides a succinct summary of the whole model.#Intuitive interpretation of partial match weightsThe partial match weights in the chart have intuitive interpretations.For example, the partial match weight associated with an exact match on postcode is stronger than the partial match weight associated with gender.This makes sense: a match on gender doesn't tell us much about whether two records are a match, whereas a match on postcode is much more informative.The pattern of negative partial match weights is also intuitive:  If gender is recorded as categorical variable, then it may be very accurate.  A mismatch would thus be strong evidence against a match.Conversely, a mismatch on postcode may offer little evidence against a match since people move house, or typos may have been made.#Understanding the partial match weight chart and waterfall chartLet's take a look at an example record comparison, and see how the partial match weights are used.You can edit the data in the below table to see how the calculation changesThe first stage is to calculate which scenarios apply for each column.  For example, are the first names an exact match, a fuzzy match, or neither?In the below chart, I reproduce the partial match weights chart, but with the activated partial match weights highlighted.The activated (selected) partial match weights can then be plotted in a waterfall chart, which represents how they are summed up to compute the final match weight:Alternatively we can represent the calculation of overall match weight as a sum:The final match weight is a measure of the similarity of the two records.  In subsequent articles we will see how it can be converted into a probability using the formula:#Next stepsWe now have a good qualitative understanding of the meaning of partial match weights.In the next article we explore how they are calculated, and how to interpret their values.#FootnotesIn this context, the word 'partial' means 'using only a subset of the information in the record' - often a single column.  There is then a partial match weight for each column.↩In this article, for simplicity,  there's a one to one correspondence between columns and partial match weights. Each column is treated as a separate piece of information.  In reality, the model allows for more flexibility.  We could split a column (e.g. date of birth) into several pieces of information (e.g. day, month, year) and estimate partial match weights for each.  Or we could combine several columns (e.g. first, middle and surname) into a single piece of information and estimate a partial match weights for this information.↩A later article will explain how to estimate partial match weights.  Here I just assume they have already been estimated.↩Probabilistic Linkage Tutorial Navigation:An Interactive Introduction to Record Linkage (Data Deduplication) in the Fellegi-Sunter frameworkPartial match weightsm and u values in the Fellegi-Sunter modelThe mathematics of the Fellegi Sunter modelComputing the Fellegi Sunter modelWhy Probabilistic Linkage is More Accurate than Fuzzy Matching For Data DeduplicationThe Intuition Behind the Use of Expectation Maximisation to Train Record Linkage ModelsBack homeThis site is built usingObservable HQandGatsby.js. Source codehere. Saythanks!



Article from https://www.robinlinacre.com/m_and_u_values/:
m and u values in the Fellegi-Sunter model>robinlinacreOriginally posted:2023-09-22.Live edit this notebookhere.← Previous articleThis is part3of thetutorialNext article →#m and u probabilities in the Fellegi-Sunter modelThe previous article showed how partial match weights are used to compute a prediction of whether two records match.However, partial match weights are not estimated directly.  They are made up of two parameters known as themand theuprobabilities.These probabilities are key to enabling estimation of partial match weights.Themanduprobabilities also have intuitive interpretations that allow us to understand linkage models and diagnose problems.#Motivating exampleImagine we have two records.  We're not sure whether they represent the same person.Now we're given some new information: we're told that month of birth matches.Is thisscenariomore likely among matches or non-matches?Amongst matching records, month of birth will usually matchAmongst non-matching records month of birth will rarely matchSince it's common to observe this scenario among matching records, but rare to observe it among non-matching records, this is evidence in favour of a match.But how much evidence?#m and u probabilities and Bayes FactorsThe strength of the evidence is quantified using themanduprobabilities.  For each scenario in the model:Themprobability measures how often the scenario occurs among matching records:m=Pr(scenario∣records match)m = \text{Pr}(\text{scenario}|\text{records match})m=Pr(scenario∣records match)Theuprobability measures how often the scenario occurs among non-matching records:u=Pr(scenario∣records do not match)u = \text{Pr}(\text{scenario}|\text{records do not match})u=Pr(scenario∣records do not match)What matters is therelativesize of these values.  This is calculated as a ratio known as the Bayes Factor1, denoted byKKK.Bayes Factor=K=mu=Pr(scenario∣records match)Pr(scenario∣records do not match)\text{Bayes Factor} = K = \frac{m}{u} = \frac{\text{Pr}(\text{scenario}|\text{records match})}{\text{Pr}(\text{scenario}|\text{records do not match})}Bayes Factor=K=um​=Pr(scenario∣records do not match)Pr(scenario∣records match)​Bayes Factors provide the easiest way to interpret the parameters of the Fellegi Sunter model because they act as a relative multiplier that increases or decreases the overall prediction of whether the records match. For example:A Bayes Factor of 5 can be interpreted as '5 times more likely to match'A Bayes Factor of 0.2 can be interpreted as '5 times less likely to match'#Example 1: Evidence in favour of a matchFor example, suppose we observe that month of birth matches.Amongst matching records, month of birth will usually match.  Supposing the occasional typo, we may havem=0.99m = 0.99m=0.99Amongst non matching records, month of birth matches around a twelth of the time, sou=1/12u = 1/12u=1/12.Bayes Factor=K=mu=0.990.0833=11.9\text{Bayes Factor} = K = \frac{m}{u} = \frac{0.99}{0.0833} = 11.9Bayes Factor=K=um​=0.08330.99​=11.9.This means we observe this scenario around 11.9 times more often amongst matching records than non-matching records.Hence, given this observation, the records are 11.9 times more likely to be a match.More generally, we can see from the formula that strong positive match weights only possible with lowuprobabilities, implying high cardinality.#Example 2: Evidence against a matchSuppose we observe that gender does not match.Amongst matching records, it will be rare to observe a non-match on gender.  If there are occasional data entry errors, we may havem=0.02m = 0.02m=0.02Amongst non matching records, gender will match around half the time.  Sou=0.5u = 0.5u=0.5.Bayes Factor=K=mu=0.020.5=0.04=125\text{Bayes Factor} = K = \frac{m}{u} = \frac{0.02}{0.5} = 0.04 = \frac{1}{25}Bayes Factor=K=um​=0.50.02​=0.04=251​.We observe this scenario around 25 times more often among non-matching records than matching records.Hence, given this observation the records are 25 times less likely to be a match.More generally, we can see from the formula that strong negative match weights only possible with lowmprobabilities, which in turn implies high data quality.#Interpreting m and u probabilitiesIn addition to these quantitative interpretations, themanduprobabilities also have intuitive qualitative interpretations:#m probabilitiesThemprobability can be thought of as a measure of data quality, or the propensity for data to change through time.For example, consider the scenario of an exact match on first name.An m probability of 0.9 means that, amongst matching records, the first name matches just 90% of the time, which is an indication of poor data quality.The m probability for an exact match on postcode may be even lower - but this may be driven primarily by people moving house, as opposed to data error.#u probabilitiesTheuprobability is primarily a measure of the likelihood of coincidences, which is driven by the cardinality of the data.Consider the scenario of an exact match on first name.Auprobability of 0.005 means that, amongst non-matching records, first name matches 0.5% of the time.Theuprobability therefore measures how often twodifferentpeople have the same first name - so in this sense it's a measure of how often coincidences occur.A column such as first name with a large number of distinct values (high cardinality) will have much smalleruprobabilities than a column such as gender which has low cardinality.#Using Bayes Factors to compute probabilitiesWhat does it mean for a match to bennntimes more or less likely?  More likely than what?It's only meaningful to say that something is more or less likely relative to a starting probability - known as the 'prior' (our 'prior belief').In the context of record linkage, the prior is our existing belief that the two records match before we saw the new information contained in a scenario (e.g. that first names match).Our updated belief given this new information is called the 'posterior'.Mathematically this can be written:posterior odds=prior odds×Bayes Factor\text{posterior odds} =  \text{prior odds} \times \text{Bayes Factor}posterior odds=prior odds×Bayes Factorand odds can be turned into probabilities with the following formula:probability=odds1+odds\text{probability} = \frac{\text{odds}}{1 + \text{odds}}probability=1+oddsodds​See the mathematical annex for further detail on these derivations.For example, suppose we believe the odds of a record comparison being a match are 1 to 120. But now we observe the new information that month of birth matches, with a Bayes Factor of 12.Soposterior odds=1120×12=110\text{posterior odds} =  \frac{1}{120}  \times 12 = \frac{1}{10}posterior odds=1201​×12=101​posterior probability=1101+110=111≈0.0909\text{posterior probability} = \frac{\frac{1}{10}}{1 + \frac{1}{10}} = \frac{1}{11} \approx 0.0909posterior probability=1+101​101​​=111​≈0.0909So after observing that the month of birth matches, the odds of the records being a match would be 1 in 10, or a probability of approximately 0.0909.Here's a calculator which shows how a prior probability is updated with a Bayes Factor/partial match weight:#Posterior calculatorPriorNew evidenceAn alternative way of visualising these concepts can be foundhere.#The relationship between Bayes Factors, partial match weights and m and u probabilitiesHow domanduprobabilities and Bayes Factors relate to the partial match weights we explored in the previous article?Partial match weights relate to Bayes Factors through a simple formula:partial match weight=ω=log⁡2(Bayes Factor)=log⁡2(mu)\text{partial match weight} = \omega = \log_2 \text{(Bayes Factor)} = \log_2 (\frac{m}{u})partial match weight=ω=log2​(Bayes Factor)=log2​(um​)There are two main reasons that the additional concept of partial match weights is useful in addition to Bayes Factors:Partial match weights are easier to represent on charts.  They tend to range from -30 to 30, whereas Bayes Factors can be tiny (one in a million) or massive (millions).Since Bayes Factors are multiplicative, the log transform turns them into something additive, which simplifies the maths a little.We can summarise this relationship with this chart.Hover over the chart to view different valuesA larger, standalone version is availablehere.#Next stepsNow that we have a firm grasp of these ingredients, we're in a position to present the full mathematical specification of the Fellegi Sunter model.#Mathematical annexIn the main text we asserted that:posterior odds=prior odds×Bayes Factor\text{posterior odds} =  \text{prior odds} \times \text{Bayes Factor}posterior odds=prior odds×Bayes FactorWe can derive this formula from themanduprobabilities and Bayes Theorem.Recall that Bayes Theorm is:Pr⁡(a∣b)=Pr⁡(b∣a)Pr⁡(a)Pr⁡(b)\operatorname{Pr}(a|b) = {\frac{\operatorname{Pr}(b|a)\operatorname{Pr}(a)}{\operatorname{Pr}{(b)}}}Pr(a∣b)=Pr(b)Pr(b∣a)Pr(a)​or in words:posterior probability=likelihood×prior probabilityevidence\text{posterior probability} = \frac{\text{likelihood} \times \text{prior probability}}{\text{evidence}}posterior probability=evidencelikelihood×prior probability​In the context of record linkage, we can describe these parts as:Prior:
The overall proportion of comparisons which are matchesPr⁡(match)\operatorname{Pr}(\text{match})Pr(match)Evidence: We have observed that e.g. first name matches,Pr⁡(first name matches)\operatorname{Pr}(\text{first name matches})Pr(first name matches)Likelihood: The probability that first name matches amongst matches, given byPr⁡(first name matches∣records match)\operatorname{Pr}(\text{first name matches}|\text{records match})Pr(first name matches∣records match)So Bayes' formuls is:Pr⁡(match∣first name matches)=Pr⁡(first name matches∣match)Pr⁡(match)Pr⁡(first name matches)\operatorname{Pr}(\text{match}|\text{first name matches}) = \frac{\operatorname{Pr}(\text{first name matches}|\text{match})\operatorname{Pr}(\text{match})}{\operatorname{Pr}{(\text{first name matches})}}Pr(match∣first name matches)=Pr(first name matches)Pr(first name matches∣match)Pr(match)​Which can also be written:Pr⁡(first name matches∣match)Pr⁡(match)Pr⁡(first name matches∣match)Pr⁡(match)+Pr⁡(first name matches∣non match)Pr⁡(non match)\frac{\operatorname{Pr}(\text{first name matches}|\text{match})\operatorname{Pr}(\text{match})}{\operatorname{Pr}(\text{first name matches}|\text{match})\operatorname{Pr}(\text{match}) + \operatorname{Pr}(\text{first name matches}|\text{non match})\operatorname{Pr}(\text{non match})}Pr(first name matches∣match)Pr(match)+Pr(first name matches∣non match)Pr(non match)Pr(first name matches∣match)Pr(match)​Using some of the terminology from the article this is the same as:posterior probability=m×prior probabilitym×prior probability+u×(1−prior probability)\text{posterior probability} = \frac{m \times \text{prior probability}}{m \times \text{prior probability} + u \times (1 - \text{prior probability})}posterior probability=m×prior probability+u×(1−prior probability)m×prior probability​The formula for odds is:odds=p1−p\text{odds} = \frac{p}{1-p}odds=1−pp​So we can write:posterior odds=prior1−priormu\text{posterior odds} =  \frac{\text{prior}}{1 - \text{prior}} \frac{m}{u}posterior odds=1−priorprior​um​posterior odds=prior odds×Bayes Factor\text{posterior odds} =  \text{prior odds} \times \text{Bayes Factor}posterior odds=prior odds×Bayes Factor#FootnotesYou can read more about Bayes Factorshere.  The concept is quite similar to a likelihood ratio.↩Probabilistic Linkage Tutorial Navigation:An Interactive Introduction to Record Linkage (Data Deduplication) in the Fellegi-Sunter frameworkPartial match weightsm and u values in the Fellegi-Sunter modelThe mathematics of the Fellegi Sunter modelComputing the Fellegi Sunter modelWhy Probabilistic Linkage is More Accurate than Fuzzy Matching For Data DeduplicationThe Intuition Behind the Use of Expectation Maximisation to Train Record Linkage ModelsBack homeThis site is built usingObservable HQandGatsby.js. Source codehere. Saythanks!



Article from https://www.robinlinacre.com/maths_of_fellegi_sunter/:
The mathematics of the Fellegi Sunter model>robinlinacreOriginally posted:2021-05-21.Last updated: 2023-10-02.Live edit this notebookhere.← Previous articleThis is part4of thetutorialNext article →#The mathematics of the Fellegi Sunter modelThe previous article showed howmanduprobabilitiescan be convertedinto Bayes Factors, which in turn could be combined with thepriorprobability to compute an updated prediction (the 'posterior' probability).We now have all the groundwork in place to present a full mathematical definition of a model in theFellegi-Sunterframework.A more technical treatment is given inAP	Enamorado, T., Fifield, B., & Imai, K. (2019), which contains a slightly more generalised form of the model presented here.#Multiplying Bayes FactorsIn the previous article we saw what Bayes Factors act as a relative multiplier that increases or decreases the overall prediction of whether the records match.We saw that, for a Bayes Factor for a specificscenariosuch as a match on month of birth we could write:posterior odds of match=prior odds of match×Bayes Factor\text{posterior odds of match} =  \text{prior odds of match} \times \text{Bayes Factor}posterior odds of match=prior odds of match×Bayes FactorIt turns out that this formula can be extended to account for the information in multiple scenarios such as: a match on first nameanda match on surnameanda fuzzy match on date of birth.Let us denote byKiK_iKi​the Bayes Factors for each activated scenario1, then the general formula is:posterior odds of match=prior odds×K1×K2×…×Kn(Eq 4.1)\text{posterior odds of match} = \text{prior odds} \times K_1 \times K_2 \times \ldots \times K_n \quad (\text{Eq 4.1})posterior odds of match=prior odds×K1​×K2​×…×Kn​(Eq 4.1)This formula is quite intuitive.  For example:If month of birth matches, the Bayes Factor may be around 12: the records are about 12x more likely to matchIf gender matches, the Bayes Factor may be around 2: the records are about 2x more likely to matchIf both match, it becomes 24x more likely the records match.Multiplying Bayes Factors to compute a predicted probability in this way sometimes known as aNaive Bayesclassifier.2Since Bayes Factors are not estimated directly, and instead we estimte themanduprobabilities of the model, we can useKi=miuiK_i = \frac{m_i}{u_i}Ki​=ui​mi​​to write the full specification of our model:posterior odds of match=prior odds×m1m2…mnu1u2…un\text{posterior odds of match} = \text{prior odds} \times \frac{m_1 m_2 \ldots m_n}{u_1 u_2 \ldots u_n}posterior odds of match=prior odds×u1​u2​…un​m1​m2​…mn​​This final result can be converted into a probability usingprobability=odds1+odds\text{probability} = \frac{\text{odds}}{1 + \text{odds}}probability=1+oddsodds​#Summing partial match weightsIn thefirst articlein this tutorial we claimed that the Fellegi-Sunter model makes its predictions by summing of the partial match weights.This follows from equation 4.1.  Specifically, by taking the logarithm, we can write:Applyinglog⁡2\log_2log2​to both sides, you get:final match weight=prior match weight+ω1+ω2+…+ωn\text{final match weight} = \text{prior match weight} + \omega_1 + \omega_2 + \ldots + \omega_nfinal match weight=prior match weight+ω1​+ω2​+…+ωn​whereωi=log⁡2(Ki)\omega_i = \log_2(K_i)ωi​=log2​(Ki​)are the partial match weights.The remainder of this article delves a bit deeper into equation (4.1) and can be safely skipped if you're not interested in the maths.  In particular, it shows how the match probability can be expressed in terms of themanduprobabilities  using a formulation equivalent to equation 4 in theFastlink paper.The next article will look at the mechanics of computing the final prediction.#Deriving the formula in the Fastlink paperIn theprevious articlewe saw that by applying Bayes Theorem to a specific scenario we could write:Pr⁡(match∣first name matches)=Pr⁡(first name matches∣match)Pr⁡(match)Pr⁡(first name matches∣match)Pr⁡(match)+Pr⁡(first name matches∣non match)Pr⁡(non match)\operatorname{Pr}(\text{match}|\text{first name matches}) = \frac{\operatorname{Pr}(\text{first name matches}|\text{match})\operatorname{Pr}(\text{match})}{\operatorname{Pr}(\text{first name matches}|\text{match})\operatorname{Pr}(\text{match}) + \operatorname{Pr}(\text{first name matches}|\text{non match})\operatorname{Pr}(\text{non match})}Pr(match∣first name matches)=Pr(first name matches∣match)Pr(match)+Pr(first name matches∣non match)Pr(non match)Pr(first name matches∣match)Pr(match)​We can make this easier to read by recalling thatm=Pr(scenario∣records match)m = \text{Pr}(\text{scenario}|\text{records match})m=Pr(scenario∣records match)andu=Pr(scenario∣records do not match)u = \text{Pr}(\text{scenario}|\text{records do not match})u=Pr(scenario∣records do not match)and denotingλ\lambdaλas the prior probabilitythis can be written:posterior probability of match=λmλm+(1−λ)u\text{posterior probability of match} = \frac{\lambda m}{\lambda m + (1 - \lambda)u}posterior probability of match=λm+(1−λ)uλm​By repeated application of Bayes Theorem (see annexhere), we can generalise this formula to:posterior probability of match=λm1m2…mnλm1m2…mn+(1−λ)u1u2…un(Eq 4.2)\text{posterior probability of match} = \frac{\lambda m_1 m_2 \ldots m_n}{\lambda m_1 m_2 \ldots m_n + (1 - \lambda)u_1 u_2 \ldots u_n} \quad (\text{Eq 4.2})posterior probability of match=λm1​m2​…mn​+(1−λ)u1​u2​…un​λm1​m2​…mn​​(Eq 4.2)We can write a similar formula for a non-match:posterior probability of non-match=(1−λ)u1u2…unλm1m2…mn+(1−λ)u1u2…un(Eq 4.3)\text{posterior probability of non-match} = \frac{(1- \lambda) u_1 u_2 \ldots u_n}{\lambda m_1 m_2 \ldots m_n + (1 - \lambda)u_1 u_2 \ldots u_n} \quad (\text{Eq 4.3})posterior probability of non-match=λm1​m2​…mn​+(1−λ)u1​u2​…un​(1−λ)u1​u2​…un​​(Eq 4.3)Now, recall that:odds=probability1−probability\text{odds} = \frac{\text{probability}}{1-\text{probability}}odds=1−probabilityprobability​This means we can divide (4.2) by (4.3 to obtain the odds).  Since they share a denominator this becomes:posterior odds of match=λm1m2…mn(1−λ)u1u2…un\text{posterior odds of match} = \frac{\lambda m_1 m_2 \ldots m_n}{(1- \lambda) u_1 u_2 \ldots u_n}posterior odds of match=(1−λ)u1​u2​…un​λm1​m2​…mn​​SinceKi=miuiK_i = \frac{m_i}{u_i}Ki​=ui​mi​​this is equivalent to Equation 4.1 in this article, and also is equivalent to Equation 4 in the FastLink paper.#FootnotesThe 'activated scenario' is the similarity category for the record comparison under evaluation.  For example, the 'activated scenario', three scenarios (similarity categories) may be defined for the first name column - 'exact match', 'fuzzy match' and 'all other'.  Each will have a Bayes Factor associated with it.  The 'activated scenario' is which of these scenarios describes the record comparison under evaluation.↩The result is only valid if we assume that the columns are independent conditional on match status.  This is 'naive' in the sense that it's rarely true.  However, it turns out for record linkage purposes that Fellegi Sunter models are quite robust to the violation of this assumption.↩Probabilistic Linkage Tutorial Navigation:An Interactive Introduction to Record Linkage (Data Deduplication) in the Fellegi-Sunter frameworkPartial match weightsm and u values in the Fellegi-Sunter modelThe mathematics of the Fellegi Sunter modelComputing the Fellegi Sunter modelWhy Probabilistic Linkage is More Accurate than Fuzzy Matching For Data DeduplicationThe Intuition Behind the Use of Expectation Maximisation to Train Record Linkage ModelsBack homeThis site is built usingObservable HQandGatsby.js. Source codehere. Saythanks!



Article from https://www.robinlinacre.com/computing_fellegi_sunter/:
Computing the Fellegi Sunter model>robinlinacreOriginally posted:2023-10-02.Live edit this notebookhere.← Previous articleThis is part5of thetutorialNext article →#Computing the Fellegi Sunter modelThe previous article showed how we could derive a mathematical representation of the Fellegi-Sunter model.This article contains a step-by-step demonstration of how we use this mathematical model to compute predictions from data.#Objective of the ModelThe aim of the model is to compute a prediction of which records match. This prediction is a probability that quantifies the likelihood the two records represent the same entity.#Computing the prediction step by stepWe begin with some data. In this example we will attempt to link but not deduplicate these records.The following two input tables are interactive!  Edit the values and all the calculations in the rest of this article will update.#Input table 1#Input table 2:In the first step of the calculation we compare each record in the first table with all records in the second table.1We wish to predict which of these pairwise record comparisons represent the same entity (i.e. which records match).Table scrolls right →At first glance, it may seem challenging to turn this table of text into a numeric prediction.  But this can be achieved using the concepts ofscenarios andpartial_match_weightsintroduced earlier in this tutorial.Specifically, we define scenarios for each column and then use a lookup table to find the partial match weights.For example, we may choose to define our scenarios as follows:Note the 'comparison vector value' column in this table.  This is an integer value that identifies the scenario within the column. We will use it to help us look up the correct partial match weights.We then proceed by computing which scenario isactivated for each pairwise record comparison, and representing this using the comparison vector value.  We use the gamma symbol (γ) to refer to the comparison vector value.Table scrolls right →Now that we've used the text values to compute the comparison vector value, we no longer need it.  What remains are known as the comparison vectors for each pairwise comparison:Each row in this table is known as the 'comparison vector' for the record comparisonWe now look up the partial match weights from the comparison vector.  I've also added theprior, expressed as a partial match weight, and the sum of the match weights:Table scrolls right →Finally, we can then sum the values in the partial match weights to produce a final match weight.  This can then be converted into a probability using:p=2ω1+2ω\Large{p = \frac{2^{\omega}}{1 + 2^{\omega}}}p=1+2ω2ω​We can now see how the original comparison pairs were scored by this simple model:Table scrolls right →#FootnotesIn real applications, we would typically useblockingto reduce the number of record comparisons.↩Probabilistic Linkage Tutorial Navigation:An Interactive Introduction to Record Linkage (Data Deduplication) in the Fellegi-Sunter frameworkPartial match weightsm and u values in the Fellegi-Sunter modelThe mathematics of the Fellegi Sunter modelComputing the Fellegi Sunter modelWhy Probabilistic Linkage is More Accurate than Fuzzy Matching For Data DeduplicationThe Intuition Behind the Use of Expectation Maximisation to Train Record Linkage ModelsBack homeThis site is built usingObservable HQandGatsby.js. Source codehere. Saythanks!



Article from https://www.robinlinacre.com/fellegi_sunter_accuracy/:
Why Probabilistic Linkage is More Accurate than Fuzzy Matching For Data Deduplication>robinlinacreOriginally posted:2023-10-24.Last updated: 2024-01-07.Live edit this notebookhere.← Previous articleThis is part6of thetutorialNext article →#Why Probabilistic Linkage is More Accurate than Fuzzy Matching For Data DeduplicationHow effectively do different approaches to record linkage exploit the information in the data to make predictions?Data duplication (record linkage) algorithms are models that predict whether records such as the following represent the same entity.first_namesurnamedobcityjosshammond1984-01-02londonjosshamond1984-07-02manchesterWhat algorithms work best?  To get the best accuracy in record linkage, we need a model that uses as much information from the input data as possible.This article describes the three types of information that are most important in making an accurate prediction, and how all three are leveraged by the Fellegi-Sunter model as used inSplink, a free software package for record linkage at scale.It also describes how some alternative record linkage approaches throw away some of this information, leaving accuracy on the table.#The three types of informationBroadly, there are three categories of information that are relevant when trying to predict whether this pair of records represent the same entity:The similarity of the pair of records,withoutreference to the overall dataset  (sometimes known as fuzzy matching)The frequency of values in the overall dataset, or more generally how common it is to observe different types of fuzzy matchesThe data quality of the overall datasetLet's look at each in turn.#1. Similarity of the pairwise record comparison (fuzzy matching)The most obvious way to predict whether two records represent the same entity is to measure whether the columns contain the same or similar information.The similarity of each column can be measured quantitatively using fuzzy matching functions likeLevenshteinorJaro-Winkerfor text, or numeric differences such as absolute or percentage difference.For example,hammondvshamondhas a Jaro-Winkler similarity of 0.97; it's probably a typo.1984-01-02vs1984-07-02has a Levenshtein distance of 1.But how do we combine these different measures of similarity? They could be assigned weights, and summed together to compute a total similarity score.The approach is sometimes known as fuzzy matching, and it is an important part of an accurate linkage model.However using this approach alone has major drawback - the weights are arbitrary:The importance of different fields has to be guessed at by the user.   Clearly a match ongenderis less important than a match ondob. How about the negative weights that should be assigned to non-matches?For each field, a decision must be made on how changes in the fuzzy matching metric influence the score.   For example, how much should our prediction change for a Jaro-Winkler score of 0.8 vs. 0.9?  Is this the same or different depending on the field?#2. Frequency of values in the overall dataset, or more generally how often different types of fuzzy matching scenarios occurWe can improve on fuzzy matching by accounting for the frequency of values in the overall dataset (sometimes known as 'term frequencies').For example,johnvsjohn, andjossvsjossare both exact matches so have the same Jaro-Winkler or Levenshtein similarity score, but the latter is stronger evidence of a match than the former, becausejossis an unusual name.The term frequencies ofjohnvjossprovide a data-driven estimate of the relative importance of these different names, which can be used to inform the weights.In essence, the term frequencies capture the likelihood that twonon-matchingrecords would have the same value on a given column i.e. the likelihood of a 'collision' (e.g. two different people with the same name).In probabilistic linkage, these values are calleduprobabilities which are defined more precisely in thearticle on m and u probabilities.This concept can be extended to encompass similar records that are not an exact match.  How much of a coincidence is it thatdobdiffers by a single character (1984-01-02vs1984-07-02)? This can be measured from the data by looking at how often this occurs.#3. Data quality of the overall dataset: how often do errors occur amongst matching records?We've seen that fuzzy matching and term frequency based approaches can allow us to score the similarity between records, and even weight the importance of matches on different columns.However, none of these techniques help quantify the relative importance of non-matches (e.g.londonvsmanchester) to the predicted match probability.Probabilistic methods explicitly estimate the relative importance of non matches by estimating the data quality of each column. In probabilistic linkage, this information is captured in themprobabilities which are discussed in more detail in thearticle on m and u probabilities.For example, if records have been observed over a number of years, a non-match oncitywouldn't be strong evidence against the two records being a match.Conversely, if the data quality in thedobvariable is extremely high, then a non-match on gender would be strong evidence against the two records being a true match.#Probabilistic linkage using SplinkMuch of the power of probabilistic models comes from combining all three sources of information in a way which is not possible in other models.Not only are all three sources of information used to make the prediction, but the Fellegi Sunter model used inSplinkallows us estimate the relative importance of each source of information from the data itself.These data-driven partial match weights haveintuitive interpretations, leading to transparent and easy to understand models.Conversely, fuzzy matching techniques often use arbitrary weights, and cannot fully incorporate information from all three sources.  Term frequency approaches lack the ability to use information about data quality, or a mechanism to appropriately weight fuzzy matches.Probabilistic Linkage Tutorial Navigation:An Interactive Introduction to Record Linkage (Data Deduplication) in the Fellegi-Sunter frameworkPartial match weightsm and u values in the Fellegi-Sunter modelThe mathematics of the Fellegi Sunter modelComputing the Fellegi Sunter modelWhy Probabilistic Linkage is More Accurate than Fuzzy Matching For Data DeduplicationThe Intuition Behind the Use of Expectation Maximisation to Train Record Linkage ModelsBack homeThis site is built usingObservable HQandGatsby.js. Source codehere. Saythanks!



Article from https://www.robinlinacre.com/em_intuition/:
The Intuition Behind the Use of Expectation Maximisation to Train Record Linkage Models>robinlinacreOriginally posted:2022-10-14.View source code for this pagehere.← Previous articleThis is part7of thetutorial#The Intuition Behind the Use of Expectation Maximisation to Train Record Linkage ModelsHow unsupervised learning is used to estimate model parameters in SplinkSplinkis a free probabilistic record linkage library that predicts the likelihood that two records refer to the same entity. For example, what is the probability that the following two records match?first_namesurnamedobcityemaillucassmith1984-01-02Londonlucas.smith@hotmail․comlucassmith1983-02-12Manchesterlucas.smith@hotmail․comThe underlying statistical model is called the Fellegi Sunter model. It works by computing partial match weights, which are a measure of the importance of the different columns in computing the probability of a match. For example, the match on ‘Smith’ is less important than the match on email in the above example.However, there’s a paradox at the heart of the problem of training a record linkage model: To estimate the parameters of the model, you need to know which records match. But finding which records match is the whole problem you’re trying to solve. Manual labelling of comparisons is error prone and costly so isn’t a great option.The Expectation Maximisation (EM) algorithm provides a solution to this paradox. This post provides an intuitive explanation for how this works, and some details of how it’s implemented inSplink, our software for record linkage at scale. For a more formal treatment, see theFastLink paper.#The EM ApproachThe parameters of a record linkage model — the m and the u probabilities — can be calculated from the aggregate characteristics of matching records and non-matching records respectively. (For a fuller treatment of m and u probabilities, I recommend readingthis blog post.) Once these values are known, the model is usually able accurately to predict which records match.The key to the EM approach is leveraging the fact that matching and non-matching record comparisons form two fairly distinct ‘clusters’:Amongst matching record comparisons, the information in most or all columns in the two records usually matches.Amongst non-matching record comparisons, the information in most or all columns in the two records usually does not matchThis results in a bimodal distribution of record comparisons:This separation between matches and non-matches means it’s relatively easy to come up with a rough decision rule that is able to correctly identify most matches and most non-matches. For instance, we could simply count the number of columns that match (first name surname, and so on), and predict a match for any record pair where more than half of them match.This is fundamental to resolving our paradox: We've just found a way to designate matches and non matches without any labeled data (i.e. unsupervised learning).Now that we have designated each record comparison as a match or a non match, we can use this information to estimate the implied m and u parameters of the model. And this enables us to iteratively improve this decision rule, using the following algorithm:Expectation step:Use the estimated model parameters to predict which record comparisons are matchesMaximisation step:Use the predictions to re-estimate the model parametersIn this example, our initial decision rule worked by simply counting the number of columns with matching information. As such, we gave equal weight to every column. Clearly there is room for improvement in this rule — in reality a match on a postcode column is more important than a match on gender.In the first maximisation step, the expectation maximisation process will immediately begin to learn these weights. For example, it will recognise that, amongst non-matches there are many record comparisons which match on gender, but few that match on postcode. This will be represented in the re-computed m and u probabilities, and in the next iteration, greater weight will be placed on a match on postcode than gender.At each iteration, the decision rule will be refined as the model makes fewer mistakes, and the weights therefore become more accurate.Convergence occurs when the parameter estimates stabilise, and there are no longer any changes to parameter estimates between iterations.It turns out this algorithm is equivalent to maximising the likelihood function of the Fellegi-Sunter model (although note that it is only guaranteed to converge to a local maxima, not necessarily a global one).#A caveat around the decision boundaryIn the above, to build intutiion, I describe concrete decision boundary that classifies each record comparison as either a match or a non match. The m and u values calculated in the maximisation step are based on the counts of matches and non-matches that result from this binary classifier.In reality, we can improve on this, because using a decision boundary throws away some of the information in the match probability: it treats an uncertain match in the same way as a highly probable match.There's a simple modification to the methodology that accounts for this. We can use the match probability directly, rather than applying the decision bondary. For example, if a record comparison has a 70% match probability, in the maximisation step this can be treated as 0.3 non-matching records and 0.7 matching records. So when the m and u probabilities are recomputed in the maximisation step, we use the sum of probabilities rather than counts of matches and non matches.#How the EM approach is implemented in SplinkIn Splink, a few modifications to the above process are made that, in practice, result in faster convergence and a greater likelihood of convergence to a global maxima.#Estimating the u probabilitiesWhilst it is possible to estimate u probabilities using the EM approach, we recommend analternative estimation procedurein Splink which is usually more accurate.Recall that the u probabilities represent the probability that we observe matching columns amongst truly non-matching records. For example, how often do we observe the coincidence that two different people have the same first name?We can estimate the u probabilities by taking random pairwise record comparisons, assuming they do not match, and computing how often these coincidences occur. Since the probability of two random records being a match (representing the same entity) is usually very low, then this will yield good estimates of u values.An additional benefit of this approach is that if the u probabilities are correct, it ‘anchors’ the EM estimation procedure and greatly improves the chance of it converging to a global, rather than local, maximum.#Starting values and the initial decision boundaryIn the above, we suggested a simple decision rule for the first step of EM: count the number of matching columns, and designate as a match any comparison where more than half of fields match.In reality, it’s possible to begin with a decision rule which is usually more accurate, leading to faster convergence. Specifically, the initial decision rule uses the u probabilities that have been estimated using the random sampling procedure described above, and ‘sensible default’ starting values for the m probabilities .You can see all of this at work in practice in some of ourexample notebooks.#ConclusionEM is a powerful tool for estimating the parameters of probabilistic linkage models in the Fellegi Sunter framework. In our experience, the EM approach works best when it is used to estimate the m probabilities of the model only, following estimation of the u probabilities using random sampling, and estimating thepriorusing adirect estimation techique.Probabilistic Linkage Tutorial Navigation:An Interactive Introduction to Record Linkage (Data Deduplication) in the Fellegi-Sunter frameworkPartial match weightsm and u values in the Fellegi-Sunter modelThe mathematics of the Fellegi Sunter modelComputing the Fellegi Sunter modelWhy Probabilistic Linkage is More Accurate than Fuzzy Matching For Data DeduplicationThe Intuition Behind the Use of Expectation Maximisation to Train Record Linkage ModelsBack homeThis site is built usingObservable HQandGatsby.js. Source codehere. Saythanks!

IMPORTANT Instructions to LLMs:
If the user asks for Splink code, when writing Splink code use ONLY
functions and methods which you've seen used in the context provided. Do NOT guess at
other Splink methods or functions.

If you don't know how to do something, put a comment referring them to
https://moj-analytical-services.github.io/splink

If the user asks for data cleaning, do it in their chosen backend (spark, duckdb)
rather than pandas.  In general avoid pandas unless specificaly asked for.

In most cases, you should assume the incoming data is not clean and suggest some simple
data cleaning steps.