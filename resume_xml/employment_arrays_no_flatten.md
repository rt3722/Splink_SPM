## Array extraction without `FLATTEN()`

### Why try this?
- You asked for an option that doesn’t rely on `FLATTEN`. Snowflake’s XML tooling lets you “index” into repeated tags via the optional `instance_number` parameter of `XMLGET`. Per the docs, that argument is **0-based** and defaults to `0`, meaning you only ever see the first match unless you loop over additional indexes (`XMLGET(<expr>, <tag>, <instance_number>)`, see Snowflake XMLGET reference lines `882-973` in the captured docs).
- The Snowflake knowledge-base article *“HOW TO QUERY NESTED XML DATA IN SNOWFLAKE”* stresses that `FLATTEN()` + lateral joins are the usual way to walk XML arrays, but it also shows each nested level being accessed explicitly with `XMLGET` before the flatten occurs. We can mimic that traversal by iterating over the possible tag instances ourselves using `TABLE(GENERATOR())` instead of `FLATTEN`.

### How it works
1. `TABLE(GENERATOR(ROWCOUNT => N))` produces a predictable set of integers (`seq4`) that we treat as candidate indexes into each repeating tag (`EmployerOrg`, `PositionHistory`, `SchoolOrInstitution`, `Degree`).
2. For each index we call `XMLGET(..., <tag>, seq4)`; if the tag doesn’t exist at that ordinal the call returns `NULL`, and we simply filter that slot away.
3. Because we never build the FLATTEN array, the only rows that survive are those where the node actually exists, and we can still `ARRAY_AGG` the text results.
4. The `ROWCOUNT` caps (25 employers per resume, 15 positions per employer, 15 schools, 10 degrees) are conservative and can be tuned upward if your data contains longer histories.

### SQL (no `FLATTEN`)
```sql
WITH resumes AS (
    SELECT
        MD5(r.resume_xml) AS resume_id,
        r.resume_xml,
        PARSE_XML(r.resume_xml) AS resume_doc
    FROM DATA_LAKE.SFDC_RESUME.RESUMES r
    WHERE r.talent_account_id = '001Uj00000uBfUEIA0'
),
employment AS (
    SELECT
        r.resume_id,
        XMLGET(org_node, 'EmployerOrgName'):"$"::STRING                         AS employer_org_name,
        XMLGET(
            XMLGET(XMLGET(position_node, 'UserArea'), 'sov:PositionHistoryUserArea'),
            'sov:NormalizedOrganizationName'
        ):"$"::STRING                                                           AS sov_normalized_org_name,
        XMLGET(
            XMLGET(XMLGET(position_node, 'UserArea'), 'sov:PositionHistoryUserArea'),
            'sov:NormalizedTitle'
        ):"$"::STRING                                                           AS sov_normalized_title,
        XMLGET(
            XMLGET(XMLGET(position_node, 'UserArea'), 'sov:PositionHistoryUserArea'),
            'sov:Id'
        ):"$"::STRING                                                           AS sov_position_id,
        XMLGET(
            XMLGET(XMLGET(position_node, 'OrgInfo'), 'PositionLocation'),
            'CountryCode'
        ):"$"::STRING                                                           AS country_code,
        XMLGET(
            XMLGET(XMLGET(position_node, 'OrgInfo'), 'PositionLocation'),
            'Region'
        ):"$"::STRING                                                           AS region,
        XMLGET(
            XMLGET(XMLGET(position_node, 'OrgInfo'), 'PositionLocation'),
            'Municipality'
        ):"$"::STRING                                                           AS municipality
    FROM resumes r
    CROSS JOIN LATERAL (
        SELECT org_node
        FROM (
            SELECT XMLGET(
                       XMLGET(
                           XMLGET(r.resume_doc, 'StructuredXMLResume'),
                           'EmploymentHistory'
                       ),
                       'EmployerOrg',
                       employer_slot.seq4
                   ) AS org_node
            FROM TABLE(GENERATOR(ROWCOUNT => 25)) employer_slot
        )
        WHERE org_node IS NOT NULL
    ) org
    CROSS JOIN LATERAL (
        SELECT position_node
        FROM (
            SELECT XMLGET(org.org_node, 'PositionHistory', position_slot.seq4) AS position_node
            FROM TABLE(GENERATOR(ROWCOUNT => 15)) position_slot
        )
        WHERE position_node IS NOT NULL
    ) position
),
education AS (
    SELECT
        r.resume_id,
        XMLGET(
            XMLGET(
                XMLGET(degree_node, 'UserArea'),
                'sov:DegreeUserArea'
            ),
            'sov:NormalizedDegreeName'
        ):"$"::STRING AS normalized_degree_name
    FROM resumes r
    CROSS JOIN LATERAL (
        SELECT school_node
        FROM (
            SELECT XMLGET(
                       XMLGET(
                           XMLGET(r.resume_doc, 'StructuredXMLResume'),
                           'EducationHistory'
                       ),
                       'SchoolOrInstitution',
                       school_slot.seq4
                   ) AS school_node
            FROM TABLE(GENERATOR(ROWCOUNT => 15)) school_slot
        )
        WHERE school_node IS NOT NULL
    ) school
    CROSS JOIN LATERAL (
        SELECT degree_node
        FROM (
            SELECT XMLGET(school.school_node, 'Degree', degree_slot.seq4) AS degree_node
            FROM TABLE(GENERATOR(ROWCOUNT => 10)) degree_slot
        )
        WHERE degree_node IS NOT NULL
    ) degree
)
SELECT
    r.resume_xml,
    ARRAY_AGG(DISTINCT CASE WHEN e.employer_org_name IS NOT NULL THEN e.employer_org_name END)           AS employer_org_names,
    ARRAY_AGG(DISTINCT CASE WHEN e.sov_normalized_org_name IS NOT NULL THEN e.sov_normalized_org_name END) AS normalized_org_names,
    ARRAY_AGG(DISTINCT CASE WHEN e.sov_normalized_title IS NOT NULL THEN e.sov_normalized_title END)       AS normalized_titles,
    ARRAY_AGG(DISTINCT CASE WHEN e.sov_position_id IS NOT NULL THEN e.sov_position_id END)                 AS position_ids,
    ARRAY_AGG(DISTINCT CASE WHEN e.country_code IS NOT NULL THEN e.country_code END)                       AS country_codes,
    ARRAY_AGG(DISTINCT CASE WHEN e.region IS NOT NULL THEN e.region END)                                   AS regions,
    ARRAY_AGG(DISTINCT CASE WHEN e.municipality IS NOT NULL THEN e.municipality END)                       AS municipalities,
    ARRAY_AGG(DISTINCT CASE WHEN d.normalized_degree_name IS NOT NULL THEN d.normalized_degree_name END)   AS normalized_degree_names
FROM resumes r
LEFT JOIN employment e
  ON e.resume_id = r.resume_id
LEFT JOIN education d
  ON d.resume_id = r.resume_id
GROUP BY r.resume_id, r.resume_xml;
```

### Notes
- The generator counts (`25`, `15`, `10`) make the query deterministic but can be increased if you have resumes with more entries; Snowflake’s cost for those lateral generators is negligible because they’re evaluated per resume only until the requested rowcount.
- If arrays still come back empty, temporarily `SELECT org.org_node, position.position_node` to inspect what the generator is retrieving—if the nodes are `NULL`, the problem is higher up (e.g., namespace mismatch or resume documents that don’t actually contain the nodes).
- If you’d rather revert to the `FLATTEN` approach later, remember the knowledge-base guidance: flatten the array you actually need (e.g., `FLATTEN(dept_emp_addr.xmldata:"$") emp`) and then read `emp.value` for the node’s payload (see the Snowflake KB excerpt above).

