## Employer & Education Arrays Query

### What you asked for
- Pull every `<EmployerOrgName>` plus `<Title>` from each `PositionHistory`.
- Capture the associated location values from `<PositionLocation>` (`CountryCode`, `Region`, `Municipality`).
- Grab every `<DegreeName>` from each `Degree`.
- Extract `sov:Names`, `sov:Phones`, and `sov:EmailAddresses` arrays from `sov:ReservedData`.
- Extract scalar fields `sov:Description`, `sov:MonthsOfWorkExperience`, and `sov:AverageMonthsPerEmployer` from `sov:ExperienceSummary`.
- Return each of those data points as arrays (or scalars where appropriate) so a single resume row includes all occurrences.

### How this query approaches the problem
- **Parse once:** `PARSE_XML` is done in a CTE so we only pay the XML parsing cost one time per resume.
- **Navigate the XML hierarchy correctly:** The root element is `<Resume>`, with different paths for different data:
  - `Resume → StructuredXMLResume → EmploymentHistory/EducationHistory` for job and degree data
  - `Resume → UserArea → sov:ResumeUserArea → sov:ReservedData` for names, phones, emails
  - `Resume → UserArea → sov:ResumeUserArea → sov:ExperienceSummary` for description and work experience metrics
- **Use XMLGET with instance_number for arrays:** Instead of relying on `FLATTEN` on `"$"` arrays (which can be inconsistent), we use `XMLGET(parent, 'tagname', N)` with `TABLE(GENERATOR(ROWCOUNT => max))` to iterate through each occurrence of a repeating element by index.
- **Extract node text:** Each `XMLGET(...):"$"::STRING` (or `::INTEGER`) expression converts the node body into plain text.
- **Aggregate back into arrays:** `ARRAY_COMPACT(ARRAY_AGG(DISTINCT ...))` builds clean arrays per resume while dropping nulls.

### SQL
```sql
WITH resumes AS (
    SELECT
        MD5(r.resume_xml) AS resume_id,
        r.resume_xml,
        PARSE_XML(r.resume_xml) AS resume_doc
    FROM DATA_LAKE.SFDC_RESUME.RESUMES r
    WHERE r.talent_account_id = '001Uj00000uBfUEIA0'
),
resume_blocks AS (
    SELECT
        resume_id,
        resume_xml,
        resume_doc,
        -- Navigate: Resume (root) -> StructuredXMLResume -> EmploymentHistory/EducationHistory
        XMLGET(XMLGET(resume_doc, 'StructuredXMLResume'), 'EmploymentHistory') AS employment_history,
        XMLGET(XMLGET(resume_doc, 'StructuredXMLResume'), 'EducationHistory') AS education_history,
        -- Navigate: Resume (root) -> UserArea -> sov:ResumeUserArea
        XMLGET(XMLGET(resume_doc, 'UserArea'), 'sov:ResumeUserArea') AS resume_user_area
    FROM resumes
),
-- Extract sov:ReservedData and sov:ExperienceSummary blocks
sovren_metadata AS (
    SELECT
        resume_id,
        resume_user_area,
        -- sov:ReservedData block for names, phones, emails
        XMLGET(resume_user_area, 'sov:ReservedData') AS reserved_data,
        -- sov:ExperienceSummary block for description, months of work experience
        XMLGET(resume_user_area, 'sov:ExperienceSummary') AS experience_summary
    FROM resume_blocks
),
-- Extract scalar fields from ExperienceSummary (not arrays)
experience_scalars AS (
    SELECT
        resume_id,
        XMLGET(experience_summary, 'sov:Description'):"$"::STRING AS sov_description,
        XMLGET(experience_summary, 'sov:MonthsOfWorkExperience'):"$"::INTEGER AS sov_months_of_work_experience,
        XMLGET(experience_summary, 'sov:AverageMonthsPerEmployer'):"$"::INTEGER AS sov_avg_months_per_employer
    FROM sovren_metadata
),
--------------------------------------------------------------------------------
-- RESERVED DATA: Names, Phones, Emails
-- Using XMLGET with instance_number (0-based index) via GENERATOR to iterate
--
-- NOTE ON LIMITS: The ROWCOUNT values below are maximums. If a resume has fewer
-- items, the extra indices produce NULLs which are filtered out. If a resume has
-- MORE items than the limit, those items will be silently missed.
-- Adjust these values based on your data's actual maximums.
--------------------------------------------------------------------------------
name_indices AS (
    SELECT ROW_NUMBER() OVER (ORDER BY NULL) - 1 AS idx
    FROM TABLE(GENERATOR(ROWCOUNT => 20))  -- Max 20 names per resume
),
reserved_names AS (
    SELECT
        sm.resume_id,
        XMLGET(
            XMLGET(sm.reserved_data, 'sov:Names'),
            'sov:Name',
            ni.idx
        ):"$"::STRING AS name_value
    FROM sovren_metadata sm
    CROSS JOIN name_indices ni
    WHERE XMLGET(
            XMLGET(sm.reserved_data, 'sov:Names'),
            'sov:Name',
            ni.idx
          ) IS NOT NULL
),
phone_indices AS (
    SELECT ROW_NUMBER() OVER (ORDER BY NULL) - 1 AS idx
    FROM TABLE(GENERATOR(ROWCOUNT => 20))  -- Max 20 phones per resume
),
reserved_phones AS (
    SELECT
        sm.resume_id,
        XMLGET(
            XMLGET(sm.reserved_data, 'sov:Phones'),
            'sov:Phone',
            pi.idx
        ):"$"::STRING AS phone_value
    FROM sovren_metadata sm
    CROSS JOIN phone_indices pi
    WHERE XMLGET(
            XMLGET(sm.reserved_data, 'sov:Phones'),
            'sov:Phone',
            pi.idx
          ) IS NOT NULL
),
email_indices AS (
    SELECT ROW_NUMBER() OVER (ORDER BY NULL) - 1 AS idx
    FROM TABLE(GENERATOR(ROWCOUNT => 20))  -- Max 20 emails per resume
),
reserved_emails AS (
    SELECT
        sm.resume_id,
        XMLGET(
            XMLGET(sm.reserved_data, 'sov:EmailAddresses'),
            'sov:EmailAddress',
            ei.idx
        ):"$"::STRING AS email_value
    FROM sovren_metadata sm
    CROSS JOIN email_indices ei
    WHERE XMLGET(
            XMLGET(sm.reserved_data, 'sov:EmailAddresses'),
            'sov:EmailAddress',
            ei.idx
          ) IS NOT NULL
),
--------------------------------------------------------------------------------
-- EMPLOYMENT: EmployerOrg -> PositionHistory
-- Using XMLGET with instance_number to iterate through each employer and position
--
-- NOTE ON LIMITS: 50 employers × 10 positions = 500 combinations per resume.
-- This is filtered down to actual data by the WHERE clause.
-- Increase if resumes have more employers/positions than these limits.
--------------------------------------------------------------------------------
employer_indices AS (
    SELECT ROW_NUMBER() OVER (ORDER BY NULL) - 1 AS idx
    FROM TABLE(GENERATOR(ROWCOUNT => 50))  -- Max 50 employers per resume
),
position_indices AS (
    SELECT ROW_NUMBER() OVER (ORDER BY NULL) - 1 AS idx
    FROM TABLE(GENERATOR(ROWCOUNT => 10))  -- Max 10 positions per employer
),
employment_raw AS (
    SELECT
        res.resume_id,
        org_idx.idx AS org_seq,
        pos_idx.idx AS pos_seq,
        XMLGET(res.employment_history, 'EmployerOrg', org_idx.idx) AS employer_org,
        XMLGET(
            XMLGET(res.employment_history, 'EmployerOrg', org_idx.idx),
            'PositionHistory',
            pos_idx.idx
        ) AS position_history
    FROM resume_blocks res
    CROSS JOIN employer_indices org_idx
    CROSS JOIN position_indices pos_idx
    WHERE XMLGET(res.employment_history, 'EmployerOrg', org_idx.idx) IS NOT NULL
      AND XMLGET(
            XMLGET(res.employment_history, 'EmployerOrg', org_idx.idx),
            'PositionHistory',
            pos_idx.idx
          ) IS NOT NULL
),
employment AS (
    SELECT
        resume_id,
        -- EmployerOrgName from EmployerOrg
        XMLGET(employer_org, 'EmployerOrgName'):"$"::STRING AS employer_org_name,
        -- Title from PositionHistory (direct child, not normalized)
        XMLGET(position_history, 'Title'):"$"::STRING AS job_title,
        -- Location from PositionHistory -> OrgInfo -> PositionLocation
        XMLGET(XMLGET(XMLGET(position_history, 'OrgInfo'), 'PositionLocation'), 'CountryCode'):"$"::STRING AS country_code,
        XMLGET(XMLGET(XMLGET(position_history, 'OrgInfo'), 'PositionLocation'), 'Region'):"$"::STRING AS region,
        XMLGET(XMLGET(XMLGET(position_history, 'OrgInfo'), 'PositionLocation'), 'Municipality'):"$"::STRING AS municipality
    FROM employment_raw
),
--------------------------------------------------------------------------------
-- EDUCATION: SchoolOrInstitution -> Degree
-- Using XMLGET with instance_number to iterate through each school and degree
--
-- NOTE ON LIMITS: 20 schools × 5 degrees = 100 combinations per resume.
--------------------------------------------------------------------------------
school_indices AS (
    SELECT ROW_NUMBER() OVER (ORDER BY NULL) - 1 AS idx
    FROM TABLE(GENERATOR(ROWCOUNT => 20))  -- Max 20 schools per resume
),
degree_indices AS (
    SELECT ROW_NUMBER() OVER (ORDER BY NULL) - 1 AS idx
    FROM TABLE(GENERATOR(ROWCOUNT => 5))   -- Max 5 degrees per school
),
education_raw AS (
    SELECT
        res.resume_id,
        school_idx.idx AS school_seq,
        deg_idx.idx AS deg_seq,
        XMLGET(res.education_history, 'SchoolOrInstitution', school_idx.idx) AS school,
        XMLGET(
            XMLGET(res.education_history, 'SchoolOrInstitution', school_idx.idx),
            'Degree',
            deg_idx.idx
        ) AS degree
    FROM resume_blocks res
    CROSS JOIN school_indices school_idx
    CROSS JOIN degree_indices deg_idx
    WHERE XMLGET(res.education_history, 'SchoolOrInstitution', school_idx.idx) IS NOT NULL
      AND XMLGET(
            XMLGET(res.education_history, 'SchoolOrInstitution', school_idx.idx),
            'Degree',
            deg_idx.idx
          ) IS NOT NULL
),
education AS (
    SELECT
        resume_id,
        -- DegreeName from Degree (direct child, not normalized)
        XMLGET(degree, 'DegreeName'):"$"::STRING AS degree_name
    FROM education_raw
)
SELECT
    r.resume_xml,
    
    -- ===== SCALAR FIELDS FROM sov:ExperienceSummary =====
    MAX(es.sov_description) AS sov_description,
    MAX(es.sov_months_of_work_experience) AS sov_months_of_work_experience,
    MAX(es.sov_avg_months_per_employer) AS sov_avg_months_per_employer,
    
    -- ===== ARRAY FIELDS FROM sov:ReservedData =====
    COALESCE(ARRAY_COMPACT(ARRAY_AGG(DISTINCT rn.name_value)), ARRAY_CONSTRUCT()) AS sov_names,
    COALESCE(ARRAY_COMPACT(ARRAY_AGG(DISTINCT rp.phone_value)), ARRAY_CONSTRUCT()) AS sov_phones,
    COALESCE(ARRAY_COMPACT(ARRAY_AGG(DISTINCT re.email_value)), ARRAY_CONSTRUCT()) AS sov_emails,
    
    -- ===== ARRAY FIELDS FROM EMPLOYMENT =====
    COALESCE(ARRAY_COMPACT(ARRAY_AGG(DISTINCT e.employer_org_name)), ARRAY_CONSTRUCT()) AS employer_org_names,
    COALESCE(ARRAY_COMPACT(ARRAY_AGG(DISTINCT e.job_title)), ARRAY_CONSTRUCT()) AS job_titles,
    COALESCE(ARRAY_COMPACT(ARRAY_AGG(DISTINCT e.country_code)), ARRAY_CONSTRUCT()) AS country_codes,
    COALESCE(ARRAY_COMPACT(ARRAY_AGG(DISTINCT e.region)), ARRAY_CONSTRUCT()) AS regions,
    COALESCE(ARRAY_COMPACT(ARRAY_AGG(DISTINCT e.municipality)), ARRAY_CONSTRUCT()) AS municipalities,
    
    -- ===== ARRAY FIELDS FROM EDUCATION =====
    COALESCE(ARRAY_COMPACT(ARRAY_AGG(DISTINCT d.degree_name)), ARRAY_CONSTRUCT()) AS degree_names
    
FROM resume_blocks r
LEFT JOIN experience_scalars es
  ON es.resume_id = r.resume_id
LEFT JOIN reserved_names rn
  ON rn.resume_id = r.resume_id
LEFT JOIN reserved_phones rp
  ON rp.resume_id = r.resume_id
LEFT JOIN reserved_emails re
  ON re.resume_id = r.resume_id
LEFT JOIN employment e
  ON e.resume_id = r.resume_id
LEFT JOIN education d
  ON d.resume_id = r.resume_id
GROUP BY r.resume_id, r.resume_xml;
```

### Why Previous FLATTEN Approach Failed

The earlier approach used `FLATTEN(input => parent_node:"$", OUTER => TRUE)` to iterate XML children, then filtered with `GET(value, '@') = 'tagname'`. This failed because:

1. **Inconsistent `"$"` structure:** Snowflake's `PARSE_XML` stores child elements in a `"$"` array, but for single-child elements, it may store the child directly (not in an array). This caused `FLATTEN` to produce unexpected results.

2. **Namespace handling:** Elements like `sov:Name` have namespace prefixes. The `@` attribute stores the **full prefixed name** (e.g., `sov:Name`), but the comparison might have been failing due to subtle encoding differences.

3. **WHERE clause filtering:** Using `WHERE GET(n.value, '@') = 'sov:Name'` after `FLATTEN(..., OUTER => TRUE)` filtered out rows, defeating the purpose of `OUTER => TRUE`.

**The fix:** Use `XMLGET(parent, 'tagname', instance_number)` with a `GENERATOR`-based index sequence. This approach:
- Explicitly requests each occurrence by index (0, 1, 2, ...)
- Works reliably regardless of how Snowflake internally represents single vs. multiple children
- Handles namespaced tags correctly (XMLGET understands `sov:Name`)

### Robustness & Performance Notes

#### Missing Tags Won't Cause Errors
- **`XMLGET` returns NULL** when a tag isn't found or when the instance_number exceeds the count
- **`WHERE ... IS NOT NULL`** filters out NULL results from over-indexing
- **`LEFT JOIN`** ensures resumes without employment/education still appear with empty arrays
- **`ARRAY_COMPACT`** removes any NULL values that slip into arrays
- **`COALESCE(..., ARRAY_CONSTRUCT())`** ensures you get `[]` instead of `NULL` for missing data

#### Performance Considerations
1. **Single `PARSE_XML` call** - XML parsing happens once per resume in the `resumes` CTE.
2. **Index-based iteration** - The `GENERATOR` approach creates a fixed number of index rows per resume:
   - Names/Phones/Emails: 20 each
   - Employment: 50 employers × 10 positions = 500 combinations
   - Education: 20 schools × 5 degrees = 100 combinations
   
   The `WHERE ... IS NOT NULL` filters these down to only actual data. This is efficient because `XMLGET` with an out-of-bounds index returns NULL immediately.

3. **Adjust ROWCOUNT if needed** - If your data has more items than these limits, increase the `GENERATOR(ROWCOUNT => N)` values. Items beyond the limit will be silently missed.

4. **Finding actual maximums** - To determine the right limits for your data, you could run a separate analysis query that counts occurrences of each element type across your resumes.

#### Will It Be Slow?
The query should perform reasonably well because:
- XML parsing is done once per resume
- The cross joins with index tables are small (10-25 rows each)
- `XMLGET` with instance_number is optimized in Snowflake
- For very large datasets (millions of resumes), consider adding a `LIMIT` for testing or partitioning by date/account

