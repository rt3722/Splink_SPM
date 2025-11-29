## Resume Data Extraction - Flat Columns Approach

### What this query does
Instead of aggregating repeating XML elements into arrays, this query extracts each occurrence into its own numbered column:
- `name_1`, `name_2`, `name_3` instead of `sov_names[]`
- `email_1`, `email_2`, `email_3` instead of `sov_emails[]`
- `phone_1`, `phone_2`, `phone_3` instead of `sov_phones[]`
- `employer_1`, `title_1`, `employer_2`, `title_2`, etc. for employment
- `degree_1`, `degree_2`, `degree_3` for education

### How it works
Each column uses `XMLGET(parent, 'tagname', instance_number)` where:
- `instance_number = 0` → first occurrence → `_1` column
- `instance_number = 1` → second occurrence → `_2` column
- `instance_number = 2` → third occurrence → `_3` column

### Will it error if data is missing?
**No.** When `XMLGET` is called with an `instance_number` that doesn't exist:
- It returns `NULL`
- The column simply has a NULL value
- No error is thrown

Example: If a resume has only 1 name:
- `name_1` = "John Doe"
- `name_2` = NULL
- `name_3` = NULL

### Trade-offs vs Array Approach

| Aspect | This (Flat Columns) | Array Approach |
|--------|---------------------|----------------|
| **Schema** | Fixed columns | Dynamic array length |
| **Nulls** | Columns may be NULL | No nulls in arrays |
| **Querying** | Direct column access | Need array functions |
| **BI Tools** | Easy to use | May need transformation |
| **Beyond limit** | Data missed | Data missed |
| **Query complexity** | Simple, single SELECT | Complex CTEs + joins |

### Column Limits in This Query
- Names: 3 columns (`name_1`, `name_2`, `name_3`)
- Phones: 3 columns (`phone_1`, `phone_2`, `phone_3`)
- Emails: 3 columns (`email_1`, `email_2`, `email_3`)
- Employers: 4 employers × 2 positions each = 8 position columns
- Education: 3 schools × 2 degrees each = 6 degree columns

Adjust the number of columns if your data needs more.

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
sovren_blocks AS (
    SELECT
        resume_id,
        resume_xml,
        employment_history,
        education_history,
        -- sov:ReservedData block for names, phones, emails
        XMLGET(resume_user_area, 'sov:ReservedData') AS reserved_data,
        -- sov:ExperienceSummary block for description, months of work experience
        XMLGET(resume_user_area, 'sov:ExperienceSummary') AS experience_summary
    FROM resume_blocks
)
SELECT
    resume_id,
    
    --------------------------------------------------------------------------------
    -- SCALAR FIELDS FROM sov:ExperienceSummary
    --------------------------------------------------------------------------------
    XMLGET(experience_summary, 'sov:Description'):"$"::STRING AS sov_description,
    XMLGET(experience_summary, 'sov:MonthsOfWorkExperience'):"$"::INTEGER AS sov_months_of_work_experience,
    XMLGET(experience_summary, 'sov:AverageMonthsPerEmployer'):"$"::INTEGER AS sov_avg_months_per_employer,
    
    --------------------------------------------------------------------------------
    -- NAMES (up to 3)
    -- Path: sov:ReservedData -> sov:Names -> sov:Name[0,1,2]
    --------------------------------------------------------------------------------
    XMLGET(XMLGET(reserved_data, 'sov:Names'), 'sov:Name', 0):"$"::STRING AS name_1,
    XMLGET(XMLGET(reserved_data, 'sov:Names'), 'sov:Name', 1):"$"::STRING AS name_2,
    XMLGET(XMLGET(reserved_data, 'sov:Names'), 'sov:Name', 2):"$"::STRING AS name_3,
    
    --------------------------------------------------------------------------------
    -- PHONES (up to 3)
    -- Path: sov:ReservedData -> sov:Phones -> sov:Phone[0,1,2]
    --------------------------------------------------------------------------------
    XMLGET(XMLGET(reserved_data, 'sov:Phones'), 'sov:Phone', 0):"$"::STRING AS phone_1,
    XMLGET(XMLGET(reserved_data, 'sov:Phones'), 'sov:Phone', 1):"$"::STRING AS phone_2,
    XMLGET(XMLGET(reserved_data, 'sov:Phones'), 'sov:Phone', 2):"$"::STRING AS phone_3,
    
    --------------------------------------------------------------------------------
    -- EMAILS (up to 3)
    -- Path: sov:ReservedData -> sov:EmailAddresses -> sov:EmailAddress[0,1,2]
    --------------------------------------------------------------------------------
    XMLGET(XMLGET(reserved_data, 'sov:EmailAddresses'), 'sov:EmailAddress', 0):"$"::STRING AS email_1,
    XMLGET(XMLGET(reserved_data, 'sov:EmailAddresses'), 'sov:EmailAddress', 1):"$"::STRING AS email_2,
    XMLGET(XMLGET(reserved_data, 'sov:EmailAddresses'), 'sov:EmailAddress', 2):"$"::STRING AS email_3,
    
    --------------------------------------------------------------------------------
    -- EMPLOYMENT (up to 4 employers, each with up to 2 positions)
    -- Path: EmploymentHistory -> EmployerOrg[N] -> EmployerOrgName
    -- Path: EmploymentHistory -> EmployerOrg[N] -> PositionHistory[M] -> Title
    -- Path: EmploymentHistory -> EmployerOrg[N] -> PositionHistory[M] -> OrgInfo -> PositionLocation -> CountryCode/Region/Municipality
    --------------------------------------------------------------------------------
    
    -- Employer 1, Position 1
    XMLGET(XMLGET(employment_history, 'EmployerOrg', 0), 'EmployerOrgName'):"$"::STRING AS employer_1,
    XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 0), 'PositionHistory', 0), 'Title'):"$"::STRING AS title_1_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 0), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'CountryCode'):"$"::STRING AS country_1_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 0), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'Region'):"$"::STRING AS region_1_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 0), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'Municipality'):"$"::STRING AS municipality_1_1,
    
    -- Employer 1, Position 2
    XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 0), 'PositionHistory', 1), 'Title'):"$"::STRING AS title_1_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 0), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'CountryCode'):"$"::STRING AS country_1_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 0), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'Region'):"$"::STRING AS region_1_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 0), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'Municipality'):"$"::STRING AS municipality_1_2,
    
    -- Employer 2, Position 1
    XMLGET(XMLGET(employment_history, 'EmployerOrg', 1), 'EmployerOrgName'):"$"::STRING AS employer_2,
    XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 1), 'PositionHistory', 0), 'Title'):"$"::STRING AS title_2_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 1), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'CountryCode'):"$"::STRING AS country_2_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 1), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'Region'):"$"::STRING AS region_2_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 1), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'Municipality'):"$"::STRING AS municipality_2_1,
    
    -- Employer 2, Position 2
    XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 1), 'PositionHistory', 1), 'Title'):"$"::STRING AS title_2_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 1), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'CountryCode'):"$"::STRING AS country_2_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 1), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'Region'):"$"::STRING AS region_2_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 1), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'Municipality'):"$"::STRING AS municipality_2_2,
    
    -- Employer 3, Position 1
    XMLGET(XMLGET(employment_history, 'EmployerOrg', 2), 'EmployerOrgName'):"$"::STRING AS employer_3,
    XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 2), 'PositionHistory', 0), 'Title'):"$"::STRING AS title_3_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 2), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'CountryCode'):"$"::STRING AS country_3_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 2), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'Region'):"$"::STRING AS region_3_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 2), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'Municipality'):"$"::STRING AS municipality_3_1,
    
    -- Employer 3, Position 2
    XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 2), 'PositionHistory', 1), 'Title'):"$"::STRING AS title_3_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 2), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'CountryCode'):"$"::STRING AS country_3_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 2), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'Region'):"$"::STRING AS region_3_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 2), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'Municipality'):"$"::STRING AS municipality_3_2,
    
    -- Employer 4, Position 1
    XMLGET(XMLGET(employment_history, 'EmployerOrg', 3), 'EmployerOrgName'):"$"::STRING AS employer_4,
    XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 3), 'PositionHistory', 0), 'Title'):"$"::STRING AS title_4_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 3), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'CountryCode'):"$"::STRING AS country_4_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 3), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'Region'):"$"::STRING AS region_4_1,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 3), 'PositionHistory', 0), 'OrgInfo'), 'PositionLocation'), 'Municipality'):"$"::STRING AS municipality_4_1,
    
    -- Employer 4, Position 2
    XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 3), 'PositionHistory', 1), 'Title'):"$"::STRING AS title_4_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 3), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'CountryCode'):"$"::STRING AS country_4_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 3), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'Region'):"$"::STRING AS region_4_2,
    XMLGET(XMLGET(XMLGET(XMLGET(XMLGET(employment_history, 'EmployerOrg', 3), 'PositionHistory', 1), 'OrgInfo'), 'PositionLocation'), 'Municipality'):"$"::STRING AS municipality_4_2,
    
    --------------------------------------------------------------------------------
    -- EDUCATION (up to 3 schools, each with up to 2 degrees)
    -- Path: EducationHistory -> SchoolOrInstitution[N] -> Degree[M] -> DegreeName
    --------------------------------------------------------------------------------
    
    -- School 1, Degree 1
    XMLGET(XMLGET(XMLGET(education_history, 'SchoolOrInstitution', 0), 'Degree', 0), 'DegreeName'):"$"::STRING AS degree_1_1,
    
    -- School 1, Degree 2
    XMLGET(XMLGET(XMLGET(education_history, 'SchoolOrInstitution', 0), 'Degree', 1), 'DegreeName'):"$"::STRING AS degree_1_2,
    
    -- School 2, Degree 1
    XMLGET(XMLGET(XMLGET(education_history, 'SchoolOrInstitution', 1), 'Degree', 0), 'DegreeName'):"$"::STRING AS degree_2_1,
    
    -- School 2, Degree 2
    XMLGET(XMLGET(XMLGET(education_history, 'SchoolOrInstitution', 1), 'Degree', 1), 'DegreeName'):"$"::STRING AS degree_2_2,
    
    -- School 3, Degree 1
    XMLGET(XMLGET(XMLGET(education_history, 'SchoolOrInstitution', 2), 'Degree', 0), 'DegreeName'):"$"::STRING AS degree_3_1,
    
    -- School 3, Degree 2
    XMLGET(XMLGET(XMLGET(education_history, 'SchoolOrInstitution', 2), 'Degree', 1), 'DegreeName'):"$"::STRING AS degree_3_2

FROM sovren_blocks;
```

### Column Reference

#### Contact Info
| Column | Description |
|--------|-------------|
| `name_1`, `name_2`, `name_3` | Names from `sov:ReservedData` |
| `phone_1`, `phone_2`, `phone_3` | Phones from `sov:ReservedData` |
| `email_1`, `email_2`, `email_3` | Emails from `sov:ReservedData` |

#### Experience Summary
| Column | Description |
|--------|-------------|
| `sov_description` | Experience description text |
| `sov_months_of_work_experience` | Total months of work experience |
| `sov_avg_months_per_employer` | Average tenure per employer |

#### Employment (naming: `field_employer_position`)
| Column | Description |
|--------|-------------|
| `employer_1` through `employer_4` | Employer organization names |
| `title_X_Y` | Job title for employer X, position Y |
| `country_X_Y` | Country code for employer X, position Y |
| `region_X_Y` | Region/state for employer X, position Y |
| `municipality_X_Y` | City for employer X, position Y |

#### Education (naming: `degree_school_degree`)
| Column | Description |
|--------|-------------|
| `degree_X_Y` | Degree name for school X, degree Y |

### Extending the Query

To add more columns, just add more `XMLGET` lines with incremented indices:

```sql
-- Add a 4th name
XMLGET(XMLGET(reserved_data, 'sov:Names'), 'sov:Name', 3):"$"::STRING AS name_4,

-- Add a 5th employer
XMLGET(XMLGET(employment_history, 'EmployerOrg', 4), 'EmployerOrgName'):"$"::STRING AS employer_5,
```

### Performance Notes

This approach is **more efficient** than the array approach because:
1. No `GENERATOR` cross joins creating extra rows
2. No aggregation (`GROUP BY`, `ARRAY_AGG`)
3. Single pass through the data
4. Each `XMLGET` with an out-of-bounds index returns NULL immediately (no error, no extra processing)

