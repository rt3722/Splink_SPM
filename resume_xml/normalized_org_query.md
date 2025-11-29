## Resume XML query with normalized organization name

```sql
WITH parsed_resumes AS (
    SELECT
        r.resume_xml,
        PARSE_XML(r.resume_xml) AS resume_doc
    FROM DATA_LAKE.SFDC_RESUME.RESUMES r
    WHERE r.talent_account_id = '001Uj00000uBfUEIA0'
),
extracted_blocks AS (
    SELECT
        resume_xml,
        XMLGET(resume_doc, 'StructuredXMLResume') AS structured_resume,
        XMLGET(structured_resume, 'EmploymentHistory') AS employment_history,
        XMLGET(employment_history, 'EmployerOrg') AS employer_org,
        XMLGET(XMLGET(employer_org, 'PositionHistory'), 'UserArea') AS user_area
    FROM parsed_resumes
)
SELECT
    resume_xml,
    employment_history,
    employer_org,
    user_area,
    XMLGET(
        XMLGET(user_area, 'sov:PositionHistoryUserArea'),
        'sov:NormalizedOrganizationName'
    )::STRING AS normalized_organization_name
FROM extracted_blocks;
```

