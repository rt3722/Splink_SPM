Select 
resume_xml,
XMLGET(XMLGET(PARSE_XML(r.RESUME_XML), 'StructuredXMLResume'), 'EmploymentHistory') as EmploymentHistory,
XMLGET(XMLGET(XMLGET(PARSE_XML(r.RESUME_XML), 'StructuredXMLResume'), 'EmploymentHistory'),'EmployerOrg')
as EmployerOrg,
XMLGET(XMLGET(XMLGET(XMLGET(PARSE_XML(r.RESUME_XML), 'StructuredXMLResume'), 'EmploymentHistory'),'EmployerOrg'),'UserArea')
as UserArea
FROM DATA_LAKE.SFDC_RESUME.RESUMES r 
WHERE 
    -- Ensure we are only looking at the correct tags
    talent_account_id  = '001Uj00000uBfUEIA0' 