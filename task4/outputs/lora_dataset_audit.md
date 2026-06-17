# LoRA Dataset Audit

**Total Samples**: 21231

## Structure
- Valid: 21231
- Invalid Messages: 0
- Missing Messages: 0
- Invalid Roles: 0

## Prompt Sections
- QUESTION: 21231
- AVAILABLE TABLES: 21231
- Relevant Columns: 21231
- KNOWN JOIN RELATIONSHIPS: 21231
- BUSINESS SEMANTICS: 21231
- DOCUMENTATION: 21231
- REFERENCE SQL PATTERN: 21231

**Missing Sections:**

## SQL Targets
- Valid SQL: 21229
- Markdown SQL: 0
- Empty SQL: 0
- Non SQL: 2

## Token Statistics
- Min: 137
- Mean: 733
- Median: 733
- P95: 918
- P99: 1099
- Max: 2302

## Complexity Distribution
- MULTI_JOIN: 810
- AGGREGATION: 4576
- RANKING: 2440
- SUBQUERY: 7941
- FILTER_QUERY: 5402
- CTE: 54
- SIMPLE_SELECT: 8

## Sample Metadata

### Sample 1
**Complexity**: SUBQUERY

**Target SQL**:
SELECT sub.block_name, sub.total_villages, sub.covered_villages, ROUND((sub.covered_villages::numeric / sub.total_villages) * 100, 2) AS block_coverage_percent FROM (SELECT dbgv.block_name, COUNT(DISTINCT dbgv.village_code) AS total_villages, COUNT(DISTINCT cr.village_code) AS covered_villages FROM public.chatbot_districts_blocks_gp_villages dbgv LEFT JOIN public.chatbot_coverage_areas cr ON cr.state_code = dbgv.state_code AND cr.district_code = dbgv.district_code AND cr.block_code = dbgv.block_code AND cr.village_code = dbgv.village_code AND cr.status = 1 AND cr.is_approved != 2 AND cr.is_draft = 0 AND cr.sector_of_operation_name ILIKE '%handloom%' WHERE dbgv.state_name ILIKE '%UTTAR PRADESH%' AND dbgv.district_name ILIKE '%LUCKNOW%' AND dbgv.status = 1 GROUP BY dbgv.block_name, dbgv.block_code) AS sub ORDER BY sub.block_name;


### Sample 2
**Complexity**: SUBQUERY

**Target SQL**:
SELECT sub.state_name, sub.district_name, sub.gram_panchayat_name, sub.total_villages_in_gp, sub.villages_covered, ROUND((sub.villages_covered::numeric / sub.total_villages_in_gp) * 100, 2) AS coverage_percent FROM (SELECT dbgv.state_name, dbgv.district_name, dbgv.gram_panchayat_name, COUNT(DISTINCT dbgv.village_code) AS total_villages_in_gp, COUNT(DISTINCT cr.village_code) AS villages_covered FROM public.chatbot_districts_blocks_gp_villages dbgv LEFT JOIN public.chatbot_coverage_areas cr ON cr.state_code = dbgv.state_code AND cr.district_code = dbgv.district_code AND cr.panchayat_code = dbgv.gram_panchayat_code AND cr.village_code = dbgv.village_code AND cr.status = 1 AND cr.is_approved != 2 AND cr.is_draft = 0 AND cr.sector_of_operation_name ILIKE '%fishery%' WHERE dbgv.state_name ILIKE '%HARYANA%' AND dbgv.district_name ILIKE '%HISAR%' AND dbgv.status = 1 GROUP BY dbgv.state_name, dbgv.district_name, dbgv.gram_panchayat_name, dbgv.gram_panchayat_code HAVING COUNT(DISTINCT cr.village_code) > 0 AND COUNT(DISTINCT cr.village_code) < COUNT(DISTINCT dbgv.village_code)) AS sub ORDER BY sub.district_name, sub.gram_panchayat_name;


### Sample 3
**Complexity**: SUBQUERY

**Target SQL**:
SELECT sub.state_name, sub.district_name, sub.gram_panchayat_name, sub.total_villages_in_gp, sub.villages_covered, ROUND((sub.villages_covered::numeric / sub.total_villages_in_gp) * 100, 2) AS coverage_percent FROM (SELECT dbgv.state_name, dbgv.district_name, dbgv.gram_panchayat_name, COUNT(DISTINCT dbgv.village_code) AS total_villages_in_gp, COUNT(DISTINCT cr.village_code) AS villages_covered FROM public.chatbot_districts_blocks_gp_villages dbgv LEFT JOIN public.chatbot_coverage_areas cr ON cr.state_code = dbgv.state_code AND cr.district_code = dbgv.district_code AND cr.panchayat_code = dbgv.gram_panchayat_code AND cr.village_code = dbgv.village_code AND cr.status = 1 AND cr.is_approved != 2 AND cr.is_draft = 0 AND cr.sector_of_operation_name ILIKE '%dairy%' WHERE dbgv.state_name ILIKE '%MAHARASHTRA%' AND dbgv.district_name ILIKE '%NASHIK%' AND dbgv.status = 1 GROUP BY dbgv.state_name, dbgv.district_name, dbgv.gram_panchayat_name, dbgv.gram_panchayat_code HAVING COUNT(DISTINCT cr.village_code) > 0 AND COUNT(DISTINCT cr.village_code) < COUNT(DISTINCT dbgv.village_code)) AS sub ORDER BY sub.district_name, sub.gram_panchayat_name;


### Sample 4
**Complexity**: AGGREGATION

**Target SQL**:
SELECT c.district_name, COUNT(*) FROM chatbot_cooperative_registrations_processing s JOIN chatbot_cooperative_registrations_master c ON s.ncd_id = c.cooperative_id WHERE c.state_name ILIKE '%KERALA%' AND c.status = 1 AND c.is_draft = 0 AND c.is_approved = 1 GROUP BY c.district_name;


### Sample 5
**Complexity**: SUBQUERY

**Target SQL**:
SELECT sub.block_name, sub.total_villages, sub.covered_villages, ROUND((sub.covered_villages::numeric / sub.total_villages) * 100, 2) AS block_coverage_percent FROM (SELECT dbgv.block_name, COUNT(DISTINCT dbgv.village_code) AS total_villages, COUNT(DISTINCT cr.village_code) AS covered_villages FROM public.chatbot_districts_blocks_gp_villages dbgv LEFT JOIN public.chatbot_coverage_areas cr ON cr.state_code = dbgv.state_code AND cr.district_code = dbgv.district_code AND cr.block_code = dbgv.block_code AND cr.village_code = dbgv.village_code AND cr.status = 1 AND cr.is_approved != 2 AND cr.is_draft = 0 AND cr.sector_of_operation_name ILIKE '%multi%' WHERE dbgv.state_name ILIKE '%MAHARASHTRA%' AND dbgv.district_name ILIKE '%THANE%' AND dbgv.status = 1 GROUP BY dbgv.block_name, dbgv.block_code) AS sub ORDER BY sub.block_name;


### Sample 6
**Complexity**: FILTER_QUERY

**Target SQL**:
SELECT * FROM chatbot_state_district_federation_private sfp JOIN chatbot_state_district_federation_tribal st ON st.state_district_federation_id = sfp.id WHERE sfp.primary_activity_name ILIKE '%Tribal%' AND sfp.state_name ILIKE '%UTTAR PRADESH%' AND sfp.is_profit_making = '1' AND sfp.is_draft = '0' AND is_approved = '0' AND status = '1';


### Sample 7
**Complexity**: SUBQUERY

**Target SQL**:
SELECT COUNT(*) AS covered_gp_count FROM (SELECT dbgv.gram_panchayat_name FROM public.chatbot_districts_blocks_gp_villages dbgv LEFT JOIN public.chatbot_coverage_areas cr ON cr.state_code = dbgv.state_code AND cr.district_code = dbgv.district_code AND cr.panchayat_code = dbgv.gram_panchayat_code AND cr.village_code = dbgv.village_code AND cr.status = 1 AND cr.is_approved != 2 AND cr.is_draft = 0 AND cr.sector_of_operation_name ILIKE '%fishery%' WHERE dbgv.state_name ILIKE '%TAMIL NADU%' AND dbgv.district_name ILIKE '%CHENNAI%' AND dbgv.status = 1 GROUP BY dbgv.gram_panchayat_name, dbgv.gram_panchayat_code HAVING COUNT(DISTINCT cr.village_code) > 0 AND COUNT(DISTINCT cr.village_code) = COUNT(DISTINCT dbgv.village_code)) AS sub;


### Sample 8
**Complexity**: FILTER_QUERY

**Target SQL**:
SELECT * FROM chatbot_state_district_federation_private sfp JOIN chatbot_state_district_federation_bee st ON st.state_district_federation_id = sfp.id WHERE sfp.primary_activity_name ILIKE '%Beekeeping%' AND sfp.state_name ILIKE '%RAJASTHAN%' AND sfp.is_profit_making = '1' AND sfp.is_draft = '0' AND is_approved = '0' AND status = '1';


### Sample 9
**Complexity**: FILTER_QUERY

**Target SQL**:
SELECT nfcd.contact_person FROM chatbot_national_federations nf JOIN chatbot_national_federations_contact_details nfcd ON nf.id = nfcd.national_federations_id WHERE nf.cooperative_society_name ILIKE '%Dairy Federation%' AND nfcd.designation_name ILIKE '%MD%' AND nf.is_new = '1' AND nf.is_draft = '0';


### Sample 10
**Complexity**: SUBQUERY

**Target SQL**:
SELECT cooperative_name, total_score, rank FROM public.chatbot_cooperative_rankings WHERE district_name ILIKE '%NIZAMABAD%' AND sector_name ILIKE '%Dairy%' AND rank_type_name = 'District' AND CAST(date_time AS DATE) = (SELECT MAX(CAST(date_time AS DATE)) FROM public.chatbot_cooperative_rankings WHERE district_name ILIKE '%NIZAMABAD%' AND sector_name ILIKE '%Dairy%' AND rank_type_name = 'District') ORDER BY flag ASC, rank ASC LIMIT 10;

