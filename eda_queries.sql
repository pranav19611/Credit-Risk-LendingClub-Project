-- =====================================================
-- Credit Risk Analysis — SQL EDA
-- Lending Club Loan Data (2007-2018)
-- Database: PostgreSQL
-- =====================================================

-- ---------------------------------------------------
-- 1. Check loan_status categories (to define "default")
-- ---------------------------------------------------
SELECT loan_status, COUNT(*)
FROM loans
GROUP BY loan_status
ORDER BY COUNT(*) DESC;

-- Default definition used throughout:
--   Default = 1  -> loan_status IN ('Charged Off', 'Default')
--   Default = 0  -> loan_status = 'Fully Paid'
--   Excluded     -> Current, Late, In Grace Period, "Does not meet credit policy" variants
--                   (unresolved / different population)

-- ---------------------------------------------------
-- 2. Overall default rate
-- ---------------------------------------------------
SELECT
    COUNT(*) AS total_resolved_loans,
    SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS default_rate_pct
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default');

-- Result: 1,303,638 resolved loans, 20.07% overall default rate

-- ---------------------------------------------------
-- 3. Default rate by loan purpose
-- ---------------------------------------------------
SELECT
    purpose,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS default_rate_pct
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
GROUP BY purpose
ORDER BY default_rate_pct DESC;

-- ---------------------------------------------------
-- 4. Default rate by loan grade
-- ---------------------------------------------------
SELECT
    grade,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS default_rate_pct
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
GROUP BY grade
ORDER BY grade;

-- Result: near-perfect monotonic relationship, A (6.09%) -> G (50.07%)

-- ---------------------------------------------------
-- 5. Default rate by home ownership
-- ---------------------------------------------------
SELECT
    home_ownership,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS default_rate_pct
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
GROUP BY home_ownership
ORDER BY default_rate_pct DESC;

-- ---------------------------------------------------
-- 6. Default rate by income bracket (custom bucketing via CASE WHEN)
-- ---------------------------------------------------
SELECT
    CASE
        WHEN annual_inc < 30000 THEN '1. Under 30K'
        WHEN annual_inc < 60000 THEN '2. 30K-60K'
        WHEN annual_inc < 90000 THEN '3. 60K-90K'
        WHEN annual_inc < 120000 THEN '4. 90K-120K'
        ELSE '5. 120K+'
    END AS income_bracket,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS default_rate_pct
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
GROUP BY income_bracket
ORDER BY income_bracket;

-- ---------------------------------------------------
-- 7. Raw employment length values (before bucketing)
-- ---------------------------------------------------
SELECT emp_length, COUNT(*)
FROM loans
GROUP BY emp_length
ORDER BY emp_length;

-- ---------------------------------------------------
-- 8. Default rate by employment length bucket
-- ---------------------------------------------------
SELECT
    CASE
        WHEN emp_length IS NULL THEN '0. Not Reported'
        WHEN emp_length IN ('< 1 year', '1 year', '2 years', '3 years') THEN '1. 0-3 years'
        WHEN emp_length IN ('4 years', '5 years', '6 years') THEN '2. 4-6 years'
        WHEN emp_length IN ('7 years', '8 years', '9 years') THEN '3. 7-9 years'
        WHEN emp_length = '10+ years' THEN '4. 10+ years'
    END AS emp_length_bucket,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS default_rate_pct
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
GROUP BY emp_length_bucket
ORDER BY emp_length_bucket;

-- Key finding: "Not Reported" defaults at 27.03% -- notably higher than any
-- reported tenure bucket (18.89%-20.31%). Missingness itself carries signal.

-- ---------------------------------------------------
-- 9. Default rate by DTI (debt-to-income ratio) bucket
-- ---------------------------------------------------
SELECT
    CASE
        WHEN dti < 10 THEN '1. Under 10'
        WHEN dti < 20 THEN '2. 10-20'
        WHEN dti < 30 THEN '3. 20-30'
        WHEN dti < 40 THEN '4. 30-40'
        ELSE '5. 40+'
    END AS dti_bucket,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS default_rate_pct
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
GROUP BY dti_bucket
ORDER BY dti_bucket;

-- ---------------------------------------------------
-- 10. Default rate by issue year (loan vintage)
-- ---------------------------------------------------
SELECT
    EXTRACT(YEAR FROM TO_DATE(issue_d, 'Mon-YYYY')) AS issue_year,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS default_rate_pct
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
GROUP BY issue_year
ORDER BY issue_year;

-- ---------------------------------------------------
-- 11. Year-over-year change in default rate (window function: LAG)
-- ---------------------------------------------------
SELECT
    issue_year,
    total_loans,
    defaults,
    default_rate_pct,
    LAG(default_rate_pct) OVER (ORDER BY issue_year) AS prev_year_rate,
    ROUND(default_rate_pct - LAG(default_rate_pct) OVER (ORDER BY issue_year), 2) AS yoy_change
FROM (
    SELECT
        EXTRACT(YEAR FROM TO_DATE(issue_d, 'Mon-YYYY')) AS issue_year,
        COUNT(*) AS total_loans,
        SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
        ROUND(100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*), 2) AS default_rate_pct
    FROM loans
    WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
    GROUP BY issue_year
) yearly_stats
ORDER BY issue_year;

-- Key finding: sharp -8.17pt drop in 2018 is a censoring artifact (loans
-- issued recently haven't had time to mature into default), not a real
-- risk improvement. Excluded from trend interpretation.
