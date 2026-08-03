# Credit Risk Prediction — Lending Club Loan Data

## 

## Problem Statement

Unsecured personal lending platforms like Lending Club face a core challenge: predicting which borrowers are likely to default before approving a loan. This project builds an end-to-end pipeline — from raw SQL exploration to a trained ML model — to predict loan default risk using Lending Club's public loan dataset (2007-2018, \~2.26M loans).

Beyond building a working model, this project specifically investigates a subtler question: how much of that predictive power comes from raw applicant/loan characteristics versus Lending Club's own proprietary risk assessment (loan grade and interest rate)? This distinction matters — a model that just re-learns "grade predicts default" adds little new insight, while a model that predicts risk independently from raw data is a genuinely useful, standalone tool.

## 

## Dataset \& Methodology

* Source: Lending Club Loan Data, \~2.26M loans issued 2007-2018
* Target definition: loans were filtered to only those with a resolved outcome — `Fully Paid` (default = 0) or `Charged Off` / `Default` (default = 1). Ongoing loans (`Current`, `Late`, `In Grace Period`) were excluded since their outcome is not yet known. This yielded 1,303,638 resolved loans with a 20.07% baseline default rate.
* Tools: PostgreSQL for data storage and exploratory analysis (SQL), Python (pandas, scikit-learn, XGBoost) for cleaning, feature engineering, and modeling.
* Feature selection: rather than using all 145 raw columns, \~20-25 features were deliberately selected based on domain reasoning and EDA findings (loan terms, credit history indicators, income/debt ratios, categorical borrower attributes) to keep the analysis interpretable and avoid noise from redundant/sparse fields.

## 

## Exploratory Data Analysis (using SQL)

All EDA was conducted in SQL against the full resolved-loans dataset before any Python/ML work began, to build a data-driven understanding of what should — and shouldn't — matter to the model.

Overall default rate: 20.07% across 1.3M resolved loans — high relative to secured lending, consistent with Lending Club's unsecured, peer-to-peer lending model.

Loan purpose: default rate ranged from 12.16% (wedding) to 29.75% (small business). `debt\_consolidation` dominates by volume (757K of 1.3M loans), meaning its risk profile disproportionately drives overall portfolio performance.



fig. (chart2\_default\_by\_purpose.png)



Loan grade: showed a near-perfect monotonic relationship with default — 6.09% (Grade A) up to 50.07% (Grade G). This confirmed Lending Club's internal risk grading is genuinely predictive, not just a marketing label, and set up the central research question of this project (see Modeling section).



fig. (chart1\_default\_by\_grade.png)



Home ownership: MORTGAGE holders default least (17.30%), RENT holders default most (23.35%) — plausible given asset-backing and financial stability signals.

Income \& DTI (debt-to-income ratio): both showed clean, monotonic relationships with default risk — default rate fell from 24.57% (under $30K income) to 15.70% ($120K+), and rose from 14.88% (DTI under 10) to 31.20% (DTI 40+).

Employment length: a notably weaker predictor on its own (19-20% default rate across all reported tenure buckets) — but borrowers with missing employment length data defaulted at 27.03%, meaningfully higher than any reported tenure group. This suggests missingness itself is a predictive signal, not just a data quality issue, and was explicitly preserved as a feature (see Feature Engineering) rather than discarded.

Loan vintage (issue year): default rate rose steadily from 2009 (12.60%) to a peak in 2016 (24.27%), then dropped sharply in 2017 (22.90%) and 2018 (14.73%). This drop is a censoring artifact, not a real risk improvement — loans issued in 2017-2018 haven't had enough time to mature into default, so the "resolved" loans from those years are biased toward faster-resolving (often safer) cases. This is a standard pitfall in credit risk vintage analysis and was explicitly excluded from being read as a genuine trend.

## 

## Feature Engineering \& Cleaning (using Python)

Key decisions made during cleaning, with reasoning:

* Missing employment length: rather than simply imputing a value and discarding the missingness, a separate `emp\_length\_missing` binary flag was created to preserve the signal identified in EDA (missing tenure correlates with higher default), while the underlying numeric field was median-imputed so the model could still use it.
* Other missing values (`mort\_acc`, `revol\_util`, `dti`, `inq\_last\_6mths`, `pub\_rec\_bankruptcies`, and later `bc\_util`, `num\_tl\_90g\_dpd\_24m`, `mo\_sin\_old\_rev\_tl\_op`) were median-imputed — all had low missingness (<5%) with no EDA evidence of a meaningful missingness pattern like employment length showed.
* Categorical encoding: `grade` was ordinal-encoded (A=0 ... G=6) since EDA confirmed a genuine, monotonic order. Unordered categoricals (`home\_ownership`, `verification\_status`, `purpose`) were one-hot encoded, since no numeric ordering between categories exists.
* Target variable: `loan\_status` was converted to a binary `default` flag (1 = Charged Off/Default, 0 = Fully Paid), matching the resolved-loans filter applied throughout.

## 

## Modeling

### Approach

Given a class imbalance of \~80% non-default / \~20% default, plain accuracy is a misleading metric — a model that always predicts "no default" would score \~80% accuracy while being useless. Models were evaluated primarily on ROC-AUC (threshold-independent ranking ability) alongside precision/recall/F1 for the minority (default) class, and class imbalance was addressed via `class\_weight='balanced'` (scikit-learn) or `scale\_pos\_weight` (XGBoost).

### 

### Central research question: does the model rely on Lending Club's own risk grade?

Since `grade` showed a near-perfect monotonic relationship with default in EDA, a natural concern is that a model using `grade` (and `int\_rate`, which is largely derived from grade) as a feature would just be re-learning Lending Club's own risk assessment rather than independently predicting risk from raw applicant data. To test this, models were trained on three feature sets:

1. Full feature set (includes `grade` and `int\_rate`)
2. Without `grade` (int\_rate retained)
3. Without `grade` and `int\_rate` (raw applicant/loan characteristics only)

### 

### Results

|Model|Feature Set|ROC-AUC|Recall (default)|Precision (default)|
|-|-|-|-|-|
|Logistic Regression (unweighted)|Full|0.709|0.09|0.53|
|Logistic Regression (balanced)|Full|0.709|0.63|0.33|
|Random Forest|Full|0.711|0.68|0.31|
|Random Forest|Without grade|0.710|—|—|
|Random Forest|Without grade + int\_rate|0.687|—|—|
|Random Forest|Expanded features (+3), Full|0.711|0.68|0.31|
|XGBoost|Expanded features (+3), Full|0.720|0.68|0.32|
|XGBoost|Without grade + int\_rate|0.709|—|—|



fig. (chart3\_model\_comparison.png)

### 

### Key findings

1\. Class imbalance handling matters enormously. The unweighted logistic regression achieved 80% accuracy while catching only 9% of actual defaults — a model that would be nearly useless in production despite its impressive-looking accuracy. Applying class weighting raised recall to 63% at the cost of precision, a trade-off discussed further below.

2\. Grade alone is largely redundant with other features. Removing `grade` cost almost nothing (0.711 → 0.710). This suggests other raw features (DTI, income, credit history) already capture most of the same signal grade does.

3\. `int\_rate` carries real independent signal beyond grade. Removing both `grade` and `int\_rate` caused a genuine drop (0.710 → 0.687), suggesting Lending Club's pricing incorporates additional risk information beyond the coarse grade label.

4\. Model complexity gave diminishing returns. Random Forest barely outperformed logistic regression (0.709 → 0.711), and feature importance confirmed the underlying relationships are largely linear/monotonic (consistent with the clean, orderly EDA trends for grade, income, and DTI) — leaving little non-linear signal for a more complex model to exploit.

5\. Adding features + XGBoost gave a small, genuine improvement (0.711 → 0.720) — but not for the reason initially hypothesized. Three additional features (`bc\_util`, `num\_tl\_90g\_dpd\_24m`, `mo\_sin\_old\_rev\_tl\_op`) were added, expecting they'd add new predictive signal. Feature importance on the final XGBoost model showed these three did not rank among the top 15 most important features — `grade\_encoded` alone accounted for 72% of the model's decisions. The gain instead came from XGBoost's sequential tree-boosting being marginally better at exploiting the existing grade/int\_rate signal, not from the new features. This was confirmed by re-running the no-grade/no-int\_rate ablation on XGBoost, which converged to the same \~0.70-0.71 ceiling seen across every other model and feature set.



fig. (chart4\_feature\_importance.png)



6\. The core, most important finding: raw applicant data alone consistently predicts default around 0.687-0.709 ROC-AUC, regardless of algorithm. Logistic regression, Random Forest, and XGBoost all converged to a similar ceiling when grade/int\_rate were excluded. This consistency across fundamentally different algorithms is strong evidence that this \~0.70 ceiling reflects genuine, reproducible signal in the raw data — not an artifact of any one model. Lending Club's proprietary signals (grade, int\_rate) add a real but modest \~0.01-0.03 ROC-AUC on top of this.

### 

### Business interpretation: precision vs. recall trade-off

For a lender, missing an actual default (false negative) typically costs more than being overly cautious (false positive) — a missed default risks the full loan principal, while a false alarm only costs a potentially-good customer. Under that framing, the higher-recall balanced models (catching \~63-68% of defaults at the cost of more false alarms) are more business-appropriate than the unweighted baseline, despite lower headline accuracy. However, if flagged applicants are rejected automatically without human review, a precision of \~0.31-0.33 means roughly 2 in 3 flagged applicants would be wrongly rejected — a real cost that argues for using model output as a risk-scoring input to human review rather than an automatic accept/reject switch.

## 

## Limitations \& Future Work

* Irreducible uncertainty: ROC-AUC in the 0.70-0.72 range is consistent with published benchmarks for unsecured personal lending risk models. Individual default outcomes are influenced by factors outside this dataset entirely (job loss, medical emergencies, other undisclosed debts), which no model trained on this data alone can capture.
* Vintage censoring: this analysis did not implement formal survival analysis to properly account for loans of different ages/maturity — a more rigorous treatment would model time-to-default directly rather than a static binary outcome.
* Additional features: only \~25 of the dataset's 145 columns were used, chosen via domain reasoning. Further features (e.g., `total\_rev\_hi\_lim`, `num\_tl\_op\_past\_12m`) and interaction terms could be explored with more rigorous feature selection (e.g., recursive feature elimination), balanced against overfitting risk from increased dimensionality.
* Explainable AI (XAI): a natural next step is integrating SHAP values with the final XGBoost model to explain individual predictions (e.g., "this applicant was flagged as high-risk primarily due to high DTI and limited credit history") — valuable both for regulatory transparency and applicant-facing communication in a real lending context.
* Deployment: a simple Streamlit interface accepting applicant details and returning a risk score would demonstrate the model as a usable tool rather than a static notebook result.

## 

## Tech Stack

* Database: PostgreSQL
* Languages/Libraries: Python, pandas, scikit-learn, XGBoost, matplotlib/seaborn
* Analysis: SQL (window functions, CASE-based bucketing/aggregation), Python (data cleaning, feature engineering, classification modelling)

