import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="centered")

# -----------------------------
# Load model artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    columns = joblib.load("model_columns.pkl")
    medians = joblib.load("imputation_medians.pkl")
    return model, columns, medians

model, model_columns, medians = load_artifacts()

st.title("💳 Credit Risk Predictor")
st.write(
    "Predicts the probability a loan will default, based on a model trained on "
    "1.3M+ resolved Lending Club loans (2007-2018). "
    "[View the full project on GitHub](https://github.com/YOUR_USERNAME/YOUR_REPO)"
)

st.divider()

# -----------------------------
# Input form
# -----------------------------
st.subheader("Applicant & Loan Details")

col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
    term = st.selectbox("Term (months)", [36, 60])
    int_rate = st.slider("Interest Rate (%)", 5.0, 31.0, 13.0, step=0.1)
    installment = st.number_input("Monthly Installment ($)", min_value=10.0, value=330.0, step=10.0)
    annual_inc = st.number_input("Annual Income ($)", min_value=0, value=65000, step=1000)
    dti = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 50.0, 18.0, step=0.5)
    grade = st.selectbox("Loan Grade (Lending Club's own risk grade)", ["A", "B", "C", "D", "E", "F", "G"])

with col2:
    home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER", "NONE"])
    verification_status = st.selectbox("Income Verification", ["Not Verified", "Source Verified", "Verified"])
    purpose = st.selectbox("Loan Purpose", [
        "debt_consolidation", "credit_card", "home_improvement", "other",
        "major_purchase", "medical", "small_business", "car", "vacation",
        "moving", "house", "renewable_energy", "wedding", "educational"
    ])
    emp_length = st.selectbox("Employment Length", [
        "Not Reported", "< 1 year", "1 year", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
    ])
    mort_acc = st.number_input("Number of Mortgage Accounts", min_value=0, value=1, step=1)
    revol_util = st.slider("Revolving Credit Utilization (%)", 0.0, 150.0, 45.0, step=1.0)

with st.expander("Additional credit history details (optional — defaults to typical values)"):
    delinq_2yrs = st.number_input("Delinquencies (last 2 years)", min_value=0, value=0, step=1)
    inq_last_6mths = st.number_input("Credit Inquiries (last 6 months)", min_value=0, value=0, step=1)
    open_acc = st.number_input("Open Credit Lines", min_value=0, value=11, step=1)
    pub_rec = st.number_input("Public Records (derogatory)", min_value=0, value=0, step=1)
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0, value=15000, step=500)
    total_acc = st.number_input("Total Credit Lines (ever)", min_value=0, value=25, step=1)
    pub_rec_bankruptcies = st.number_input("Public Record Bankruptcies", min_value=0, value=0, step=1)
    bc_util = st.slider("Bankcard Utilization (%)", 0.0, 150.0, 50.0, step=1.0)
    num_tl_90g_dpd_24m = st.number_input("Accounts 90+ Days Delinquent (24mo)", min_value=0, value=0, step=1)
    mo_sin_old_rev_tl_op = st.number_input("Age of Oldest Revolving Account (months)", min_value=0, value=150, step=1)

show_grade_comparison = st.checkbox(
    "Also show prediction WITHOUT using loan grade (raw applicant data only)",
    value=True,
    help="Demonstrates this project's core finding: how much predictive power comes from "
         "Lending Club's own risk grade vs. raw applicant characteristics."
)

# -----------------------------
# Build feature row matching training data
# -----------------------------
def build_input_row(include_grade=True):
    emp_length_map = {
        'Not Reported': None, '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8,
        '9 years': 9, '10+ years': 10
    }
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

    emp_clean = emp_length_map[emp_length]
    emp_missing = 1 if emp_clean is None else 0
    if emp_clean is None:
        emp_clean = medians['emp_length_clean']

    row = {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': installment,
        'annual_inc': annual_inc,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'inq_last_6mths': inq_last_6mths,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'mort_acc': mort_acc,
        'pub_rec_bankruptcies': pub_rec_bankruptcies,
        'bc_util': bc_util,
        'num_tl_90g_dpd_24m': num_tl_90g_dpd_24m,
        'mo_sin_old_rev_tl_op': mo_sin_old_rev_tl_op,
        'emp_length_clean': emp_clean,
        'emp_length_missing': emp_missing,
        'grade_encoded': grade_map[grade] if include_grade else medians.get('grade_encoded', 3),
    }

    # One-hot columns — default all to 0, then set the selected category to 1
    onehot_defaults = {c: 0 for c in model_columns if c.startswith(('home_ownership_', 'verification_status_', 'purpose_'))}
    row.update(onehot_defaults)

    ho_col = f"home_ownership_{home_ownership}"
    if ho_col in row:
        row[ho_col] = 1

    vs_col = f"verification_status_{verification_status}"
    if vs_col in row:
        row[vs_col] = 1

    purpose_col = f"purpose_{purpose}"
    if purpose_col in row:
        row[purpose_col] = 1

    if not include_grade:
        row['int_rate'] = medians.get('int_rate', int_rate)  # neutralize int_rate too, matching the ablation study

    # Build dataframe in the exact column order the model expects
    input_df = pd.DataFrame([row])
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]
    return input_df

# -----------------------------
# Predict
# -----------------------------
st.divider()
if st.button("Predict Default Risk", type="primary", use_container_width=True):
    input_df = build_input_row(include_grade=True)
    proba = model.predict_proba(input_df)[0][1]

    risk_label = "🟢 Low Risk" if proba < 0.15 else ("🟡 Medium Risk" if proba < 0.35 else "🔴 High Risk")

    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Default Probability", f"{proba:.1%}")
    c2.metric("Risk Category", risk_label)

    if show_grade_comparison:
        input_df_ng = build_input_row(include_grade=False)
        proba_ng = model.predict_proba(input_df_ng)[0][1]
        st.caption(
            f"**Without grade/interest rate (raw applicant data only):** "
            f"{proba_ng:.1%} predicted default probability — showing how much of the prediction "
            f"relies on Lending Club's own risk assessment vs. raw applicant characteristics."
        )

    st.info(
        "This is a demo model built for a portfolio project and should not be used for actual "
        "lending decisions. See the GitHub repo for full methodology, limitations, and evaluation metrics."
    )
