import streamlit as st
import pandas as pd
import joblib

# ========== Load model, encoder, feature order ==========
@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_default_random_forest.pkl")
    encoder = joblib.load("target_encoder.pkl")
    feature_order = joblib.load("feature_order.pkl")
    return model, encoder, feature_order

model, encoder, feature_order = load_artifacts()

# ========== Your original function (slightly wrapped) ==========
def predict_new_loan(new_loan: pd.DataFrame) -> str:
    """
    new_loan: DataFrame with ONE row and at least the columns in feature_order.
    """

    # Ensure all expected columns exist
    for col in feature_order:
        if col not in new_loan.columns:
            # For unknown missing columns, fill with a simple default
            new_loan[col] = "Unknown"

    # Encode categorical columns using fitted encoder
    new_encoded = encoder.transform(new_loan)

    # Reorder columns to match training order
    new_encoded = new_encoded[feature_order]

    # Predict
    pred = model.predict(new_encoded)[0]
    return "Charged Off (Bad Loan)" if pred == 1 else "Fully Paid (Good Loan)"


# ========== Streamlit UI ==========
st.title("📊 Loan Default Prediction App")
st.write("Enter loan information below to predict if it will likely be **Fully Paid** or **Charged Off**.")

st.sidebar.header("Loan Input Parameters")

# We’ll build inputs only for the most important / common columns,
# and fill any missing ones automatically.

inputs = {}

# Helper: only create widget if column exists in feature_order
def has(col):
    return col in feature_order

# ---- Numeric fields ----
if has("loan_amnt"):
    inputs["loan_amnt"] = st.sidebar.number_input(
        "Loan Amount ($)", min_value=500.0, max_value=50000.0, value=10000.0, step=500.0
    )

if has("term"):
    term_option = st.sidebar.selectbox("Term (months)", [36, 60])
    inputs["term"] = term_option

if has("int_rate"):
    inputs["int_rate"] = st.sidebar.number_input(
        "Interest Rate (%)", min_value=5.0, max_value=30.0, value=13.5, step=0.1
    )

if has("installment"):
    inputs["installment"] = st.sidebar.number_input(
        "Monthly Installment ($)", min_value=50.0, max_value=1500.0, value=350.0, step=10.0
    )

if has("emp_length"):
    inputs["emp_length"] = st.sidebar.number_input(
        "Employment Length (years)", min_value=0, max_value=40, value=5, step=1
    )

if has("annual_inc"):
    inputs["annual_inc"] = st.sidebar.number_input(
        "Annual Income ($)", min_value=10000.0, max_value=300000.0, value=60000.0, step=1000.0
    )

if has("dti"):
    inputs["dti"] = st.sidebar.number_input(
        "Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=50.0, value=18.0, step=0.1
    )

if has("earliest_cr_line"):
    inputs["earliest_cr_line"] = st.sidebar.number_input(
        "Earliest Credit Line Year", min_value=1980, max_value=2025, value=2005, step=1
    )

if has("issue_d"):
    inputs["issue_d"] = st.sidebar.number_input(
        "Loan Issue Year", min_value=2000, max_value=2025, value=2015, step=1
    )

if has("open_acc"):
    inputs["open_acc"] = st.sidebar.number_input(
        "Open Accounts", min_value=0, max_value=50, value=10, step=1
    )

if has("pub_rec"):
    inputs["pub_rec"] = st.sidebar.number_input(
        "Public Records", min_value=0, max_value=10, value=0, step=1
    )

if has("revol_bal"):
    inputs["revol_bal"] = st.sidebar.number_input(
        "Revolving Balance", min_value=0.0, max_value=200000.0, value=8000.0, step=500.0
    )

if has("revol_util"):
    inputs["revol_util"] = st.sidebar.number_input(
        "Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=45.0, step=1.0
    )

if has("total_acc"):
    inputs["total_acc"] = st.sidebar.number_input(
        "Total Accounts", min_value=0, max_value=200, value=25, step=1
    )

if has("mort_acc"):
    inputs["mort_acc"] = st.sidebar.number_input(
        "Mortgage Accounts", min_value=0, max_value=50, value=1, step=1
    )

if has("pub_rec_bankruptcies"):
    inputs["pub_rec_bankruptcies"] = st.sidebar.number_input(
        "Public Record Bankruptcies", min_value=0, max_value=5, value=0, step=1
    )

# ---- Categorical fields ----
if has("grade"):
    inputs["grade"] = st.sidebar.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])

if has("sub_grade"):
    inputs["sub_grade"] = st.sidebar.text_input("Subgrade (e.g., B3, C4)", "B3")

if has("home_ownership"):
    inputs["home_ownership"] = st.sidebar.selectbox(
        "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

if has("verification_status"):
    inputs["verification_status"] = st.sidebar.selectbox(
        "Verification Status", ["Verified", "Source Verified", "Not Verified"]
    )

if has("purpose"):
    inputs["purpose"] = st.sidebar.selectbox(
        "Loan Purpose",
        [
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "major_purchase",
            "small_business",
            "car",
            "other",
        ],
    )

if has("initial_list_status"):
    inputs["initial_list_status"] = st.sidebar.selectbox(
        "Initial List Status", ["w", "f"]
    )

if has("application_type"):
    inputs["application_type"] = st.sidebar.selectbox(
        "Application Type", ["INDIVIDUAL", "JOINT", "DIRECT_PAY"]
    )

if has("address"):
    inputs["address"] = st.sidebar.text_input(
        "Address (can be rough, not exact)", "123 Example St"
    )

if has("emp_title"):
    inputs["emp_title"] = st.sidebar.text_input(
        "Job Title", "Manager"
    )

if has("title"):
    inputs["title"] = st.sidebar.text_input(
        "Loan Title", "Debt consolidation"
    )

# Fill any remaining expected columns with defaults
for col in feature_order:
    if col not in inputs:
        # default fallback (Unknown or 0 depending on type)
        inputs[col] = "Unknown"

# Turn to DataFrame
input_df = pd.DataFrame([inputs])

st.subheader("📄 Input Preview")
st.write(input_df)

if st.button("Predict Loan Outcome"):
    prediction = predict_new_loan(input_df)
    if "Charged Off" in prediction:
        st.error(f"🚨 Prediction: {prediction}")
    else:
        st.success(f"✅ Prediction: {prediction}")
