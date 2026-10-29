
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.joblib"

st.set_page_config(
    page_title="ChurnIQ | AI Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- THEME ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #f5f7fc;
    --navy: #101a32;
    --muted: #77839a;
    --blue: #5277f7;
    --border: #e7ebf3;
}

.stApp {
    background: var(--bg);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(245,247,252,.92);
}

.block-container {
    max-width: 1600px;
    padding: 1.8rem 2.4rem 3rem;
}

section[data-testid="stSidebar"] {
    background: #101a32;
}

section[data-testid="stSidebar"] * {
    color: #e9efff;
}

.brand {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -1px;
    padding: 8px 0 2px;
}

.brand span {
    color: #83a0ff;
}

.sidebar-note {
    color: #93a1bf !important;
    font-size: .78rem;
    margin-bottom: 22px;
}

.hero {
    background: linear-gradient(115deg, #121e3b 0%, #203f7d 65%, #3159a4 100%);
    border-radius: 22px;
    padding: 30px 32px;
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 12px 32px rgba(23, 46, 95, .14);
}

.hero-label {
    color: #b9ccff;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: .72rem;
    font-weight: 700;
}

.hero h1 {
    color: white !important;
    font-size: 2rem;
    letter-spacing: -1px;
    margin: 9px 0;
}

.hero p {
    color: #d6e2ff;
    margin: 0;
    font-size: .93rem;
}

.section-heading {
    color: #182641;
    font-size: 1.1rem;
    font-weight: 750;
    margin: 20px 0 12px;
}

.section-caption {
    color: #7b879c;
    font-size: .82rem;
    margin-top: -7px;
    margin-bottom: 16px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    padding: 20px 20px 17px;
    border-radius: 17px;
    box-shadow: 0 5px 18px rgba(27, 45, 80, .035);
}

div[data-testid="stMetricLabel"] {
    color: #748198;
    font-size: .84rem;
}

div[data-testid="stMetricValue"] {
    color: #172541;
    font-weight: 800;
}

div[data-testid="stPlotlyChart"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 10px 10px 0;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #b7c6e4;
    border-radius: 17px;
    padding: 15px;
}

.stButton button, .stDownloadButton button {
    border-radius: 10px;
    font-weight: 650;
    min-height: 43px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}

.footer {
    color: #8490a5;
    font-size: .77rem;
    border-top: 1px solid #e5eaf3;
    padding-top: 17px;
    margin-top: 30px;
}

@media (max-width: 800px) {
    .block-container {
        padding: 1rem .8rem 2rem;
    }
    .hero {
        padding: 23px 20px;
    }
    .hero h1 {
        font-size: 1.55rem;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    FEATURES = list(model.feature_names_in_)
except Exception as exc:
    st.error(f"Model load nahi ho saka: {exc}")
    st.stop()


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        '<div class="brand">◈ Churn<span>IQ</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-note">AI CUSTOMER INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Customer Explorer",
            "Single Customer Prediction",
            "Model Performance",
            "Model Information",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.caption("MODEL")
    st.markdown("Logistic Regression")
    st.caption("19 input features")
    st.markdown("---")
    st.caption("Local ML workspace")


# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <div class="hero-label">Customer Intelligence Platform</div>
    <h1>Understand churn. Take action.</h1>
    <p>Explore customer risk predictions from your trained machine learning model.</p>
</div>
""", unsafe_allow_html=True)


# ---------- SINGLE CUSTOMER PREDICTION ----------
if page == "Single Customer Prediction":
    st.markdown(
        '<div class="section-heading">Single customer analysis</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Enter one customer's details to estimate their churn probability."
    )

    with st.form("single_customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input(
                "Tenure (months)", 0, 100, 12
            )
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )

        with c2:
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )

        with c3:
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input(
                "Monthly Charges", min_value=0.0, value=70.0
            )
            total_charges = st.number_input(
                "Total Charges", min_value=0.0, value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            type="primary",
            use_container_width=True,
        )

    threshold_value = st.slider(
        "Classification threshold (%)",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        key="single_customer_threshold",
        help="Probability is compared with this threshold to decide the class.",
    )

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges,
        }])

        try:
            customer = customer[FEATURES]
            probabilities = model.predict_proba(customer)
            classes = list(model.classes_)
            churn_index = classes.index("Yes")
            probability = float(probabilities[0, churn_index]) * 100

            st.metric("Estimated churn probability", f"{probability:.2f}%")

            if probability >= threshold_value:
                st.warning("Prediction: Churn")
            else:
                st.success("Prediction: Not Churn")

            st.caption(
                "This is a model estimate, not a guarantee of future behavior."
            )

        except Exception as exc:
            st.error(f"Prediction error: {exc}")

    st.stop()


# ---------- MODEL PERFORMANCE ----------
if page == "Model Performance":
    st.markdown(
        '<div class="section-heading">Model Performance</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Metrics calculated from the held-out test split.</div>',
        unsafe_allow_html=True,
    )

    DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn-cleaned.csv"
    MODEL_FILES = {
        "Logistic Regression": "logistic_regression.joblib",
        "Default Decision Tree": "decision_tree_default.joblib",
        "Tuned Decision Tree": "decision_tree_tuned.joblib",
    }

    try:
        df_eval = pd.read_csv(DATA_PATH)
        X_eval = df_eval.drop(columns=["customerID", "Churn"])
        y_eval = df_eval["Churn"]
        _, X_test, _, y_test = train_test_split(
            X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval
        )

        model_results = {}
        comparison_rows = []
        for model_name, filename in MODEL_FILES.items():
            saved_model = joblib.load(BASE_DIR / "models" / filename)
            y_pred = saved_model.predict(X_test)
            model_results[model_name] = {
                "predictions": y_pred,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, pos_label="Yes", zero_division=0),
                "recall": recall_score(y_test, y_pred, pos_label="Yes", zero_division=0),
                "f1": f1_score(y_test, y_pred, pos_label="Yes", zero_division=0),
            }
            comparison_rows.append({
                "Model": model_name,
                "Accuracy (%)": accuracy_score(y_test, y_pred) * 100,
                "Precision (%)": precision_score(y_test, y_pred, pos_label="Yes", zero_division=0) * 100,
                "Recall (%)": recall_score(y_test, y_pred, pos_label="Yes", zero_division=0) * 100,
                "F1-score (%)": f1_score(y_test, y_pred, pos_label="Yes", zero_division=0) * 100,
            })

        selected_name = "Logistic Regression"
        selected = model_results[selected_name]
        st.info(
            f"Evaluation uses {len(y_test):,} held-out customers from the project's saved dataset. "
            "These are test-set metrics, not scores calculated from uploaded batch predictions."
        )

        st.markdown("### Logistic Regression")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{selected['accuracy']:.2%}")
        m2.metric("Churn precision", f"{selected['precision']:.2%}")
        m3.metric("Churn recall", f"{selected['recall']:.2%}")
        m4.metric("Churn F1-score", f"{selected['f1']:.2%}")

        st.markdown("### Confusion Matrix")
        confusion = pd.DataFrame(
            confusion_matrix(y_test, selected["predictions"], labels=["No", "Yes"]),
            index=["Actual Not Churn", "Actual Churn"],
            columns=["Predicted Not Churn", "Predicted Churn"],
        )
        fig_cm = px.imshow(
            confusion, text_auto=True, aspect="auto",
            color_continuous_scale=["#eaf0ff", "#3159a4"],
            labels={"x": "Predicted class", "y": "Actual class", "color": "Customers"},
        )
        fig_cm.update_layout(
            title="Actual vs Predicted Customers", paper_bgcolor="white",
            plot_bgcolor="white", font=dict(color="#263651", family="Inter"),
            margin=dict(l=20, r=20, t=65, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        tn, fp, fn, tp = confusion_matrix(y_test, selected["predictions"], labels=["No", "Yes"]).ravel()
        st.markdown("### What do these numbers mean?")
        st.write(f"**Accuracy:** {selected['accuracy']:.2%} of test customers were classified correctly overall.")
        st.write(f"**Precision:** Of customers predicted to churn, {selected['precision']:.2%} actually churned.")
        st.write(f"**Recall:** The model identified {tp} of {tp + fn} actual churn customers and missed {fn}.")
        st.write(f"**F1-score:** {selected['f1']:.2%}, a combined measure of churn precision and recall.")

        a, b = st.columns(2)
        a.metric("Correctly identified churn", f"{tp:,}")
        b.metric("Missed churn customers", f"{fn:,}")
        c, d = st.columns(2)
        c.metric("Correctly identified non-churn", f"{tn:,}")
        d.metric("False churn predictions", f"{fp:,}")
        st.warning(
            "Business implication: missed churn customers may not receive retention attention. "
            "Consider the trade-off between missed churn and false alarms before operational use."
        )

        st.markdown("### Model comparison")
        comparison = pd.DataFrame(comparison_rows)
        st.dataframe(
            comparison.round(2), use_container_width=True, hide_index=True,
            column_config={
                "Accuracy (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Precision (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Recall (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "F1-score (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
    except Exception as exc:
        st.error(f"Model evaluation could not be completed: {exc}")
        st.caption("Check that the cleaned dataset and all three saved models exist in the project folders.")

    st.stop()


# ---------- MODEL INFO ----------
if page == "Model Information":
    st.markdown(
        '<div class="section-heading">Model information</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "This application uses your saved Logistic Regression pipeline. "
        "Categorical encoding and numerical scaling are handled by the pipeline."
    )
    st.info(
        "Predictions are estimates, not guarantees. "
        "A high predicted probability does not prove a customer will leave."
    )
    st.markdown("**Input requirements**")
    st.write("CSV columns must include all 19 trained input features.")
    st.write("The app does not use customerID or the actual Churn column as input.")
    st.stop()


# ---------- UPLOAD ----------
st.markdown(
    '<div class="section-heading">01 / Import customer data</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">Upload a CSV to generate customer-level predictions.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Drag and drop your CSV here, or browse files",
    type=["csv"],
    help="The CSV must contain all 19 model input columns.",
    key="customer_csv",
)

if uploaded_file is None:
    st.markdown("### Your analytics workspace is ready")
    st.write("Upload customer data to see KPIs, risk distribution and detailed results.")
    st.markdown(
        '<div class="footer">ChurnIQ · Local analytics workspace · Model estimates only</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ---------- READ & VALIDATE ----------
try:
    customers = pd.read_csv(uploaded_file)

    if customers.empty:
        st.error("CSV mein koi customer rows nahi hain.")
        st.stop()

    missing = [col for col in FEATURES if col not in customers.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()

    X = customers[FEATURES].copy()

    # Empty input fields should not silently be interpreted as valid values.
    empty_columns = X.columns[X.isna().any()].tolist()
    if empty_columns:
        st.error(
            "Missing values found in: "
            + ", ".join(empty_columns)
            + ". Please correct the CSV and upload it again."
        )
        st.stop()

except Exception as exc:
    st.error(f"CSV read nahi ho saki: {exc}")
    st.stop()


# ---------- PREDICTION ----------
if st.button("Run AI Churn Analysis", type="primary", use_container_width=True):
    try:
        with st.spinner("Analyzing customer records..."):
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)

        classes = list(model.classes_)
        churn_index = classes.index("Yes")

        results = customers.copy()
        results["Predicted Churn"] = predictions
        results["Churn Probability (%)"] = (
            probabilities[:, churn_index] * 100
        ).round(2)

        st.session_state["churniq_results"] = results
        st.session_state["churniq_source"] = uploaded_file.name

    except Exception as exc:
        st.error(f"Prediction nahi ho saki: {exc}")
        st.stop()


if "churniq_results" not in st.session_state:
    st.markdown("#### Preview")
    st.caption(f"{len(customers):,} customer records loaded.")
    st.dataframe(customers.head(8), use_container_width=True, hide_index=True)
    st.stop()

results = st.session_state["churniq_results"]

# ---------- CHURN THRESHOLD ----------
st.markdown(
    '<div class="section-heading">Churn decision settings</div>',
    unsafe_allow_html=True,
)
threshold = st.slider(
    "Churn classification threshold (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help="Higher threshold means a customer needs a higher predicted probability to be classified as churn.",
)

# Reclassify customers using the selected threshold.
results = results.copy()
results["Predicted Churn"] = (
    results["Churn Probability (%)"] >= threshold
).map({True: "Yes", False: "No"})

st.caption(
    f"Customers with churn probability of {threshold}% or higher "
    "are classified as predicted churn."
)

# If a different file is uploaded, require fresh analysis.
if st.session_state.get("churniq_source") != uploaded_file.name:
    st.info("New file upload hui hai. Updated results ke liye analysis dobara run karein.")
    st.dataframe(customers.head(8), use_container_width=True, hide_index=True)
    st.stop()


# ---------- KPIs ----------
total = len(results)
yes_count = int((results["Predicted Churn"] == "Yes").sum())
no_count = total - yes_count
share = yes_count / total * 100 if total else 0
avg_probability = results["Churn Probability (%)"].mean() if total else 0

st.markdown(
    '<div class="section-heading">02 / Portfolio overview</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">Summary of predictions for the uploaded batch.</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total customers", f"{total:,}")
k2.metric("Predicted churn", f"{yes_count:,}")
k3.metric("Predicted not churn", f"{no_count:,}")
k4.metric("Average churn probability", f"{avg_probability:.1f}%")

# ---------- CHARTS ----------
st.markdown(
    '<div class="section-heading">03 / Risk analytics</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.35])

with left:
    pie_data = pd.DataFrame({
        "Prediction": ["Predicted churn", "Predicted not churn"],
        "Customers": [yes_count, no_count],
    })

    fig_pie = px.pie(
        pie_data,
        names="Prediction",
        values="Customers",
        hole=0.70,
        color="Prediction",
        color_discrete_map={
            "Predicted churn": "#5277f7",
            "Predicted not churn": "#c8d3e5",
        },
    )
    fig_pie.update_layout(
        title="Prediction distribution",
        margin=dict(l=15, r=15, t=55, b=15),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#263651", family="Inter"),
        legend=dict(orientation="h", y=-0.08),
    )
    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with right:
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

    risk = pd.cut(
        results["Churn Probability (%)"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    risk_counts = (
        risk.value_counts(sort=False)
        .reindex(labels, fill_value=0)
        .rename_axis("Probability range")
        .reset_index(name="Customers")
    )

    fig_risk = px.bar(
        risk_counts,
        x="Probability range",
        y="Customers",
        color="Customers",
        color_continuous_scale=["#dbe6ff", "#5277f7"],
        title="Churn probability distribution",
    )
    fig_risk.update_layout(
        margin=dict(l=15, r=15, t=55, b=15),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#263651", family="Inter"),
        coloraxis_showscale=False,
        xaxis_title="Estimated churn probability",
        yaxis_title="Number of customers",
    )
    st.plotly_chart(fig_risk, use_container_width=True)


# ---------- CUSTOMER EXPLORER ----------
st.markdown(
    '<div class="section-heading">04 / Customer explorer</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">Filter predictions and inspect individual records.</div>',
    unsafe_allow_html=True,
)

# ---------- ADVANCED FILTERS ----------
search = st.text_input(
    "Search customer records",
    placeholder="Search a customer ID or any text value...",
)

f1, f2, f3 = st.columns(3)

with f1:
    filter_value = st.selectbox(
        "Prediction status",
        ["All customers", "Predicted churn", "Predicted not churn"],
    )

with f2:
    contract_options = ["All contracts"]
    if "Contract" in results.columns:
        contract_options += sorted(
            results["Contract"].dropna().astype(str).unique().tolist()
        )

    contract_filter = st.selectbox(
        "Contract type",
        contract_options,
    )

with f3:
    if "MonthlyCharges" in results.columns:
        min_charge = float(results["MonthlyCharges"].min())
        max_charge = float(results["MonthlyCharges"].max())

        if min_charge < max_charge:
            charge_range = st.slider(
                "Monthly Charges",
                min_value=min_charge,
                max_value=max_charge,
                value=(min_charge, max_charge),
            )
        else:
            charge_range = (min_charge, max_charge)
    else:
        charge_range = None
        st.caption("MonthlyCharges column unavailable")

filtered = results.copy()

if filter_value == "Predicted churn":
    filtered = filtered[filtered["Predicted Churn"] == "Yes"]
elif filter_value == "Predicted not churn":
    filtered = filtered[filtered["Predicted Churn"] == "No"]

if contract_filter != "All contracts" and "Contract" in filtered.columns:
    filtered = filtered[
        filtered["Contract"].astype(str) == contract_filter
    ]

if charge_range is not None and "MonthlyCharges" in filtered.columns:
    charges = pd.to_numeric(
        filtered["MonthlyCharges"], errors="coerce"
    )
    filtered = filtered[
        charges.between(charge_range[0], charge_range[1])
    ]

if search.strip():
    mask = filtered.astype(str).apply(
        lambda col: col.str.contains(
            search.strip(), case=False, na=False
        )
    ).any(axis=1)
    filtered = filtered[mask]

st.caption(f"Showing {len(filtered):,} of {total:,} records.")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Churn Probability (%)": st.column_config.ProgressColumn(
            "Churn Probability",
            format="%.2f%%",
            min_value=0,
            max_value=100,
        ),
    },
)

st.download_button(
    "Download all predictions as CSV",
    data=results.to_csv(index=False).encode("utf-8-sig"),
    file_name="churn_predictions.csv",
    mime="text/csv",
    type="primary",
    use_container_width=True,
)

st.download_button(
    "Download filtered customers as CSV",
    data=filtered.to_csv(index=False).encode("utf-8-sig"),
    file_name="filtered_churn_customers.csv",
    mime="text/csv",
    use_container_width=True,
    key="download_filtered",
)

st.markdown(
    '<div class="footer">ChurnIQ · Powered by your trained Logistic Regression model. '
    'Batch statistics describe uploaded records only; predictions are not guarantees.</div>',
    unsafe_allow_html=True,
)
