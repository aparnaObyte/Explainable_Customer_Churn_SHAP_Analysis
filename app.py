import streamlit as st
import pandas as pd
import pickle
import shap
import plotly.graph_objects as go
from streamlit_shap import st_shap

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Churn Dashboard", page_icon="📡", layout="wide")

# --- APPLICATION HEADERS ---
st.title("📡 Live Customer Churn Dashboard")
st.markdown("### Updates instantly as you adjust the customer profile")
st.divider()

# --- CACHED RESOURCE LOADING ---
@st.cache_resource
def load_assets():
    model = pickle.load(open('model.pkl', 'rb'))
    explainer = pickle.load(open('explainer.pkl', 'rb'))
    feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
    ref_stats = pickle.load(open('reference_stats.pkl', 'rb'))
    return model, explainer, feature_columns, ref_stats

model, explainer, feature_columns, ref_stats = load_assets()

# --- DATA MAPPINGS ---
gender_map = {'Female': 0, 'Male': 1}
partner_map = {'No': 0, 'Yes': 1}
dependents_map = {'No': 0, 'Yes': 1}
phone_service_map = {'No': 0, 'Yes': 1}
multiple_lines_map = {'No': 0, 'No phone service': 1, 'Yes': 2}
internet_service_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
online_security_map = {'No': 0, 'No internet service': 1, 'Yes': 2}
online_backup_map = {'No': 0, 'No internet service': 1, 'Yes': 2}
device_protection_map = {'No': 0, 'No internet service': 1, 'Yes': 2}
tech_support_map = {'No': 0, 'No internet service': 1, 'Yes': 2}
streaming_tv_map = {'No': 0, 'No internet service': 1, 'Yes': 2}
streaming_movies_map = {'No': 0, 'No internet service': 1, 'Yes': 2}
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
paperless_billing_map = {'No': 0, 'Yes': 1}
payment_method_map = {
    'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1,
    'Electronic check': 2, 'Mailed check': 3
}

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⚙️ Customer Profile")
    gender = st.selectbox("Gender", list(gender_map.keys()))
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", list(partner_map.keys()))
    dependents = st.selectbox("Dependents", list(dependents_map.keys()))
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    st.divider()
    st.subheader("📞 Services")
    phone_service = st.selectbox("Phone Service", list(phone_service_map.keys()))
    multiple_lines = st.selectbox("Multiple Lines", list(multiple_lines_map.keys()))
    internet_service = st.selectbox("Internet Service", list(internet_service_map.keys()))
    online_security = st.selectbox("Online Security", list(online_security_map.keys()))
    online_backup = st.selectbox("Online Backup", list(online_backup_map.keys()))
    device_protection = st.selectbox("Device Protection", list(device_protection_map.keys()))
    tech_support = st.selectbox("Tech Support", list(tech_support_map.keys()))
    streaming_tv = st.selectbox("Streaming TV", list(streaming_tv_map.keys()))
    streaming_movies = st.selectbox("Streaming Movies", list(streaming_movies_map.keys()))

    st.divider()
    st.subheader("💳 Account")
    contract = st.selectbox("Contract", list(contract_map.keys()))
    paperless_billing = st.selectbox("Paperless Billing", list(paperless_billing_map.keys()))
    payment_method = st.selectbox("Payment Method", list(payment_method_map.keys()))
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 1000.0)

# --- INFERENCE VECTOR ---
input_dict = {
    'gender': gender_map[gender], 'SeniorCitizen': senior_citizen,
    'Partner': partner_map[partner], 'Dependents': dependents_map[dependents],
    'tenure': tenure, 'PhoneService': phone_service_map[phone_service],
    'MultipleLines': multiple_lines_map[multiple_lines],
    'InternetService': internet_service_map[internet_service],
    'OnlineSecurity': online_security_map[online_security],
    'OnlineBackup': online_backup_map[online_backup],
    'DeviceProtection': device_protection_map[device_protection],
    'TechSupport': tech_support_map[tech_support],
    'StreamingTV': streaming_tv_map[streaming_tv],
    'StreamingMovies': streaming_movies_map[streaming_movies],
    'Contract': contract_map[contract],
    'PaperlessBilling': paperless_billing_map[paperless_billing],
    'PaymentMethod': payment_method_map[payment_method],
    'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_dict])[feature_columns]
pred_proba = model.predict_proba(input_df)[0][1]

if pred_proba >= 0.7:
    risk_label, risk_color = "HIGH RISK", "#FF4B6E"
elif pred_proba >= 0.4:
    risk_label, risk_color = "MEDIUM RISK", "#FFA94D"
else:
    risk_label, risk_color = "LOW RISK", "#4ADE80"

# --- Row 1: KPI cards (Pure Python Containers) ---
k1, k2, k3, k4 = st.columns(4)
for col, label, value in zip(
    [k1, k2, k3, k4],
    ["Churn Probability", "Risk Level", "Tenure vs Avg", "Monthly Charges vs Avg"],
    [f"{pred_proba:.1%}", risk_label,
     f"{tenure} mo ({tenure - ref_stats['avg_tenure']:+.0f} vs avg)",
     f"${monthly_charges:.0f} ({monthly_charges - ref_stats['avg_monthly_charges']:+.0f} vs avg)"]
):
    with col:
        with st.container(border=True):
            if label == "Risk Level":
                if value == "HIGH RISK":
                    st.markdown(f"### :red[{value}]")
                elif value == "MEDIUM RISK":
                    st.markdown(f"### :orange[{value}]")
                else:
                    st.markdown(f"### :green[{value}]")
            else:
                st.markdown(f"### {value}")
            st.caption(label)

st.write("")

# --- Row 2: Gauge + Radar side by side ---
g1, g2 = st.columns(2)

with g1:
    with st.container(border=True):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pred_proba * 100,
            number={'suffix': "%", 'font': {'color': 'white', 'size': 32}},
            title={'text': "Churn Risk Gauge", 'font': {'color': '#F3F4F6', 'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#E5E7EB"},
                'bar': {'color': risk_color},
                'bgcolor': "#1C1F26",
                'steps': [
                    {'range': [0, 40], 'color': '#1f4d2c'},
                    {'range': [40, 70], 'color': '#4d3b1f'},
                    {'range': [70, 100], 'color': '#4d1f28'}
                ],
            }))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#F3F4F6"},
            height=280,
            margin=dict(t=40, b=10, l=10, r=10)
        )
        st.plotly_chart(fig, width="stretch")

with g2:
    with st.container(border=True):
        radar_categories = ['Tenure', 'Monthly Charges', 'Contract Commitment', 'Has Security', 'Has Tech Support']
        radar_values = [
            tenure / 72 * 100,
            monthly_charges / 120 * 100,
            (contract_map[contract] / 2) * 100,
            100 if online_security == 'Yes' else 0,
            100 if tech_support == 'Yes' else 0
        ]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=radar_values, theta=radar_categories, fill='toself',
            line_color=risk_color, fillcolor=risk_color, opacity=0.4
        ))
        fig2.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.03)",
                radialaxis=dict(visible=True, range=[0, 100], color="#E5E7EB", gridcolor="#374151"),
                angularaxis=dict(color="#E5E7EB", gridcolor="#374151")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "#F3F4F6"},
            title="Customer Profile Radar",
            height=280,
            margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(fig2, width="stretch")

# --- Row 3: SHAP waterfall + Top factors ---
st.write("")
shap_values_input = explainer.shap_values(input_df)
shap_row = shap_values_input[0, :, 1]

s1, s2 = st.columns([2, 1])

with s1:
    with st.container(border=True):
        st.markdown("##### 🔍 SHAP Explanation")
        st_shap(shap.plots.waterfall(
            shap.Explanation(
                values=shap_row, base_values=explainer.expected_value[1],
                data=input_df.iloc[0], feature_names=feature_columns
            )
        ), height=380)

with s2:
    with st.container(border=True):
        st.markdown("##### 📌 Top Factors")
        factor_df = pd.DataFrame({'feature': feature_columns, 'shap': shap_row})
        factor_df = factor_df.reindex(factor_df['shap'].abs().sort_values(ascending=False).index).head(5)
        for _, row in factor_df.iterrows():
            if row['shap'] > 0:
                color_syntax = ":red[▲ increases risk]"
            else:
                color_syntax = ":green[▼ reduces risk]"
            st.markdown(f"**{row['feature']}**: {color_syntax}")