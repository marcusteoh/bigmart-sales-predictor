import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.main .block-container { padding-top: 2rem; max-width: 1100px; }
.hero {
    background: linear-gradient(135deg, #1f3a5f 0%, #2c5f8a 100%);
    padding: 1.75rem 2rem; border-radius: 14px; color: white; margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 1.9rem; }
.hero p { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1rem; }
.pred-card {
    background: #f0f7ff; border: 1px solid #cfe3fb; border-radius: 14px;
    padding: 1.5rem; text-align: center;
}
[data-theme="dark"] .pred-card, .pred-card { color: inherit; }
.section-label {
    font-weight: 600; font-size: 0.95rem; color: #2c5f8a; margin-bottom: 0.3rem;
    text-transform: uppercase; letter-spacing: 0.03em;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return joblib.load("models/best_pipeline.joblib")


@st.cache_data
def load_metadata():
    with open("models/feature_metadata.json") as f:
        return json.load(f)


pipeline = load_pipeline()
metadata = load_metadata()

NUMERIC_RANGES = metadata["numeric_ranges"]
CAT_OPTIONS = metadata["categorical_options"]
FEATURES = metadata["features"]
DATASET_MEAN_SALES = 2181.29  # historical mean Item_Outlet_Sales, for comparison context

st.markdown(
    """
    <div class="hero">
        <h1>🛒 BigMart Sales Predictor</h1>
        <p>Forecast <b>Item_Outlet_Sales</b> for any product/outlet combination — helping
        category managers plan inventory and prioritise stock with data, not guesswork.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📊 About this model")
    st.metric("Test RMSE", f"${metadata['test_rmse']:,.0f}")
    st.metric("Test R²", f"{metadata['test_r2']:.3f}")
    st.metric("Test MAE", f"${metadata['test_mae']:,.0f}")
    st.markdown("---")
    st.markdown(
        """
        **Model:** Gradient Boosting Regressor (tuned via RandomizedSearchCV)

        **Dataset:** [BigMart Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/bigmartsalesdataset)
        — 8,523 historical retail transactions

        **Why RMSE?** It's in dollars and penalises large forecasting misses more heavily,
        matching the real cost of inventory over/under-stocking.
        """
    )

col_inputs, col_result = st.columns([1.3, 1], gap="large")

with col_inputs:
    st.markdown('<div class="section-label">Product Details</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        item_weight = st.number_input(
            "Item Weight (kg)",
            min_value=round(NUMERIC_RANGES["Item_Weight"]["min"], 1),
            max_value=round(NUMERIC_RANGES["Item_Weight"]["max"], 1),
            value=round(NUMERIC_RANGES["Item_Weight"]["mean"], 1),
            step=0.1,
            help="Weight of a single unit of the product.",
        )
        item_mrp = st.number_input(
            "Item MRP ($)",
            min_value=round(NUMERIC_RANGES["Item_MRP"]["min"], 2),
            max_value=round(NUMERIC_RANGES["Item_MRP"]["max"], 2),
            value=round(NUMERIC_RANGES["Item_MRP"]["mean"], 2),
            step=1.0,
            help="Maximum Retail Price — the single strongest driver of sales.",
        )
    with p2:
        item_visibility = st.slider(
            "Item Visibility (shelf share)",
            min_value=0.0,
            max_value=round(NUMERIC_RANGES["Item_Visibility"]["max"], 3),
            value=round(NUMERIC_RANGES["Item_Visibility"]["mean"], 3),
            step=0.001,
            format="%.3f",
            help="Fraction of total display area allocated to this product in the store.",
        )
        item_fat_content = st.selectbox("Item Fat Content", CAT_OPTIONS["Item_Fat_Content"])

    item_type = st.selectbox("Item Type", CAT_OPTIONS["Item_Type"])

    st.markdown('<div class="section-label">Outlet Details</div>', unsafe_allow_html=True)
    o1, o2 = st.columns(2)
    with o1:
        outlet_age = st.slider(
            "Outlet Age (years)",
            min_value=int(NUMERIC_RANGES["Outlet_Age"]["min"]),
            max_value=int(NUMERIC_RANGES["Outlet_Age"]["max"]),
            value=int(NUMERIC_RANGES["Outlet_Age"]["mean"]),
            help=f"Years since the outlet opened (reference year: {metadata['reference_year']}).",
        )
        outlet_size = st.selectbox("Outlet Size", CAT_OPTIONS["Outlet_Size"])
    with o2:
        outlet_location_type = st.selectbox("Outlet Location Type", CAT_OPTIONS["Outlet_Location_Type"])
        outlet_type = st.selectbox("Outlet Type", CAT_OPTIONS["Outlet_Type"])

input_row = pd.DataFrame([{
    "Item_Weight": item_weight,
    "Item_Visibility": item_visibility,
    "Item_MRP": item_mrp,
    "Outlet_Age": outlet_age,
    "Item_Fat_Content": item_fat_content,
    "Item_Type": item_type,
    "Outlet_Size": outlet_size,
    "Outlet_Location_Type": outlet_location_type,
    "Outlet_Type": outlet_type,
}])[FEATURES]

prediction_error = None
predicted_sales = None
try:
    if item_mrp <= 0 or item_weight <= 0:
        prediction_error = "Item MRP and Item Weight must be greater than zero."
    else:
        predicted_sales = float(pipeline.predict(input_row)[0])
        predicted_sales = max(predicted_sales, 0.0)  # sales cannot be negative
except Exception:
    prediction_error = "Couldn't generate a prediction for these inputs. Please review your selections and try again."

with col_result:
    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
    if prediction_error:
        st.error(f"⚠️ {prediction_error}")
    else:
        delta_pct = (predicted_sales - DATASET_MEAN_SALES) / DATASET_MEAN_SALES * 100
        st.markdown(
            f"""
            <div class="pred-card">
                <div style="font-size:1rem; opacity:0.75;">Predicted Item Outlet Sales</div>
                <div style="font-size:2.6rem; font-weight:700; color:#1f3a5f;">
                    ${predicted_sales:,.2f}
                </div>
                <div style="font-size:0.95rem;">
                    {'▲' if delta_pct >= 0 else '▼'} {abs(delta_pct):.1f}% vs. average outlet sales (${DATASET_MEAN_SALES:,.0f})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<div class="section-label">What drives this prediction?</div>', unsafe_allow_html=True)

    @st.cache_data
    def get_importance_chart_data(_pipeline):
        pre = _pipeline.named_steps["pre"]
        model = _pipeline.named_steps["model"]
        raw_names = pre.get_feature_names_out()
        importances = model.feature_importances_

        agg = {}
        for raw_name, imp in zip(raw_names, importances):
            base = raw_name.split("__", 1)[1].split("_")[0] if "__" in raw_name else raw_name
            for feat in FEATURES:
                if raw_name.split("__", 1)[1].startswith(feat):
                    base = feat
                    break
            agg[base] = agg.get(base, 0.0) + imp

        imp_df = pd.DataFrame(list(agg.items()), columns=["feature", "importance"])
        return imp_df.sort_values("importance", ascending=False)

    imp_df = get_importance_chart_data(pipeline)
    st.bar_chart(imp_df.set_index("feature")["importance"], horizontal=True)

st.markdown("---")
st.markdown('<div class="section-label">What-if: sensitivity to price (Item MRP)</div>', unsafe_allow_html=True)
st.caption("Holding every other input fixed at your current selections, how would predicted sales change if this product were priced differently?")

mrp_lo, mrp_hi = NUMERIC_RANGES["Item_MRP"]["min"], NUMERIC_RANGES["Item_MRP"]["max"]
mrp_range = np.linspace(mrp_lo, mrp_hi, 30)
scenario_rows = pd.concat([input_row] * len(mrp_range), ignore_index=True)
scenario_rows["Item_MRP"] = mrp_range
scenario_preds = pipeline.predict(scenario_rows)

chart_df = pd.DataFrame({"Item_MRP": mrp_range, "Predicted Sales ($)": scenario_preds}).set_index("Item_MRP")
st.line_chart(chart_df)
