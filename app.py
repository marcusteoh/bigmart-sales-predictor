"""BigMart Sales Predictor — Streamlit front end for the tuned scikit-learn pipeline.

Run with:  streamlit run app.py

The exported pipeline does its own cleaning and imputation, so this app sends *raw* inputs
(the nine fields a category manager can actually know) and lets the pipeline handle the rest.
That is what keeps training and serving identical — see Section 5 of the notebook.
"""

import json
import sys
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Resolve everything relative to this file rather than the working directory, so the app runs
# identically whether launched from the project root, from elsewhere, or by Streamlit Cloud.
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Required even though nothing here calls it directly: joblib stores a reference to the
# transformer classes, so the module must be importable for the pipeline to unpickle.
import bigmart_preprocessing  # noqa: E402,F401
from bigmart_preprocessing import NON_CONSUMABLE_ITEM_TYPES  # noqa: E402

st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling is kept deliberately light and theme-aware: colours are derived from Streamlit's
# own CSS variables where possible so the app stays legible in both light and dark mode.
CUSTOM_CSS = """
<style>
.main .block-container { padding-top: 2rem; max-width: 1150px; }
.hero {
    background: linear-gradient(135deg, #1f3a5f 0%, #2c5f8a 100%);
    padding: 1.6rem 2rem; border-radius: 14px; color: #ffffff; margin-bottom: 1.25rem;
}
.hero h1 { margin: 0; font-size: 1.85rem; color: #ffffff; }
.hero p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 0.98rem; color: #ffffff; }
.pred-card {
    border: 1px solid rgba(128, 160, 200, 0.45);
    background: rgba(120, 165, 220, 0.12);
    border-radius: 14px; padding: 1.4rem 1.2rem; text-align: center;
}
.pred-value { font-size: 2.5rem; font-weight: 700; line-height: 1.1; }
.pred-interval { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }
.pred-sub { font-size: 0.9rem; opacity: 0.75; }
.section-label {
    font-weight: 600; font-size: 0.9rem; margin-bottom: 0.35rem;
    text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.75;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load the point-estimate pipeline plus the two quantile pipelines."""
    return {
        "point": joblib.load(MODEL_DIR / "best_pipeline.joblib"),
        "lower": joblib.load(MODEL_DIR / "quantile_lower_pipeline.joblib"),
        "upper": joblib.load(MODEL_DIR / "quantile_upper_pipeline.joblib"),
    }


@st.cache_data
def load_metadata():
    with open(MODEL_DIR / "feature_metadata.json") as f:
        return json.load(f)


try:
    models = load_models()
    meta = load_metadata()
except FileNotFoundError:
    st.error(
        "**Model files not found.** Run every cell of `BigMart_Sales_Prediction.ipynb` first — "
        "its final section exports `models/best_pipeline.joblib`, the two quantile pipelines, "
        "and `models/feature_metadata.json`."
    )
    st.stop()

RANGES = meta["numeric_ranges"]
OPTIONS = meta["categorical_options"]
APP_COLUMNS = meta["app_input_columns"]
REFERENCE_YEAR = meta["reference_year"]
TARGET_MEAN = meta["target_mean"]


def predict_with_interval(rows: pd.DataFrame):
    """Return (point, lower, upper) predictions, floored at zero.

    Sales cannot be negative, and the quantile models are fitted independently of the point
    model, so we also enforce lower <= point <= upper rather than assuming it holds.
    """
    point = np.clip(models["point"].predict(rows), 0, None)
    lower = np.clip(models["lower"].predict(rows), 0, None)
    upper = np.clip(models["upper"].predict(rows), 0, None)
    lower = np.minimum(lower, point)
    upper = np.maximum(upper, point)
    return point, lower, upper


# --------------------------------------------------------------------------------------
# Header + sidebar model card
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛒 BigMart Sales Predictor</h1>
        <p>Forecast <b>Item_Outlet_Sales</b> for any product/outlet combination — so category
        managers can plan inventory and prioritise stock with evidence, not guesswork.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📊 Model card")
    st.metric("Test RMSE", f"${meta['test_rmse']:,.0f}")
    st.metric("Test MAE", f"${meta['test_mae']:,.0f}")
    st.metric("Test R²", f"{meta['test_r2']:.3f}")
    st.caption(
        f"Held-out estimates from a 20% test split. The deployed model is the same "
        f"configuration refit on all {meta['n_training_rows']:,} rows, so these are honest "
        f"estimates of its accuracy rather than scores measured on the shipped object."
    )

    st.divider()
    st.markdown(f"**Model:** {meta['model_name']}")
    st.markdown(
        f"**vs. no model:** a flat-average forecast scores RMSE "
        f"${meta['naive_baseline_rmse']:,.0f} — this model removes "
        f"{(1 - meta['test_mae'] / meta['naive_baseline_mae']) * 100:.0f}% of the typical "
        f"forecast error."
    )
    st.markdown(
        f"**Prediction intervals:** nominal {meta['interval_nominal_coverage']:.0%}, "
        f"measured {meta['interval_empirical_coverage']:.1%} coverage on held-out data."
    )

    st.divider()
    st.markdown(
        """
        **Why RMSE?** It is denominated in dollars and penalises large misses more heavily,
        matching the real cost of a stock-out or of overstocking.

        **Dataset:** [BigMart Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/bigmartsalesdataset)
        — 8,523 historical retail records.
        """
    )
    st.caption(f"scikit-learn {meta['sklearn_version']}")


tab_predict, tab_whatif, tab_batch = st.tabs(
    ["🎯 Single prediction", "📈 Price sensitivity", "📦 Batch scoring"]
)


# --------------------------------------------------------------------------------------
# Tab 1 — single prediction
# --------------------------------------------------------------------------------------
with tab_predict:
    col_inputs, col_result = st.columns([1.25, 1], gap="large")

    with col_inputs:
        st.markdown('<div class="section-label">Product details</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            item_weight = st.number_input(
                "Item weight (kg)",
                min_value=float(round(RANGES["Item_Weight"]["min"], 1)),
                max_value=float(round(RANGES["Item_Weight"]["max"], 1)),
                value=float(round(RANGES["Item_Weight"]["mean"], 1)),
                step=0.1,
                help="Weight of a single unit of the product.",
            )
            item_mrp = st.number_input(
                "Item MRP ($)",
                min_value=float(round(RANGES["Item_MRP"]["min"], 2)),
                max_value=float(round(RANGES["Item_MRP"]["max"], 2)),
                value=float(round(RANGES["Item_MRP"]["mean"], 2)),
                step=1.0,
                help="Maximum retail price — the strongest single driver of sales.",
            )
        with p2:
            # Lower bound comes from the cleaned data, so a visibility of exactly 0 cannot be
            # entered: the notebook established that zero shelf share is not physically real.
            item_visibility = st.slider(
                "Item visibility (shelf share)",
                min_value=float(round(RANGES["Item_Visibility"]["min"], 3)),
                max_value=float(round(RANGES["Item_Visibility"]["max"], 3)),
                value=float(round(RANGES["Item_Visibility"]["mean"], 3)),
                step=0.001,
                format="%.3f",
                help="Fraction of the store's total display area given to this product.",
            )
            item_type = st.selectbox("Item type", OPTIONS["Item_Type"])

        is_non_consumable = item_type in NON_CONSUMABLE_ITEM_TYPES
        if is_non_consumable:
            item_fat_content = "Non-Edible"
            st.info(
                f"**{item_type}** is a non-consumable category, so fat content does not apply. "
                "The model treats these products as *Non-Edible* — matching how they were "
                "labelled during training.",
                icon="ℹ️",
            )
        else:
            item_fat_content = st.radio(
                "Item fat content",
                meta["fat_content_input_options"],
                horizontal=True,
            )

        st.markdown('<div class="section-label">Outlet details</div>', unsafe_allow_html=True)
        o1, o2 = st.columns(2)
        with o1:
            outlet_age = st.slider(
                "Outlet age (years)",
                min_value=int(RANGES["Outlet_Age"]["min"]),
                max_value=int(RANGES["Outlet_Age"]["max"]),
                value=int(RANGES["Outlet_Age"]["mean"]),
                help=f"Years since the outlet opened (reference year {REFERENCE_YEAR}).",
            )
            outlet_size = st.selectbox("Outlet size", OPTIONS["Outlet_Size"])
        with o2:
            outlet_location_type = st.selectbox(
                "Outlet location tier", OPTIONS["Outlet_Location_Type"])
            outlet_type = st.selectbox("Outlet type", OPTIONS["Outlet_Type"])

    input_row = pd.DataFrame([{
        "Item_Weight": item_weight,
        "Item_Fat_Content": item_fat_content,
        "Item_Visibility": item_visibility,
        "Item_Type": item_type,
        "Item_MRP": item_mrp,
        "Outlet_Establishment_Year": REFERENCE_YEAR - outlet_age,
        "Outlet_Size": outlet_size,
        "Outlet_Location_Type": outlet_location_type,
        "Outlet_Type": outlet_type,
    }])[APP_COLUMNS]

    with col_result:
        st.markdown('<div class="section-label">Forecast</div>', unsafe_allow_html=True)

        if item_mrp <= 0 or item_weight <= 0:
            st.error("⚠️ Item MRP and item weight must both be greater than zero.")
        else:
            try:
                point, lower, upper = predict_with_interval(input_row)
                predicted, lo, hi = point[0], lower[0], upper[0]
                delta_pct = (predicted - TARGET_MEAN) / TARGET_MEAN * 100
                arrow = "▲" if delta_pct >= 0 else "▼"

                st.markdown(
                    f"""
                    <div class="pred-card">
                        <div class="pred-sub">Predicted item outlet sales</div>
                        <div class="pred-value">${predicted:,.2f}</div>
                        <div class="pred-interval">
                            80% likely between <b>${lo:,.0f}</b> and <b>${hi:,.0f}</b>
                        </div>
                        <div class="pred-sub" style="margin-top:0.6rem;">
                            {arrow} {abs(delta_pct):.1f}% vs. the ${TARGET_MEAN:,.0f} average
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    "The range is a calibrated 80% prediction interval, not a margin of error "
                    "on the point estimate — use its upper bound when sizing safety stock."
                )
            except Exception as exc:  # noqa: BLE001 - surface a usable message, never a traceback
                st.error(
                    "⚠️ Couldn't generate a prediction for these inputs. Please review your "
                    f"selections and try again.\n\nDetails: `{type(exc).__name__}`"
                )

        st.markdown("")
        st.markdown('<div class="section-label">What drives the model</div>',
                    unsafe_allow_html=True)

        @st.cache_data
        def importance_by_feature(_pipeline):
            """Aggregate one-hot importances back to business features.

            Uses the encoder's own `categories_` to size each feature's block of columns,
            rather than parsing generated column names — the names are not guaranteed to be
            unambiguous when one feature name is a prefix of another.
            """
            model = _pipeline.named_steps["model"]
            if not hasattr(model, "feature_importances_"):
                return None

            encoder = _pipeline.named_steps["pre"].named_steps["encode"]
            numeric = meta["numeric_features"]
            categorical = meta["categorical_features"]
            importances = model.feature_importances_

            totals, cursor = {}, 0
            for name in numeric:
                totals[name] = float(importances[cursor])
                cursor += 1
            for name, categories in zip(categorical,
                                        encoder.named_transformers_["cat"].categories_):
                width = len(categories)
                totals[name] = float(importances[cursor:cursor + width].sum())
                cursor += width

            return (pd.DataFrame({"feature": list(totals), "importance": list(totals.values())})
                    .sort_values("importance", ascending=False))

        imp_df = importance_by_feature(models["point"])
        if imp_df is None:
            st.caption("This model does not expose feature importances.")
        else:
            st.bar_chart(imp_df.set_index("feature")["importance"], horizontal=True)
            st.caption(
                "Impurity-based importance from the deployed model. The notebook cross-checks "
                "this against permutation importance, which is the more reliable measure."
            )


# --------------------------------------------------------------------------------------
# Tab 2 — price sensitivity
# --------------------------------------------------------------------------------------
with tab_whatif:
    st.markdown('<div class="section-label">What-if: sensitivity to price</div>',
                unsafe_allow_html=True)
    st.caption(
        "Holding every other input fixed at the selections on the first tab, how would the "
        "forecast change if this product were priced differently?"
    )

    mrp_grid = np.linspace(RANGES["Item_MRP"]["min"], RANGES["Item_MRP"]["max"], 40)
    scenarios = pd.concat([input_row] * len(mrp_grid), ignore_index=True)
    scenarios["Item_MRP"] = mrp_grid

    s_point, s_lower, s_upper = predict_with_interval(scenarios)
    chart_df = pd.DataFrame({
        "Item MRP ($)": mrp_grid,
        "Predicted sales": s_point,
        "Lower (10th pct)": s_lower,
        "Upper (90th pct)": s_upper,
    }).set_index("Item MRP ($)")

    st.line_chart(chart_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("At current price", f"${s_point[np.argmin(np.abs(mrp_grid - item_mrp))]:,.0f}",
              help=f"Your selected MRP of ${item_mrp:,.2f}")
    c2.metric("At lowest MRP in data", f"${s_point[0]:,.0f}",
              help=f"${mrp_grid[0]:,.2f}")
    c3.metric("At highest MRP in data", f"${s_point[-1]:,.0f}",
              help=f"${mrp_grid[-1]:,.2f}")

    st.warning(
        "**This is a model sensitivity curve, not a pricing recommendation.** It shows what the "
        "model predicts for otherwise-identical products at different price points — which is a "
        "*correlation* learned from historical data. It does not establish that raising this "
        "product's price would raise its revenue, because price is not randomly assigned in the "
        "data: expensive products differ from cheap ones in ways the model cannot see.",
        icon="⚠️",
    )


# --------------------------------------------------------------------------------------
# Tab 3 — batch scoring
# --------------------------------------------------------------------------------------
with tab_batch:
    st.markdown('<div class="section-label">Score a whole product range at once</div>',
                unsafe_allow_html=True)
    st.caption(
        "Upload a CSV to forecast many product/outlet combinations in one pass — the realistic "
        "workflow for a category manager planning a season, rather than typing one row at a time."
    )

    template = pd.DataFrame([{
        "Item_Weight": 12.9, "Item_Fat_Content": "Low Fat", "Item_Visibility": 0.07,
        "Item_Type": "Snack Foods", "Item_MRP": 180.0,
        "Outlet_Establishment_Year": 1999, "Outlet_Size": "Medium",
        "Outlet_Location_Type": "Tier 1", "Outlet_Type": "Supermarket Type1",
    }])[APP_COLUMNS]

    st.download_button(
        "⬇️ Download CSV template",
        template.to_csv(index=False),
        file_name="bigmart_batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded is not None:
        try:
            batch = pd.read_csv(StringIO(uploaded.getvalue().decode("utf-8")))
        except Exception:
            st.error("⚠️ Couldn't read that file. Please upload a valid UTF-8 encoded CSV.")
            batch = None

        if batch is not None:
            missing = [c for c in APP_COLUMNS if c not in batch.columns]
            if missing:
                st.error(
                    "⚠️ The uploaded file is missing required column(s): "
                    f"`{'`, `'.join(missing)}`. Download the template above for the exact format."
                )
            elif len(batch) == 0:
                st.error("⚠️ That file has no data rows.")
            elif len(batch) > 5000:
                st.error(f"⚠️ {len(batch):,} rows is above the 5,000-row limit for this demo.")
            else:
                try:
                    scored = batch.copy()
                    point, lower, upper = predict_with_interval(batch[APP_COLUMNS])
                    scored["Predicted_Sales"] = point.round(2)
                    scored["Interval_Low"] = lower.round(2)
                    scored["Interval_High"] = upper.round(2)

                    st.success(f"✅ Scored {len(scored):,} rows.")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total forecast sales", f"${point.sum():,.0f}")
                    m2.metric("Mean per line", f"${point.mean():,.0f}")
                    m3.metric("Highest single line", f"${point.max():,.0f}")

                    st.dataframe(scored, width="stretch", height=340)
                    st.download_button(
                        "⬇️ Download predictions",
                        scored.to_csv(index=False),
                        file_name="bigmart_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        "⚠️ Couldn't score that file. Check that the value in every column is of "
                        "the expected type — for example `Item_MRP` must be numeric.\n\n"
                        f"Details: `{type(exc).__name__}`"
                    )
    else:
        st.info(
            "The pipeline cleans and imputes uploaded data with the same rules used in training, "
            "so rows with a missing `Item_Weight` or `Outlet_Size` will still score correctly.",
            icon="ℹ️",
        )
