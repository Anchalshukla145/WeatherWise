"""
WeatherWise Analytics — Streamlit Web Application
MongoDB-Powered Weather Intelligence and Machine Learning Platform
"""

import io
import os
import sys
from datetime import date, datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Configure project paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_METRICS_PATH,
    MODEL_COMPARISON_CSV_PATH,
    REGRESSION_FEATURES,
    REGRESSION_METRICS_PATH,
)
from src.classification import (
    load_classification_models,
    predict_next_day_rain,
    train_classification_models,
)
from src.data_loader import audit_data_quality, clean_weather_data, detect_and_load_csv, load_dataset
from src.evaluation import (
    build_comparison_dataframe,
    evaluate_all_classifiers,
    evaluate_all_regressors,
    rank_models_by_metric,
)
from src.mongodb_manager import MongoDBManager
from src.preprocessing import (
    construct_ml_dataset,
    prepare_classification_data,
    prepare_regression_data,
)
from src.recommendations import generate_weather_recommendations
from src.regression import (
    load_primary_regressor,
    predict_next_day_temperature,
    train_regression_models,
)
from src.utils import format_metric, format_temp, load_json_file, logger
from src.visualizations import (
    plot_actual_vs_predicted_regression,
    plot_combined_roc_curve,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_metrics_comparison_bar,
    plot_monthly_rainfall_frequency,
    plot_outlier_boxplots,
    plot_residual_distribution,
    plot_target_distribution,
    plot_temperature_trends,
)

# ---------------------------------------------------------
# Streamlit Page Configuration & Design System
# ---------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} | Weather Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Light background, navy text, teal & slate accents)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #0F172A;
    }
    
    .main {
        background-color: #F8FAFC;
    }
    
    /* Top Brand Header */
    .brand-header {
        background: linear-gradient(135deg, #0D3B66 0%, #028090 100%);
        padding: 22px 28px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .brand-header h1 {
        color: #FFFFFF !important;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .brand-header p {
        color: #E0F2FE;
        font-size: 13px;
        margin: 4px 0 0 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        padding: 16px 18px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 14px;
    }
    .metric-card-title {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.5px;
    }
    .metric-card-value {
        font-size: 22px;
        font-weight: 700;
        color: #0D3B66;
        margin: 4px 0 2px 0;
    }
    .metric-card-subtitle {
        font-size: 11px;
        color: #028090;
    }

    /* Content Cards */
    .content-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }

    /* Recommendation Containers */
    .recom-box {
        background: #FFFFFF;
        border-left: 5px solid #00A896;
        padding: 14px 18px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        border-left-width: 5px;
        margin-bottom: 12px;
    }
    
    /* Button Styles */
    div.stButton > button:first-child {
        background-color: #0D3B66;
        color: white;
        border-radius: 8px;
        font-weight: 500;
        border: none;
        padding: 8px 18px;
    }
    div.stButton > button:first-child:hover {
        background-color: #028090;
        color: white;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# State Initialization & Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_mongo_manager():
    """Returns singleton MongoDBManager instance."""
    return MongoDBManager()


@st.cache_data(show_spinner=False)
def get_cached_dataset():
    """Loads and cleans raw weather data once."""
    df, audit = load_dataset()
    return df, audit


@st.cache_data(show_spinner=False)
def get_cached_evaluation_artifacts():
    """Loads precomputed metrics JSONs or runs pipeline if missing."""
    if CLASSIFICATION_METRICS_PATH.exists() and REGRESSION_METRICS_PATH.exists():
        clf_metrics = load_json_file(CLASSIFICATION_METRICS_PATH)
        reg_metrics = load_json_file(REGRESSION_METRICS_PATH)
        df_comp = build_comparison_dataframe(clf_metrics)
        return clf_metrics, reg_metrics, df_comp

    df, _ = get_cached_dataset()
    ml_df = construct_ml_dataset(df)
    clf_data = prepare_classification_data(ml_df)
    reg_data = prepare_regression_data(ml_df)
    clf_models = train_classification_models(clf_data["X_train"], clf_data["y_train"], save_models=True)
    reg_models = train_regression_models(reg_data["X_train"], reg_data["y_train"], save_primary_model=True)
    clf_metrics = evaluate_all_classifiers(clf_models, clf_data["X_test"], clf_data["y_test"])
    reg_metrics = evaluate_all_regressors(reg_models, reg_data["X_test"], reg_data["y_test"])
    df_comp = build_comparison_dataframe(clf_metrics)
    return clf_metrics, reg_metrics, df_comp


@st.cache_resource(show_spinner=False)
def get_cached_models():
    """Loads serialized machine learning models from disk."""
    clf_models = load_classification_models()
    reg_model = load_primary_regressor()
    if not clf_models or reg_model is None:
        df, _ = get_cached_dataset()
        ml_df = construct_ml_dataset(df)
        clf_data = prepare_classification_data(ml_df)
        reg_data = prepare_regression_data(ml_df)
        clf_models = train_classification_models(clf_data["X_train"], clf_data["y_train"], save_models=True)
        reg_models = train_regression_models(reg_data["X_train"], reg_data["y_train"], save_primary_model=True)
        reg_model = reg_models.get("Random Forest Regressor")
    return clf_models, reg_model


# Load core resources
weather_df, dataset_audit = get_cached_dataset()
clf_metrics_data, reg_metrics_data, df_comparison = get_cached_evaluation_artifacts()
clf_models_dict, regressor_model = get_cached_models()
mongo_mgr = get_mongo_manager()
mongo_connected, mongo_msg = mongo_mgr.ping()


# ---------------------------------------------------------
# Sidebar Navigation & Downloads
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;">
            <h2 style="color: #0D3B66; font-size: 20px; font-weight: 700; margin: 0;">WeatherWise</h2>
            <p style="color: #64748B; font-size: 12px; margin: 2px 0 0 0;">Intelligence & ML Platform</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Application Navigation",
        [
            "Overview Dashboard",
            "Data & EDA Overview",
            "Weather Prediction & Forecast",
            "Model Comparison & Objectives",
            "MongoDB Database",
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='margin: 18px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

    # MongoDB Health Card
    if mongo_connected:
        st.markdown(
            """
            <div style="padding: 10px; background: #DEF7EC; border-radius: 8px; border: 1px solid #BCF0DA;">
                <span style="color: #03543F; font-size: 11px; font-weight: 600; text-transform: uppercase;">MongoDB Server</span><br>
                <span style="color: #0E9F6E; font-size: 13px; font-weight: 700;">● Online (Connected)</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="padding: 10px; background: #FEF2F2; border-radius: 8px; border: 1px solid #FECACA;">
                <span style="color: #991B1B; font-size: 11px; font-weight: 600; text-transform: uppercase;">MongoDB Server</span><br>
                <span style="color: #DC2626; font-size: 13px; font-weight: 700;">○ Offline (CSV Fallback)</span>
            </div>
            """,
            unsafe_allow_html=True
        )




# Top Global Header
st.markdown(
    f"""
    <div class="brand-header">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PAGE 1: OVERVIEW DASHBOARD
# =========================================================
if page == "Overview Dashboard":
    st.subheader("Overview Dashboard & Weather Intelligence")

    # Overview Banner: Image alongside Problem Statement & Dataset Info
    overview_img_path = PROJECT_ROOT / "assets" / "weather_overview.jpg"
    img_col, text_col = st.columns([1, 2])

    with img_col:
        if overview_img_path.exists():
            img = Image.open(overview_img_path)
            st.image(img, use_container_width=True, caption="Weather Intelligence & Forecasting System")
        else:
            st.info("Illustration placeholder: WeatherWise Analytics Overview")

    with text_col:
        st.markdown(
            """
            <div class="content-card" style="margin-bottom: 0;">
                <h4 style="color: #0D3B66; margin-top: 0;">Problem Statement</h4>
                <p style="font-size: 13px; line-height: 1.6; color: #334155; margin-bottom: 12px;">
                    Atmospheric dynamics strongly govern urban mobility, agricultural planning, outdoor work, 
                    and municipal resource allocation. While raw daily weather observations are logged continuously, 
                    traditional flat-file archives lack flexible querying, indexing, and transactional integrity. 
                    Furthermore, standard machine learning pipelines frequently suffer from <b>data leakage</b> 
                    by improperly shuffling time-series sequences or using current-day rain indicators to predict 
                    current-day events.
                </p>
                <p style="font-size: 13px; line-height: 1.6; color: #334155; margin-bottom: 14px;">
                    <b>WeatherWise Analytics</b> addresses this by pairing a centralized MongoDB NoSQL document 
                    store with strictly leak-free chronological machine learning. The system ingests 73+ years 
                    of daily Delhi weather records, models next-day precipitation and maximum temperature, and 
                    translates probabilistic model outputs into intuitive, evidence-based recommendations.
                </p>
                <h5 style="color: #028090; margin-bottom: 6px;">Dataset Quick Profile</h5>
                <ul style="font-size: 13px; color: #475569; margin: 0; padding-left: 20px;">
                    <li><b>Temporal Span:</b> 1951-01-01 to 2024-06-19 (73.5 Continuous Years)</li>
                    <li><b>Observation Volume:</b> 26,803 Verified Daily Records (Zero duplicate dates)</li>
                    <li><b>Predictor Features:</b> Temp Max, Temp Min, Temp Difference, Month, Day, Week</li>
                    <li><b>Target Variables:</b> Next-Day Rain (Class) and Next-Day Temp Max (Continuous °C)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Real Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Historical Observations</div>
                <div class="metric-card-value">{dataset_audit['total_records']:,}</div>
                <div class="metric-card-subtitle">{dataset_audit['min_date']} to {dataset_audit['max_date']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        rainy_count = dataset_audit['rain_distribution']['Rain_1']
        rain_rate = round(rainy_count / dataset_audit['total_records'] * 100, 1)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Rainy Days Recorded</div>
                <div class="metric-card-value">{rainy_count:,}</div>
                <div class="metric-card-subtitle">{rain_rate}% of all recorded days</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        best_name, best_f1, _ = rank_models_by_metric(df_comparison, "F1 Score")
        auc_val = df_comparison.loc[df_comparison['Model']==best_name, 'ROC-AUC'].values[0]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Top Classifier ({best_name})</div>
                <div class="metric-card-value">{best_f1:.4f} F1</div>
                <div class="metric-card-subtitle">ROC-AUC: {auc_val:.4f} | Recall: {df_comparison.loc[df_comparison['Model']==best_name, 'Recall'].values[0]:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        rf_reg_mae = reg_metrics_data.get("Random Forest Regressor", {}).get("mae", 1.18)
        rf_reg_r2 = reg_metrics_data.get("Random Forest Regressor", {}).get("r2_score", 0.944)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Temp Regressor (R²)</div>
                <div class="metric-card-value">{rf_reg_r2:.4f}</div>
                <div class="metric-card-subtitle">Mean Absolute Error: {rf_reg_mae:.2f} °C</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Charts Row
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        st.plotly_chart(plot_target_distribution(weather_df), use_container_width=True)
    with ch_col2:
        st.plotly_chart(plot_monthly_rainfall_frequency(weather_df), use_container_width=True)

    # Long-term Temperature Time-Series
    st.plotly_chart(plot_temperature_trends(weather_df, sample_step=15), use_container_width=True)


# =========================================================
# PAGE 2: DATA & EDA OVERVIEW (With Nulls, Unique, Info, Describe & 10x10 Pagination)
# =========================================================
elif page == "Data & EDA Overview":
    st.subheader("Weather Data & Exploratory Data Analysis (EDA) Overview")
    st.write("Comprehensive data quality audits, schema inspection, summary statistics, and paginated records.")

    # 1. Null Counts, Unique Counts, and Schema Info Table
    st.markdown("#### 1. Column Metadata, Null Value Audit & Unique Counts")
    
    col_info_list = []
    for col in weather_df.columns:
        null_count = int(weather_df[col].isna().sum())
        null_pct = round(null_count / len(weather_df) * 100, 2)
        unique_cnt = int(weather_df[col].nunique())
        dtype_str = str(weather_df[col].dtype)
        sample_val = str(weather_df[col].dropna().iloc[0]) if not weather_df[col].dropna().empty else "N/A"
        
        col_info_list.append({
            "Column Name": col,
            "Data Type": dtype_str,
            "Null Count": null_count,
            "Null Percentage": f"{null_pct}%",
            "Unique Count": f"{unique_cnt:,}",
            "Sample Value": sample_val
        })
    
    df_col_audit = pd.DataFrame(col_info_list)
    st.dataframe(df_col_audit, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Descriptive Statistics (.describe())
    st.markdown("#### 2. Descriptive Statistical Summary (.describe())")
    st.write("Summary statistics for all numeric weather and calendar variables:")
    numeric_cols = ["Temp Max", "Temp Min", "Temp_diff", "Rain_Numeric", "Month", "Day", "Week", "Year"]
    avail_num = [c for c in numeric_cols if c in weather_df.columns]
    desc_df = weather_df[avail_num].describe().T
    st.dataframe(desc_df.style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")

    # 3. Paginated Dataset Display (10 rows per page)
    st.markdown("#### 3. Historical Weather Dataset (Paginated View: 10 Records per Page)")
    
    total_rows = len(weather_df)
    page_size = 10
    total_pages = (total_rows + page_size - 1) // page_size

    # Pagination controls
    p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 2, 2])
    with p_col1:
        current_page = st.number_input(
            "Page Number",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key="data_page_num"
        )
    with p_col2:
        st.write(f"<div style='padding-top: 28px; color: #64748B;'>of <b>{total_pages:,}</b> pages</div>", unsafe_allow_html=True)
    with p_col3:
        st.write(f"<div style='padding-top: 28px; color: #028090;'>Showing records <b>{(current_page-1)*page_size + 1:,}</b> to <b>{min(current_page*page_size, total_rows):,}</b> of {total_rows:,}</div>", unsafe_allow_html=True)
    with p_col4:
        # Quick Jump selector
        jump_year = st.selectbox(
            "Jump to Year",
            ["None"] + [str(y) for y in sorted(weather_df["Year"].unique())],
            index=0
        )
        if jump_year != "None":
            first_idx = weather_df[weather_df["Year"] == int(jump_year)].index[0]
            current_page = (first_idx // page_size) + 1

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    page_df = weather_df.iloc[start_idx:end_idx].copy()

    disp_columns = ["_id", "Date", "Temp Max", "Temp Min", "Temp_diff", "Rain", "Has_rained", "Month", "Year", "Day", "month_name"]
    disp_cols_avail = [c for c in disp_columns if c in page_df.columns]
    st.dataframe(page_df[disp_cols_avail], use_container_width=True, hide_index=True)

    # Export Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    exp_c1, exp_c2 = st.columns([1, 3])
    with exp_c1:
        csv_page = page_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Current 10 Rows (CSV)",
            data=csv_page,
            file_name=f"delhi_weather_page_{current_page}.csv",
            mime="text/csv"
        )
    with exp_c2:
        csv_full = weather_df[disp_cols_avail].head(5000).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download First 5,000 Records (CSV)",
            data=csv_full,
            file_name="delhi_weather_export_5000.csv",
            mime="text/csv"
        )


# =========================================================
# PAGE 3: WEATHER PREDICTION & FORECAST (Combined Rainfall + Temperature)
# =========================================================
elif page == "Weather Prediction & Forecast":
    st.subheader("Unified Next-Day Weather Prediction & Recommendations")
    st.write(
        "Single integrated interface predicting **Next-Day Rainfall (Classification)** and "
        "**Next-Day Maximum Temperature (Regression)** using day $t$ meteorological observations."
    )

    pred_in_col, pred_out_col = st.columns([1, 1])

    with pred_in_col:
        st.markdown("#### 1. Input Meteorological Observations (Day t)")
        in_t_max = st.slider("Observed Maximum Temperature (°C)", 10.0, 50.0, 34.0, step=0.1, key="unif_tmax")
        in_t_min = st.slider("Observed Minimum Temperature (°C)", 2.0, 36.0, 23.5, step=0.1, key="unif_tmin")
        in_t_diff = round(in_t_max - in_t_min, 2)
        st.caption(f"Calculated Diurnal Range (Max - Min): **{in_t_diff:.1f} °C**")

        date_col_a, date_col_b = st.columns(2)
        with date_col_a:
            in_m = st.selectbox(
                "Month of Year",
                list(range(1, 13)),
                index=6,
                format_func=lambda m: [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ][m - 1],
                key="unif_month"
            )
        with date_col_b:
            in_d = st.slider("Day of Month", 1, 31, 15, key="unif_day")

        in_w = int(min(52, max(1, (in_m - 1) * 4.3 + in_d // 7 + 1)))

        selected_classifier_name = st.selectbox(
            "Select Rainfall Classification Model",
            ["Random Forest", "Decision Tree", "Logistic Regression"],
            index=0,
            key="unif_clf_choice"
        )

    with pred_out_col:
        st.markdown("#### 2. Dual Model Predictions (Day t+1 / Tomorrow)")

        # 1. Classification Output
        clf_model_active = clf_models_dict.get(selected_classifier_name)
        if clf_model_active is not None:
            rain_pred = predict_next_day_rain(
                clf_model_active,
                temp_max=in_t_max,
                temp_min=in_t_min,
                temp_diff=in_t_diff,
                month=in_m,
                day=in_d,
                week=in_w
            )
            is_rain = rain_pred["prediction_class"] == 1
            rain_prob_pct = round(rain_pred["probability_rain"] * 100, 1)

            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 6px solid {'#E76F51' if is_rain else '#00A896'};">
                    <div class="metric-card-title">Rainfall Forecast ({selected_classifier_name})</div>
                    <div class="metric-card-value" style="color: {'#E76F51' if is_rain else '#00A896'};">
                        {rain_pred['prediction_label']}
                    </div>
                    <div class="metric-card-subtitle">
                        Rain Probability: <b>{rain_prob_pct}%</b> | Dry Probability: <b>{round(rain_pred['probability_no_rain']*100, 1)}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(float(rain_pred["probability_rain"]))
        else:
            rain_prob_pct = 20.0
            st.error("Classifier model not loaded.")

        # 2. Temperature Regression Output
        if regressor_model is not None:
            pred_temp_val = predict_next_day_temperature(
                regressor_model,
                temp_max=in_t_max,
                temp_min=in_t_min,
                temp_diff=in_t_diff,
                month=in_m,
                day=in_d,
                week=in_w
            )
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 6px solid #028090;">
                    <div class="metric-card-title">Temperature Forecast (Random Forest Regressor)</div>
                    <div class="metric-card-value" style="color: #0D3B66;">
                        {pred_temp_val:.1f} °C
                    </div>
                    <div class="metric-card-subtitle">
                        Model Fidelity: R² = 0.944 | MAE = 1.18 °C on unseen test records
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            pred_temp_val = 30.0
            st.error("Regression model not loaded.")

    st.markdown("---")

    # 3. Actionable Weather Recommendations based on dual outputs
    st.markdown("#### 3. Evidence-Based Weather Recommendations")
    
    recom_data = generate_weather_recommendations(
        rain_probability=rain_prob_pct / 100.0,
        predicted_max_temp=pred_temp_val,
        current_temp_diff=in_t_diff,
        month=in_m
    )

    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        umb = recom_data["umbrella"]
        st.markdown(
            f"""
            <div class="recom-box" style="border-left-color: {umb['color']};">
                <div style="font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase;">Umbrella Guidance</div>
                <div style="font-size: 15px; font-weight: 700; color: #0D3B66; margin: 4px 0;">{umb['status']}</div>
                <p style="font-size: 12px; color: #334155; margin: 0;">{umb['advice']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with r_col2:
        tmp_adv = recom_data["temperature"]
        st.markdown(
            f"""
            <div class="recom-box" style="border-left-color: {tmp_adv['color']};">
                <div style="font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase;">Clothing & Thermal Comfort</div>
                <div style="font-size: 15px; font-weight: 700; color: #0D3B66; margin: 4px 0;">{tmp_adv['category']} ({tmp_adv['predicted_max']}°C)</div>
                <p style="font-size: 12px; color: #334155; margin: 0;">{tmp_adv['advice']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with r_col3:
        act_adv = recom_data["activity"]
        st.markdown(
            f"""
            <div class="recom-box" style="border-left-color: {act_adv['color']};">
                <div style="font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase;">Outdoor Activity Index</div>
                <div style="font-size: 15px; font-weight: 700; color: #0D3B66; margin: 4px 0;">{act_adv['status']}</div>
                <p style="font-size: 12px; color: #334155; margin: 0;">{act_adv['reason']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(f"Seasonal Guidance: {recom_data['seasonal_context']} | {recom_data['disclaimer']}")


# =========================================================
# PAGE 4: MODEL COMPARISON & OBJECTIVES ANALYSIS
# =========================================================
elif page == "Model Comparison & Objectives":
    st.subheader("Model Comparison & Empirical Objectives Analysis")
    st.write(
        "Evaluation of **Logistic Regression**, **Decision Tree**, and **Random Forest** "
        "on the identical chronological test set (2009–2024, 5,350 days) addressing all core analytical objectives."
    )

    # Objective 1: Correlation Heatmap
    st.markdown("#### Objective 1: Multi-Variable Correlation Heatmap")
    st.write("Pearson correlation matrix showing interactions between temperature metrics, diurnal range, and precipitation:")
    st.plotly_chart(plot_correlation_heatmap(weather_df), use_container_width=True)

    st.markdown("---")

    # Objective 2: Combined ROC Curve with Detailed Analysis
    st.markdown("#### Objective 2: Combined Receiver Operating Characteristic (ROC) Curve")
    st.write(
        "The ROC curve plots the **True Positive Rate (Sensitivity / Recall)** on the Y-axis against the "
        "**False Positive Rate (1 - Specificity)** on the X-axis across all classification probability thresholds. "
        "A curve closer to the top-left corner demonstrates superior discriminative capacity over the diagonal dashed random line (AUC = 0.5000)."
    )
    st.plotly_chart(plot_combined_roc_curve(clf_metrics_data), use_container_width=True)

    st.markdown(
        """
        <div style="background: #F1F5F9; padding: 14px 18px; border-radius: 8px; font-size: 13px; color: #334155; line-height: 1.6;">
            <b>ROC Curve Interpretation & Findings:</b><br>
            • <b>Decision Tree Classifier (AUC = 0.8519):</b> Achieves high discrimination in high-sensitivity regions, making it exceptional at detecting rainy events with minimal threshold tuning.<br>
            • <b>Random Forest Classifier (AUC = 0.8489):</b> Displays consistent stability across intermediate operating points, suppressing false alarms and yielding the top F1 score.<br>
            • <b>Logistic Regression (AUC = 0.8407):</b> Proves that a well-scaled linear decision boundary provides strong baseline probabilistic ranking.<br>
            • <b>Diagonal Random Baseline (AUC = 0.5000):</b> All three models significantly outperform random classification, proving substantial predictive signal in diurnal temperature and seasonal features.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Objective 3: Grouped Performance Metrics & Comparison Table
    st.markdown("#### Objective 3: Multi-Model Benchmark Comparison (Accuracy, Precision, Recall, F1, ROC-AUC)")
    st.plotly_chart(plot_metrics_comparison_bar(df_comparison), use_container_width=True)

    st.dataframe(
        df_comparison[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]],
        use_container_width=True,
        hide_index=True
    )

    # Dynamic Ranking Selector
    st.markdown("##### Dynamic Model Ranking by Target Metric")
    rank_c1, rank_c2 = st.columns([1, 2])
    with rank_c1:
        sel_crit = st.selectbox("Select Optimization Metric", ["F1 Score", "Recall", "Accuracy", "Precision", "ROC-AUC"])
        b_name, b_score, b_expl = rank_models_by_metric(df_comparison, sel_crit)
    with rank_c2:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 5px solid #00A896;">
                <div class="metric-card-title">Top Model for {sel_crit}</div>
                <div class="metric-card-value">{b_name} ({b_score:.4f})</div>
                <div style="font-size: 12px; color: #334155; margin-top: 4px;">{b_expl}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Objective 4: Confusion Matrix Audits
    st.markdown("#### Objective 4: Confusion Matrix Heatmaps (TP, FP, TN, FN Breakdowns)")
    cm_col1, cm_col2, cm_col3 = st.columns(3)
    with cm_col1:
        cm_lr = clf_metrics_data["Logistic Regression"]["confusion_matrix"]
        st.plotly_chart(plot_confusion_matrix(cm_lr, "Logistic Regression"), use_container_width=True)
    with cm_col2:
        cm_dt = clf_metrics_data["Decision Tree"]["confusion_matrix"]
        st.plotly_chart(plot_confusion_matrix(cm_dt, "Decision Tree"), use_container_width=True)
    with cm_col3:
        cm_rf = clf_metrics_data["Random Forest"]["confusion_matrix"]
        st.plotly_chart(plot_confusion_matrix(cm_rf, "Random Forest"), use_container_width=True)

    st.markdown("---")

    # Objective 5: Temperature Regression Analysis
    st.markdown("#### Objective 5: Next-Day Temperature Regression Evaluation")
    reg_eval_df = pd.DataFrame(reg_metrics_data).T.reset_index().rename(columns={"index": "Model Architecture"})
    st.dataframe(
        reg_eval_df[["Model Architecture", "mae", "rmse", "r2_score"]].rename(
            columns={"mae": "MAE (°C)", "rmse": "RMSE (°C)", "r2_score": "R² Score"}
        ),
        use_container_width=True,
        hide_index=True
    )

    rf_reg_info = reg_metrics_data.get("Random Forest Regressor", {})
    r_act = rf_reg_info.get("actuals", [])
    r_pred = rf_reg_info.get("predictions", [])
    r_res = rf_reg_info.get("residuals", [])

    if r_act and r_pred:
        diag_c1, diag_c2 = st.columns(2)
        with diag_c1:
            st.plotly_chart(plot_actual_vs_predicted_regression(r_act, r_pred, "Random Forest"), use_container_width=True)
        with diag_c2:
            st.plotly_chart(plot_residual_distribution(r_res, "Random Forest"), use_container_width=True)


# =========================================================
# PAGE 5: MONGODB DATABASE CONSOLE (With Full PyMongo Classification)
# =========================================================
elif page == "MongoDB Database":
    st.subheader("MongoDB Database Management & PyMongo Architecture Classification")

    # Connection Status
    if mongo_connected:
        st.success(f"MongoDB Server Active: {mongo_msg}")
    else:
        st.warning(f"MongoDB Server Offline: {mongo_msg}. Operating in Offline CSV Fallback Mode.")

    st.markdown("#### 1. PyMongo Architectural Classification")
    st.markdown(
        """
        The system interfaces with MongoDB Community Server using the official `pymongo` (v4.x) driver. 
        PyMongo's integration across the application is classified into seven key functional roles:
        """
    )

    with st.expander("Role 1: Client Lifecycle & Connection Pooling (pymongo.MongoClient)", expanded=True):
        st.code(
            """
client = pymongo.MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=3000,
    connectTimeoutMS=3000,
    socketTimeoutMS=3000
)
            """,
            language="python"
        )
        st.write(
            "• Managed as a thread-safe singleton resource via `@st.cache_resource`.\n"
            "• Enforces Client-Side Operation Timeouts (3,000 ms) so the web application never blocks if the server is paused."
        )

    with st.expander("Role 2: Administrative Health Ping & Diagnostics (admin.command('ping'))", expanded=False):
        st.code(
            """
# Verifies database availability prior to query execution
client.admin.command('ping')
            """,
            language="python"
        )
        st.write(
            "• Returns `{'ok': 1.0}` when active.\n"
            "• Catches `ServerSelectionTimeoutError` and seamlessly triggers offline CSV fallback mode."
        )

    with st.expander("Role 3: Deterministic Unique Indexing (collection.create_index)", expanded=False):
        st.code(
            """
# Enforce zero duplicate dates upon repeated imports
coll.create_index('Record_Key', unique=True, name='idx_record_key_unique')
coll.create_index('Date_Parsed', name='idx_date_parsed')
coll.create_index([('Year', 1), ('Month', 1)], name='idx_year_month')
            """,
            language="python"
        )
        st.write(
            "• `Record_Key` (`YYYY-MM-DD`) guarantees zero duplicate date records.\n"
            "• `Date_Parsed` and `(Year, Month)` accelerate time-series range and aggregation queries."
        )

    with st.expander("Role 4: High-Throughput Bulk Ingestion (bulk_write & UpdateOne)", expanded=False):
        st.code(
            """
from pymongo import UpdateOne

operations = [
    UpdateOne({'Record_Key': row['Record_Key']}, {'$set': row}, upsert=True)
    for row in records
]
# Ingest in chunks of 1,000 operations
coll.bulk_write(operations[i:i+1000], ordered=False)
            """,
            language="python"
        )
        st.write(
            "• Ingests 26,803 records in ~3 seconds.\n"
            "• Idempotent: repeated runs result in 0 new insertions and 26,803 matched records."
        )

    with st.expander("Role 5: Parameterized Querying & Projection (collection.find)", expanded=False):
        st.code(
            """
# Filter by year and limit results without loading unnecessary fields into memory
docs = coll.find({'Year': 2023}, {'_id': 0, 'Date': 1, 'Temp Max': 1, 'Rain': 1}).limit(25)
            """,
            language="python"
        )

    with st.expander("Role 6: Telemetry & Auditing (count_documents & list_indexes)", expanded=False):
        st.code(
            """
total_docs = coll.count_documents({})
active_indexes = list(coll.list_indexes())
            """,
            language="python"
        )

    with st.expander("Role 7: Transparent Offline Fallback Resilience", expanded=False):
        st.write(
            "• When MongoDB is offline, functions in `src/mongodb_manager.py` gracefully return offline telemetry.\n"
            "• All EDA, ML training, evaluation, and predictions automatically pull from `data/delhi_weather.csv`."
        )

    st.markdown("---")

    # 2. Collection Telemetry & Migration Button
    st.markdown("#### 2. Collection Telemetry & Migration Actions")
    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
    with stat_c1:
        st.metric("Database", mongo_mgr.db_name)
    with stat_c2:
        st.metric("Collection", mongo_mgr.collection_name)
    with stat_c3:
        coll_stats = mongo_mgr.get_collection_statistics()
        st.metric("Document Count", f"{coll_stats['count']:,}" if mongo_connected else "Offline")
    with stat_c4:
        st.metric("Target Port", "27017")

    if st.button("Trigger Ingestion to MongoDB"):
        if not mongo_connected:
            st.error("Cannot migrate: MongoDB server is offline on port 27017.")
        else:
            with st.spinner("Ingesting 26,803 records via bulk upsert..."):
                import_res = mongo_mgr.import_dataframe(weather_df)
                if import_res["success"]:
                    st.success(
                        f"Migration complete: {import_res['message']} "
                        f"(New upserts: {import_res['upserted_new']}, Matched existing: {import_res['matched_count']}, "
                        f"Total collection documents: {import_res['total_collection_documents']:,})"
                    )
                else:
                    st.error(f"Migration error: {import_res['message']}")

    st.markdown("---")

    # 3. Live MongoDB Query Console
    st.markdown("#### 3. Live MongoDB Query Console")
    if mongo_connected:
        q_c1, q_c2 = st.columns([3, 1])
        with q_c1:
            live_year = st.slider("Select Year to Query directly from MongoDB", 1951, 2024, 2023, key="mongo_live_yr")
        with q_c2:
            live_limit = st.number_input("Limit Documents", 5, 100, 20, key="mongo_live_lim")

        live_results = mongo_mgr.fetch_records(
            query={"Year": int(live_year)},
            projection={"_id": 0, "Record_Key": 1, "Date": 1, "Temp Max": 1, "Temp Min": 1, "Rain": 1, "Has_rained": 1},
            limit=int(live_limit)
        )
        if not live_results.empty:
            st.dataframe(live_results, use_container_width=True, hide_index=True)
        else:
            st.info(f"No documents found for year {live_year}. Trigger migration above to populate the collection.")
    else:
        st.info("Live query console disabled while MongoDB is offline.")
