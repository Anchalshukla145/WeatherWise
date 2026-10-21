"""
Visualization Module for WeatherWise Analytics.
Generates interactive Plotly figures and static figures conforming to the
teal, navy, and slate business design system.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Design Palette Tokens
COLOR_PRIMARY_NAVY = "#0D3B66"
COLOR_ACCENT_TEAL = "#00A896"
COLOR_ACCENT_BLUE = "#028090"
COLOR_ORANGE = "#F4A261"
COLOR_CORAL = "#E76F51"
COLOR_BG_CARD = "#FFFFFF"
COLOR_GRID = "#E2E8F0"

MODEL_COLORS = {
    "Logistic Regression": "#028090",
    "Decision Tree": "#F4A261",
    "Random Forest": "#00A896",
    "Linear Regression": "#028090",
    "Decision Tree Regressor": "#F4A261",
    "Random Forest Regressor": "#00A896",
}


def plot_combined_roc_curve(clf_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Renders the combined ROC curve for all evaluated classification models on a single chart.
    """
    fig = go.Figure()

    # Diagonal random baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#94A3B8", dash="dash", width=1.5),
            name="Random Guess (AUC = 0.5000)",
            hoverinfo="skip"
        )
    )

    for name, res in clf_results.items():
        roc_data = res.get("roc_curve", {})
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])
        auc = res.get("roc_auc")
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        color = MODEL_COLORS.get(name, "#3B82F6")

        if fpr and tpr:
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC = {auc_str})",
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>"
                )
            )

    fig.update_layout(
        title="<b>Receiver Operating Characteristic (ROC) Comparison</b>",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        xaxis=dict(range=[-0.01, 1.01], gridcolor=COLOR_GRID, zeroline=False),
        yaxis=dict(range=[-0.01, 1.01], gridcolor=COLOR_GRID, zeroline=False),
        legend=dict(x=0.55, y=0.08, bgcolor="rgba(255,255,255,0.85)", bordercolor=COLOR_GRID, borderwidth=1),
        plot_bgcolor=COLOR_BG_CARD,
        paper_bgcolor=COLOR_BG_CARD,
        margin=dict(l=50, r=30, t=50, b=50),
        height=500
    )
    return fig


def plot_metrics_comparison_bar(df_comp: pd.DataFrame) -> go.Figure:
    """
    Renders grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fig = go.Figure()

    for metric in metrics:
        if metric in df_comp.columns:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=df_comp["Model"],
                    y=df_comp[metric],
                    text=[f"{v:.3f}" if pd.notna(v) else "N/A" for v in df_comp[metric]],
                    textposition="auto"
                )
            )

    fig.update_layout(
        title="<b>Classification Performance Metrics by Model</b>",
        barmode="group",
        yaxis_title="Score (0.0 to 1.0)",
        yaxis=dict(range=[0, 1.05], gridcolor=COLOR_GRID),
        xaxis=dict(gridcolor=COLOR_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor=COLOR_BG_CARD,
        paper_bgcolor=COLOR_BG_CARD,
        margin=dict(l=50, r=30, t=60, b=40),
        height=450
    )
    return fig


def plot_confusion_matrix(cm_dict: Dict[str, Any], model_name: str) -> go.Figure:
    """
    Plots an annotated confusion matrix heatmap for a given model.
    """
    tn = cm_dict["true_negative"]
    fp = cm_dict["false_positive"]
    fn = cm_dict["false_negative"]
    tp = cm_dict["true_positive"]

    z = [[tn, fp], [fn, tp]]
    x_labels = ["Predicted: No Rain (0)", "Predicted: Rain (1)"]
    y_labels = ["Actual: No Rain (0)", "Actual: Rain (1)"]

    annotations = []
    text_labels = [
        [f"True Negative (TN)<br><b>{tn:,}</b>", f"False Positive (FP)<br><b>{fp:,}</b>"],
        [f"False Negative (FN)<br><b>{fn:,}</b>", f"True Positive (TP)<br><b>{tp:,}</b>"]
    ]

    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=text_labels[i][j],
                    font=dict(color="white" if (i == j and z[i][j] > (tn + tp) * 0.2) else "#0F172A", size=13),
                    showarrow=False
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, "#E2E8F0"], [0.5, "#38BDF8"], [1, "#0D3B66"]],
        showscale=False
    ))

    fig.update_layout(
        title=f"<b>Confusion Matrix — {model_name}</b>",
        annotations=annotations,
        yaxis=dict(autorange="reversed"),
        plot_bgcolor=COLOR_BG_CARD,
        paper_bgcolor=COLOR_BG_CARD,
        margin=dict(l=60, r=40, t=50, b=40),
        height=380
    )
    return fig


def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plots rain vs no-rain distribution.
    """
    if "Has_rained_Binary" in df.columns:
        counts = df["Has_rained_Binary"].value_counts().sort_index()
        labels = ["No Rain (0)", "Rain (1)"]
        values = [counts.get(0, 0), counts.get(1, 0)]
    else:
        labels, values = ["No Rain", "Rain"], [1, 1]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=["#028090", "#00A896"]),
        textinfo="label+percent+value",
        insidetextorientation="horizontal"
    )])

    fig.update_layout(
        title="<b>Rain vs No-Rain Historical Class Balance</b>",
        plot_bgcolor=COLOR_BG_CARD,
        paper_bgcolor=COLOR_BG_CARD,
        margin=dict(l=30, r=30, t=50, b=30),
        height=380
    )
    return fig


def plot_monthly_rainfall_frequency(df: pd.DataFrame) -> go.Figure:
    """
    Plots monthly count of rainy days showing monsoon seasonality.
    """
    monthly = df[df["Has_rained_Binary"] == 1].groupby("month_name")["Has_rained_Binary"].count()
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    monthly = monthly.reindex(month_order).fillna(0)

    fig = go.Figure(data=[go.Bar(
        x=month_order,
        y=monthly.values,
        marker=dict(
            color=monthly.values,
            colorscale=[[0, "#E0F2FE"], [0.5, "#028090"], [1, "#0D3B66"]],
            showscale=False
        ),
        text=[f"{int(v):,}" for v in monthly.values],
        textposition="auto"
    )])

    fig.update_layout(
        title="<b>Total Historical Rainy Days by Month (Monsoon Seasonality)</b>",
        xaxis_title="Month",
        yaxis_title="Total Rainy Days Recorded",
        yaxis=dict(gridcolor=COLOR_GRID),
        plot_bgcolor=COLOR_BG_CARD,
        paper_bgcolor=COLOR_BG_CARD,
        margin=dict(l=40, r=20, t=50, b=40),
        height=400
    )
    return fig


def plot_temperature_trends(df: pd.DataFrame, sample_step: int = 10) -> go.Figure:
    """
    Plots historical time-series of maximum and minimum temperatures.
    """
    sampled = df.iloc[::sample_step].copy()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sampled["Date_Parsed"],
        y=sampled["Temp Max"],
        mode="lines",
        name="Temp Max (°C)",
        line=dict(color="#E76F51", width=1.2),
        opacity=0.85
    ))

    fig.add_trace(go.Scatter(
        x=sampled["Date_Parsed"],
        y=sampled["Temp Min"],
        mode="lines",
        name="Temp Min (°C)",
        line=dict(color="#028090", width=1.2),
        opacity=0.85
    ))

    fig.update_layout(
        title="<b>Delhi Historical Daily Temperature Records (1951 - 2024)</b>",
        xaxis_title="Observation Date",
        yaxis_title="Temperature (°C)",
        xaxis=dict(gridcolor=COLOR_GRID),
        yaxis=dict(gridcolor=COLOR_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor=COLOR_BG_CARD,
        paper_bgcolor=COLOR_BG_CARD,
        margin=dict(l=50, r=30, t=60, b=40),
        height=420
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Plots correlation matrix of valid numeric weather features.
    """
    num_cols = ["Temp Max", "Temp Min", "Temp_diff", "Rain_Numeric", "Month", "Week"]
    valid_cols = [c for c in num_cols if c in df.columns]
    corr = df[valid_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=valid_cols,
        y=valid_cols,
        colorscale=[[0, "#028090"], [0.5, "#F8F9FA"], [1, "#E76F51"]],
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11}
    ))

    fig.update_layout(
        title="<b>Correlation Matrix (Numeric Weather Variables)</b>",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor=COLOR_BG_CARD,
        paper_bgcolor=COLOR_BG_CARD,
        margin=dict(l=60, r=40, t=50, b=40),
        height=420
    )
    return fig


def plot_outlier_boxplots(df: pd.DataFrame) -> go.Figure:
    """
    Renders boxplots for numeric temperature features showing IQR distribution.
    """
    fig = go.Figure()
    for col, color in [("Temp Max", "#E76F51"), ("Temp Min", "#028090"), ("Temp_diff", "#00A896")]:
        if col in df.columns:
            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                marker_color=color,
                boxpoints="outliers"
            ))

    fig.update_layout(
        title="<b>Temperature Distributions and Outlier Inspection (IQR Analysis)</b>",
        yaxis_title="Temperature (°C)",
        yaxis=dict(gridcolor=COLOR_GRID),
        plot_bgcolor=COLOR_BG_CARD,
        paper_bgcolor=COLOR_BG_CARD,
        margin=dict(l=50, r=30, t=50, b=40),
        height=400
    )
    return fig


def plot_actual_vs_predicted_regression(actuals: List[float], preds: List[float], model_name: str) -> go.Figure:
    """
    Renders actual vs predicted scatter plot for temperature regression.
    """
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=actuals,
        y=preds,
        mode="markers",
        name="Observations",
        marker=dict(color="#00A896", size=5, opacity=0.6)
    ))

    # Perfect prediction 45-degree line
    min_v = min(min(actuals), min(preds))
    max_v = max(max(actuals), max(preds))
    fig.add_trace(go.Scatter(
        x=[min_v, max_v],
        y=[min_v, max_v],
        mode="lines",
        name="Ideal 1:1 Line",
        line=dict(color="#E76F51", dash="dash", width=2)
    ))

    fig.update_layout(
        title=f"<b>Actual vs Predicted Next-Day Maximum Temperature — {model_name}</b>",
        xaxis_title="Actual Next-Day Temp Max (°C)",
        yaxis_title="Predicted Next-Day Temp Max (°C)",
        xaxis=dict(gridcolor=COLOR_GRID),
        yaxis=dict(gridcolor=COLOR_GRID),
        plot_bgcolor=COLOR_BG_CARD,
        paper_bgcolor=COLOR_BG_CARD,
        margin=dict(l=50, r=30, t=50, b=40),
        height=450
    )
    return fig


def plot_residual_distribution(residuals: List[float], model_name: str) -> go.Figure:
    """
    Renders histogram of temperature prediction residuals.
    """
    fig = go.Figure(data=[go.Histogram(
        x=residuals,
        nbinsx=40,
        marker=dict(color="#028090", line=dict(color="#FFFFFF", width=1))
    )])

    fig.update_layout(
        title=f"<b>Residual Error Distribution (Actual - Predicted) — {model_name}</b>",
        xaxis_title="Residual Error (°C)",
        yaxis_title="Frequency",
        xaxis=dict(gridcolor=COLOR_GRID),
        yaxis=dict(gridcolor=COLOR_GRID),
        plot_bgcolor=COLOR_BG_CARD,
        paper_bgcolor=COLOR_BG_CARD,
        margin=dict(l=50, r=30, t=50, b=40),
        height=400
    )
    return fig
