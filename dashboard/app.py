"""
Superstore Sales Analytics & Forecasting Platform
Production-grade Streamlit dashboard: KPIs, 3-D Plotly visuals,
ensemble forecast, filterable product metrics, CSV export.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Analytics & Forecasting Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/superstore-mlops-forecast",
        "Report a bug": "https://github.com/your-org/superstore-mlops-forecast/issues",
        "About": "Superstore Sales Analytics & Forecasting Platform v1.0",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS — professional navy / white / blue palette, no emojis
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* ---- Force light-mode palette regardless of OS / browser theme ---- */
    html, body {
        color: #1F2937 !important;
        background-color: #EEF2F7 !important;
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
    }

    /* Streamlit root containers */
    .stApp,
    .stApp > div,
    section[data-testid="stMain"],
    section[data-testid="stMain"] > div,
    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"] {
        color: #1F2937 !important;
        background-color: #EEF2F7 !important;
    }

    /* Generic text — only target plain text nodes, not headings (headings
       are handled per-component below so custom colours aren't clobbered) */
    .stApp p,
    .stApp span,
    .stApp label,
    .stMarkdown p,
    .stMarkdown span {
        color: #1F2937 !important;
    }

    /* Streamlit metric / markdown overrides */
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        color: #1F2937 !important;
    }

    /* Keep sidebar text white — sidebar has its own dark background */
    [data-testid="stSidebar"] *:not(button):not(input):not(select) {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stMultiSelect span,
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] span {
        color: #1F2937 !important;
    }

    .stApp {
        background-color: #EEF2F7;
    }

    /* ---- Hide default Streamlit chrome ---- */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }

    /* ---- Page header banner ---- */
    .page-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #2563EB 100%) !important;
        border-radius: 10px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.2rem;
        color: #ffffff !important;
    }
    .page-header h1 {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.3rem;
        color: #ffffff !important;
    }
    .page-header p {
        font-size: 0.85rem;
        color: #93C5FD !important;
        margin: 0;
    }

    /* ---- KPI cards ---- */
    .kpi-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
        height: 100%;
    }
    .kpi-card.green  { border-left-color: #059669; }
    .kpi-card.amber  { border-left-color: #D97706; }
    .kpi-card.indigo { border-left-color: #4F46E5; }
    .kpi-card.slate  { border-left-color: #475569; }

    .kpi-label {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6B7280;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.70rem;
        font-weight: 800;
        color: #1B3A6B;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #6B7280;
        margin-top: 0.3rem;
    }
    .kpi-delta-pos { color: #059669; font-size: 0.78rem; font-weight: 600; }
    .kpi-delta-neg { color: #DC2626; font-size: 0.78rem; font-weight: 600; }

    /* ---- Section title ---- */
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1B3A6B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #DBEAFE;
        margin: 1.5rem 0 0.8rem;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #1B3A6B;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] hr { border-color: #2D5A9E; }

    /* ---- Alert / info boxes ---- */
    .info-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #1E40AF;
        margin: 0.5rem 0;
    }

    /* ---- Footer ---- */
    .page-footer {
        background: #1B3A6B !important;
        color: #93C5FD !important;
        text-align: center;
        padding: 0.9rem 1rem;
        border-radius: 8px;
        font-size: 0.78rem;
        margin-top: 2rem;
        letter-spacing: 0.01em;
    }
    .page-footer * { color: #93C5FD !important; }

    /* ---- Plotly chart containers ---- */
    .element-container { margin-bottom: 0.5rem; }

    /* ---- Dataframe overrides ---- */
    .stDataFrame { border-radius: 6px; overflow: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    path = DATA_DIR / "clean.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    path = DATA_DIR / "processed.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


@st.cache_data(show_spinner=False)
def load_forecast_data() -> pd.DataFrame:
    path = DATA_DIR / "forecast.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    path = MODELS_DIR / "metadata.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def fmt_usd(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:.0f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def kpi_card(label: str, value: str, sub: str = "", delta: str = "", variant: str = "") -> str:
    delta_cls = ""
    if delta:
        delta_cls = "kpi-delta-pos" if not delta.startswith("-") else "kpi-delta-neg"
    delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card {variant}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
        {delta_html}
    </div>
    """


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Segoe UI, Helvetica Neue, Arial", color="#1F2937", size=11),
    margin=dict(l=40, r=20, t=50, b=40),
)

BLUE_SCALE = "Blues"
CORP_COLORS = ["#1B3A6B", "#2563EB", "#60A5FA", "#3B82F6", "#93C5FD", "#DBEAFE"]


def chart_3d_surface_region_subcat(df: pd.DataFrame) -> go.Figure:
    """3D Surface: Sales by Sub-Category vs Region."""
    pivot = (
        df.groupby(["sub_category", "region"])["sales"]
        .sum()
        .unstack(fill_value=0)
    )
    z = pivot.values.tolist()
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=BLUE_SCALE,
                showscale=True,
                colorbar=dict(title="Sales ($)", thickness=12, len=0.7),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="#2563EB", project_z=True)
                ),
            )
        ]
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Sales Distribution — Sub-Category vs Region", font=dict(size=13, color="#1B3A6B")),
        height=480,
        scene=dict(
            xaxis=dict(title="Region", backgroundcolor="#EEF2F7", gridcolor="#E2E8F0"),
            yaxis=dict(title="Sub-Category", backgroundcolor="#EEF2F7", gridcolor="#E2E8F0"),
            zaxis=dict(title="Total Sales ($)", backgroundcolor="#EEF2F7", gridcolor="#E2E8F0"),
            camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0)),
        ),
    )
    return fig


def chart_3d_surface_time(daily_df: pd.DataFrame) -> go.Figure:
    """3D Surface: Average Daily Sales by Month vs Day-of-Week."""
    if daily_df.empty:
        return go.Figure()
    tmp = daily_df.copy()
    tmp["month"] = tmp["order_date"].dt.month
    tmp["day_of_week"] = tmp["order_date"].dt.dayofweek
    pivot = tmp.pivot_table(
        values="sales", index="month", columns="day_of_week",
        aggfunc="mean", fill_value=0,
    )
    # Ensure all 7 DoW columns exist
    for col in range(7):
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot.sort_index(axis=1)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    y_labels = [month_labels[i - 1] for i in pivot.index]

    fig = go.Figure(
        data=[
            go.Surface(
                z=pivot.values.tolist(),
                x=dow_labels,
                y=y_labels,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Avg Sales ($)", thickness=12, len=0.7),
                opacity=0.92,
            )
        ]
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Avg Daily Sales — Month vs Day-of-Week", font=dict(size=13, color="#1B3A6B")),
        height=480,
        scene=dict(
            xaxis=dict(title="Day of Week", backgroundcolor="#EEF2F7"),
            yaxis=dict(title="Month", backgroundcolor="#EEF2F7"),
            zaxis=dict(title="Avg Sales ($)", backgroundcolor="#EEF2F7"),
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9)),
        ),
    )
    return fig


def chart_sunburst(df: pd.DataFrame) -> go.Figure:
    """Sunburst: Revenue by Region > Category > Sub-Category."""
    sunburst_df = (
        df.groupby(["region", "category", "sub_category"])["sales"]
        .sum()
        .reset_index()
    )
    fig = px.sunburst(
        sunburst_df,
        path=["region", "category", "sub_category"],
        values="sales",
        color="sales",
        color_continuous_scale=BLUE_SCALE,
        title="Revenue Hierarchy — Region / Category / Sub-Category",
    )
    fig.update_traces(
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>Sales: $%{value:,.0f}<extra></extra>",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=480,
        title_font=dict(size=13, color="#1B3A6B"),
        coloraxis_showscale=False,
    )
    return fig


def chart_bubble(df: pd.DataFrame) -> go.Figure:
    """Scatter: Sales vs Profit, bubble size = Quantity, colour = Category."""
    bubble_df = (
        df.groupby(["product_name", "category"])
        .agg(sales=("sales", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum"))
        .reset_index()
    )
    fig = px.scatter(
        bubble_df,
        x="sales",
        y="profit",
        size="quantity",
        color="category",
        hover_name="product_name",
        color_discrete_sequence=CORP_COLORS,
        labels={
            "sales": "Total Sales ($)",
            "profit": "Total Profit ($)",
            "quantity": "Units Sold",
            "category": "Category",
        },
        title="Sales vs Profit — Bubble Size: Units Sold",
        size_max=50,
    )
    fig.update_traces(
        marker=dict(opacity=0.75, line=dict(width=0.5, color="#FFFFFF")),
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Sales: $%{x:,.0f}<br>Profit: $%{y:,.0f}<extra></extra>"
        ),
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=480,
        title_font=dict(size=13, color="#1B3A6B"),
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    return fig


def chart_forecast(daily_df: pd.DataFrame, forecast_df: pd.DataFrame, lookback_days: int = 90) -> go.Figure:
    """Line chart: historical sales + forecast with confidence band."""
    fig = go.Figure()

    if not daily_df.empty:
        cutoff = daily_df["order_date"].max() - pd.Timedelta(days=lookback_days)
        hist = daily_df[daily_df["order_date"] >= cutoff]
        # 7-day rolling average for clarity
        hist_smooth = hist.copy()
        hist_smooth["sales_smooth"] = hist_smooth["sales"].rolling(7, min_periods=1).mean()

        fig.add_trace(
            go.Scatter(
                x=hist["order_date"],
                y=hist["sales"],
                name="Actual Sales (Daily)",
                line=dict(color="#94A3B8", width=1),
                mode="lines",
                opacity=0.6,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist_smooth["order_date"],
                y=hist_smooth["sales_smooth"],
                name="7-Day Rolling Average",
                line=dict(color="#1B3A6B", width=2),
                mode="lines",
            )
        )

    if not forecast_df.empty:
        # Confidence band (filled area)
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df["date"], forecast_df["date"].iloc[::-1]]),
                y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(37,99,235,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% Confidence Interval",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["forecast_sales"],
                name="30-Day Forecast",
                line=dict(color="#2563EB", width=2.5, dash="dot"),
                mode="lines+markers",
                marker=dict(size=5, color="#2563EB"),
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Actual vs Forecast — 30-Day Horizon",
            font=dict(size=13, color="#1B3A6B"),
        ),
        height=380,
        xaxis=dict(title="Date", gridcolor="#E5E7EB", showgrid=True),
        yaxis=dict(title="Sales ($)", gridcolor="#E5E7EB", showgrid=True),
        legend=dict(orientation="h", y=-0.2, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig


def chart_sales_trend(daily_df: pd.DataFrame) -> go.Figure:
    """Monthly revenue trend bar chart."""
    monthly = (
        daily_df.copy()
        .assign(month=daily_df["order_date"].dt.to_period("M"))
        .groupby("month")["sales"]
        .sum()
        .reset_index()
    )
    monthly["month_str"] = monthly["month"].dt.strftime("%b %Y")
    monthly["ma3"] = monthly["sales"].rolling(3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly["month_str"],
            y=monthly["sales"],
            name="Monthly Revenue",
            marker=dict(
                color=monthly["sales"],
                colorscale=BLUE_SCALE,
                showscale=False,
            ),
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month_str"],
            y=monthly["ma3"],
            name="3-Month Moving Avg",
            line=dict(color="#DC2626", width=2),
            mode="lines",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Monthly Revenue Trend", font=dict(size=13, color="#1B3A6B")),
        height=320,
        xaxis=dict(tickangle=-45, gridcolor="#E5E7EB"),
        yaxis=dict(title="Sales ($)", gridcolor="#E5E7EB"),
        legend=dict(orientation="h", y=-0.3, x=0),
        barmode="overlay",
    )
    return fig


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    # --- Load data ---
    with st.spinner("Loading data…"):
        clean_df = load_clean_data()
        daily_df = load_processed_data()
        forecast_df = load_forecast_data()
        metadata = load_metadata()

    data_missing = clean_df.empty

    # --- Page header ---
    st.markdown(
        """
        <div class="page-header">
            <h1 style="color:#ffffff!important;font-size:1.65rem;font-weight:700;letter-spacing:-0.02em;margin:0 0 0.3rem 0;">Superstore Sales Analytics &amp; Forecasting Platform</h1>
            <p style="color:#93C5FD!important;font-size:0.85rem;margin:0;">XGBoost + Prophet Ensemble | MLflow Tracked | Walk-Forward Cross-Validation | 30-Day Forecast</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data_missing:
        st.error(
            "No data found in `data/`. "
            "Download `SampleSuperstore.csv` from Kaggle and run `make prepare` to generate datasets."
        )
        st.code(
            "# Step 1 — Download dataset from:\n"
            "# https://www.kaggle.com/datasets/vivek468/superstore-dataset-final\n"
            "# Place SampleSuperstore.csv in the data/ directory, then:\n\n"
            "make prepare    # ETL pipeline\n"
            "make train      # Model training (XGBoost + Prophet + Optuna)\n"
            "make dashboard  # Launch this dashboard"
        )
        return

    # -----------------------------------------------------------------------
    # Sidebar — filters
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#93C5FD;font-size:1rem;font-weight:700;"
            "letter-spacing:0.05em;text-transform:uppercase;'>Filter Parameters</h2>",
            unsafe_allow_html=True,
        )

        all_regions = sorted(clean_df["region"].dropna().unique())
        sel_regions = st.multiselect("Region", all_regions, default=all_regions)

        all_cats = sorted(clean_df["category"].dropna().unique())
        sel_cats = st.multiselect("Category", all_cats, default=all_cats)

        avail_subcats = sorted(
            clean_df[clean_df["category"].isin(sel_cats)]["sub_category"].dropna().unique()
        )
        sel_subcats = st.multiselect("Sub-Category", avail_subcats, default=avail_subcats)

        st.markdown("<hr>", unsafe_allow_html=True)

        min_date = clean_df["order_date"].min().date()
        max_date = clean_df["order_date"].max().date()
        date_range = st.slider(
            "Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        if metadata:
            st.markdown(
                f"<p style='color:#93C5FD;font-size:0.75rem;'>"
                f"Validation MAPE: {metadata.get('ensemble_val_mape', 0):.2f}%<br>"
                f"Model trained through: {metadata.get('data_end_date', 'N/A')}"
                f"</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='info-box' style='background:#1E3A6B;border-color:#2D5A9E;"
                "color:#93C5FD;'>No model artefacts found.<br>"
                "Run <code>make train</code> to enable forecasting.</div>",
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # Apply filters
    # -----------------------------------------------------------------------
    if not sel_regions:
        sel_regions = all_regions
    if not sel_cats:
        sel_cats = all_cats
    if not sel_subcats:
        sel_subcats = avail_subcats

    filtered = clean_df[
        clean_df["region"].isin(sel_regions)
        & clean_df["category"].isin(sel_cats)
        & clean_df["sub_category"].isin(sel_subcats)
        & (clean_df["order_date"].dt.date >= date_range[0])
        & (clean_df["order_date"].dt.date <= date_range[1])
    ]

    if filtered.empty:
        st.warning("No records match the selected filters. Adjust the sidebar criteria.")
        return

    # -----------------------------------------------------------------------
    # KPI cards
    # -----------------------------------------------------------------------
    section_title("Key Performance Indicators")

    total_sales = filtered["sales"].sum()
    total_profit = filtered["profit"].sum()
    profit_margin = (total_profit / total_sales * 100) if total_sales else 0
    total_orders = filtered["order_id"].nunique()
    aov = total_sales / total_orders if total_orders else 0
    top_cat = (
        filtered.groupby("category")["sales"].sum().idxmax()
        if not filtered.empty
        else "N/A"
    )
    top_cat_sales = (
        filtered.groupby("category")["sales"].sum().max()
        if not filtered.empty
        else 0
    )
    # Prefer the business-day (non-zero) WMAPE — more meaningful for retail
    if metadata:
        nz = metadata.get("ensemble_val_mape_nonzero", metadata.get("ensemble_val_mape", 0))
        forecast_mape = f"{nz:.2f}% (biz-days)"
        cv_mape = f"{metadata['cv_mape_mean']:.2f}% (CV-WMAPE)"
    else:
        forecast_mape = "N/A"
        cv_mape = "N/A"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            kpi_card("Total Revenue", fmt_usd(total_sales), sub=f"{total_orders:,} orders"),
            unsafe_allow_html=True,
        )
    with c2:
        variant = "green" if profit_margin >= 0 else "amber"
        st.markdown(
            kpi_card("Profit Margin", fmt_pct(profit_margin), sub=fmt_usd(total_profit), variant=variant),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            kpi_card("Avg Order Value", fmt_usd(aov), sub="per unique order", variant="indigo"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            kpi_card("Top Category", top_cat, sub=fmt_usd(top_cat_sales), variant="slate"),
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            kpi_card("Forecast MAPE", forecast_mape, sub="validation set"),
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            kpi_card("CV Accuracy", cv_mape, sub="walk-forward (5-fold)", variant="green"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Monthly revenue trend (full width)
    # -----------------------------------------------------------------------
    section_title("Monthly Revenue Trend")
    filtered_daily = daily_df[
        (daily_df["order_date"].dt.date >= date_range[0])
        & (daily_df["order_date"].dt.date <= date_range[1])
    ]
    if not filtered_daily.empty:
        st.plotly_chart(
            chart_sales_trend(filtered_daily),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # -----------------------------------------------------------------------
    # Top 20 Products table
    # -----------------------------------------------------------------------
    section_title("Top 20 Products by Revenue")

    top_products = (
        filtered.groupby("product_name")
        .agg(
            Sales=("sales", "sum"),
            Profit=("profit", "sum"),
            Orders=("order_id", "nunique"),
            Units=("quantity", "sum"),
        )
        .assign(
            **{"Margin (%)": lambda x: (x["Profit"] / x["Sales"] * 100).round(2)}
        )
        .sort_values("Sales", ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"product_name": "Product"})
    )
    top_products["Sales"] = top_products["Sales"].round(2)
    top_products["Profit"] = top_products["Profit"].round(2)

    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.dataframe(
            top_products.style.format(
                {
                    "Sales": "${:,.2f}",
                    "Profit": "${:,.2f}",
                    "Margin (%)": "{:.2f}%",
                    "Orders": "{:,}",
                    "Units": "{:,}",
                }
            )
            .background_gradient(subset=["Sales"], cmap="Blues")
            .background_gradient(subset=["Margin (%)"], cmap="RdYlGn"),
            use_container_width=True,
            height=420,
        )
    with col_right:
        cat_breakdown = (
            filtered.groupby("category")["sales"]
            .sum()
            .reset_index()
            .rename(columns={"category": "Category", "sales": "Sales"})
        )
        fig_pie = px.pie(
            cat_breakdown,
            names="Category",
            values="Sales",
            color_discrete_sequence=CORP_COLORS,
            title="Revenue by Category",
            hole=0.45,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            height=420,
            showlegend=False,
            title_font=dict(size=12, color="#1B3A6B"),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # CSV export
    csv_data = top_products.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Top Products — CSV",
        data=csv_data,
        file_name="top_products.csv",
        mime="text/csv",
    )

    # -----------------------------------------------------------------------
    # 3-D and analytical charts (2 x 2)
    # -----------------------------------------------------------------------
    section_title("Sales Analytics — Multi-Dimensional Views")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(
            chart_3d_surface_region_subcat(filtered),
            use_container_width=True,
            config={"displayModeBar": True},
        )
    with r1c2:
        st.plotly_chart(
            chart_3d_surface_time(filtered_daily if not filtered_daily.empty else daily_df),
            use_container_width=True,
            config={"displayModeBar": True},
        )

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(
            chart_sunburst(filtered),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with r2c2:
        st.plotly_chart(
            chart_bubble(filtered),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # -----------------------------------------------------------------------
    # Forecast section
    # -----------------------------------------------------------------------
    section_title("30-Day Sales Forecast")

    if forecast_df.empty:
        st.markdown(
            "<div class='info-box'>Forecast artefacts not found. "
            "Run <code>make train</code> to generate the 30-day forecast.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(
            chart_forecast(filtered_daily if not filtered_daily.empty else daily_df, forecast_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown("**30-Day Prediction Table**")
        disp_fc = forecast_df.copy()
        disp_fc.columns = ["Date", "Forecast Sales ($)", "Lower Bound ($)", "Upper Bound ($)"]
        disp_fc["Date"] = disp_fc["Date"].dt.strftime("%Y-%m-%d")
        disp_fc["Forecast Sales ($)"] = disp_fc["Forecast Sales ($)"].round(2)
        disp_fc["Lower Bound ($)"] = disp_fc["Lower Bound ($)"].round(2)
        disp_fc["Upper Bound ($)"] = disp_fc["Upper Bound ($)"].round(2)

        fc_col1, fc_col2 = st.columns([2, 1])
        with fc_col1:
            st.dataframe(
                disp_fc.style.format(
                    {
                        "Forecast Sales ($)": "${:,.2f}",
                        "Lower Bound ($)": "${:,.2f}",
                        "Upper Bound ($)": "${:,.2f}",
                    }
                ).background_gradient(subset=["Forecast Sales ($)"], cmap="Blues"),
                use_container_width=True,
                height=320,
            )
        with fc_col2:
            fc_kpis = {
                "Total Forecast Revenue": fmt_usd(disp_fc["Forecast Sales ($)"].sum()),
                "Avg Daily Forecast": fmt_usd(disp_fc["Forecast Sales ($)"].mean()),
                "Peak Day": disp_fc.loc[disp_fc["Forecast Sales ($)"].idxmax(), "Date"],
                "Peak Sales": fmt_usd(disp_fc["Forecast Sales ($)"].max()),
            }
            for lbl, val in fc_kpis.items():
                st.markdown(kpi_card(lbl, val), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

        csv_fc = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Forecast — CSV",
            data=csv_fc,
            file_name="forecast_30day.csv",
            mime="text/csv",
        )

    # -----------------------------------------------------------------------
    # Segment performance (bonus)
    # -----------------------------------------------------------------------
    section_title("Performance by Segment")

    seg_col1, seg_col2 = st.columns(2)
    with seg_col1:
        region_perf = (
            filtered.groupby("region")
            .agg(Sales=("sales", "sum"), Profit=("profit", "sum"), Orders=("order_id", "nunique"))
            .assign(**{"Margin (%)": lambda x: (x["Profit"] / x["Sales"] * 100).round(1)})
            .sort_values("Sales", ascending=False)
            .reset_index()
        )
        fig_reg = px.bar(
            region_perf,
            x="region",
            y="Sales",
            color="Margin (%)",
            color_continuous_scale=BLUE_SCALE,
            title="Revenue & Margin by Region",
            labels={"region": "Region", "Sales": "Total Sales ($)"},
            text_auto=".2s",
        )
        fig_reg.update_layout(**PLOTLY_LAYOUT, height=300, title_font=dict(size=12, color="#1B3A6B"))
        st.plotly_chart(fig_reg, use_container_width=True, config={"displayModeBar": False})

    with seg_col2:
        cat_perf = (
            filtered.groupby(["category", "sub_category"])
            .agg(Sales=("sales", "sum"))
            .reset_index()
            .sort_values("Sales", ascending=True)
            .tail(15)
        )
        fig_cat = px.bar(
            cat_perf,
            x="Sales",
            y="sub_category",
            color="category",
            orientation="h",
            color_discrete_sequence=CORP_COLORS,
            title="Top Sub-Categories by Revenue",
            labels={"sub_category": "", "Sales": "Total Sales ($)"},
        )
        fig_cat.update_layout(**PLOTLY_LAYOUT, height=300, title_font=dict(size=12, color="#1B3A6B"))
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    mlflow_status = "MLflow Tracked" if metadata else "MLflow — Run 'make train'"
    st.markdown(
        f"""
        <div class="page-footer" style="color:#93C5FD!important;">
            Deployed via Docker &nbsp;|&nbsp; {mlflow_status} &nbsp;|&nbsp;
            XGBoost + Prophet Ensemble &nbsp;|&nbsp;
            Version 1.0 &nbsp;|&nbsp;
            Forecast MAPE: {forecast_mape}
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
