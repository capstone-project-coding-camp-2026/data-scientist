import zipfile
import requests
from io import BytesIO
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Config

DATASET_URL = "https://raw.githubusercontent.com/Capstone-Project-Coding-Camp-2026/Data-Scientist/main/dataset_fix.zip"

COLORS = {
    "just_buy":    "#2ec4b6",
    "buy_careful": "#ffb703",
    "dont_buy":    "#e71d36",
}

RECOMMENDATIONS = ["just_buy", "buy_careful", "dont_buy"]
IMPULSE_ORDER   = ["low", "medium", "high"]
IMPULSE_LABELS  = ["Rendah", "Sedang", "Tinggi"]
EXPENSE_COLS    = [
    "expense_housing", "expense_food", "expense_transport",
    "expense_entertainment", "expense_health", "expense_education",
]
MONTH_LABELS    = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                   "Jul", "Ags", "Sep", "Okt", "Nov", "Des"]
MONTH_TICK_VALS = list(range(1, 13))
TOP_N_PERSONAS  = 6
SCATTER_SAMPLE  = 5_000
CHART_HEIGHT    = 450

# Data

@st.cache_data(show_spinner=False)
def load_data():
    resp = requests.get(DATASET_URL, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        df_a = pd.read_csv(zf.open("dataset_a_nlp.csv"))
        df_b = pd.read_csv(zf.open("dataset_b_forecasting.csv"))
        df_c = pd.read_csv(zf.open("dataset_c_whatif (1).csv"))
    try:
        df_b["bulan_dt"] = pd.to_datetime(df_b["bulan"])
    except Exception:
        df_b["bulan_dt"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
            df_b["bulan_index"] - 1, unit="M"
        )
    return df_a, df_b, df_c

# Filter

def render_sidebar_filters(df_b, df_c):
    st.sidebar.header("Filters")

    st.sidebar.subheader("Time Period")
    min_date, max_date = df_b["bulan_dt"].min(), df_b["bulan_dt"].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
    )

    st.sidebar.subheader("Recommendation")
    selected_recs = st.sidebar.multiselect(
        "Select Recommendations",
        options=RECOMMENDATIONS,
        default=RECOMMENDATIONS,
    )

    return {
        "date_range":      date_range,
        "recommendations": selected_recs or RECOMMENDATIONS,
    }


def apply_filters(df_b, df_c, filters):
    df_b_f, df_c_f = df_b.copy(), df_c.copy()

    if len(filters["date_range"]) == 2:
        start, end = (pd.to_datetime(d) for d in filters["date_range"])
        df_b_f = df_b_f[df_b_f["bulan_dt"].between(start, end)]

    if filters["recommendations"]:
        df_c_f = df_c_f[df_c_f["label_whatif"].isin(filters["recommendations"])]

    return df_b_f, df_c_f

# Chart business question 1

def _stacked_bar(df, row_col, x_labels, title, x_title):
    cross = pd.crosstab(df[row_col], df["label_whatif"], normalize="index") * 100
    fig = go.Figure()
    for rec in RECOMMENDATIONS:
        if rec not in cross.columns:
            continue
        y_vals = cross[rec].reindex(cross.index).fillna(0)
        fig.add_trace(go.Bar(
            name=rec,
            x=x_labels,
            y=y_vals.values,
            marker_color=COLORS[rec],
            text=y_vals.apply(lambda v: f"{v:.1f}%"),
            textposition="inside",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title=x_title),
        yaxis=dict(title="Proporsi Persentase (%)", tickformat=".0f"),
        barmode="stack",
        height=CHART_HEIGHT,
        legend=dict(title="Rekomendasi", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def plot_impulse_spending(df_c):
    df_sorted = df_c.copy()
    df_sorted["impulse_spending_tendency"] = pd.Categorical(
        df_sorted["impulse_spending_tendency"], categories=IMPULSE_ORDER, ordered=True
    )
    return _stacked_bar(
        df_sorted.sort_values("impulse_spending_tendency"),
        row_col="impulse_spending_tendency",
        x_labels=IMPULSE_LABELS,
        title="Dampak Belanja Impulsif terhadap Rekomendasi Pembelian",
        x_title="Tingkat Belanja Impulsif",
    )


def plot_pinjol_impact(df_c):
    return _stacked_bar(
        df_c, row_col="pinjol_active",
        x_labels=["Tidak Aktif", "Aktif"],
        title="Dampak Status Pinjol Aktif terhadap Rekomendasi Pembelian",
        x_title="Status Pinjol Aktif",
    )


def plot_credit_card_utilization(df_c):
    fig = go.Figure()
    for rec, color in COLORS.items():
        fig.add_trace(go.Box(
            y=df_c.loc[df_c["label_whatif"] == rec, "credit_card_utilization"],
            name=rec, marker_color=color, boxmean="sd",
        ))
    fig.update_layout(
        title=dict(text="Pemakaian Kartu Kredit vs Rekomendasi Pembelian", font=dict(size=16)),
        xaxis=dict(title="Rekomendasi Keputusan Finansial"),
        yaxis=dict(title="Rasio Penggunaan Kartu Kredit", tickformat=".0%"),
        height=CHART_HEIGHT,
    )
    return fig

# Chart business question 2

def plot_seasonal_trend(df_b):
    monthly = (
        df_b.groupby("bulan_dt")[["total_expense", "savings_rate"]]
        .mean().sort_index().reset_index()
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=monthly["bulan_dt"], y=monthly["total_expense"],
        mode="lines+markers", name="Total Pengeluaran",
        line=dict(color=COLORS["dont_buy"], width=2),
        marker=dict(size=6, color=COLORS["buy_careful"]),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly["bulan_dt"], y=monthly["savings_rate"],
        mode="lines+markers", name="Rasio Tabungan",
        line=dict(color=COLORS["just_buy"], width=2, dash="dash"),
        marker=dict(size=6, color=COLORS["just_buy"]),
    ), secondary_y=True)
    fig.update_layout(
        title=dict(text="Tren Pengeluaran vs Rasio Tabungan (Pola Musiman)", font=dict(size=16)),
        xaxis=dict(title="Bulan", tickformat="%b %Y", dtick="M1"),
        height=CHART_HEIGHT, hovermode="x unified",
    )
    fig.update_yaxes(title_text="Rata-rata Pengeluaran (Rp)", tickformat=",.0f", secondary_y=False)
    fig.update_yaxes(title_text="Rata-rata Rasio Tabungan (%)", tickformat=".0%", secondary_y=True)
    return fig


def plot_expense_heatmap(df_b):
    corr = df_b[EXPENSE_COLS + ["total_expense"]].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale=[[0, COLORS["dont_buy"]], [0.5, "#fdf0d5"], [1, COLORS["just_buy"]]],
        text=corr.values.round(2), texttemplate="%{text}",
        textfont={"size": 10}, zmin=-1, zmax=1,
    ))
    fig.update_layout(
        title=dict(text="Korelasi Antar Kategori Pengeluaran", font=dict(size=16)),
        height=500, width=600,
    )
    return fig

def plot_income_vs_savings(df_b):
    sample = df_b.sample(min(SCATTER_SAMPLE, len(df_b)), random_state=42)
    fig = px.scatter(
        sample, x="monthly_income", y="savings_rate", opacity=0.5,
        color="savings_rate",
        color_continuous_scale=[COLORS["dont_buy"], COLORS["buy_careful"], COLORS["just_buy"]],
        title="Hubungan Pendapatan vs Tingkat Tabungan",
    )
    fig.update_layout(
        xaxis=dict(title="Pendapatan Bulanan (Rp)", tickformat=",.0f"),
        yaxis=dict(title="Tingkat Tabungan (%)", tickformat=".0%"),
        height=CHART_HEIGHT,
    )
    return fig

# Main

def _chart(fig, caption=None):
    st.plotly_chart(fig, use_container_width=True)
    if caption:
        st.caption(f" *{caption}*")


def main():
    st.set_page_config(
        page_title="FinTime Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.spinner("Loading data..."):
        df_a, df_b, df_c = load_data()

    filters = render_sidebar_filters(df_b, df_c)
    df_b_f, df_c_f = apply_filters(df_b, df_c, filters)

    st.title("FinTime Financial Dashboard")
    st.markdown("---")

    st.header("Perilaku yang memiliki dampak paling signifikan "
                "dalam meningkatkan probabilitas pengguna mendapatkan rekomendasi `dont_buy`")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        _chart(plot_impulse_spending(df_c_f),
               "Belanja impulsif tinggi, rendah, maupun sedang menghasilkan rekomendasi yang mirip dengan perbedaan persentase yang tidak terlalu besar")
    with col2:
        _chart(plot_pinjol_impact(df_c_f),
               "Pengguna dengan pinjol aktif hampir tidak pernah mendapat rekomendasi just_buy.")
    _chart(plot_credit_card_utilization(df_c_f),
           "Distribusi CC utilization ketiga kategori hampir sama, credit card utilization bukan faktor pembeda yang kuat untuk rekomendasi pembelian")

    st.markdown("---")

    st.header("Pola musiman pada total pengeluaran dan rasio tabungan "
                "pada bulan-bulan tertentu")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        _chart(plot_seasonal_trend(df_b_f))
    with col2:
        _chart(plot_expense_heatmap(df_b_f))

    col1, col2 = st.columns(2)
    with col1:
        _chart(plot_income_vs_savings(df_b_f))
    with col2:
        st.info("""
        | Periode | Pola |
        |---------|------|
        | **Maret–April** | Lonjakan pengeluaran karena ramadan dan idul fitri |
        | **Juli** | Kenaikan konsisten akibat tahun ajaran baru |
        | **Desember** | Lonjakan pengeluaran karena masa liburan akhir tahun |
        """)

    st.markdown("---")
    st.caption(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()