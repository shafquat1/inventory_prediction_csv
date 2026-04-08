"""
Inventory Prediction App
Upload per-product sales CSV → predict restock needs, trends & peak moments
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventory Prediction",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0d0d14;
    color: #e8e8f0;
}

.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #c8ff00;
    margin-bottom: 0;
    line-height: 1.1;
}

.sub-header {
    color: #888;
    font-size: 0.9rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.metric-card {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #666;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #c8ff00;
    line-height: 1;
}

.metric-sub {
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.2rem;
}

.alert-critical {
    background: #2a0a0a;
    border: 1px solid #ff4444;
    border-left: 4px solid #ff4444;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #ffaaaa;
}

.alert-warning {
    background: #1f1800;
    border: 1px solid #ffaa00;
    border-left: 4px solid #ffaa00;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #ffd080;
}

.alert-ok {
    background: #0a1a0a;
    border: 1px solid #44cc44;
    border-left: 4px solid #44cc44;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #aaffaa;
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #666;
    border-bottom: 1px solid #2a2a3a;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.insight-item {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    color: #ccc;
}

.tag {
    display: inline-block;
    background: #c8ff0022;
    color: #c8ff00;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.1em;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-right: 0.4rem;
    border: 1px solid #c8ff0044;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0a12;
    border-right: 1px solid #1e1e2e;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #aaa;
}

/* Multiselect tag text color override */
.stMultiSelect [data-baseweb="tag"] {
    color: #000000 !important;
}
.stMultiSelect [data-baseweb="tag"] span {
    color: #000000 !important;
}
.stMultiSelect [data-baseweb="tag"] svg {
    fill: #000000 !important;
}

/* Streamlit components overrides */
.stButton > button {
    background: #c8ff00;
    color: #0d0d14;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    width: 100%;
}

.stButton > button:hover {
    background: #d4ff33;
    color: #0d0d14;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    color: #666;
}

.stTabs [aria-selected="true"] {
    color: #c8ff00 !important;
}

.stDataFrame {
    border: 1px solid #2a2a3a;
    border-radius: 8px;
}

div[data-testid="metric-container"] {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 1rem;
}

div[data-testid="metric-container"] label {
    color: #888 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #c8ff00 !important;
    font-family: 'Space Mono', monospace !important;
}

.stSelectbox label, .stSlider label, .stFileUploader label {
    color: #888 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

</style>
""", unsafe_allow_html=True)


# ─── Plotly Dark Theme Base ──────────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor='#0d0d14',
    plot_bgcolor='#16161f',
    font=dict(color='#888', family='monospace', size=11),
    legend=dict(bgcolor='rgba(0,0,0,0.4)', bordercolor='#2a2a3a',
                borderwidth=1, font=dict(color='#ccc', size=10)),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor='#2a2a3a', gridwidth=0.5, tickfont=dict(color='#555'),
               showspikes=True, spikecolor='#444', spikethickness=1,
               spikemode='across', spikesnap='cursor'),
    yaxis=dict(gridcolor='#2a2a3a', gridwidth=0.5, tickfont=dict(color='#555'),
               showspikes=True, spikecolor='#444', spikethickness=1),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='#1e1e2e', bordercolor='#3a3a4a',
                    font=dict(color='#e8e8f0', size=12)),
)


# ─── CSV PARSING ─────────────────────────────────────────────────────────────

def detect_csv_format(df):
    """
    Auto-detect CSV format. Checks in priority order:
      - 'stock_long'  : date, product, consumed, restocked, stock_balance  (multi-product with stock)
      - 'stock_simple': date, consumed, restocked, stock_balance            (single product with stock)
      - 'wide'        : date + numeric product columns
      - 'long'        : date, product, sales
      - 'simple'      : date, sales
    Returns: (format_type, info_dict)
    """
    cols     = [c.strip().lower() for c in df.columns]
    col_map  = {c.strip().lower(): c for c in df.columns}

    date_keys     = ['date', 'day', 'week', 'period', 'time', 'timestamp']
    consumed_keys = ['consumed', 'used', 'usage', 'consumption', 'daily_consumed']
    restock_keys  = ['restocked', 'received', 'replenished', 'restock', 'order_received']
    balance_keys  = ['stock_balance', 'balance', 'closing_stock', 'stock_level', 'inventory_level']
    sales_keys    = ['sales', 'quantity', 'qty', 'units', 'demand', 'sold', 'revenue']
    product_keys  = ['product', 'item', 'sku', 'name', 'material']
    stock_keys    = ['stock', 'inventory', 'current_stock', 'on_hand']
    unit_type_keys    = ['unit_type']
    unit_measure_keys = ['unit_measure', 'unit', 'uom']
    lead_time_keys    = ['lead_time_days', 'lead_time', 'leadtime', 'supplier_lead_days']

    has_date     = any(k in cols for k in date_keys)
    has_consumed = any(k in cols for k in consumed_keys)
    has_balance  = any(k in cols for k in balance_keys)
    has_product  = any(k in cols for k in product_keys)
    has_sales    = any(k in cols for k in sales_keys)

    date_col         = next((col_map[k] for k in date_keys        if k in cols), None)
    consumed_col     = next((col_map[k] for k in consumed_keys     if k in cols), None)
    restock_col      = next((col_map[k] for k in restock_keys      if k in cols), None)
    balance_col      = next((col_map[k] for k in balance_keys      if k in cols), None)
    product_col      = next((col_map[k] for k in product_keys      if k in cols), None)
    sales_col        = next((col_map[k] for k in sales_keys        if k in cols), None)
    stock_col        = next((col_map[k] for k in stock_keys        if k in cols), None)
    unit_type_col    = next((col_map[k] for k in unit_type_keys    if k in cols), None)
    unit_measure_col = next((col_map[k] for k in unit_measure_keys if k in cols), None)
    lead_time_col    = next((col_map[k] for k in lead_time_keys    if k in cols), None)

    base = dict(date_col=date_col, consumed_col=consumed_col, restock_col=restock_col,
                balance_col=balance_col, unit_type_col=unit_type_col,
                unit_measure_col=unit_measure_col, lead_time_col=lead_time_col)

    # ── Stock-aware long format (multi-product) ──────────────────────────────
    if has_date and has_consumed and has_balance and has_product:
        return 'stock_long', {**base, 'product_col': product_col}

    # ── Stock-aware simple format (single product) ───────────────────────────
    if has_date and has_consumed and has_balance and not has_product:
        return 'stock_simple', base

    # ── Wide format: date + numeric product columns ──────────────────────────
    if has_date and not has_product and not has_sales:
        numeric_cols = [c for c in df.columns if c != date_col
                        and pd.to_numeric(df[c], errors='coerce').notna().sum() > len(df) * 0.5]
        if numeric_cols:
            return 'wide', {'date_col': date_col, 'product_cols': numeric_cols, 'stock_col': stock_col}

    # ── Long format with product column ─────────────────────────────────────
    if has_date and has_product and has_sales:
        return 'long', {'date_col': date_col, 'product_col': product_col,
                        'sales_col': sales_col, 'stock_col': stock_col}

    # ── Simple two-column ────────────────────────────────────────────────────
    if has_date and has_sales:
        return 'simple', {'date_col': date_col, 'sales_col': sales_col, 'stock_col': stock_col}

    # ── Numeric-only fallback ────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        return 'numeric_only', {'sales_col': numeric_cols[0], 'product_cols': numeric_cols}

    return 'unknown', {}


def parse_to_products(df, fmt, info, current_stock_override=None):
    """
    Parse any CSV format into a dict of {product_name: dict}.
    Each value dict always contains 'df' (DataFrame with at least date+sales columns).
    Stock-aware formats also include 'unit_type', 'unit_measure', 'current_stock'.
    """
    products = {}

    # ── Stock-aware formats ──────────────────────────────────────────────────
    if fmt in ('stock_long', 'stock_simple'):
        date_col        = info.get('date_col')
        consumed_col    = info.get('consumed_col')
        restock_col     = info.get('restock_col')
        balance_col     = info.get('balance_col')
        unit_type_col   = info.get('unit_type_col')
        unit_measure_col= info.get('unit_measure_col')

        lead_time_col = info.get('lead_time_col')

        def _build_sub(grp_df, unit_type_src_df=None):
            """Build a cleaned sub-DataFrame from a group."""
            keep = [date_col, consumed_col]
            if restock_col and restock_col in grp_df.columns:
                keep.append(restock_col)
            keep.append(balance_col)
            sub = grp_df[keep].copy()
            rename = {date_col: 'date', consumed_col: 'sales', balance_col: 'stock_balance'}
            if restock_col and restock_col in grp_df.columns:
                rename[restock_col] = 'restocked'
            sub = sub.rename(columns=rename)
            sub['sales']         = pd.to_numeric(sub['sales'],         errors='coerce').fillna(0)
            sub['stock_balance'] = pd.to_numeric(sub['stock_balance'], errors='coerce').fillna(0)
            if 'restocked' in sub.columns:
                sub['restocked'] = pd.to_numeric(sub['restocked'], errors='coerce').fillna(0)
            try:
                sub['date'] = pd.to_datetime(sub['date'])
            except Exception:
                sub['date'] = pd.date_range(end=datetime.today(), periods=len(sub), freq='D')
            sub = sub.sort_values('date').reset_index(drop=True)
            src = unit_type_src_df if unit_type_src_df is not None else grp_df
            unit_type     = str(src[unit_type_col].iloc[0])    if (unit_type_col    and unit_type_col    in src.columns) else 'unit'
            unit_measure  = str(src[unit_measure_col].iloc[0]) if (unit_measure_col and unit_measure_col in src.columns) else 'units'
            current_stock = float(sub['stock_balance'].iloc[-1]) if len(sub) > 0 else 0.0
            lead_time     = int(pd.to_numeric(src[lead_time_col].iloc[0], errors='coerce')) \
                            if (lead_time_col and lead_time_col in src.columns) else None
            return sub, unit_type, unit_measure, current_stock, lead_time

        if fmt == 'stock_long':
            product_col = info['product_col']
            for prod, grp in df.groupby(product_col):
                sub, unit_type, unit_measure, current_stock, lead_time = _build_sub(grp)
                entry = {
                    'df': sub,
                    'unit_type': unit_type,
                    'unit_measure': unit_measure,
                    'current_stock': current_stock,
                }
                if lead_time is not None:
                    entry['lead_time_days'] = lead_time
                products[str(prod)] = entry
        else:  # stock_simple
            sub, unit_type, unit_measure, current_stock, lead_time = _build_sub(df, unit_type_src_df=df)
            name_keys = ['product', 'item', 'sku', 'name', 'product_name', 'item_name', 'material']
            product_name_col = next((c for c in df.columns if c.strip().lower() in name_keys), None)
            if product_name_col:
                non_null = df[product_name_col].dropna()
                label = str(non_null.iloc[0]) if not non_null.empty else 'Product'
            else:
                label = 'Product'
            entry = {
                'df': sub,
                'unit_type': unit_type,
                'unit_measure': unit_measure,
                'current_stock': current_stock,
            }
            if lead_time is not None:
                entry['lead_time_days'] = lead_time
            products[label] = entry
        return products

    # ── Wide format ──────────────────────────────────────────────────────────
    if fmt == 'wide':
        date_col = info['date_col']
        for pcol in info['product_cols']:
            sub = df[[date_col, pcol]].copy()
            sub.columns = ['date', 'sales']
            sub['sales'] = pd.to_numeric(sub['sales'], errors='coerce').fillna(0)
            try:
                sub['date'] = pd.to_datetime(sub['date'])
            except Exception:
                sub['date'] = pd.date_range(end=datetime.today(), periods=len(sub), freq='D')
            sub = sub.sort_values('date').reset_index(drop=True)
            products[pcol] = {'df': sub}

    elif fmt == 'long':
        date_col, product_col, sales_col = info['date_col'], info['product_col'], info['sales_col']
        for prod, grp in df.groupby(product_col):
            sub = grp[[date_col, sales_col]].copy()
            sub.columns = ['date', 'sales']
            sub['sales'] = pd.to_numeric(sub['sales'], errors='coerce').fillna(0)
            try:
                sub['date'] = pd.to_datetime(sub['date'])
            except Exception:
                sub['date'] = pd.date_range(end=datetime.today(), periods=len(sub), freq='D')
            sub = sub.sort_values('date').reset_index(drop=True)
            products[str(prod)] = {'df': sub}

    elif fmt == 'simple':
        date_col, sales_col = info['date_col'], info['sales_col']
        sub = df[[date_col, sales_col]].copy()
        sub.columns = ['date', 'sales']
        sub['sales'] = pd.to_numeric(sub['sales'], errors='coerce').fillna(0)
        try:
            sub['date'] = pd.to_datetime(sub['date'])
        except Exception:
            sub['date'] = pd.date_range(end=datetime.today(), periods=len(sub), freq='D')
        sub = sub.sort_values('date').reset_index(drop=True)
        name_keys = ['product', 'item', 'sku', 'name', 'product_name', 'item_name']
        product_name_col = next((c for c in df.columns if c.strip().lower() in name_keys), None)
        if product_name_col:
            non_null = df[product_name_col].dropna()
            product_label = str(non_null.iloc[0]) if not non_null.empty else 'Product'
        else:
            product_label = 'Product'
        products[product_label] = {'df': sub}

    elif fmt == 'numeric_only':
        for col in info['product_cols']:
            sub = pd.DataFrame({
                'date': pd.date_range(end=datetime.today(), periods=len(df), freq='D'),
                'sales': pd.to_numeric(df[col], errors='coerce').fillna(0)
            })
            products[col] = {'df': sub}

    return products


# ─── FORECASTING ENGINE — Statistical (Holt-Winters + weekly seasonality) ────

def forecast_demand(sales: np.ndarray, horizon: int = 30, freq: str = "D"):
    """
    Statistical demand forecasting using Holt's double-exponential smoothing
    with weekly seasonality decomposition and widening confidence intervals.

    Returns (point_forecast, lower_bound, upper_bound) as numpy arrays.
    No external ML dependencies — runs instantly on any machine.
    """
    n = len(sales)

    if n < 2:
        flat = np.full(horizon, float(sales[0]) if n == 1 else 1.0)
        return flat, flat * 0.8, flat * 1.2

    sales = sales.astype(float)

    # ── Weekly seasonality indices (day-of-week pattern) ────────────────────
    if n >= 14:
        dow_sum    = np.zeros(7)
        dow_count  = np.zeros(7)
        for i, v in enumerate(sales):
            dow_sum[i % 7]   += v
            dow_count[i % 7] += 1
        dow_avgs = np.where(dow_count > 0, dow_sum / dow_count, np.mean(sales))
        mean_d   = dow_avgs.mean()
        seasonal = np.where(mean_d > 0, dow_avgs / mean_d, np.ones(7))
    else:
        seasonal = np.ones(7)

    # ── Holt's double exponential smoothing ─────────────────────────────────
    alpha, beta = 0.25, 0.05
    level = sales[0]
    # Initialise trend from first 7 points (or fewer)
    init_w = min(7, n)
    trend  = float(np.polyfit(np.arange(init_w), sales[:init_w], 1)[0])

    for i, v in enumerate(sales):
        prev_level = level
        level = alpha * v + (1.0 - alpha) * (level + trend)
        trend = beta  * (level - prev_level) + (1.0 - beta) * trend

    # ── Generate point forecast with seasonal modulation ────────────────────
    last_dow = (n - 1) % 7
    preds = np.array([
        max((level + trend * (i + 1)) * seasonal[(last_dow + i + 1) % 7], 0.0)
        for i in range(horizon)
    ])

    # ── Uncertainty bands (widen with sqrt of horizon steps) ────────────────
    # Use in-sample residuals for scale; fall back to rolling std
    smooth = np.convolve(sales, np.ones(min(7, n)) / min(7, n), mode='same')
    residuals = np.abs(sales - smooth)
    base_std  = max(np.std(residuals), np.std(sales) * 0.05, 0.01)
    widths    = base_std * np.sqrt(np.arange(1, horizon + 1))
    lo = np.maximum(preds - 1.5 * widths, 0.0)
    hi = preds + 1.5 * widths

    return preds, lo, hi


def detect_peaks(sales: np.ndarray, dates: pd.Series):
    """Return indices of peak days above 75th percentile"""
    threshold = np.percentile(sales, 75)
    peaks = [(i, dates.iloc[i], sales[i]) for i in range(len(sales)) if sales[i] >= threshold]
    return peaks


def days_until_stockout(current_stock: float, forecast: np.ndarray):
    """How many days until cumulative demand exceeds stock"""
    cumulative = np.cumsum(forecast)
    breach = np.where(cumulative >= current_stock)[0]
    if len(breach) == 0:
        return len(forecast), False  # won't run out in horizon
    return breach[0] + 1, True


def reorder_recommendation(avg_daily: float, lead_time: int, safety_days: int):
    reorder_point = avg_daily * (lead_time + safety_days)
    suggested_qty = avg_daily * 30  # one month supply
    return reorder_point, suggested_qty


# ─── HOLIDAY & SEASONAL HELPERS ──────────────────────────────────────────────

# Fixed public holidays (month, day, name)
_FIXED_HOLIDAYS = [
    (1,  1,  "New Year's Day"),
    (1,  26, "Republic Day"),
    (5,  1,  "Labour Day"),
    (8,  15, "Independence Day"),
    (10, 2,  "Gandhi Jayanti"),
    (10, 24, "Diwali"),
    (12, 25, "Christmas"),
    (12, 26, "Boxing Day"),
]


def get_holidays(dates_series: pd.Series) -> list:
    """Return [(pd.Timestamp, name), …] for known holidays within the date range."""
    if dates_series.empty:
        return []
    lo, hi = dates_series.min(), dates_series.max()
    result = []
    for year in range(lo.year, hi.year + 1):
        for m, d, name in _FIXED_HOLIDAYS:
            try:
                h = pd.Timestamp(year, m, d)
                if lo <= h <= hi:
                    result.append((h, name))
            except Exception:
                pass
    return sorted(result)


def plot_monthly_seasonality(sales: np.ndarray, dates: pd.Series):
    """Bar chart of average daily consumption by calendar month."""
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    avgs = []
    for m in range(1, 13):
        vals = [sales[i] for i in range(len(sales)) if dates.iloc[i].month == m]
        avgs.append(float(np.mean(vals)) if vals else 0.0)

    mean_val = np.mean([v for v in avgs if v > 0]) or 1.0
    colors  = ['#c8ff00' if v == max(avgs) else '#f0a500' if v == min([x for x in avgs if x > 0]) else '#4488ff' for v in avgs]
    pct     = [f"{((v/mean_val-1)*100):+.1f}% vs avg" if mean_val > 0 and v > 0 else 'No data' for v in avgs]

    fig = go.Figure(go.Bar(
        x=month_names, y=avgs,
        marker_color=colors,
        text=[f"{v:.1f}" if v > 0 else '' for v in avgs],
        textposition='outside',
        textfont=dict(color='#ccc', size=10),
        customdata=pct,
        hovertemplate='<b>%{x}</b><br>Avg Daily: <b>%{y:.2f}</b><br>%{customdata}<extra></extra>',
    ))
    layout = dict(**_LAYOUT)
    layout.update(
        title=dict(text='Seasonal Pattern — Avg Daily Consumption by Month',
                   font=dict(color='#e8e8f0', size=13)),
        yaxis=dict(**_LAYOUT['yaxis'],
                   title=dict(text='Avg Daily Qty', font=dict(color='#888'))),
        hovermode='x',
        height=300,
        bargap=0.25,
    )
    fig.update_layout(**layout)
    return fig


# ─── PLOTS ───────────────────────────────────────────────────────────────────

def plot_sales_and_forecast(hist_df, forecast, lo, hi, product_name, unit_measure="units", holidays=None):
    pred_dates = pd.date_range(
        start=hist_df['date'].iloc[-1] + timedelta(days=1), periods=len(forecast), freq='D'
    )
    last_date = hist_df['date'].iloc[-1]
    mean_sales = hist_df['sales'].mean()

    fig = go.Figure()

    # Confidence band (drawn first so it sits behind)
    fig.add_trace(go.Scatter(
        x=list(pred_dates) + list(pred_dates[::-1]),
        y=list(hi) + list(lo[::-1]),
        fill='toself',
        fillcolor='rgba(255,107,107,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band',
        hoverinfo='skip',
        showlegend=True,
    ))

    # Historical sales
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df['sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='#4488ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(68,136,255,0.07)',
        hovertemplate=(
            '<b>%{x|%b %d, %Y}</b>  %{x|%A}<br>'
            f'Sales: <b>%{{y:.2f}} {unit_measure}</b><extra></extra>'
        ),
    ))

    # Peak markers
    peaks = detect_peaks(hist_df['sales'].values, hist_df['date'])
    if peaks:
        peak_dates = [p[1] for p in peaks]
        peak_vals  = [p[2] for p in peaks]
        peak_text  = [
            f"+{((p[2]/mean_sales - 1)*100):.0f}% vs avg" if mean_sales > 0 else ''
            for p in peaks
        ]
        fig.add_trace(go.Scatter(
            x=peak_dates, y=peak_vals,
            mode='markers',
            name='Peak Moments',
            marker=dict(color='#c8ff00', size=9, symbol='circle',
                        line=dict(color='#0d0d14', width=1)),
            text=peak_text,
            hovertemplate=(
                '<b>⚡ PEAK</b>  %{x|%b %d, %Y} (%{x|%A})<br>'
                f'Sales: <b>%{{y:.2f}} {unit_measure}</b><br>'
                '%{text}<extra></extra>'
            ),
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=pred_dates, y=forecast,
        mode='lines',
        name='Forecast',
        line=dict(color='#ff6b6b', width=2.5, dash='dash'),
        hovertemplate=(
            '<b>📈 FORECAST</b>  %{x|%b %d, %Y} (%{x|%A})<br>'
            f'Predicted: <b>%{{y:.2f}} {unit_measure}</b><extra></extra>'
        ),
    ))

    # NOW divider
    fig.add_vline(x=last_date, line_dash='dot', line_color='#555', line_width=1.5)
    fig.add_annotation(
        x=last_date, y=1, yref='paper',
        text='NOW', font=dict(color='#888', size=9),
        showarrow=False, xanchor='left', yanchor='top',
    )

    # Holiday markers (on both historical and forecast range)
    for hdate, hname in (holidays or []):
        fig.add_vline(x=hdate, line_dash='dot', line_color='rgba(200,255,0,0.35)', line_width=1)
        fig.add_annotation(
            x=hdate, y=0.98, yref='paper',
            text=f'🎌 {hname}', font=dict(color='rgba(200,255,0,0.6)', size=8),
            showarrow=False, xanchor='left', yanchor='top', textangle=-90,
        )

    layout = dict(**_LAYOUT)
    layout.update(
        title=dict(text=f'{product_name} — Consumption History & Forecast',
                   font=dict(color='#e8e8f0', size=13)),
        yaxis=dict(**_LAYOUT['yaxis'],
                   title=dict(text=f'Qty ({unit_measure})', font=dict(color='#888'))),
        height=440,
    )
    fig.update_layout(**layout)
    return fig


def plot_inventory_projection(current_stock, forecast, lead_time, safety_days, avg_daily,
                              unit_measure="units", hist_stock_df=None):
    """
    Inventory projection chart.

    Parameters
    ----------
    hist_stock_df : DataFrame with columns ['date', 'stock_balance'] (and optionally 'restocked'),
                    representing the historical record.  When provided, the chart shows the
                    actual historical stock as a solid teal line **before** today, then the
                    projected stock as a dashed orange line going forward.
    """
    stock_proj = [current_stock]
    for d in forecast:
        stock_proj.append(max(stock_proj[-1] - d, 0))

    today = pd.Timestamp(datetime.today().date())
    proj_dates = pd.date_range(start=today, periods=len(stock_proj), freq='D')
    reorder_pt   = avg_daily * (lead_time + safety_days)
    safety_stock = avg_daily * safety_days

    fig = go.Figure()

    # ── Danger-zone shading ──────────────────────────────────────────────────
    fig.add_hrect(y0=0, y1=max(safety_stock, 0.001), fillcolor='rgba(255,68,68,0.06)', line_width=0)

    # ── Historical stock (solid line) ────────────────────────────────────────
    if hist_stock_df is not None and 'stock_balance' in hist_stock_df.columns and len(hist_stock_df) > 0:
        hist_x = hist_stock_df['date']
        hist_y = hist_stock_df['stock_balance']
        fig.add_trace(go.Scatter(
            x=hist_x, y=hist_y,
            mode='lines',
            name='Actual Stock',
            line=dict(color='#00d4aa', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,212,170,0.07)',
            hovertemplate=(
                '<b>%{x|%b %d, %Y}</b>  %{x|%A}<br>'
                f'Actual Stock: <b>%{{y:.1f}} {unit_measure}</b><extra></extra>'
            ),
        ))

        # Restock event markers
        if 'restocked' in hist_stock_df.columns:
            restock_events = hist_stock_df[hist_stock_df['restocked'] > 0]
            if not restock_events.empty:
                fig.add_trace(go.Scatter(
                    x=restock_events['date'],
                    y=restock_events['stock_balance'],
                    mode='markers',
                    name='Restock Event',
                    marker=dict(color='#c8ff00', size=8, symbol='triangle-up',
                                line=dict(color='#ffffff', width=1)),
                    customdata=restock_events['restocked'],
                    hovertemplate=(
                        '<b>%{x|%b %d, %Y}</b><br>'
                        f'Restocked: <b>+%{{customdata:.0f}} {unit_measure}</b><br>'
                        f'Balance after: <b>%{{y:.1f}} {unit_measure}</b><extra></extra>'
                    ),
                ))

        # TODAY divider
        fig.add_vline(x=today, line_color='rgba(200,255,0,0.6)', line_dash='dot', line_width=1.5)
        fig.add_annotation(
            x=today, y=1, yref='paper',
            text='Today', font=dict(color='#c8ff00', size=10),
            showarrow=False, xanchor='left', yanchor='top',
        )

    # ── Projected stock (dashed line) ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=proj_dates, y=stock_proj,
        mode='lines',
        name='Projected Stock',
        line=dict(color='#f0a500', width=2.5, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(240,165,0,0.06)',
        hovertemplate=(
            '<b>%{x|%b %d, %Y}</b>  %{x|%A}<br>'
            f'Projected: <b>%{{y:.1f}} {unit_measure}</b><extra></extra>'
        ),
    ))

    # ── Reorder point line ───────────────────────────────────────────────────
    fig.add_hline(y=reorder_pt, line_dash='dash', line_color='#ff6b6b', line_width=1.5)
    fig.add_annotation(
        x=0, xref='paper', y=reorder_pt,
        text=f'Reorder Point  {reorder_pt:.1f} {unit_measure}',
        font=dict(color='#ff6b6b', size=10), showarrow=False,
        xanchor='left', yanchor='bottom',
    )

    # ── Safety stock line ─────────────────────────────────────────────────────
    fig.add_hline(y=safety_stock, line_dash='dot', line_color='#ff4444', line_width=1.5)
    fig.add_annotation(
        x=0, xref='paper', y=safety_stock,
        text=f'Safety Stock  {safety_stock:.1f} {unit_measure}',
        font=dict(color='#ff4444', size=10), showarrow=False,
        xanchor='left', yanchor='top',
    )

    # ── Stockout marker ───────────────────────────────────────────────────────
    stockout_day, will_stockout = days_until_stockout(current_stock, forecast)
    if will_stockout:
        stockout_date = today + timedelta(days=int(stockout_day))
        fig.add_vline(x=stockout_date, line_color='#ff4444', line_width=2)
        fig.add_annotation(
            x=stockout_date, y=1, yref='paper',
            text=f'Stockout Day {stockout_day}',
            font=dict(color='#ff6b6b', size=10), showarrow=False,
            xanchor='right', yanchor='top',
        )

    layout = dict(**_LAYOUT)
    layout.update(
        title=dict(text='Inventory Projection', font=dict(color='#e8e8f0', size=13)),
        yaxis=dict(**_LAYOUT['yaxis'],
                   title=dict(text=f'Stock ({unit_measure})', font=dict(color='#888'))),
        height=420,
    )
    fig.update_layout(**layout)
    return fig


def plot_weekly_heatmap(sales: np.ndarray, dates: pd.Series):
    """Day-of-week average sales bar chart."""
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_avgs = []
    for dow in range(7):
        vals = [sales[i] for i in range(len(sales)) if dates.iloc[i].weekday() == dow]
        dow_avgs.append(np.mean(vals) if vals else 0)

    mean_val = np.mean(dow_avgs) if any(dow_avgs) else 1
    colors   = ['#c8ff00' if v == max(dow_avgs) else '#4488ff' for v in dow_avgs]
    pct_text = [f"{((v / mean_val - 1) * 100):+.1f}% vs avg" if mean_val > 0 else '' for v in dow_avgs]

    fig = go.Figure(go.Bar(
        x=dow_names, y=dow_avgs,
        marker_color=colors,
        text=[f"{v:.1f}" for v in dow_avgs],
        textposition='outside',
        textfont=dict(color='#ccc', size=10),
        customdata=pct_text,
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Avg Sales: <b>%{y:.2f}</b><br>'
            '%{customdata}<extra></extra>'
        ),
    ))

    layout = dict(**_LAYOUT)
    layout.update(
        title=dict(text='Avg Sales by Day of Week', font=dict(color='#e8e8f0', size=13)),
        yaxis=dict(**_LAYOUT['yaxis'],
                   title=dict(text='Avg Units', font=dict(color='#888'))),
        hovermode='x',
        height=280,
        bargap=0.3,
    )
    fig.update_layout(**layout)
    return fig


# ─── INVENTORY CATALOG FORMAT ────────────────────────────────────────────────

_INVENTORY_REQUIRED = {'item_name', 'pre_sales_week1', 'pre_sales_week2', 'pre_sales_week3', 'pre_sales_week4'}

def is_inventory_catalog(df):
    """Return True if df looks like an inventory catalog (not a time-series)."""
    cols = {c.strip().lower() for c in df.columns}
    return _INVENTORY_REQUIRED.issubset(cols)


def parse_inventory_catalog(df):
    """
    Parse an inventory catalog CSV (item_name, unit_type, unit_measure,
    pre_sales_week1-4, current_stock, etc.) into per-product time-series dicts.
    Returns: {item_name: {df, unit_type, unit_measure, current_stock, ...}}
    """
    result = {}
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for _, row in df.iterrows():
        name = str(row.get('item_name', f"Item {row.get('item_id', '')}")).strip()

        # Weekly sales totals → daily averages
        weeks = []
        for w in range(1, 5):
            col = f'pre_sales_week{w}'
            val = row.get(col, 0)
            weeks.append(float(val) if pd.notna(val) else 0.0)

        # Synthetic 60-day time series from weekly data
        np.random.seed(abs(hash(name)) % (2 ** 31))
        days = 60
        dates = pd.date_range(end=datetime.today(), periods=days, freq='D')

        seasonal = float(row.get('seasonal_factor', 1.0)) if pd.notna(row.get('seasonal_factor', 1.0)) else 1.0
        turnover = float(row.get('stock_turnover_rate', 5.0)) if pd.notna(row.get('stock_turnover_rate', 5.0)) else 5.0
        daily_avgs = [w / 7.0 for w in weeks]

        sales = []
        for i in range(days):
            week_idx = min(i // 15, 3)
            base = daily_avgs[week_idx]
            seasonal_factor = 1 + (seasonal - 1) * np.sin(2 * np.pi * i / 30)
            dow = dates[i].weekday()
            dow_factor = 1.3 if dow >= 4 else 0.9
            noise = np.random.normal(0, base * 0.12 * (turnover / 5.0))
            sales.append(max(base * seasonal_factor * dow_factor + noise, 0))

        hist_df = pd.DataFrame({'date': dates, 'sales': sales})

        def _safe_float(key, default=0.0):
            v = row.get(key, default)
            return float(v) if pd.notna(v) else default

        result[name] = {
            'df': hist_df,
            'unit_type': str(row.get('unit_type', 'unit')).strip().lower(),
            'unit_measure': str(row.get('unit_measure', 'units')).strip(),
            'current_stock': _safe_float('current_stock', 100.0),
            'reorder_point': _safe_float('reorder_point', 20.0),
            'lead_time_days': int(_safe_float('lead_time_days', 7)),
            'category': str(row.get('category', '')).strip(),
            'supplier_id': str(row.get('supplier_id', '')).strip(),
            'unit_cost': _safe_float('unit_cost', 0.0),
            'selling_price': _safe_float('selling_price', 0.0),
            'avg_monthly_sales': _safe_float('avg_monthly_sales', 0.0),
        }

    return result


def format_qty(value, unit_type, unit_measure):
    """Format a quantity with its unit label.

    Handles two kinds of unit_measure:
      • Simple units  (L, kg, each, pack …) → "{value} {um}"
      • Package sizes (500mL, 2L, 300mL …) → "{value} × {um}"
        These start with a digit and describe the size of one container,
        so "75 500mL" is wrong — it should read "75 × 500mL".
    """
    ut = (unit_type or 'unit').lower()
    um = (unit_measure or 'units').strip()

    # Package-size descriptor (e.g. "500mL", "2L", "1L", "250mL", "300mL")
    if um and um[0].isdigit():
        return f"{int(round(value))} × {um}"

    if ut == 'liquid':
        return f"{value:.1f} {um}"
    elif ut == 'weight':
        if 'g' in um.lower() and 'kg' not in um.lower():
            return f"{value:.0f} {um}"
        return f"{value:.2f} {um}"
    else:
        return f"{value:.0f} {um}"


def unit_type_badge(unit_type):
    """Return an HTML pill badge for the unit type."""
    ut = (unit_type or 'unit').lower()
    styles = {
        'liquid': ('💧 LIQUID', '#1a3a5c', '#4488ff'),
        'weight': ('⚖️ WEIGHT', '#3a2a0a', '#f0a500'),
        'unit':   ('📦 UNIT',   '#0a2a0a', '#44cc44'),
    }
    label, bg, fg = styles.get(ut, styles['unit'])
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg}44;'
        f'font-size:0.7rem;padding:0.15rem 0.5rem;border-radius:4px;'
        f'font-family:monospace;letter-spacing:0.1em">{label}</span>'
    )


def unit_emoji(unit_type):
    return {'liquid': '💧', 'weight': '⚖️'}.get((unit_type or 'unit').lower(), '📦')


# ─── SAMPLE CSV ──────────────────────────────────────────────────────────────

def make_stock_sample_csv():
    """
    Return the wheelchair factory stock CSV (stock_long format).
    Reads wheelchair_factory_stock.csv from disk if it exists, else generates
    it on the fly.

    Columns: date, product, unit_type, unit_measure, consumed, restocked, stock_balance
    One row per product per day — 365 days × 12 materials = 4 380 rows.
    """
    import os
    stock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wheelchair_factory_stock.csv')
    if os.path.exists(stock_path):
        with open(stock_path) as f:
            return f.read()

    # ── Generate on the fly if the file is missing ───────────────────────────
    np.random.seed(2024)
    start = datetime(datetime.today().year - 1, 1, 1)
    dates = pd.date_range(start=start, periods=365, freq='D')
    holiday_set = {(m, d) for m, d, _ in _FIXED_HOLIDAYS}
    seasonal = {1:1.10, 2:1.08, 3:1.05, 4:0.98, 5:0.92, 6:0.85,
                7:0.80, 8:0.83, 9:0.97, 10:1.18, 11:1.30, 12:1.22}

    def _daily_chairs(d):
        if (d.month, d.day) in holiday_set or d.weekday() == 6:
            return 0.0
        base = 5.0 if d.weekday() == 5 else 10.0
        return max(base * seasonal[d.month] * np.random.normal(1.0, 0.10), 0.0)

    chairs_per_day = np.array([_daily_chairs(d.to_pydatetime()) for d in dates])

    # (name, unit_type, unit_measure, per_chair, noise_std, initial_stock, reorder_days, order_qty_days, lead_time_days)
    materials = [
        ("Steel Tubing",        "weight", "kg",    8.50, 0.05, 3000,  10, 25, 14),
        ("Aluminum Profile",    "weight", "kg",    2.20, 0.04,  800,  10, 25, 14),
        ("Rear Wheel Assembly", "unit",   "units", 2.00, 0.00,  500,  14, 30, 21),
        ("Front Caster Wheel",  "unit",   "units", 2.00, 0.00,  500,  14, 30, 21),
        ("Seat Foam Pad",       "weight", "kg",    1.50, 0.06,  400,  10, 25, 10),
        ("Upholstery Fabric",   "weight", "m",     2.50, 0.04,  600,  10, 25, 10),
        ("Fastener Pack",       "unit",   "packs", 1.00, 0.00,  300,  10, 20,  7),
        ("Wheel Bearings",      "unit",   "units", 4.00, 0.00,  800,  14, 30, 21),
        ("Push Rims",           "unit",   "units", 2.00, 0.00,  500,  14, 30, 21),
        ("Brake Assembly",      "unit",   "units", 2.00, 0.00,  400,  14, 30, 21),
        ("Powder Coat Paint",   "liquid", "L",     0.80, 0.07,  200,  10, 25,  7),
        ("Welding Wire",        "weight", "kg",    0.30, 0.06,   80,  10, 25,  7),
    ]

    rows = []
    for name, unit_type, unit_measure, per_chair, noise_std, init_stock, reorder_days, order_days, lead_time_days in materials:
        noise = np.random.normal(1.0, noise_std, len(dates)) if noise_std > 0 else np.ones(len(dates))
        consumed_arr = np.round(chairs_per_day * per_chair * noise, 2)
        avg_daily = max(np.mean(consumed_arr), 0.01)
        reorder_point = avg_daily * reorder_days
        order_qty     = avg_daily * order_days

        stock = float(init_stock)
        for i, (d, consumed) in enumerate(zip(dates, consumed_arr)):
            restocked = 0.0
            if stock <= reorder_point:
                restocked = round(order_qty, 2)
                stock += restocked
            stock = max(stock - consumed, 0.0)
            if consumed > 0:   # skip Sundays / holidays (zero-production days)
                rows.append({
                    'date':            d.strftime('%Y-%m-%d'),
                    'product':         name,
                    'unit_type':       unit_type,
                    'unit_measure':    unit_measure,
                    'consumed':        round(consumed, 2),
                    'restocked':       restocked,
                    'stock_balance':   round(stock, 2),
                    'lead_time_days':  lead_time_days,
                })

    result_df = pd.DataFrame(rows, columns=['date','product','unit_type','unit_measure',
                                             'consumed','restocked','stock_balance','lead_time_days'])
    return result_df.to_csv(index=False)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def _run_product_analysis(product_name, data, lead_time, safety_days, forecast_horizon):
    """Render full analysis for a single product inside an expander."""
    hist_df = data['df']
    unit_type = data.get('unit_type', 'unit')
    unit_measure = data.get('unit_measure', 'units')
    prod_lead = int(data.get('lead_time_days', lead_time))

    # Use last stock_balance from CSV if available; else fall back to sidebar default or data key
    if 'stock_balance' in hist_df.columns and len(hist_df) > 0:
        prod_stock = float(hist_df['stock_balance'].iloc[-1])
    else:
        prod_stock = data.get('current_stock', 0.0)

    # Historical stock series for inventory projection chart
    if 'stock_balance' in hist_df.columns:
        hist_stock_df = hist_df[['date', 'stock_balance'] +
                                 (['restocked'] if 'restocked' in hist_df.columns else [])].copy()
    else:
        hist_stock_df = None
    category = data.get('category', '')

    sales = hist_df['sales'].values.astype(float)
    dates = hist_df['date']

    forecast, lo, hi = forecast_demand(sales, forecast_horizon)
    avg_daily = np.mean(forecast)
    total_forecasted = np.sum(forecast)
    stockout_day, will_stockout = days_until_stockout(prod_stock, forecast)
    reorder_pt, suggested_qty = reorder_recommendation(avg_daily, prod_lead, safety_days)
    hist_avg = np.mean(sales[-min(14, len(sales)):])
    trend_pct = ((avg_daily - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0

    # Urgency
    if will_stockout and stockout_day <= prod_lead:
        urgency_html = '<div class="alert-critical">🔴 <strong>CRITICAL</strong> — Stockout expected before next delivery. Order NOW.</div>'
    elif will_stockout and stockout_day <= prod_lead + safety_days:
        urgency_html = '<div class="alert-warning">🟡 <strong>REORDER NOW</strong> — Stock will hit safety threshold within lead time window.</div>'
    elif prod_stock < reorder_pt:
        urgency_html = '<div class="alert-warning">🟡 <strong>REORDER ADVISED</strong> — Current stock is below reorder point.</div>'
    else:
        urgency_html = '<div class="alert-ok">🟢 <strong>STOCK OK</strong> — Inventory adequate for forecast horizon.</div>'

    st.markdown(urgency_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Current Stock", format_qty(prod_stock, unit_type, unit_measure))
    with k2:
        st.metric("Days to Stockout", f"{stockout_day}d" if will_stockout else f">{forecast_horizon}d")
    with k3:
        st.metric(f"Avg Daily ({unit_measure})", f"{avg_daily:.2f}")
    with k4:
        st.metric(f"{forecast_horizon}d Total", format_qty(total_forecasted, unit_type, unit_measure))
    with k5:
        st.metric("Demand Trend", f"{trend_pct:+.1f}%", delta=f"{trend_pct:+.1f}%")

    # Optional catalog metadata row
    if data.get('unit_cost') or data.get('selling_price'):
        st.markdown(
            f'<div style="display:flex;gap:1.5rem;margin:0.5rem 0 1rem">'
            f'{unit_type_badge(unit_type)}'
            f'<span style="color:#888;font-size:0.85rem">Unit Cost: <strong style="color:#ccc">${data.get("unit_cost",0):.2f}</strong></span>'
            f'<span style="color:#888;font-size:0.85rem">Sell Price: <strong style="color:#ccc">${data.get("selling_price",0):.2f}</strong></span>'
            + (f'<span style="color:#888;font-size:0.85rem">Category: <strong style="color:#ccc">{category}</strong></span>' if category else '')
            + '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Compute holidays once for the full date span (history + forecast)
    all_dates = pd.concat([
        dates,
        pd.Series(pd.date_range(start=dates.iloc[-1] + timedelta(days=1), periods=forecast_horizon, freq='D'))
    ], ignore_index=True)
    holidays = get_holidays(all_dates)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📦 Inventory", "📅 Seasons & Peaks", "💡 Recommendations"])

    with tab1:
        fig = plot_sales_and_forecast(hist_df, forecast, lo, hi, product_name, unit_measure, holidays=holidays)
        st.plotly_chart(fig, width='stretch', key=f"{product_name}_forecast")

        st.markdown('<div class="section-title">Day-by-Day Forecast</div>', unsafe_allow_html=True)
        pred_dates = pd.date_range(start=hist_df['date'].iloc[-1] + timedelta(days=1), periods=forecast_horizon, freq='D')
        forecast_table = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in pred_dates],
            'Day': [d.strftime('%A') for d in pred_dates],
            f'Forecast ({unit_measure})': [f"{v:.2f}" for v in forecast],
            'Low Estimate': [f"{v:.2f}" for v in lo],
            'High Estimate': [f"{v:.2f}" for v in hi],
        })
        st.dataframe(forecast_table, hide_index=True, width="stretch", height=350)

    with tab2:
        fig2 = plot_inventory_projection(prod_stock, forecast, prod_lead, safety_days, avg_daily,
                                         unit_measure, hist_stock_df=hist_stock_df)
        st.plotly_chart(fig2, width='stretch', key=f"{product_name}_inventory")

        st.markdown('<div class="section-title">Inventory Milestones</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Reorder Point", format_qty(reorder_pt, unit_type, unit_measure))
        with c2:
            st.metric("Safety Stock", format_qty(avg_daily * safety_days, unit_type, unit_measure))
        with c3:
            st.metric("Suggested Order Qty", format_qty(suggested_qty, unit_type, unit_measure))

    with tab3:
        # Monthly seasonality
        if len(sales) >= 30:
            st.markdown('<div class="section-title">Monthly Seasonal Pattern</div>', unsafe_allow_html=True)
            fig_season = plot_monthly_seasonality(sales, dates)
            st.plotly_chart(fig_season, width='stretch', key=f"{product_name}_seasonal")

        # Weekly pattern
        st.markdown('<div class="section-title">Day-of-Week Pattern</div>', unsafe_allow_html=True)
        if len(sales) >= 7:
            fig3 = plot_weekly_heatmap(sales, dates)
            st.plotly_chart(fig3, width='stretch', key=f"{product_name}_weekly")
        else:
            st.info("Need at least 7 days of data for weekly patterns.")

        # Holidays in forecast window
        forecast_holidays = [(h, n) for h, n in holidays
                             if h > hist_df['date'].iloc[-1]]
        if forecast_holidays:
            st.markdown('<div class="section-title">Upcoming Holidays in Forecast Window</div>', unsafe_allow_html=True)
            hol_df = pd.DataFrame([{
                'Date': h.strftime('%Y-%m-%d'),
                'Day': h.strftime('%A'),
                'Holiday': n,
                'Note': '⚠️ Expect zero / reduced output'
            } for h, n in forecast_holidays])
            st.dataframe(hol_df, hide_index=True, width="stretch")

        st.markdown('<div class="section-title">Historical Peak Days (Top 25%)</div>', unsafe_allow_html=True)
        peaks = detect_peaks(sales, dates)
        if peaks:
            peaks_sorted = sorted(peaks, key=lambda x: -x[2])
            peak_df = pd.DataFrame([{
                'Date': p[1].strftime('%Y-%m-%d'),
                'Day': p[1].strftime('%A'),
                'Month': p[1].strftime('%B'),
                f'Qty ({unit_measure})': f"{p[2]:.2f}",
                'vs Avg': f"+{((p[2] / np.mean(sales) - 1) * 100):.0f}%"
            } for p in peaks_sorted[:20]])
            st.dataframe(peak_df, hide_index=True, width="stretch")

        st.markdown('<div class="section-title">Forecasted Peak Days</div>', unsafe_allow_html=True)
        pred_dates_list = pd.date_range(start=hist_df['date'].iloc[-1] + timedelta(days=1), periods=forecast_horizon, freq='D')
        threshold = np.percentile(forecast, 75)
        forecast_peaks = [(d, v) for d, v in zip(pred_dates_list, forecast) if v >= threshold]
        if forecast_peaks:
            fp_df = pd.DataFrame([{
                'Date': d.strftime('%Y-%m-%d'),
                'Day': d.strftime('%A'),
                'Holiday': next((n for h, n in holidays if h == d), '—'),
                f'Predicted ({unit_measure})': f"{v:.2f}",
            } for d, v in sorted(forecast_peaks, key=lambda x: -x[1])[:15]])
            st.dataframe(fp_df, hide_index=True, width="stretch")

    with tab4:
        st.markdown('<div class="section-title">Restock Recommendation</div>', unsafe_allow_html=True)

        if will_stockout and stockout_day <= prod_lead:
            st.markdown(f"""
            <div class="alert-critical">
            <strong>⚡ URGENT ORDER REQUIRED</strong><br><br>
            Current stock of <strong>{format_qty(prod_stock, unit_type, unit_measure)}</strong> will run out in
            <strong>{stockout_day} days</strong>, within your <strong>{prod_lead}-day lead time</strong>.<br><br>
            → Order <strong>{format_qty(suggested_qty, unit_type, unit_measure)}</strong> immediately.
            </div>""", unsafe_allow_html=True)
        elif prod_stock < reorder_pt:
            restock_date = datetime.today() + timedelta(days=prod_lead)
            st.markdown(f"""
            <div class="alert-warning">
            <strong>🛒 REORDER NOW</strong><br><br>
            Current stock ({format_qty(prod_stock, unit_type, unit_measure)}) is below reorder point
            ({format_qty(reorder_pt, unit_type, unit_measure)}).<br>
            Order <strong>{format_qty(suggested_qty, unit_type, unit_measure)}</strong> today to receive by
            <strong>{restock_date.strftime('%b %d, %Y')}</strong>.
            </div>""", unsafe_allow_html=True)
        else:
            next_reorder = int(stockout_day) - prod_lead - safety_days
            reorder_date = datetime.today() + timedelta(days=max(next_reorder, 0))
            st.markdown(f"""
            <div class="alert-ok">
            <strong>✅ STOCK ADEQUATE</strong><br><br>
            Next recommended reorder: <strong>{reorder_date.strftime('%b %d, %Y')}</strong>
            (Day {max(next_reorder, 0)}) for <strong>{format_qty(suggested_qty, unit_type, unit_measure)}</strong>.
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)

        insights = []
        if abs(trend_pct) > 5:
            direction = "rising" if trend_pct > 0 else "falling"
            insights.append(f"📈 Demand is <strong>{direction} by {abs(trend_pct):.1f}%</strong> vs last 14-day avg — adjust order quantities.")
        cv = np.std(sales) / np.mean(sales) if np.mean(sales) > 0 else 0
        if cv > 0.3:
            insights.append(f"⚡ High demand volatility (CV={cv:.2f}). Maintain a larger safety buffer.")
        if peaks:
            peak_day = max(peaks, key=lambda x: x[2])
            insights.append(f"🎯 Historically highest sales on <strong>{peak_day[1].strftime('%A')}s</strong>.")
        if will_stockout:
            insights.append(f"⚠️ Stock depleted in <strong>{stockout_day} days</strong> based on forecast.")
        if avg_daily > 0:
            insights.append(f"📦 Current stock covers approximately <strong>{prod_stock / avg_daily:.0f} days</strong> at forecasted demand.")
        if not insights:
            insights.append("✅ No major issues. Monitor weekly and reorder at the reorder point.")
        for ins in insights:
            st.markdown(f'<div class="insight-item">{ins}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Parameter Summary</div>', unsafe_allow_html=True)
        param_df = pd.DataFrame({
            'Parameter': ['Current Stock', 'Avg Daily Demand', 'Reorder Point', 'Safety Stock',
                          'Suggested Order Qty', 'Supplier Lead Time', 'Days to Stockout'],
            'Value': [
                format_qty(prod_stock, unit_type, unit_measure),
                f"{avg_daily:.2f} {unit_measure}/day",
                format_qty(reorder_pt, unit_type, unit_measure),
                format_qty(avg_daily * safety_days, unit_type, unit_measure),
                format_qty(suggested_qty, unit_type, unit_measure),
                f"{prod_lead} days",
                f"{stockout_day} days" if will_stockout else f">{forecast_horizon} days",
            ]
        })
        st.dataframe(param_df, hide_index=True, width="stretch")

    return {
        'Product': product_name,
        'Type': unit_type.upper(),
        'Unit': unit_measure,
        'Urgency': '🔴 CRITICAL' if (will_stockout and stockout_day <= prod_lead)
                   else ('🟡 REORDER' if (will_stockout and stockout_day <= prod_lead + safety_days) or prod_stock < reorder_pt
                         else '🟢 OK'),
        'Current Stock': format_qty(prod_stock, unit_type, unit_measure),
        'Days to Stockout': f"{stockout_day}d" if will_stockout else f">{forecast_horizon}d",
        f'Avg Daily': format_qty(avg_daily, unit_type, unit_measure),
        f'Forecast Total': format_qty(total_forecasted, unit_type, unit_measure),
        'Trend': f"{trend_pct:+.1f}%",
        'Reorder Point': format_qty(reorder_pt, unit_type, unit_measure),
        'Suggested Order': format_qty(suggested_qty, unit_type, unit_measure),
    }


def main():
    st.markdown('<div class="main-header">INVENTORY DEMAND FORECAST</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">raw material prediction · seasonal & holiday awareness · restock alerts · multi-product</div>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙ SETTINGS")

        uploaded_files = st.file_uploader(
            "Upload Inventory CSV(s)",
            type=['csv'],
            accept_multiple_files=True,
            help=(
                "Upload one or more dated CSVs.\n\n"
                "**Wide (recommended)**: date column + one column per material/product.\n\n"
                "**Long**: date, product, quantity columns.\n\n"
                "**Simple**: date, quantity (single material).\n\n"
                "Dates must be in YYYY-MM-DD format. Each row = one day."
            )
        )

        st.markdown("---")
        safety_days = st.slider("Safety Stock Buffer (days)", 1, 14, 5)
        forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30, step=7)
        lead_time = 7  # fallback for CSVs without a lead_time_days column

        st.markdown("---")
        st.markdown('<div style="color:#888;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em">Sample CSV</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Sample CSV",
            data=make_stock_sample_csv(),
            file_name="wheelchair_factory_stock.csv",
            mime="text/csv",
            help="365-day stock-aware CSV · 12 wheelchair raw materials · daily consumed, restocked & running stock balance",
        )

        run = st.button("🚀 Run Analysis", type="primary")

    # ── Welcome screen ───────────────────────────────────────────────────────
    if not uploaded_files:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class="metric-card">
            <div class="metric-label">Step 1</div><div style="font-size:1.8rem">📁</div>
            <div style="color:#ccc;margin-top:0.5rem">Upload a <strong>dated CSV</strong> — one row per day, columns for each raw material or product</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="metric-card">
            <div class="metric-label">Step 2</div><div style="font-size:1.8rem">⚙️</div>
            <div style="color:#ccc;margin-top:0.5rem">Select materials to forecast — filter by type (💧 liquid · ⚖️ weight · 📦 unit)</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="metric-card">
            <div class="metric-label">Step 3</div><div style="font-size:1.8rem">📊</div>
            <div style="color:#ccc;margin-top:0.5rem">AI demand forecast with <strong>holiday markers</strong>, seasonal trends, restock alerts, and peak detection</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Accepted CSV Formats — All Must Have Dates</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Stock (long) ✨ — recommended**")
            st.code("date,product,unit_type,unit_measure,consumed,restocked,stock_balance\n2025-01-02,Steel Tubing,weight,kg,107.1,0,2778.6\n2025-01-03,Steel Tubing,weight,kg,97.4,0,2681.2\n2025-01-10,Steel Tubing,weight,kg,95.2,3000,5585.8", language='text')
        with c2:
            st.markdown("**Wide — one row per day**")
            st.code("date, Steel Tubing (kg), Rear Wheels\n2025-01-02, 107.1, 26\n2025-01-03, 97.4, 24\n2025-01-04, 42.0, 11", language='text')
        with c3:
            st.markdown("**Long format**")
            st.code("date, material, quantity\n2025-01-02, Steel Tubing, 107.1\n2025-01-02, Rear Wheels, 26", language='text')
        with c4:
            st.markdown("**Simple — single material**")
            st.code("date, quantity\n2025-01-02, 107.1\n2025-01-03, 97.4", language='text')

        st.info("💡 Download the **Sample CSV** from the sidebar — 365 days of daily consumption, restock events, and running stock balance for 12 wheelchair raw materials, ready to upload.")
        return

    # ── Parse all uploaded CSVs ───────────────────────────────────────────────
    all_products = {}
    parse_errors = []

    for uploaded in uploaded_files:
        try:
            df = pd.read_csv(uploaded)
            if is_inventory_catalog(df):
                catalog = parse_inventory_catalog(df)
                for name, pdata in catalog.items():
                    key = name if name not in all_products else f"{uploaded.name}: {name}"
                    all_products[key] = pdata
            else:
                fmt, info = detect_csv_format(df)
                if fmt == 'unknown':
                    parse_errors.append(f"⚠️ Could not parse **{uploaded.name}** — ensure it has a date column and numeric sales columns.")
                    continue
                products = parse_to_products(df, fmt, info)
                for name, pdata in products.items():
                    key = name if name not in all_products else f"{uploaded.name}: {name}"
                    # pdata is always a dict with at least {'df': DataFrame}
                    all_products[key] = {
                        'df': pdata['df'],
                        'unit_type': pdata.get('unit_type', 'unit'),
                        'unit_measure': pdata.get('unit_measure', 'units'),
                        'current_stock': pdata.get('current_stock', 0.0),
                        'reorder_point': 0.0,
                        'lead_time_days': lead_time,
                        'category': '',
                        'supplier_id': '',
                        'unit_cost': 0.0,
                        'selling_price': 0.0,
                        'avg_monthly_sales': 0.0,
                    }
        except Exception as e:
            parse_errors.append(f"⚠️ **{uploaded.name}**: {e}")

    for err in parse_errors:
        st.error(err)

    if not all_products:
        st.error("No products could be loaded. Please check your CSV files.")
        return

    # ── Product selection ─────────────────────────────────────────────────────
    product_names = sorted(all_products.keys())

    filter_col, select_col = st.columns([1, 3])
    with filter_col:
        filter_unit = st.radio("Filter by type", ["All", "💧 Liquid", "⚖️ Weight", "📦 Unit"], horizontal=False)

    filter_map = {"All": None, "💧 Liquid": "liquid", "⚖️ Weight": "weight", "📦 Unit": "unit"}
    active_filter = filter_map[filter_unit]
    filtered_names = [p for p in product_names
                      if active_filter is None or all_products[p].get('unit_type', 'unit') == active_filter]

    with select_col:
        selected_products = st.multiselect(
            f"Select Products to Analyze  ({len(filtered_names)} available / {len(product_names)} total)",
            filtered_names,
            default=filtered_names[:min(5, len(filtered_names))],
            help="Select one or more products. Use the type filter on the left to narrow the list."
        )

    if not selected_products:
        st.info("Select at least one product from the list above.")
        return

    # ── Preview (before run) ─────────────────────────────────────────────────
    if not run:
        preview_rows = []
        for name in selected_products:
            d = all_products[name]
            ut, um = d.get('unit_type', 'unit'), d.get('unit_measure', 'units')
            preview_rows.append({
                'Product': name,
                'Type': f"{unit_emoji(ut)} {ut.upper()}",
                'Unit': um,
                'Category': d.get('category', '—'),
                'Current Stock': format_qty(d.get('current_stock', 0.0), ut, um),
                'Lead Time': f"{d.get('lead_time_days', lead_time)}d",
                'Data Points': len(d['df']),
            })
        st.info(
            f"✓ Loaded **{len(all_products)}** products from **{len(uploaded_files)}** file(s). "
            f"**{len(selected_products)}** selected. Click **Run Analysis** to forecast."
        )
        st.dataframe(pd.DataFrame(preview_rows), hide_index=True, width="stretch")
        with st.expander("Raw data preview (first selected product)"):
            st.dataframe(all_products[selected_products[0]]['df'].head(20), width="stretch")
        return

    # ── Run analysis ─────────────────────────────────────────────────────────
    summary_rows = []

    for product_name in selected_products:
        data = all_products[product_name]
        ut = data.get('unit_type', 'unit')
        em = unit_emoji(ut)
        cat = data.get('category', '')
        label = f"{em} **{product_name}**" + (f"  ·  {cat}" if cat else "")

        with st.expander(label, expanded=(len(selected_products) == 1)):
            row = _run_product_analysis(product_name, data, lead_time, safety_days, forecast_horizon)
            summary_rows.append(row)

        st.markdown("")

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(selected_products) > 1:
        st.markdown("---")
        st.markdown('<div class="section-title">Analysis Summary — All Selected Products</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
