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
    Auto-detect if CSV is:
      - 'wide': product names as columns, rows = dates/days
      - 'long': columns like [date, product, sales] or [date, sales] for a single product
      - 'simple': just [date/day, sales] two-column
    Returns: (format_type, info_dict)
    """
    cols = [c.strip().lower() for c in df.columns]
    col_map = {c.strip().lower(): c for c in df.columns}

    date_keys = ['date', 'day', 'week', 'period', 'time', 'timestamp']
    sales_keys = ['sales', 'quantity', 'qty', 'units', 'demand', 'sold', 'revenue']
    product_keys = ['product', 'item', 'sku', 'name']
    stock_keys = ['stock', 'inventory', 'current_stock', 'on_hand']

    has_date = any(k in cols for k in date_keys)
    has_product = any(k in cols for k in product_keys)
    has_sales = any(k in cols for k in sales_keys)
    has_stock = any(k in cols for k in stock_keys)

    date_col = next((col_map[k] for k in date_keys if k in cols), None)
    product_col = next((col_map[k] for k in product_keys if k in cols), None)
    sales_col = next((col_map[k] for k in sales_keys if k in cols), None)
    stock_col = next((col_map[k] for k in stock_keys if k in cols), None)

    # Wide format: first col is date, rest are product names
    if has_date and not has_product and not has_sales:
        numeric_cols = [c for c in df.columns if c != date_col and pd.to_numeric(df[c], errors='coerce').notna().sum() > len(df) * 0.5]
        if numeric_cols:
            return 'wide', {
                'date_col': date_col,
                'product_cols': numeric_cols,
                'stock_col': stock_col
            }

    # Long format with product column
    if has_date and has_product and has_sales:
        return 'long', {
            'date_col': date_col,
            'product_col': product_col,
            'sales_col': sales_col,
            'stock_col': stock_col
        }

    # Simple two-column: date + sales (single product)
    if has_date and has_sales:
        return 'simple', {
            'date_col': date_col,
            'sales_col': sales_col,
            'stock_col': stock_col
        }

    # Last resort: numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 1:
        return 'numeric_only', {
            'sales_col': numeric_cols[0],
            'product_cols': numeric_cols
        }

    return 'unknown', {}


def parse_to_products(df, fmt, info, current_stock_override=None):
    """
    Parse any CSV format into a dict of {product_name: DataFrame(date, sales)}
    """
    products = {}

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
            products[pcol] = sub

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
            products[str(prod)] = sub

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
        # Extract product name from any name-like column
        name_keys = ['product', 'item', 'sku', 'name', 'product_name', 'item_name']
        product_name_col = next((c for c in df.columns if c.strip().lower() in name_keys), None)
        if product_name_col:
            non_null = df[product_name_col].dropna()
            product_label = str(non_null.iloc[0]) if not non_null.empty else 'Product'
        else:
            product_label = 'Product'
        products[product_label] = sub

    elif fmt == 'numeric_only':
        for col in info['product_cols']:
            sub = pd.DataFrame({
                'date': pd.date_range(end=datetime.today(), periods=len(df), freq='D'),
                'sales': pd.to_numeric(df[col], errors='coerce').fillna(0)
            })
            products[col] = sub

    return products


# ─── FORECASTING ENGINE — Salesforce Moirai-1.0-R-Base ──────────────────────

try:
    from gluonts.dataset.pandas import PandasDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    MOIRAI_AVAILABLE = True
except ImportError:
    MOIRAI_AVAILABLE = False


@st.cache_resource(show_spinner="Loading Moirai-1.0-R-Large model (downloads once)...")
def load_moirai_model():
    """Download & cache Moirai-1.0-R-Large from Hugging Face. ~311M params."""
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-large")
    return module


def _statistical_fallback(sales: np.ndarray, horizon: int):
    """Simple fallback when uni2ts is not installed."""
    n = len(sales)
    alpha = 0.3
    s = sales[0]
    for v in sales:
        s = alpha * v + (1 - alpha) * s
    window = min(14, n)
    slope = np.polyfit(np.arange(window), sales[-window:], 1)[0]
    std = np.std(sales[-window:])
    preds = np.array([max(s + slope * (i + 1), 0) for i in range(horizon)])
    return preds, np.maximum(preds - 1.5 * std, 0), preds + 1.5 * std


def forecast_demand(sales: np.ndarray, horizon: int = 30, freq: str = "D"):
    """
    Run Salesforce Moirai-1.0-R-Base zero-shot forecast.
    Returns (point_forecast, lower_bound, upper_bound) as numpy arrays.
    Falls back to statistical model if uni2ts is not installed.
    """
    n = len(sales)

    # Need at least 2 points
    if n < 2:
        flat = np.full(horizon, float(sales[0]) if n == 1 else 1.0)
        return flat, flat * 0.8, flat * 1.2

    if not MOIRAI_AVAILABLE:
        st.warning(
            "⚠️ `uni2ts` library not found — using statistical fallback. "
            "Install with: `pip install uni2ts gluonts`",
            icon="⚠️"
        )
        return _statistical_fallback(sales, horizon)

    try:
        module = load_moirai_model()

        # Context length: Moirai works best with ≥ 32 points; cap at 512
        ctx_len = min(max(len(sales), 32), 512)

        # Pad with the series mean if we have fewer points than ctx_len
        if len(sales) < ctx_len:
            pad = np.full(ctx_len - len(sales), np.mean(sales))
            sales_ctx = np.concatenate([pad, sales])
        else:
            sales_ctx = sales[-ctx_len:]

        # Build a minimal GluonTS PandasDataset from the context window
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=ctx_len, freq=freq)
        df_ctx = pd.DataFrame({"target": sales_ctx.astype(float)}, index=idx)
        ds = PandasDataset({"series": df_ctx}, target="target", freq=freq)

        # Instantiate MoiraiForecast for this prediction length
        model = MoiraiForecast(
            module=module,
            prediction_length=horizon,
            context_length=ctx_len,
            patch_size="auto",
            num_samples=100,          # 100 Monte-Carlo samples → rich uncertainty
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        predictor = model.create_predictor(batch_size=1)
        forecast_it = predictor.predict(ds)
        fc = next(iter(forecast_it))

        # fc.samples shape: (num_samples, horizon)
        samples = fc.samples                        # (100, horizon)
        point   = samples.mean(axis=0)             # median or mean — mean is smoother
        lo      = np.percentile(samples, 10, axis=0)   # 10th percentile
        hi      = np.percentile(samples, 90, axis=0)   # 90th percentile

        return (
            np.maximum(point, 0),
            np.maximum(lo, 0),
            np.maximum(hi, 0),
        )

    except Exception as e:
        st.warning(f"Moirai inference failed ({e}) — falling back to statistical model.")
        return _statistical_fallback(sales, horizon)


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


# ─── PLOTS ───────────────────────────────────────────────────────────────────

def plot_sales_and_forecast(hist_df, forecast, lo, hi, product_name, unit_measure="units"):
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

    layout = dict(**_LAYOUT)
    layout.update(
        title=dict(text=f'{product_name} — Sales History & Forecast',
                   font=dict(color='#e8e8f0', size=13)),
        yaxis=dict(**_LAYOUT['yaxis'],
                   title=dict(text=f'Sold ({unit_measure})', font=dict(color='#888'))),
        height=420,
    )
    fig.update_layout(**layout)
    return fig


def plot_inventory_projection(current_stock, forecast, lead_time, safety_days, avg_daily, unit_measure="units"):
    stock_proj = [current_stock]
    for d in forecast:
        stock_proj.append(max(stock_proj[-1] - d, 0))

    proj_dates = pd.date_range(start=datetime.today(), periods=len(stock_proj), freq='D')
    reorder_pt  = avg_daily * (lead_time + safety_days)
    safety_stock = avg_daily * safety_days

    fig = go.Figure()

    # Danger-zone shading
    fig.add_hrect(y0=0, y1=safety_stock, fillcolor='rgba(255,68,68,0.06)', line_width=0)

    # Projected stock
    fig.add_trace(go.Scatter(
        x=proj_dates, y=stock_proj,
        mode='lines',
        name='Projected Stock',
        line=dict(color='#f0a500', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(240,165,0,0.08)',
        hovertemplate=(
            '<b>%{x|%b %d, %Y}</b>  %{x|%A}<br>'
            f'Stock: <b>%{{y:.1f}} {unit_measure}</b><extra></extra>'
        ),
    ))

    # Reorder point
    fig.add_hline(y=reorder_pt, line_dash='dash', line_color='#ff6b6b', line_width=1.5)
    fig.add_annotation(
        x=0, xref='paper', y=reorder_pt,
        text=f'Reorder Point  {reorder_pt:.1f} {unit_measure}',
        font=dict(color='#ff6b6b', size=10), showarrow=False,
        xanchor='left', yanchor='bottom',
    )

    # Safety stock
    fig.add_hline(y=safety_stock, line_dash='dot', line_color='#ff4444', line_width=1.5)
    fig.add_annotation(
        x=0, xref='paper', y=safety_stock,
        text=f'Safety Stock  {safety_stock:.1f} {unit_measure}',
        font=dict(color='#ff4444', size=10), showarrow=False,
        xanchor='left', yanchor='top',
    )

    # Stockout marker
    stockout_day, will_stockout = days_until_stockout(current_stock, forecast)
    if will_stockout:
        stockout_date = datetime.today() + timedelta(days=int(stockout_day))
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
        height=380,
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
    """Format a quantity with its unit label."""
    ut = (unit_type or 'unit').lower()
    um = unit_measure or 'units'
    if ut == 'liquid':
        if any(s in um.lower() for s in ['ml', 'oz', 'fl']):
            return f"{value:.0f} {um}"
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

def make_sample_csv():
    dates = pd.date_range(end=datetime.today(), periods=60, freq='D')
    np.random.seed(42)
    base = 25
    sales_a = np.maximum(base + np.random.normal(0, 5, 60) + np.sin(np.arange(60) * 0.3) * 8, 0).astype(int)
    sales_b = np.maximum(15 + np.random.normal(0, 4, 60) + np.linspace(0, 10, 60), 0).astype(int)
    df = pd.DataFrame({'date': dates, 'Widget A': sales_a, 'Widget B': sales_b})
    return df.to_csv(index=False)


def make_inventory_sample_csv():
    """Return the bundled sample inventory catalog CSV."""
    import os
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_store_inventory.csv')
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            return f.read()
    # Minimal fallback
    return (
        "item_id,item_name,category,supplier_id,unit_cost,selling_price,current_stock,"
        "reorder_point,lead_time_days,pre_sales_week1,pre_sales_week2,pre_sales_week3,"
        "pre_sales_week4,avg_monthly_sales,seasonal_factor,stock_turnover_rate,unit_type,unit_measure\n"
        "LQ001,Whole Milk,Dairy,SUP011,1.20,2.49,320,80,2,210,198,225,215,848,1.10,8.50,liquid,L\n"
        "WG001,Chicken Breast,Meat & Poultry,SUP031,4.20,8.99,85,25,1,52,48,55,50,205,1.05,7.25,weight,kg\n"
        "UN001,Canned Tomatoes (400g),Canned Goods,SUP061,0.55,1.29,350,90,7,185,178,192,188,743,0.95,6.36,unit,each\n"
    )


# ─── MAIN ────────────────────────────────────────────────────────────────────

def _run_product_analysis(product_name, data, default_stock, lead_time, safety_days, forecast_horizon):
    """Render full analysis for a single product inside an expander."""
    hist_df = data['df']
    unit_type = data.get('unit_type', 'unit')
    unit_measure = data.get('unit_measure', 'units')
    prod_stock = data.get('current_stock', float(default_stock))
    prod_lead = int(data.get('lead_time_days', lead_time))
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

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📦 Inventory", "📅 Peaks", "💡 Recommendations"])

    with tab1:
        fig = plot_sales_and_forecast(hist_df, forecast, lo, hi, product_name, unit_measure)
        st.plotly_chart(fig, width='stretch')

        st.markdown('<div class="section-title">Day-by-Day Forecast</div>', unsafe_allow_html=True)
        pred_dates = pd.date_range(start=hist_df['date'].iloc[-1] + timedelta(days=1), periods=forecast_horizon, freq='D')
        forecast_table = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in pred_dates],
            'Day': [d.strftime('%A') for d in pred_dates],
            f'Forecast ({unit_measure})': [f"{v:.2f}" for v in forecast],
            'Low Estimate': [f"{v:.2f}" for v in lo],
            'High Estimate': [f"{v:.2f}" for v in hi],
        })
        st.dataframe(forecast_table, hide_index=True, use_container_width=True, height=350)

    with tab2:
        fig2 = plot_inventory_projection(prod_stock, forecast, prod_lead, safety_days, avg_daily, unit_measure)
        st.plotly_chart(fig2, width='stretch')

        st.markdown('<div class="section-title">Inventory Milestones</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Reorder Point", format_qty(reorder_pt, unit_type, unit_measure))
        with c2:
            st.metric("Safety Stock", format_qty(avg_daily * safety_days, unit_type, unit_measure))
        with c3:
            st.metric("Suggested Order Qty", format_qty(suggested_qty, unit_type, unit_measure))

    with tab3:
        st.markdown('<div class="section-title">Weekly Pattern</div>', unsafe_allow_html=True)
        if len(sales) >= 7:
            fig3 = plot_weekly_heatmap(sales, dates)
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("Need at least 7 days of data for weekly patterns.")

        st.markdown('<div class="section-title">Historical Peak Days (Top 25%)</div>', unsafe_allow_html=True)
        peaks = detect_peaks(sales, dates)
        if peaks:
            peaks_sorted = sorted(peaks, key=lambda x: -x[2])
            peak_df = pd.DataFrame([{
                'Date': p[1].strftime('%Y-%m-%d'),
                'Day': p[1].strftime('%A'),
                f'Sales ({unit_measure})': f"{p[2]:.2f}",
                'vs Avg': f"+{((p[2] / np.mean(sales) - 1) * 100):.0f}%"
            } for p in peaks_sorted[:20]])
            st.dataframe(peak_df, hide_index=True, use_container_width=True)

        st.markdown('<div class="section-title">Forecasted Peak Days</div>', unsafe_allow_html=True)
        pred_dates_list = pd.date_range(start=hist_df['date'].iloc[-1] + timedelta(days=1), periods=forecast_horizon, freq='D')
        threshold = np.percentile(forecast, 75)
        forecast_peaks = [(d, v) for d, v in zip(pred_dates_list, forecast) if v >= threshold]
        if forecast_peaks:
            fp_df = pd.DataFrame([{
                'Date': d.strftime('%Y-%m-%d'),
                'Day': d.strftime('%A'),
                f'Predicted ({unit_measure})': f"{v:.2f}",
            } for d, v in sorted(forecast_peaks, key=lambda x: -x[1])[:15]])
            st.dataframe(fp_df, hide_index=True, use_container_width=True)

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
        st.dataframe(param_df, hide_index=True, use_container_width=True)

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
    st.markdown('<div class="main-header">INVENTORY PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">demand forecasting · restock prediction · peak detection · multi-product</div>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙ SETTINGS")

        uploaded_files = st.file_uploader(
            "Upload Sales CSV(s)",
            type=['csv'],
            accept_multiple_files=True,
            help=(
                "Upload one or more CSVs.\n\n"
                "**Inventory catalog**: item_name, unit_type, unit_measure, pre_sales_week1-4, current_stock …\n\n"
                "**Time-series**: wide (date + product columns), long (date, product, sales), or simple (date, sales)"
            )
        )

        st.markdown("---")
        default_stock = st.number_input("Default Current Stock", min_value=0, value=200, step=10,
                                        help="Used for time-series CSVs without embedded stock data")
        lead_time = st.slider("Default Lead Time (days)", 1, 30, 7)
        safety_days = st.slider("Safety Stock Buffer (days)", 1, 14, 5)
        forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30, step=7)

        st.markdown("---")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            st.download_button("📥 Time-Series CSV", data=make_sample_csv(),
                               file_name="sample_timeseries.csv", mime="text/csv")
        with c2:
            st.download_button("📥 Inventory CSV", data=make_inventory_sample_csv(),
                               file_name="sample_inventory_catalog.csv", mime="text/csv")

        run = st.button("🚀 Run Analysis", type="primary")

    # ── Welcome screen ───────────────────────────────────────────────────────
    if not uploaded_files:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class="metric-card">
            <div class="metric-label">Step 1</div><div style="font-size:1.8rem">📁</div>
            <div style="color:#ccc;margin-top:0.5rem">Upload one or more CSVs — time-series or inventory catalog with liquid/weight/unit items</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="metric-card">
            <div class="metric-label">Step 2</div><div style="font-size:1.8rem">⚙️</div>
            <div style="color:#ccc;margin-top:0.5rem">Select products to analyze — filter by type (💧 liquid · ⚖️ weight · 📦 unit)</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="metric-card">
            <div class="metric-label">Step 3</div><div style="font-size:1.8rem">📊</div>
            <div style="color:#ccc;margin-top:0.5rem">Get unit-aware forecasts, restock alerts, trend analysis and peak detection</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Accepted CSV Formats</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Inventory Catalog** *(recommended)*")
            st.code("item_name, unit_type, unit_measure,\ncurrent_stock, pre_sales_week1..4\nMilk, liquid, L, 320, 210...", language='text')
        with c2:
            st.markdown("**Wide (multi-product)**")
            st.code("date, Widget A, Widget B\n2024-01-01, 30, 12\n2024-01-02, 28, 15", language='text')
        with c3:
            st.markdown("**Long format**")
            st.code("date, product, sales\n2024-01-01, Widget A, 30\n2024-01-01, Widget B, 12", language='text')
        with c4:
            st.markdown("**Simple (single product)**")
            st.code("date, sales\n2024-01-01, 30\n2024-01-02, 28", language='text')

        st.info("💡 Download the **Inventory CSV** sample in the sidebar — it includes 39 real-world products across liquid, weight, and unit categories.")
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
                for name, hist_df in products.items():
                    key = name if name not in all_products else f"{uploaded.name}: {name}"
                    all_products[key] = {
                        'df': hist_df,
                        'unit_type': 'unit',
                        'unit_measure': 'units',
                        'current_stock': float(default_stock),
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
                'Current Stock': format_qty(d.get('current_stock', default_stock), ut, um),
                'Lead Time': f"{d.get('lead_time_days', lead_time)}d",
                'Data Points': len(d['df']),
            })
        st.info(
            f"✓ Loaded **{len(all_products)}** products from **{len(uploaded_files)}** file(s). "
            f"**{len(selected_products)}** selected. Click **Run Analysis** to forecast."
        )
        st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)
        with st.expander("Raw data preview (first selected product)"):
            st.dataframe(all_products[selected_products[0]]['df'].head(20), use_container_width=True)
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
            row = _run_product_analysis(product_name, data, default_stock, lead_time, safety_days, forecast_horizon)
            summary_rows.append(row)

        st.markdown("")

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(selected_products) > 1:
        st.markdown("---")
        st.markdown('<div class="section-title">Analysis Summary — All Selected Products</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
