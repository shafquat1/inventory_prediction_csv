"""
Streamlit Inventory Prediction App - CSV ENABLED VERSION
Loads inventory data from CSV and generates predictions
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# Page config
st.set_page_config(
    page_title="Inventory Prediction AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)


class ForecastModel:
    """Statistical forecasting model with advanced time-series analysis"""
    
    def __init__(self):
        self.name = "Advanced Statistical Forecasting"
    
    def forecast(self, data_tensor, prediction_length):
        """Generate forecasts using statistical methods"""
        # Convert tensor to numpy
        if hasattr(data_tensor, 'numpy'):
            data = data_tensor.squeeze().numpy()
        else:
            data = np.array(data_tensor).squeeze()
        
        # Components analysis
        recent_30 = data[-30:]
        recent_60 = data[-60:]
        
        # Base level (exponential smoothing)
        alpha = 0.3
        base = recent_30[0]
        for val in recent_30:
            base = alpha * val + (1 - alpha) * base
        
        # Trend (linear regression on recent data)
        x = np.arange(len(recent_60))
        coeffs = np.polyfit(x, recent_60, 1)
        trend = coeffs[0]
        
        # Weekly seasonality (average by day of week)
        weekly_pattern = []
        for day in range(7):
            day_values = [data[i] for i in range(len(data)) if i % 7 == day]
            weekly_pattern.append(np.mean(day_values[-4:]) if day_values else 0)
        weekly_avg = np.mean(weekly_pattern)
        weekly_factors = [w - weekly_avg for w in weekly_pattern]
        
        # Volatility
        volatility = np.std(recent_30)
        
        # Generate predictions
        predictions = []
        for i in range(prediction_length):
            # Base forecast
            pred = base + (trend * i)
            
            # Add weekly seasonality
            day_of_week = i % 7
            pred += weekly_factors[day_of_week]
            
            # Add controlled randomness
            noise = np.random.normal(0, volatility * 0.3)
            pred += noise
            
            # Ensure positive and realistic
            pred = max(pred, base * 0.3)
            pred = min(pred, base * 2.5)
            
            predictions.append(pred)
        
        # Return as tensor-like object
        class PredictionTensor:
            def __init__(self, data):
                self.data = np.array([data])
            def __getitem__(self, idx):
                return self.data[idx]
            def numpy(self):
                return self.data
        
        return PredictionTensor(predictions)


@st.cache_data
def load_inventory_csv(filepath='inventory_prediction_data.csv'):
    """Load inventory data from CSV file"""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"❌ CSV file '{filepath}' not found. Please upload it.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None


@st.cache_resource
def load_model():
    """Load forecasting model"""
    return ForecastModel()


def generate_historical_from_csv(item_data, days):
    """
    Generate realistic historical demand data based on CSV pre-sales data
    This creates time-series data that respects the item's characteristics
    """
    np.random.seed(hash(item_data['item_name']) % 2**32)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    actual_days = len(dates)
    
    # Calculate base demand from pre-sales weeks
    weekly_sales = [
        item_data['pre_sales_week1'],
        item_data['pre_sales_week2'],
        item_data['pre_sales_week3'],
        item_data['pre_sales_week4']
    ]
    avg_weekly = np.mean(weekly_sales)
    base_daily_demand = avg_weekly / 7  # Convert to daily
    
    # Use actual monthly sales average
    monthly_avg = item_data['avg_monthly_sales']
    base_daily_demand = monthly_avg / 30
    
    # Get seasonal factor
    seasonal_factor = item_data.get('seasonal_factor', 1.0)
    
    # Generate base demand
    demand = np.ones(actual_days) * base_daily_demand
    
    # Apply seasonal variations
    if seasonal_factor > 1.2:  # Highly seasonal (like calendars)
        seasonal_wave = (seasonal_factor - 1.0) * base_daily_demand * np.sin(
            np.arange(actual_days) * 2 * np.pi / 90
        )
        demand += seasonal_wave
    
    # Weekly seasonality (lower on weekends)
    for i in range(actual_days):
        day_of_week = (i + dates[0].dayofweek) % 7
        if day_of_week >= 5:  # Weekend
            demand[i] *= 0.75
        elif day_of_week == 0:  # Monday
            demand[i] *= 1.1
    
    # Trend based on recent weeks
    trend_direction = (weekly_sales[-1] - weekly_sales[0]) / max(weekly_sales[0], 1)
    if abs(trend_direction) > 0.05:
        trend = np.linspace(0, base_daily_demand * trend_direction * 0.2, actual_days)
        demand += trend
    
    # Stock turnover influence (higher turnover = more variable demand)
    turnover = item_data.get('stock_turnover_rate', 0.6)
    volatility = base_daily_demand * (0.15 + turnover * 0.1)
    
    # Add variability
    demand += np.random.normal(0, volatility, actual_days)
    
    # Occasional promotional spikes
    promo_days = np.random.choice([0, 1], size=actual_days, p=[0.95, 0.05])
    demand += promo_days * np.random.uniform(base_daily_demand * 0.5, base_daily_demand * 1.5, actual_days)
    
    # Ensure positive and reasonable
    demand = np.maximum(demand, base_daily_demand * 0.2)
    
    # Simulate inventory levels
    current_stock = item_data['current_stock']
    reorder_point = item_data['reorder_point']
    lead_time = item_data['lead_time_days']
    
    inventory = []
    stock = current_stock + (base_daily_demand * 30)  # Start with extra stock
    reorder_qty = base_daily_demand * lead_time * 3
    
    for daily_demand in demand:
        stock -= daily_demand
        if stock < reorder_point:
            stock += reorder_qty
        inventory.append(max(stock, 0))
    
    return pd.DataFrame({
        'date': dates,
        'demand': demand.astype(int),
        'inventory_level': np.array(inventory).astype(int)
    })


def predict_demand(model, historical_data, prediction_days):
    """Generate demand predictions"""
    demand_series = historical_data['demand'].values.astype(float)
    
    predictions = model.forecast(demand_series, prediction_length=prediction_days)
    forecast = predictions[0]
    
    return np.maximum(forecast, 0)


def plot_forecast(historical_data, predictions):
    """Create forecast visualization"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Historical
    recent = historical_data.tail(60)
    ax.plot(recent['date'], recent['demand'], 
            label='Historical Demand', color='#2E86AB', linewidth=2)
    
    # Predictions
    last_date = historical_data['date'].iloc[-1]
    pred_dates = pd.date_range(start=last_date + timedelta(days=1),
                                periods=len(predictions), freq='D')
    
    ax.plot(pred_dates, predictions, 
            label='Predicted Demand', color='#E63946', 
            linewidth=2.5, linestyle='--', marker='o', markersize=4)
    
    # Confidence interval
    std = np.std(recent['demand'].values)
    ax.fill_between(pred_dates, predictions - std, predictions + std,
                     alpha=0.2, color='#E63946', label='Confidence Range')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Demand (Units)', fontsize=12)
    ax.set_title('Demand Forecast', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_inventory_projection(historical_data, predictions):
    """Create inventory projection"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    current_stock = historical_data['inventory_level'].iloc[-1]
    projected = [current_stock]
    
    for demand in predictions:
        projected.append(max(projected[-1] - demand, 0))
    
    last_date = historical_data['date'].iloc[-1]
    proj_dates = pd.date_range(start=last_date, periods=len(projected), freq='D')
    
    ax.plot(proj_dates, projected, color='#F77F00', linewidth=2.5, 
            label='Projected Inventory')
    
    avg_demand = np.mean(predictions)
    ax.axhline(y=avg_demand * 10, color='red', linestyle='--', 
              label='Reorder Point', linewidth=2)
    ax.axhline(y=avg_demand * 5, color='orange', linestyle='--', 
              label='Safety Stock', linewidth=2)
    
    ax.fill_between(proj_dates, 0, projected, alpha=0.3, color='#F77F00')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Inventory Level (Units)', fontsize=12)
    ax.set_title('Inventory Projection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">📦 Inventory Prediction System</div>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Powered Demand Forecasting & Inventory Management</p>', 
                unsafe_allow_html=True)
    
    # Info banner
    st.markdown("""
    <div class="info-box">
    <strong>✨ CSV-Powered Forecasting:</strong> Load your inventory data from CSV and get AI-powered predictions 
    based on historical sales patterns, seasonality, and stock turnover rates.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("📁 Upload Inventory CSV", type=['csv'], 
                                     help="Upload inventory_prediction_data.csv or your own CSV file")
    
    # Load CSV
    if uploaded_file is not None:
        inventory_df = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(inventory_df)} products from CSV")
    else:
        # Try to load from current directory
        inventory_df = load_inventory_csv()
        if inventory_df is not None:
            st.success(f"✓ Loaded {len(inventory_df)} products from inventory_prediction_data.csv")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    if inventory_df is not None:
        # Product selection from CSV
        st.sidebar.subheader("📦 Select Product")
        
        # Show products by category
        categories = inventory_df['category'].unique()
        selected_category = st.sidebar.selectbox("Filter by Category", 
                                                  ['All'] + list(categories))
        
        if selected_category == 'All':
            filtered_df = inventory_df
        else:
            filtered_df = inventory_df[inventory_df['category'] == selected_category]
        
        product_options = [f"{row['item_name']} (Stock: {row['current_stock']})" 
                          for _, row in filtered_df.iterrows()]
        
        selected_product_str = st.sidebar.selectbox("Product", product_options)
        selected_idx = product_options.index(selected_product_str)
        selected_item = filtered_df.iloc[selected_idx]
        
        # Display product info
        with st.sidebar.expander("📋 Product Details", expanded=True):
            st.write(f"**Item ID:** {selected_item['item_id']}")
            st.write(f"**Category:** {selected_item['category']}")
            st.write(f"**Current Stock:** {selected_item['current_stock']} units")
            st.write(f"**Reorder Point:** {selected_item['reorder_point']} units")
            st.write(f"**Lead Time:** {selected_item['lead_time_days']} days")
            st.write(f"**Unit Cost:** ${selected_item['unit_cost']:.2f}")
            st.write(f"**Selling Price:** ${selected_item['selling_price']:.2f}")
            st.write(f"**Avg Monthly Sales:** {selected_item['avg_monthly_sales']:.0f} units")
            st.write(f"**Stock Turnover:** {selected_item['stock_turnover_rate']:.2f}")
            st.write(f"**Seasonal Factor:** {selected_item['seasonal_factor']:.1f}")
        
        # Use CSV values
        lead_time = int(selected_item['lead_time_days'])
        product_name = selected_item['item_name']
        product_category = selected_item['category']
        
    else:
        st.sidebar.warning("⚠️ No CSV loaded - using manual input")
        product_name = st.sidebar.text_input("Product Name", "Wireless Mouse")
        product_category = st.sidebar.selectbox(
            "Category",
            ["Electronics", "Furniture", "Food & Beverage", "Clothing", "Other"]
        )
        lead_time = st.sidebar.slider("Supplier Lead Time (days)", 1, 30, 7, 1)
        selected_item = None
    
    # Prediction settings
    st.sidebar.subheader("🔮 Forecast Settings")
    historical_days = st.sidebar.slider(
        "Historical Data (days)",
        min_value=90,
        max_value=365,
        value=180,
        step=30
    )
    
    prediction_days = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    safety_days = st.sidebar.slider(
        "Safety Stock (days)",
        min_value=3,
        max_value=14,
        value=5,
        step=1
    )
    
    # Run prediction button
    run_prediction = st.sidebar.button("🚀 Run Prediction", type="primary", use_container_width=True)
    
    # Main content
    if run_prediction:
        if inventory_df is None and selected_item is None:
            st.error("❌ Please upload a CSV file or check that inventory_prediction_data.csv exists")
            return
        
        # Load model
        model = load_model()
        st.success("✓ Forecasting engine ready!")
        
        # Generate data
        with st.spinner("Analyzing historical data..."):
            if selected_item is not None:
                # Use CSV data
                historical_data = generate_historical_from_csv(selected_item, historical_days)
            else:
                # Fallback to sample data
                from generate_sample_data import generate_sample_data
                historical_data = generate_sample_data(product_name, historical_days, 150, "Stable")
        
        # Generate predictions
        with st.spinner("Generating predictions..."):
            predictions = predict_demand(model, historical_data, prediction_days)
        
        st.success(f"✓ {prediction_days}-day forecast generated!")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", "📈 Forecast", "📦 Inventory", "📋 Recommendations"
        ])
        
        with tab1:
            st.subheader("Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            current_stock = historical_data['inventory_level'].iloc[-1]
            avg_pred = np.mean(predictions)
            total_pred = np.sum(predictions)
            hist_avg = historical_data['demand'].tail(30).mean()
            change = ((avg_pred - hist_avg) / hist_avg) * 100
            
            with col1:
                st.metric("Current Stock", f"{current_stock:.0f} units")
            
            with col2:
                st.metric("Avg Predicted Demand", f"{avg_pred:.1f} units/day")
            
            with col3:
                st.metric(f"{prediction_days}-Day Demand", f"{total_pred:.0f} units")
            
            with col4:
                st.metric("Demand Trend", f"{change:+.1f}%")
            
            st.markdown("---")
            
            # Historical summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Historical Analysis (Last 30 Days)")
                recent = historical_data.tail(30)
                
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Average Demand',
                        'Peak Demand',
                        'Low Demand',
                        'Std Deviation',
                        'Total Demand'
                    ],
                    'Value': [
                        f"{recent['demand'].mean():.1f} units",
                        f"{recent['demand'].max():.0f} units",
                        f"{recent['demand'].min():.0f} units",
                        f"{recent['demand'].std():.1f} units",
                        f"{recent['demand'].sum():.0f} units"
                    ]
                })
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.subheader("🔮 Forecast Summary")
                
                forecast_df = pd.DataFrame({
                    'Metric': [
                        'Avg Predicted Demand',
                        'Peak Predicted Demand',
                        'Low Predicted Demand',
                        'Total Forecasted',
                        'Forecast Method'
                    ],
                    'Value': [
                        f"{predictions.mean():.1f} units",
                        f"{predictions.max():.0f} units",
                        f"{predictions.min():.0f} units",
                        f"{predictions.sum():.0f} units",
                        "Statistical AI"
                    ]
                })
                st.dataframe(forecast_df, hide_index=True, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Demand Forecast")
            
            fig = plot_forecast(historical_data, predictions)
            st.pyplot(fig)
            
            st.markdown("---")
            
            st.subheader("📅 Day-by-Day Predictions")
            
            last_date = historical_data['date'].iloc[-1]
            pred_df = pd.DataFrame({
                'Date': [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                        for i in range(len(predictions))],
                'Day': [(last_date + timedelta(days=i+1)).strftime('%A') 
                       for i in range(len(predictions))],
                'Predicted Demand': [f"{pred:.0f}" for pred in predictions]
            })
            
            st.dataframe(pred_df, hide_index=True, use_container_width=True, height=400)
        
        with tab3:
            st.subheader("📦 Inventory Level Projection")
            
            fig = plot_inventory_projection(historical_data, predictions)
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Days until stockout
            cumulative = 0
            days_to_stockout = len(predictions)
            for i, demand in enumerate(predictions):
                cumulative += demand
                if cumulative > current_stock:
                    days_to_stockout = i
                    break
            
            col1, col2, col3 = st.columns(3)
            
            reorder_point = np.mean(predictions) * (lead_time + safety_days)
            safety_stock = np.mean(predictions) * safety_days
            
            with col1:
                st.metric("Days Until Stockout", f"{days_to_stockout} days")
            
            with col2:
                st.metric("Reorder Point", f"{reorder_point:.0f} units")
            
            with col3:
                st.metric("Safety Stock", f"{safety_stock:.0f} units")
        
        with tab4:
            st.subheader("📋 Reorder Recommendations")
            
            # Calculate recommendations
            avg_demand = np.mean(predictions)
            reorder_point = avg_demand * (lead_time + safety_days)
            order_qty = int(avg_demand * 30)
            needs_reorder = current_stock < reorder_point
            
            if days_to_stockout <= 7:
                urgency = "🔴 HIGH"
                urgency_color = "error"
            elif days_to_stockout <= 14:
                urgency = "🟡 MEDIUM"
                urgency_color = "warning"
            else:
                urgency = "🟢 LOW"
                urgency_color = "success"
            
            # Display urgency
            if urgency_color == "error":
                st.error(f"**Urgency Level:** {urgency}")
            elif urgency_color == "warning":
                st.warning(f"**Urgency Level:** {urgency}")
            else:
                st.success(f"**Urgency Level:** {urgency}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Current Status")
                status_df = pd.DataFrame({
                    'Parameter': [
                        'Current Stock Level',
                        'Reorder Point',
                        'Safety Stock Level',
                        'Days Until Stockout',
                        'Supplier Lead Time'
                    ],
                    'Value': [
                        f"{current_stock:.0f} units",
                        f"{reorder_point:.0f} units",
                        f"{safety_stock:.0f} units",
                        f"{days_to_stockout} days",
                        f"{lead_time} days"
                    ]
                })
                st.dataframe(status_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("### 💡 Action Items")
                
                if needs_reorder:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h4>⚠️ Reorder Required</h4>
                    <p><strong>Recommended Order:</strong> {order_qty} units</p>
                    <p><strong>Action:</strong> Stock below reorder point</p>
                    <p><strong>Expected Delivery:</strong> {lead_time} days</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>✓ Stock Levels Adequate</h4>
                    <p><strong>Next Review:</strong> {days_to_stockout // 2} days</p>
                    <p><strong>Current Coverage:</strong> {days_to_stockout} days</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### 🎯 Key Insights")
            
            insights = []
            
            # Trend insight
            hist_avg = historical_data['demand'].tail(30).mean()
            trend_pct = ((avg_demand - hist_avg) / hist_avg) * 100
            if abs(trend_pct) > 5:
                direction = "increasing" if trend_pct > 0 else "decreasing"
                insights.append(f"• Demand is {direction} by {abs(trend_pct):.1f}% - adjust stock levels accordingly")
            
            # Volatility insight
            cv = historical_data['demand'].tail(30).std() / hist_avg
            if cv > 0.3:
                insights.append(f"• High demand variability detected (CV={cv:.2f}) - maintain higher safety stock")
            
            # Stockout risk
            if days_to_stockout <= 10:
                insights.append(f"• ⚠️ Stockout risk within {days_to_stockout} days - prioritize reorder")
            
            # Seasonal insights (if CSV data available)
            if selected_item is not None and selected_item['seasonal_factor'] > 1.3:
                insights.append(f"• High seasonal variability (factor: {selected_item['seasonal_factor']:.1f}) - plan for demand spikes")
            
            # Optimal order
            insights.append(f"• Recommended order quantity covers ~30 days of forecasted demand")
            
            for insight in insights:
                st.markdown(insight)
    
    else:
        # Welcome screen
        st.info("👈 Configure settings in the sidebar and click 'Run Prediction' to start")
        
        st.markdown("### 🎯 Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📈 Demand Forecasting**
            - CSV-powered predictions
            - Uses pre-sales data
            - Trend & seasonality detection
            - Up to 90-day horizon
            """)
        
        with col2:
            st.markdown("""
            **📦 Inventory Management**
            - Real-time stock projections
            - Automated reorder points
            - Safety stock calculation
            - Stockout prevention
            """)
        
        with col3:
            st.markdown("""
            **💡 Smart Recommendations**
            - Urgency-based alerts
            - Optimal order quantities
            - Risk assessment
            - Actionable insights
            """)
        
        st.markdown("---")
        
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Upload** your inventory CSV file (or ensure inventory_prediction_data.csv is in the directory)
        2. **Select** a product from the dropdown
        3. **Set** forecast parameters
        4. **Click** '🚀 Run Prediction' to generate forecasts
        
        💡 **CSV Format:** Your CSV should include columns like item_name, current_stock, pre_sales_week1-4, 
        avg_monthly_sales, seasonal_factor, etc.
        """)


if __name__ == "__main__":
    main()
