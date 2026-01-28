"""
Inventory Prediction App - CSV ENABLED VERSION
Loads inventory data from CSV and generates predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')


def load_moirai_model(model_size="small"):
    """
    Load Moirai model correctly using the Hugging Face repo ID directly
    """
    print(f"🤖 Loading Moirai-1.0-R-{model_size} from Hugging Face...")
    print(f"   This will download ~100MB on first run\n")
    
    try:
        from uni2ts.model.moirai import MoiraiForecast
        import torch
        
        # Use the Hugging Face repo ID directly - this is the correct way
        repo_id = f"Salesforce/moirai-1.0-R-{model_size}"
        
        # Load model - uni2ts will handle the download automatically
        model = MoiraiForecast.load_from_checkpoint(
            checkpoint_path=repo_id,
            model_size=model_size,
            patch_size="auto",
            context_length=512,
            prediction_length=90,
        )
        
        print("✓ Model loaded successfully!\n")
        return model
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return None


def use_simple_forecast():
    """Simple statistical forecasting fallback"""
    print("📊 Using Simple Statistical Forecasting")
    print("   (This works without the Moirai model)\n")
    
    class SimpleForecast:
        def forecast(self, data_tensor, prediction_length):
            import torch
            data = data_tensor.squeeze().numpy()
            
            # Calculate trend and seasonality
            recent = data[-30:]
            base = np.mean(recent)
            
            # Linear trend
            x = np.arange(len(data))
            y = data
            trend_coef = np.polyfit(x[-60:], y[-60:], 1)[0]
            
            # Generate predictions
            predictions = []
            for i in range(prediction_length):
                # Base + trend + weekly pattern
                pred = base + (trend_coef * i)
                pred += -12 * np.sin((i % 7) * 2 * np.pi / 7)
                pred += np.random.normal(0, 5)
                predictions.append(max(pred, 10))
            
            return torch.tensor([predictions])
    
    return SimpleForecast()


def load_inventory_csv(filepath='inventory_prediction_data.csv'):
    """Load inventory data from CSV file"""
    try:
        print(f"📂 Loading inventory data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} products from CSV\n")
        return df
    except FileNotFoundError:
        print(f"✗ Error: {filepath} not found")
        print("   Please ensure the CSV file is in the current directory\n")
        return None
    except Exception as e:
        print(f"✗ Error loading CSV: {e}\n")
        return None


def generate_historical_from_csv(item_data, days=180):
    """
    Generate historical demand data using pre-sales information and item characteristics
    This creates realistic time-series data based on the CSV data
    """
    np.random.seed(hash(item_data['item_name']) % 2**32)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    actual_days = len(dates)
    
    # Calculate average from pre-sales weeks
    weekly_sales = [
        item_data['pre_sales_week1'],
        item_data['pre_sales_week2'],
        item_data['pre_sales_week3'],
        item_data['pre_sales_week4']
    ]
    avg_weekly = np.mean(weekly_sales)
    
    # Use actual monthly sales average
    monthly_avg = item_data['avg_monthly_sales']
    base_daily_demand = monthly_avg / 30
    
    # Get seasonal factor
    seasonal_factor = item_data.get('seasonal_factor', 1.0)
    
    # Generate demand pattern
    demand = np.ones(actual_days) * base_daily_demand
    
    # Apply seasonal variations based on seasonal_factor
    if seasonal_factor > 1.2:  # Highly seasonal
        seasonal_wave = (seasonal_factor - 1.0) * base_daily_demand * np.sin(
            np.arange(actual_days) * 2 * np.pi / 90
        )
        demand += seasonal_wave
    
    # Weekly seasonality (lower on weekends)
    for i in range(actual_days):
        day_of_week = (i + dates[0].dayofweek) % 7
        if day_of_week >= 5:  # Weekend
            demand[i] *= 0.7
    
    # Trend based on recent weeks
    trend_direction = (weekly_sales[-1] - weekly_sales[0]) / max(weekly_sales[0], 1)
    if abs(trend_direction) > 0.05:  # Significant trend
        trend = np.linspace(0, base_daily_demand * trend_direction * 0.3, actual_days)
        demand += trend
    
    # Stock turnover influence (higher turnover = more variable demand)
    turnover = item_data.get('stock_turnover_rate', 0.6)
    volatility = base_daily_demand * (0.15 + turnover * 0.15)
    
    # Add variability
    demand += np.random.normal(0, volatility, actual_days)
    
    # Occasional promotional spikes
    promo_days = np.random.choice([0, 1], size=actual_days, p=[0.95, 0.05])
    demand += promo_days * np.random.uniform(base_daily_demand * 0.5, base_daily_demand * 1.5, actual_days)
    
    # Ensure positive
    demand = np.maximum(demand, base_daily_demand * 0.2)
    
    # Simulate inventory levels
    current_stock = item_data['current_stock']
    reorder_point = item_data['reorder_point']
    lead_time = item_data['lead_time_days']
    
    # Calculate historical inventory
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


def predict_demand(model, historical_data, prediction_days=30):
    """Generate forecast"""
    import torch
    
    demand_series = historical_data['demand'].values.astype(float)
    demand_tensor = torch.tensor(demand_series).float().unsqueeze(0)
    
    print(f"🔮 Forecasting {prediction_days} days...")
    
    with torch.no_grad():
        predictions = model.forecast(
            demand_tensor,
            prediction_length=prediction_days,
        )
    
    forecast = predictions[0].numpy()
    return np.maximum(forecast, 0)


def calculate_metrics(current_stock, predictions, lead_time=7, safety_days=5):
    """Calculate inventory metrics"""
    avg_demand = np.mean(predictions)
    total_demand = np.sum(predictions)
    
    # Reorder calculations
    lead_time_demand = np.sum(predictions[:lead_time])
    safety_stock = avg_demand * safety_days
    reorder_point = lead_time_demand + safety_stock
    order_qty = int(avg_demand * 30)
    
    # Days until stockout
    cumulative = 0
    days_to_stockout = len(predictions)
    for i, demand in enumerate(predictions):
        cumulative += demand
        if cumulative > current_stock:
            days_to_stockout = i
            break
    
    # Urgency
    needs_reorder = current_stock < reorder_point
    if days_to_stockout <= 7:
        urgency = "🔴 HIGH"
    elif days_to_stockout <= 14:
        urgency = "🟡 MEDIUM"
    else:
        urgency = "🟢 LOW"
    
    return {
        'current_stock': current_stock,
        'avg_daily_demand': avg_demand,
        'total_30day_demand': total_demand,
        'reorder_point': reorder_point,
        'safety_stock': safety_stock,
        'order_qty': order_qty,
        'days_to_stockout': days_to_stockout,
        'needs_reorder': needs_reorder,
        'urgency': urgency,
        'lead_time': lead_time
    }


def print_report(product_name, historical_data, predictions, metrics, item_data=None):
    """Print comprehensive report"""
    print("=" * 80)
    print(f"INVENTORY PREDICTION - {product_name}")
    print("=" * 80)
    
    # Print CSV data if available
    if item_data is not None:
        print(f"\n📋 PRODUCT INFORMATION (from CSV)")
        print("-" * 80)
        print(f"Item ID:                 {item_data['item_id']}")
        print(f"Category:                {item_data['category']}")
        print(f"Supplier:                {item_data['supplier_id']}")
        print(f"Unit Cost:               ${item_data['unit_cost']:.2f}")
        print(f"Selling Price:           ${item_data['selling_price']:.2f}")
        print(f"Stock Turnover Rate:     {item_data['stock_turnover_rate']:.2f}")
        print(f"Seasonal Factor:         {item_data['seasonal_factor']:.1f}")
        print(f"\n  Pre-Sales History:")
        print(f"    Week 1: {item_data['pre_sales_week1']} units")
        print(f"    Week 2: {item_data['pre_sales_week2']} units")
        print(f"    Week 3: {item_data['pre_sales_week3']} units")
        print(f"    Week 4: {item_data['pre_sales_week4']} units")
    
    recent = historical_data.tail(30)
    
    print(f"\n📊 HISTORICAL ANALYSIS (Last 30 Days)")
    print("-" * 80)
    print(f"Average Daily Demand:    {recent['demand'].mean():.1f} units")
    print(f"Peak Demand:             {recent['demand'].max():.0f} units")
    print(f"Total Demand:            {recent['demand'].sum():.0f} units")
    print(f"Volatility (Std Dev):    {recent['demand'].std():.1f} units")
    
    print(f"\n🔮 DEMAND FORECAST (Next 30 Days)")
    print("-" * 80)
    print(f"Predicted Avg Demand:    {metrics['avg_daily_demand']:.1f} units/day")
    print(f"Total Forecasted:        {metrics['total_30day_demand']:.0f} units")
    print(f"Peak Predicted:          {predictions.max():.0f} units")
    print(f"Low Predicted:           {predictions.min():.0f} units")
    
    # Trend
    hist_avg = recent['demand'].mean()
    pred_avg = metrics['avg_daily_demand']
    trend_pct = ((pred_avg - hist_avg) / hist_avg) * 100
    trend_dir = "📈 INCREASING" if trend_pct > 0 else "📉 DECREASING"
    print(f"Trend:                   {trend_dir} ({abs(trend_pct):.1f}%)")
    
    print(f"\n📦 INVENTORY STATUS")
    print("-" * 80)
    print(f"Current Stock:           {metrics['current_stock']:.0f} units")
    print(f"Reorder Point:           {metrics['reorder_point']:.0f} units")
    print(f"Safety Stock:            {metrics['safety_stock']:.0f} units")
    print(f"Days Until Stockout:     {metrics['days_to_stockout']} days")
    print(f"Supplier Lead Time:      {metrics['lead_time']} days")
    
    print(f"\n🚨 URGENCY: {metrics['urgency']}")
    
    if metrics['needs_reorder']:
        print(f"\n⚠️  RECOMMENDATION: Order {metrics['order_qty']} units NOW")
        print(f"   Stock is below reorder point!")
        delivery_date = datetime.now() + timedelta(days=metrics['lead_time'])
        print(f"   Expected delivery: {delivery_date.strftime('%Y-%m-%d')}")
        if item_data is not None:
            order_cost = metrics['order_qty'] * item_data['unit_cost']
            print(f"   Estimated order cost: ${order_cost:,.2f}")
    else:
        print(f"\n✓ Stock levels adequate")
        print(f"   Current supply covers {metrics['days_to_stockout']} days")
        print(f"   Next review: {metrics['days_to_stockout'] // 2} days")
    
    print("\n" + "=" * 80 + "\n")


def visualize(historical_data, predictions, product_name, metrics):
    """Create visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Left: Demand forecast
    ax1 = axes[0]
    recent = historical_data.tail(60)
    ax1.plot(recent['date'], recent['demand'], 
            label='Historical Demand', color='#2E86AB', linewidth=2)
    
    last_date = historical_data['date'].iloc[-1]
    pred_dates = pd.date_range(start=last_date + timedelta(days=1),
                               periods=len(predictions), freq='D')
    
    ax1.plot(pred_dates, predictions, 
            label='Predicted Demand', color='#E63946', 
            linewidth=2.5, linestyle='--', marker='o', markersize=3)
    
    # Confidence interval
    std = recent['demand'].std()
    ax1.fill_between(pred_dates, predictions - std, predictions + std,
                     alpha=0.2, color='#E63946', label='Confidence Range')
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Demand (Units)', fontsize=11)
    ax1.set_title(f'Demand Forecast - {product_name}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Right: Inventory projection
    ax2 = axes[1]
    projected = [metrics['current_stock']]
    for demand in predictions:
        projected.append(max(projected[-1] - demand, 0))
    
    proj_dates = pd.date_range(start=last_date, periods=len(projected), freq='D')
    
    ax2.plot(proj_dates, projected, color='#F77F00', linewidth=2.5, 
            label='Projected Inventory')
    
    # Reference lines
    ax2.axhline(y=metrics['reorder_point'], color='red', linestyle='--', 
               label='Reorder Point', linewidth=2)
    ax2.axhline(y=metrics['safety_stock'], color='orange', linestyle='--', 
               label='Safety Stock', linewidth=1.5)
    
    ax2.fill_between(proj_dates, 0, projected, alpha=0.3, color='#F77F00')
    
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Inventory Level (Units)', fontsize=11)
    ax2.set_title('Inventory Projection', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    filename = f"{product_name.replace(' ', '_').lower()}_forecast.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {filename}\n")
    plt.close()


def main():
    """Main application"""
    print("\n" + "=" * 80)
    print("INVENTORY PREDICTION - CSV MODE")
    print("Powered by AI Time-Series Forecasting")
    print("=" * 80 + "\n")
    
    # Load CSV data
    inventory_df = load_inventory_csv('inventory_prediction_data.csv')
    
    if inventory_df is None:
        print("⚠️  CSV not found. Make sure 'inventory_prediction_data.csv' is in the current directory")
        print("    Exiting...\n")
        return
    
    # Display available products
    print("Available Products:")
    print("-" * 80)
    for idx, row in inventory_df.iterrows():
        print(f"{idx+1:2d}. {row['item_name']:30s} | {row['category']:15s} | Stock: {row['current_stock']:4d} units")
    print("-" * 80)
    
    # Get user selection
    print("\nSelect products to analyze:")
    print("  - Enter numbers separated by commas (e.g., 1,5,10)")
    print("  - Enter 'all' to analyze all products")
    print("  - Enter a range (e.g., 1-5)")
    
    selection = input("\nYour selection: ").strip()
    
    if selection.lower() == 'all':
        selected_indices = list(range(len(inventory_df)))
    elif '-' in selection:
        # Handle range
        start, end = selection.split('-')
        selected_indices = list(range(int(start)-1, int(end)))
    else:
        # Handle comma-separated
        try:
            selected_indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_indices = [i for i in selected_indices if 0 <= i < len(inventory_df)]
        except:
            print("Invalid selection. Analyzing first 3 products.")
            selected_indices = [0, 1, 2]
    
    print(f"\n✓ Selected {len(selected_indices)} product(s)\n")
    
    # Try to load Moirai
    print("Attempting to load Moirai AI model...\n")
    model = load_moirai_model("small")
    
    # Fallback to simple forecasting
    if model is None:
        print("⚠️  Moirai model not available")
        response = input("Use simple statistical forecasting instead? (y/n): ")
        if response.lower() != 'y':
            print("\nExiting. To use Moirai:")
            print("  pip install uni2ts torch")
            return
        model = use_simple_forecast()
    
    # Process each selected product
    for i, idx in enumerate(selected_indices, 1):
        item = inventory_df.iloc[idx]
        
        print(f"\n{'='*80}")
        print(f"PRODUCT {i}/{len(selected_indices)}: {item['item_name']}")
        print("=" * 80 + "\n")
        
        try:
            # Generate historical data from CSV
            print(f"📈 Generating 180 days of historical data from CSV...")
            historical_data = generate_historical_from_csv(item, days=180)
            print(f"✓ Generated {len(historical_data)} days of data\n")
            
            # Predict
            predictions = predict_demand(model, historical_data, 30)
            print(f"✓ Generated 30-day forecast\n")
            
            # Calculate metrics
            current_stock = historical_data['inventory_level'].iloc[-1]
            metrics = calculate_metrics(
                current_stock, 
                predictions,
                lead_time=int(item['lead_time_days']),
                safety_days=5
            )
            
            # Report
            print_report(item['item_name'], historical_data, predictions, metrics, item)
            
            # Visualize
            print("📊 Creating visualization...")
            visualize(historical_data, predictions, item['item_name'], metrics)
            
        except Exception as e:
            print(f"✗ Error processing {item['item_name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if i < len(selected_indices):
            input("Press Enter for next product...")
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    for idx in selected_indices:
        item = inventory_df.iloc[idx]
        filename = f"{item['item_name'].replace(' ', '_').lower()}_forecast.png"
        print(f"  - {filename}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
