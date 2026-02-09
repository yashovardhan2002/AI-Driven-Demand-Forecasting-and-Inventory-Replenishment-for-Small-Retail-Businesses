import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
try:
    from prophet import Prophet
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "prophet", "scipy", "matplotlib", "seaborn", "-q"])
    from prophet import Prophet
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns

CONFIG = {'num_skus': 50, 'train_test_split_days': 90, 'sma_window': 30, 'lead_time_days': 7, 'service_level': 0.95, 'unit_cost': 15.0, 'holding_cost_rate': 0.25, 'stockout_cost_per_unit': 3.0}
HIGH_VOLUME_RANGE, MEDIUM_VOLUME_RANGE, LOW_VOLUME_RANGE = range(0, 15), range(15, 35), range(35, 50)

def get_volume_category_by_id(sku_id):
    return 'High' if sku_id in HIGH_VOLUME_RANGE else ('Medium' if sku_id in MEDIUM_VOLUME_RANGE else 'Low')

def generate_sku_sales_data(sku_id, start_date='2023-01-01', end_date='2025-06-30'):
    np.random.seed(42 + sku_id)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    if sku_id < 15: base_demand, volatility, lumpy_prob = np.random.randint(80, 150), 0.15, 0.0
    elif sku_id < 35: base_demand, volatility, lumpy_prob = np.random.randint(30, 80), 0.25, 0.05
    else: base_demand, volatility, lumpy_prob = np.random.randint(5, 30), 0.40, 0.30
    trend = np.linspace(0, base_demand * 0.3, len(date_range))
    weekly = base_demand * 0.15 * np.sin(2 * np.pi * np.arange(len(date_range)) / 7)
    annual = base_demand * 0.20 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365)
    noise = np.random.normal(0, base_demand * volatility, len(date_range))
    sales = base_demand + trend + weekly + annual + noise
    for i, date in enumerate(date_range):
        if date.month == 11 and 20 <= date.day <= 27: sales[i] *= np.random.uniform(1.5, 2.5)
        elif date.month == 12 and date.day >= 15: sales[i] *= np.random.uniform(1.3, 1.8)
        elif date.weekday() == 5: sales[i] *= 1.2
    if lumpy_prob > 0:
        sales[np.random.random(len(date_range)) < lumpy_prob] = 0
        sales[np.random.random(len(date_range)) < 0.03] *= np.random.uniform(2.5, 4.0)
    df = pd.DataFrame({'date': date_range, 'sales': np.maximum(sales, 0), 'is_promo': 0})
    df.loc[(df['date'].dt.month == 11) & (df['date'].dt.day >= 20) & (df['date'].dt.day <= 27), 'is_promo'] = 1
    df.loc[(df['date'].dt.month == 12) & (df['date'].dt.day >= 15), 'is_promo'] = 1
    return df

def simple_moving_average_forecast(train_data, forecast_periods, window=30):
    sma = train_data['sales'].rolling(window=window).mean()
    return np.full(forecast_periods, sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else train_data['sales'].mean())

def train_prophet_model(train_data, forecast_periods):
    prophet_df = train_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    holidays_df = pd.DataFrame({'holiday': 'promotion', 'ds': train_data[train_data['is_promo'] == 1]['date'], 'lower_window': 0, 'upper_window': 0})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode='additive', changepoint_prior_scale=0.05, holidays=holidays_df if len(holidays_df) > 0 else None)
    model.fit(prophet_df)
    forecast = model.predict(model.make_future_dataframe(periods=forecast_periods))
    return np.maximum(forecast.tail(forecast_periods)['yhat'].values, 0), forecast

def calculate_forecast_metrics(actual, predicted, method_name="Model"):
    non_zero_mask = actual > 0.01
    if non_zero_mask.sum() == 0: return {'MAE': np.nan, 'MAPE': np.nan, 'RMSE': np.nan, 'Bias': np.nan}
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    return {'Method': method_name, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'Bias': np.mean(predicted - actual), 'Accuracy_%': 100 - mape}

def calculate_dynamic_rop(forecast_values, lead_time_days, forecast_df, z_score):
    avg_daily_demand = np.mean(forecast_values[:lead_time_days])
    forecast_std = (forecast_df['yhat_upper'] - forecast_df['yhat_lower']).mean() / (2 * 1.28)
    safety_stock = z_score * forecast_std * np.sqrt(lead_time_days)
    return {'avg_daily_demand': avg_daily_demand, 'safety_stock': safety_stock, 'rop': (avg_daily_demand * lead_time_days) + safety_stock}

def calculate_traditional_rop(historical_sales, lead_time_days, z_score):
    avg_demand, std_demand = historical_sales.mean(), historical_sales.std()
    safety_stock = z_score * std_demand * np.sqrt(lead_time_days)
    return {'avg_daily_demand': avg_demand, 'safety_stock': safety_stock, 'rop': (avg_demand * lead_time_days) + safety_stock}

def simulate_inventory(actual, rop, avg_daily_demand, lead_time_days):
    on_hand, pipeline_orders = rop, [0] * lead_time_days
    stockout_days, lost_units, on_hand_history = 0, 0, []
    target_level = rop + (avg_daily_demand * lead_time_days)
    for t in range(len(actual)):
        on_hand += pipeline_orders[0]
        pipeline_orders = pipeline_orders[1:] + [0]
        demand = actual[t]
        if on_hand >= demand: on_hand -= demand
        else: lost_units += (demand - on_hand); stockout_days += 1; on_hand = 0
        on_hand_history.append(on_hand)
        inventory_position = on_hand + sum(pipeline_orders)
        if inventory_position <= rop: pipeline_orders[-1] += max(0, target_level - inventory_position)
    return {'stockout_days': stockout_days, 'lost_units': lost_units, 'avg_on_hand': np.mean(on_hand_history)}

def calculate_financial_impact(rop_trad, rop_ai, actual, sma_pred, prophet_pred, config):
    trad_sim = simulate_inventory(actual, rop_trad['rop'], rop_trad['avg_daily_demand'], config['lead_time_days'])
    ai_sim = simulate_inventory(actual, rop_ai['rop'], rop_ai['avg_daily_demand'], config['lead_time_days'])
    annualize = 365 / len(actual)
    holding_cost_trad = trad_sim['avg_on_hand'] * config['unit_cost'] * config['holding_cost_rate']
    stockout_cost_trad = trad_sim['lost_units'] * config['stockout_cost_per_unit'] * annualize
    holding_cost_ai = ai_sim['avg_on_hand'] * config['unit_cost'] * config['holding_cost_rate']
    stockout_cost_ai = ai_sim['lost_units'] * config['stockout_cost_per_unit'] * annualize
    holding_cost_savings, stockout_cost_savings = holding_cost_trad - holding_cost_ai, stockout_cost_trad - stockout_cost_ai
    return {'holding_cost_savings': holding_cost_savings, 'stockout_cost_savings': stockout_cost_savings, 'total_annual_savings': holding_cost_savings + stockout_cost_savings, 'sma_stockout_days': trad_sim['stockout_days'], 'prophet_stockout_days': ai_sim['stockout_days'], 'stockout_reduction_%': ((trad_sim['stockout_days'] - ai_sim['stockout_days']) / max(trad_sim['stockout_days'], 1)) * 100}

def analyze_single_sku(sku_id, df, config, z_score):
    split = -config['train_test_split_days']
    train, test = df[:split].copy(), df[split:].copy()
    actual = test['sales'].values
    sma_pred = simple_moving_average_forecast(train, len(test), config['sma_window'])
    prophet_pred, forecast = train_prophet_model(train, len(test))
    sma_metrics = calculate_forecast_metrics(actual, sma_pred, "SMA")
    prophet_metrics = calculate_forecast_metrics(actual, prophet_pred, "Prophet")
    rop_trad = calculate_traditional_rop(train['sales'], config['lead_time_days'], z_score)
    rop_ai = calculate_dynamic_rop(prophet_pred, config['lead_time_days'], forecast.tail(config['train_test_split_days']), z_score)
    finance = calculate_financial_impact(rop_trad, rop_ai, actual, sma_pred, prophet_pred, config)
    vol_cat, avg_sales = get_volume_category_by_id(sku_id), df['sales'].mean()
    return {'SKU_ID': f'SKU_{sku_id:03d}', 'Volume_Category': vol_cat, 'Avg_Daily_Sales': avg_sales, 'SMA_MAE': sma_metrics['MAE'], 'SMA_MAPE': sma_metrics['MAPE'], 'SMA_Accuracy': sma_metrics['Accuracy_%'], 'Prophet_MAE': prophet_metrics['MAE'], 'Prophet_MAPE': prophet_metrics['MAPE'], 'Prophet_Accuracy': prophet_metrics['Accuracy_%'], 'MAE_Improvement_%': ((sma_metrics['MAE'] - prophet_metrics['MAE']) / sma_metrics['MAE'] * 100), 'MAPE_Improvement_%': ((sma_metrics['MAPE'] - prophet_metrics['MAPE']) / sma_metrics['MAPE'] * 100), 'Traditional_Safety_Stock': rop_trad['safety_stock'], 'AI_Safety_Stock': rop_ai['safety_stock'], 'Safety_Stock_Reduction_%': ((rop_trad['safety_stock'] - rop_ai['safety_stock']) / max(rop_trad['safety_stock'], 1) * 100), 'Traditional_ROP': rop_trad['rop'], 'AI_ROP': rop_ai['rop'], 'Annual_Savings_USD': finance['total_annual_savings'], 'Holding_Savings_USD': finance['holding_cost_savings'], 'Stockout_Savings_USD': finance['stockout_cost_savings'], 'SMA_Stockout_Days': finance['sma_stockout_days'], 'Prophet_Stockout_Days': finance['prophet_stockout_days']}

def main():
    print("="*80 + "\nAI-DRIVEN DEMAND FORECASTING FOR SME RETAIL\nBSc Thesis Implementation\n" + "="*80)
    z_score = stats.norm.ppf(CONFIG['service_level'])
    print(f"\nConfiguration:\n  SKUs: {CONFIG['num_skus']}\n  Test Period: {CONFIG['train_test_split_days']} days\n  Service Level: {CONFIG['service_level']*100:.0f}%\n  Z-Score: {z_score:.4f}\n  Lead Time: {CONFIG['lead_time_days']} days")
    print(f"\nProcessing {CONFIG['num_skus']} SKUs...")
    results_df = pd.DataFrame([analyze_single_sku(i, generate_sku_sales_data(i), CONFIG, z_score) for i in range(CONFIG['num_skus'])])
    print("Analysis complete!")
    print("\n" + "="*80 + "\nSTATISTICAL SIGNIFICANCE TEST\n" + "="*80)
    t_stat, p_value = stats.ttest_rel(results_df['SMA_MAE'].values, results_df['Prophet_MAE'].values, alternative='greater')
    print(f"\nPaired t-test (one-tailed, H1: SMA MAE > Prophet MAE):\n  t-statistic: {t_stat:.4f}\n  p-value: {p_value:.6f}\n  Result: {'Prophet is significantly better (p < 0.05) ✓' if p_value < 0.05 else 'No significant difference (p >= 0.05)'}")
    print("\n" + "="*80 + "\nRESULTS BY VOLUME CATEGORY\n" + "="*80)
    category_counts = results_df['Volume_Category'].value_counts()
    print(f"\nVolume Category Distribution:\n  High: {category_counts.get('High', 0)} SKUs\n  Medium: {category_counts.get('Medium', 0)} SKUs\n  Low: {category_counts.get('Low', 0)} SKUs")
    print("\n" + results_df.groupby('Volume_Category').agg({'Avg_Daily_Sales': 'mean', 'SMA_MAPE': 'mean', 'Prophet_MAPE': 'mean', 'MAPE_Improvement_%': 'mean', 'Safety_Stock_Reduction_%': 'mean', 'Annual_Savings_USD': 'sum'}).round(2).to_string())
    print("\n" + "="*80 + "\nOVERALL SUMMARY - {} SKUs\n".format(CONFIG['num_skus']) + "="*80)
    total_sma, total_prophet = results_df['SMA_Stockout_Days'].sum(), results_df['Prophet_Stockout_Days'].sum()
    stockout_change = ((total_prophet - total_sma) / total_sma * 100) if total_sma > 0 else (0 if total_prophet == 0 else 100)
    sma_mape_mean, prophet_mape_mean = results_df['SMA_MAPE'].mean(), results_df['Prophet_MAPE'].mean()
    mape_improvement_aggregate = ((sma_mape_mean - prophet_mape_mean) / sma_mape_mean * 100)
    print(f"\nForecast Accuracy:\n  SMA MAPE: {sma_mape_mean:.2f}%\n  Prophet MAPE: {prophet_mape_mean:.2f}%\n  Improvement: {mape_improvement_aggregate:.2f}%")
    print(f"\nInventory Optimization:\n  Avg Traditional Safety Stock: {results_df['Traditional_Safety_Stock'].mean():.1f} units\n  Avg AI Safety Stock: {results_df['AI_Safety_Stock'].mean():.1f} units\n  Safety Stock Reduction: {results_df['Safety_Stock_Reduction_%'].mean():.1f}%\n  Total Annual Savings: ${results_df['Annual_Savings_USD'].sum():,.2f}")
    print(f"\nStockout Analysis:\n  SMA Total Stockout Days: {total_sma}\n  Prophet Total Stockout Days: {total_prophet}\n  Change: {'+' if stockout_change > 0 else ''}{stockout_change:.1f}%")
    print("\n" + "="*80 + "\nSAVING RESULTS\n" + "="*80)
    results_df.to_csv('thesis_results.csv', index=False)
    print("\n✓ Saved: thesis_results.csv")
    pd.DataFrame({'Metric': ['Number of SKUs', 'Test Period (days)', 'Lead Time (days)', 'Service Level (%)', 'Z-Score', 'SMA MAPE (%)', 'Prophet MAPE (%)', 'MAPE Improvement (%)', 'Avg Traditional Safety Stock', 'Avg AI Safety Stock', 'Safety Stock Reduction (%)', 'Total Annual Savings ($)', 'SMA Stockout Days', 'Prophet Stockout Days', 'Stockout Change (%)', 'p-value', 'Significant?'], 'Value': [CONFIG['num_skus'], CONFIG['train_test_split_days'], CONFIG['lead_time_days'], f"{CONFIG['service_level']*100:.0f}", f"{z_score:.4f}", f"{sma_mape_mean:.2f}", f"{prophet_mape_mean:.2f}", f"{mape_improvement_aggregate:.2f}", f"{results_df['Traditional_Safety_Stock'].mean():.1f}", f"{results_df['AI_Safety_Stock'].mean():.1f}", f"{results_df['Safety_Stock_Reduction_%'].mean():.1f}", f"{results_df['Annual_Savings_USD'].sum():,.2f}", f"{total_sma}", f"{total_prophet}", f"{stockout_change:.1f}", f"{p_value:.6f}", "Yes" if p_value < 0.05 else "No"]}).to_csv('thesis_summary.csv', index=False)
    print("✓ Saved: thesis_summary.csv")
    print("\nGenerating visualizations...")
    sns.set_style("whitegrid"); plt.rcParams['figure.dpi'] = 300; vol_order = ['Low', 'Medium', 'High']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    results_df.groupby('Volume_Category')[['SMA_MAPE', 'Prophet_MAPE']].mean().reindex(vol_order).plot(kind='bar', ax=axes[0], color=['#e74c3c', '#27ae60']); axes[0].set_title('Forecast Accuracy (MAPE)', fontweight='bold'); axes[0].set_xlabel('Volume Category'); axes[0].set_ylabel('MAPE (%)'); axes[0].legend(['SMA', 'Prophet']); axes[0].set_xticklabels(vol_order, rotation=0)
    results_df.groupby('Volume_Category')[['SMA_Accuracy', 'Prophet_Accuracy']].mean().reindex(vol_order).plot(kind='bar', ax=axes[1], color=['#e74c3c', '#27ae60']); axes[1].set_title('Forecast Accuracy %', fontweight='bold'); axes[1].set_xlabel('Volume Category'); axes[1].set_ylabel('Accuracy (%)'); axes[1].legend(['SMA', 'Prophet']); axes[1].set_xticklabels(vol_order, rotation=0)
    plt.tight_layout(); plt.savefig('figure_1_forecast_accuracy.png', dpi=300, bbox_inches='tight'); print("   ✓ Saved: figure_1_forecast_accuracy.png"); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    results_df.groupby('Volume_Category')[['Traditional_Safety_Stock', 'AI_Safety_Stock']].mean().reindex(vol_order).plot(kind='bar', ax=axes[0], color=['#e74c3c', '#27ae60']); axes[0].set_title('Safety Stock Requirements', fontweight='bold'); axes[0].set_xlabel('Volume Category'); axes[0].set_ylabel('Safety Stock (units)'); axes[0].legend(['Traditional', 'AI']); axes[0].set_xticklabels(vol_order, rotation=0)
    results_df.groupby('Volume_Category')['Annual_Savings_USD'].sum().reindex(vol_order).plot(kind='bar', ax=axes[1], color='#3498db'); axes[1].set_title('Annual Savings by Category', fontweight='bold'); axes[1].set_xlabel('Volume Category'); axes[1].set_ylabel('Savings ($)'); axes[1].set_xticklabels(vol_order, rotation=0)
    plt.tight_layout(); plt.savefig('figure_2_inventory_optimization.png', dpi=300, bbox_inches='tight'); print("   ✓ Saved: figure_2_inventory_optimization.png"); plt.close()
    fig, ax = plt.subplots(figsize=(10, 6)); ax.axis('tight'); ax.axis('off')
    table_data = [['Metric', 'SMA', 'Prophet', 'Change'], ['MAPE (%)', f"{sma_mape_mean:.1f}", f"{prophet_mape_mean:.1f}", f"{mape_improvement_aggregate:.1f}% ↓"], ['Accuracy (%)', f"{results_df['SMA_Accuracy'].mean():.1f}", f"{results_df['Prophet_Accuracy'].mean():.1f}", f"{results_df['Prophet_Accuracy'].mean() - results_df['SMA_Accuracy'].mean():.1f}% ↑"], ['Safety Stock', f"{results_df['Traditional_Safety_Stock'].mean():.1f}", f"{results_df['AI_Safety_Stock'].mean():.1f}", f"{results_df['Safety_Stock_Reduction_%'].mean():.1f}% ↓"], ['Annual Savings', 'Baseline', f"${results_df['Annual_Savings_USD'].sum():,.0f}", f"${results_df['Annual_Savings_USD'].sum():,.0f}"], ['Stockouts', f"{total_sma} days", f"{total_prophet} days", f"{'+' if stockout_change > 0 else ''}{stockout_change:.1f}%"]]
    table = ax.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.3, 0.25, 0.25, 0.2]); table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 2)
    for i in range(4): table[(0, i)].set_facecolor('#34495e'); table[(0, i)].set_text_props(weight='bold', color='white')
    plt.title('Summary: AI vs Traditional ({} SKUs)'.format(CONFIG['num_skus']), fontweight='bold', fontsize=12, pad=20); plt.savefig('figure_3_summary_table.png', dpi=300, bbox_inches='tight'); print("   ✓ Saved: figure_3_summary_table.png"); plt.close()
    print("\n" + "="*80 + "\nCOMPLETE\n" + "="*80 + "\nGenerated files:\n  1. thesis_results.csv\n  2. thesis_summary.csv\n  3. figure_1_forecast_accuracy.png\n  4. figure_2_inventory_optimization.png\n  5. figure_3_summary_table.png\n" + "="*80)

if __name__ == "__main__":
    main()
