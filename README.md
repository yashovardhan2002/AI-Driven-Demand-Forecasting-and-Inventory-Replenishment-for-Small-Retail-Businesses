# AI-Driven-Demand-Forecasting-and-Inventory-Replenishment-for-Small-Retail-Businesses
AI-driven demand forecasting thesis comparing Prophet vs SMA for retail inventory optimization. Analyzes 50 SKUs across volume categories, calculates dynamic ROP/safety stock, performs statistical validation, and generates financial impact reports with visualizations showing cost savings.
# AI-Driven Demand Forecasting and Inventory Replenishment for Small Retail Businesses
## 📋 Overview

This BSc thesis project implements and compares AI-driven demand forecasting using Facebook Prophet against traditional Simple Moving Average (SMA) methods for inventory optimization in small to medium-sized retail enterprises (SMEs).

## 🎯 Key Features

- **Demand Forecasting**: Facebook Prophet vs Simple Moving Average
- **Multi-Category Analysis**: High, Medium, and Low volume SKU categories
- **Inventory Optimization**: Dynamic Reorder Point (ROP) and safety stock calculation
- **Financial Impact Analysis**: Holding costs, stockout costs, and annual savings
- **Statistical Validation**: Paired t-tests for significance testing
- **Visualizations**: Publication-ready charts and comparison tables

## 📊 Results Summary

The AI-driven approach demonstrates:
- Improved forecast accuracy (MAPE reduction)
- Reduced safety stock requirements
- Decreased stockout occurrences
- Significant annual cost savings
- Statistical significance validated (p < 0.05)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yashovardhan2002/AI-Driven-Demand-Forecasting-and-Inventory-Replenishment-for-Small-Retail-Businesses.git
cd AI-Driven-Demand-Forecasting-and-Inventory-Replenishment-for-Small-Retail-Businesses
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

Run the main analysis:
```bash
python thesis.py
```

The script will:
1. Generate synthetic sales data for 50 SKUs
2. Train and compare forecasting models
3. Calculate inventory optimization metrics
4. Perform statistical analysis
5. Generate output files and visualizations

## 📁 Output Files

The script generates the following files:

| File | Description |
|------|-------------|
| `thesis_results.csv` | Detailed results for all 50 SKUs |
| `thesis_summary.csv` | Aggregated summary metrics |
| `figure_1_forecast_accuracy.png` | MAPE and accuracy comparison charts |
| `figure_2_inventory_optimization.png` | Safety stock and savings visualizations |
| `figure_3_summary_table.png` | Summary comparison table |

## ⚙️ Configuration

Modify the `CONFIG` dictionary in `thesis.py` to adjust parameters:
```python
CONFIG = {
    'num_skus': 50,                    # Number of SKUs to analyze
    'train_test_split_days': 90,       # Test period duration
    'sma_window': 30,                  # Moving average window
    'lead_time_days': 7,               # Supplier lead time
    'service_level': 0.95,             # Target service level (95%)
    'unit_cost': 15.0,                 # Cost per unit
    'holding_cost_rate': 0.25,         # Annual holding cost rate
    'stockout_cost_per_unit': 3.0      # Penalty cost per stockout
}
```

## 📈 Methodology

### Data Generation
- Synthetic sales data spanning 2.5 years (2023-2025)
- Three volume categories: High (80-150 units), Medium (30-80 units), Low (5-30 units)
- Incorporated patterns: trend, weekly seasonality, annual seasonality, promotional effects
- Realistic noise and intermittent demand for low-volume SKUs

### Forecasting Models
1. **Simple Moving Average (SMA)**: 30-day rolling average baseline
2. **Facebook Prophet**: Time series model with seasonality and holiday effects

### Inventory Optimization
- **Traditional ROP**: Based on historical average and standard deviation
- **AI-Driven ROP**: Uses Prophet forecast uncertainty for dynamic safety stock
- Service level: 95% (z-score = 1.645)
- Lead time: 7 days

### Evaluation Metrics
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)
- Forecast accuracy percentage
- Financial impact (holding costs, stockout costs, total savings)

## 🔬 Statistical Analysis

Paired t-test validates that Prophet significantly outperforms SMA in forecast accuracy (p < 0.05).

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **prophet**: Time series forecasting
- **scipy**: Statistical functions
- **matplotlib**: Visualization
- **seaborn**: Statistical data visualization

## 🤝 Contributing

This is a thesis project, but suggestions and feedback are welcome! Please open an issue to discuss proposed changes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Yashovardhan**
- GitHub: [@yashovardhan2002](https://github.com/yashovardhan2002)

## 🙏 Acknowledgments

- Facebook Prophet development team
- Academic supervisors and advisors
- Open source community

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out via GitHub.

---

**Note**: This implementation uses synthetic data for demonstration purposes. For production use, replace with actual retail sales data.
