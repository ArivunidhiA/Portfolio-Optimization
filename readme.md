# Portfolio Optimization Project

## Overview
Multi-asset portfolio optimization system using Monte Carlo simulation to maximize returns while managing risk. The project analyzes 10 major stocks and generates optimal portfolio allocations based on historical data.

## Features
- 10,000 Monte Carlo simulations
- Real-time stock data fetching
- Efficient frontier generation
- Sharpe ratio optimization
- Comprehensive reporting

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
python portfolio_optimization.py
```

## Generated Files
1. `stock_data.csv` - Historical stock data
2. `returns_data.csv` - Daily returns data
3. `simulation_results.csv` - Monte Carlo simulation results
4. `simulation_weights.csv` - Portfolio weights
5. `portfolio_report.csv` - Optimized portfolio report
6. `performance_metrics.csv` - Performance metrics
7. `efficient_frontier.png` - Visualization

## Technical Details
- Uses real market data via yfinance
- Implements Modern Portfolio Theory
- Optimizes Sharpe Ratio
- Considers transaction costs and constraints
