import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self, initial_investment=10000000):
        """
        Initialize Portfolio Optimizer
        initial_investment: Portfolio value in dollars
        """
        self.initial_investment = initial_investment
        self.stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'BRK-B', 
                      'JPM', 'JNJ', 'V', 'PG', 'XOM']
        self.risk_free_rate = 0.02  # 2% risk-free rate
        
    def fetch_data(self, start_date='2020-01-01', end_date=None):
        """Fetch historical stock data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print("Fetching historical stock data...")
        self.stock_data = pd.DataFrame()
        
        for stock in self.stocks:
            try:
                data = yf.download(stock, start=start_date, end=end_date)['Adj Close']
                self.stock_data[stock] = data
            except Exception as e:
                print(f"Error fetching {stock}: {str(e)}")
                
        # Calculate daily returns
        self.returns = self.stock_data.pct_change()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        # Save data to CSV
        self.stock_data.to_csv('stock_data.csv')
        self.returns.to_csv('returns_data.csv')
        
        return self.stock_data
        
    def portfolio_performance(self, weights):
        """Calculate portfolio performance metrics"""
        returns = np.sum(self.mean_returns * weights) * 252
        risk = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
        )
        sharpe_ratio = (returns - self.risk_free_rate) / risk
        return returns, risk, sharpe_ratio
        
    def negative_sharpe(self, weights):
        """Negative Sharpe Ratio for optimization"""
        return -self.portfolio_performance(weights)[2]
        
    def optimize_portfolio(self):
        """Optimize portfolio weights using Sharpe Ratio"""
        num_assets = len(self.stocks)
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        )
        bounds = tuple((0, 1) for asset in range(num_assets))
        
        # Equal weights as initial guess
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            self.negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result
        
    def run_monte_carlo(self, num_simulations=10000):
        """Run Monte Carlo simulations"""
        print(f"Running {num_simulations} Monte Carlo simulations...")
        
        # Store results
        results = np.zeros((num_simulations, 3))  # returns, risk, sharpe
        all_weights = np.zeros((num_simulations, len(self.stocks)))
        
        for i in range(num_simulations):
            # Generate random weights
            weights = np.random.random(len(self.stocks))
            weights = weights / np.sum(weights)
            
            # Calculate portfolio performance
            returns, risk, sharpe = self.portfolio_performance(weights)
            
            # Store results
            results[i] = [returns, risk, sharpe]
            all_weights[i] = weights
            
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1} simulations...")
        
        # Convert results to DataFrame
        self.simulation_results = pd.DataFrame(
            results,
            columns=['Returns', 'Risk', 'Sharpe']
        )
        self.simulation_weights = pd.DataFrame(
            all_weights,
            columns=self.stocks
        )
        
        # Save results
        self.simulation_results.to_csv('simulation_results.csv')
        self.simulation_weights.to_csv('simulation_weights.csv')
        
        return self.simulation_results, self.simulation_weights
        
    def generate_efficient_frontier(self):
        """Generate efficient frontier plot"""
        plt.figure(figsize=(12, 8))
        plt.scatter(
            self.simulation_results['Risk'],
            self.simulation_results['Returns'],
            c=self.simulation_results['Sharpe'],
            cmap='viridis',
            marker='o',
            s=10,
            alpha=0.3
        )
        
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Risk (Volatility)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        
        # Save plot
        plt.savefig('efficient_frontier.png')
        plt.close()
        
    def generate_portfolio_report(self):
        """Generate comprehensive portfolio report"""
        # Get optimized portfolio
        optimization_result = self.optimize_portfolio()
        optimal_weights = optimization_result['x']
        optimal_returns, optimal_risk, optimal_sharpe = self.portfolio_performance(optimal_weights)
        
        # Calculate portfolio values
        optimal_portfolio_value = self.initial_investment * (1 + optimal_returns)
        value_increase = optimal_portfolio_value - self.initial_investment
        percentage_increase = (value_increase / self.initial_investment) * 100
        
        # Create report DataFrame
        report_data = {
            'Asset': self.stocks,
            'Optimal Weight': optimal_weights,
            'Allocation': optimal_weights * self.initial_investment
        }
        
        report_df = pd.DataFrame(report_data)
        report_df['Allocation'] = report_df['Allocation'].round(2)
        
        # Save report
        report_df.to_csv('portfolio_report.csv')
        
        # Performance metrics
        metrics = pd.DataFrame({
            'Metric': ['Initial Investment', 'Optimized Value', 'Return (%)', 
                      'Risk (%)', 'Sharpe Ratio'],
            'Value': [
                f"${self.initial_investment:,.2f}",
                f"${optimal_portfolio_value:,.2f}",
                f"{optimal_returns*100:.2f}%",
                f"{optimal_risk*100:.2f}%",
                f"{optimal_sharpe:.2f}"
            ]
        })
        
        metrics.to_csv('performance_metrics.csv')
        
        return report_df, metrics

def main():
    # Initialize optimizer
    optimizer = PortfolioOptimizer(initial_investment=10000000)
    
    # Fetch historical data
    data = optimizer.fetch_data()
    
    # Run Monte Carlo simulations
    simulation_results, weights = optimizer.run_monte_carlo()
    
    # Generate efficient frontier plot
    optimizer.generate_efficient_frontier()
    
    # Generate portfolio report
    report, metrics = optimizer.generate_portfolio_report()
    
    print("\nPortfolio Optimization Complete!")
    print("\nPerformance Metrics:")
    print(metrics.to_string(index=False))
    
    print("\nFiles generated:")
    print("1. stock_data.csv - Historical stock data")
    print("2. returns_data.csv - Daily returns data")
    print("3. simulation_results.csv - Monte Carlo simulation results")
    print("4. simulation_weights.csv - Portfolio weights from simulations")
    print("5. portfolio_report.csv - Optimized portfolio report")
    print("6. performance_metrics.csv - Performance metrics")
    print("7. efficient_frontier.png - Efficient frontier plot")

if __name__ == "__main__":
    main()
