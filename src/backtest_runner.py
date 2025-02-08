import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from .data_collector import DataCollector
from .price_predictor import PricePredictor
from .risk_manager import RiskManager
from .market_regime_analyzer import MarketRegimeAnalyzer
from .analysis import StockAnalyzer
from .config import SYMBOL

class BacktestRunner:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = 0
        self.trades = []
        self.collector = DataCollector()
        self.predictor = PricePredictor()
        self.risk_manager = RiskManager(initial_capital)
        self.market_analyzer = MarketRegimeAnalyzer()
        self.stock_analyzer = StockAnalyzer()
        self.model_trained = False
        
    def run_backtest(self, start_date, end_date):
        try:
            # Ensure we only use historical data
            if isinstance(end_date, datetime):
                end_date = min(end_date, datetime.now() - timedelta(days=1))
            if isinstance(start_date, datetime):
                start_date = min(start_date, end_date)
                
            print(f"\nCollecting data for period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            data = self.collector.collect_historical_data('NVDA', start_date, end_date)
            if data is None or len(data) == 0:
                print(f"No data available for period {start_date} to {end_date}")
                return None
                
            print(f"Collected {len(data)} data points")
            min_data_points = 5  # Further reduced minimum required data points
            if len(data) < min_data_points:
                print(f"Insufficient data points ({len(data)}) for period {start_date} to {end_date}")
                return None
                
            # Extract features first
            features = self.predictor._extract_features(data)
            if features is None or len(features) < min_data_points:
                print(f"Error extracting features or insufficient feature data points")
                return None
                
            # Print data shape information for debugging
            print(f"Data shape: {data.shape}, Features shape: {features.shape}")
            print(f"Available features: {features.columns.tolist()}")
                
            # Dynamically adjust sequence length based on available data
            original_sequence_length = self.predictor.sequence_length
            if len(features) < 50:
                self.predictor.sequence_length = max(3, min(10, len(features) // 3))
                print(f"Adjusted sequence length to {self.predictor.sequence_length} for {len(features)} data points")
                
            # Train the model if not already trained
            if not self.model_trained:
                self.predictor.scaler.fit(features)
                X, y = self.predictor._create_sequences(features)
                if len(X) > 0 and len(y) > 0:
                    self.predictor.train(X, y)
                    self.model_trained = True
                    self.predictor.feature_columns = features.columns.tolist()  # Changed to public attribute
                else:
                    print("Failed to create valid sequences for training")
                    return None
                    
            results = {
                'dates': [],
                'actual_prices': [],
                'predicted_prices': [],
                'positions': [],
                'returns': [],
                'capital': []
            }
            
            try:
                window_size = min(20, len(data) // 4)  # Reduced window size for shorter periods
                for i in range(window_size, len(data) - 1):
                    try:
                        current_date = data.index[i]
                        window_data = data.iloc[i-window_size:i]
                        
                        prediction = self.predictor.predict_next_day(window_data)
                        if prediction is None:
                            continue
                            
                        actual_next_price = data.iloc[i+1]['Close']
                        current_price = data.iloc[i]['Close']
                        
                        # Calculate volatility using available data
                        volatility = window_data['Close'].pct_change().std()
                        if pd.notnull(volatility):
                            volatility *= np.sqrt(252)  # Annualize volatility
                        else:
                            volatility = 0.02  # Default 2% volatility if can't calculate
                            
                        # Determine market regime based on recent price action
                        recent_return = window_data['Close'].pct_change().mean()
                        if recent_return > 0.01:
                            market_regime = 'bullish'
                        elif recent_return < -0.01:
                            market_regime = 'bearish'
                        else:
                            market_regime = 'normal'
                        
                        # More conservative thresholds for shorter periods
                        buy_threshold = 1.005 if len(data) < 60 else 1.01
                        sell_threshold = 0.995 if len(data) < 60 else 0.99
                        
                        if prediction > current_price * buy_threshold:  # Buy signal
                            if self.positions == 0:
                                shares, metrics = self.risk_manager.calculate_position_size(
                                    current_price, volatility, market_regime
                                )
                                cost = shares * current_price
                                if cost <= self.capital:
                                    self.positions = shares
                                    self.capital -= cost
                                    self.trades.append({
                                        'date': current_date,
                                        'type': 'buy',
                                        'price': current_price,
                                        'shares': shares,
                                        'cost': cost,
                                        'prediction': prediction,
                                        'confidence': metrics.get('confidence', 0.5)
                                    })
                        
                        elif prediction < current_price * sell_threshold:  # Sell signal
                            if self.positions > 0:
                                proceeds = self.positions * current_price
                                self.capital += proceeds
                                self.trades.append({
                                    'date': current_date,
                                    'type': 'sell',
                                    'price': current_price,
                                    'shares': self.positions,
                                    'proceeds': proceeds,
                                    'prediction': prediction
                                })
                                self.positions = 0
                        
                        portfolio_value = self.capital + (self.positions * current_price)
                        daily_return = (portfolio_value - self.initial_capital) / self.initial_capital
            
                        results['dates'].append(current_date)
                        results['actual_prices'].append(actual_next_price)
                        results['predicted_prices'].append(prediction)
                        results['positions'].append(self.positions)
                        results['returns'].append(daily_return)
                        results['capital'].append(portfolio_value)
                    except Exception as e:
                        print(f"Error processing window at {current_date}: {str(e)}")
                        continue
                        
                if len(results['dates']) > 0:
                    return pd.DataFrame(results)
                else:
                    print("No valid results generated during backtesting")
                    return None
            except Exception as e:
                print(f"Error during backtest window setup: {str(e)}")
                return None
        except Exception as e:
            print(f"Error during backtest initialization: {str(e)}")
            return None
    
    def calculate_metrics(self, results):
        if results is None or len(results) == 0:
            return None
            
        returns = pd.Series(results['returns'])
        metrics = {
            'total_return': ((results['capital'].iloc[-1] - self.initial_capital) 
                           / self.initial_capital * 100),
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() 
                          if returns.std() != 0 else 0,
            'max_drawdown': (returns.cummax() - returns).max() * 100,
            'win_rate': len([t for t in self.trades if t.get('proceeds', 0) > t.get('cost', 0)]) 
                       / len(self.trades) if len(self.trades) > 0 else 0,
            'total_trades': len(self.trades),
            'prediction_accuracy': np.mean(
                np.sign(np.diff(results['predicted_prices'])) == 
                np.sign(np.diff(results['actual_prices']))
            ) * 100
        }
        return metrics

def run_full_backtest(periods=None):
    if periods is None:
        end_date = datetime.now()
        periods = [
            (end_date - timedelta(days=365), end_date, "1 Year"),
            (end_date - timedelta(days=90), end_date, "3 Months"),
            (end_date - timedelta(days=60), end_date, "2 Months"),
            (end_date - timedelta(days=30), end_date, "1 Month")
        ]
    
    all_results = {}
    all_metrics = {}
    
    for start_date, end_date, period_name in periods:
        print(f"\nRunning backtest for {period_name} period...")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        try:
            runner = BacktestRunner()
            results = runner.run_backtest(start_date, end_date)
            
            if results is not None:
                metrics = runner.calculate_metrics(results)
                if metrics:  # Only process if we have valid metrics
                    all_results[period_name] = results
                    all_metrics[period_name] = metrics
                    
                    # Plot results for this period
                    plt.figure(figsize=(15, 10))
                    plt.subplot(2, 1, 1)
                    plt.plot(results['dates'], results['actual_prices'], label='Actual Price', color='blue')
                    plt.plot(results['dates'], results['predicted_prices'], label='Predicted Price', color='red', linestyle='--')
                    plt.title(f'{SYMBOL} Price Prediction vs Actual - {period_name}')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.subplot(2, 1, 2)
                    plt.plot(results['dates'], results['capital'], label='Portfolio Value', color='green')
                    plt.title(f'Portfolio Value Over Time - {period_name}')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plot_filename = f'backtest_results_{period_name.lower().replace(" ", "_")}.png'
                    plt.savefig(plot_filename)
                    plt.close()
                    
                    print(f"\nBacktest Results for {period_name}:")
                    print("-" * 50)
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.2f}")
                else:
                    print(f"No valid metrics generated for {period_name}")
        except Exception as e:
            print(f"Error running backtest for {period_name}: {str(e)}")
            continue
    
    return all_results, all_metrics

if __name__ == '__main__':
    all_results, all_metrics = run_full_backtest()
    if all_metrics:
        for period_name, metrics in all_metrics.items():
            print(f"\nBacktest Results for {SYMBOL} - {period_name}:")
            print("-" * 50)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")
            print(f"\nPlot saved as backtest_results_{period_name.lower().replace(' ', '_')}.png")
