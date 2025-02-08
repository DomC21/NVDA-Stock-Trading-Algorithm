import logging
from datetime import datetime, timedelta
logging.basicConfig(level=logging.INFO)

from src.backtest_runner import run_full_backtest

def main():
    print('\nRunning Extended Backtests for NVDA...')
    print('=' * 50)
    
    # Use dates up to yesterday for historical analysis
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    
    # Define test periods - all ending yesterday
    periods = [
        (end_date - timedelta(days=365), end_date, "Past Year"),
        (end_date - timedelta(days=180), end_date, "Past 6 Months"),
        (end_date - timedelta(days=90), end_date, "Past 3 Months"),
        (end_date - timedelta(days=30), end_date, "Past Month")
    ]
    
    print(f"Running backtests using historical data up to {end_date.strftime('%Y-%m-%d')}")
    print("Testing periods:")
    for start, end, name in periods:
        print(f"- {name}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    results, metrics = run_full_backtest(periods)
    
    if metrics:
        print('\nSummary of Results Across All Periods:')
        print('=' * 50)
        for period_name, period_metrics in metrics.items():
            print(f'\n{period_name}:')
            print('-' * 30)
            for metric, value in period_metrics.items():
                if isinstance(value, float):
                    print(f'{metric}: {value:.2f}')
                else:
                    print(f'{metric}: {value}')
            
            if period_name in results:
                period_data = results[period_name]
                if not period_data.empty:
                    total_trades = len([p for p in period_data['positions'] if p > 0])
                    print(f'Total Trades: {total_trades}')
                    print(f'Data Points: {len(period_data)}')
    else:
        print('No valid backtest results generated')
        
    print('\nBacktest plots have been saved as:')
    print('- backtest_results_1_year.png')
    print('- backtest_results_3_months.png')
    print('- backtest_results_2_months.png')
    print('- backtest_results_1_month.png')

if __name__ == '__main__':
    main()
