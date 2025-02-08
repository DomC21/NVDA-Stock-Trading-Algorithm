from data_collector import DataCollector
import pandas as pd
pd.set_option('display.max_rows', 10)

def main():
    collector = DataCollector()
    data = collector.collect_all_data()
    
    print("\nPolygon.io Data Sample:")
    if not data['polygon'].empty:
        print(data['polygon'].head())
    else:
        print("Failed to fetch Polygon data")
        
    print("\nAlpha Vantage Data Sample:")
    if not data['alpha_vantage'].empty:
        print(data['alpha_vantage'].head())
    else:
        print("Failed to fetch Alpha Vantage data")
        
    print("\nYFinance Data Sample:")
    if not data['yfinance'].empty:
        print(data['yfinance'].head())
    else:
        print("Failed to fetch YFinance data")

if __name__ == "__main__":
    main()
