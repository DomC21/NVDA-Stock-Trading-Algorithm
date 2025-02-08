from trading_algorithm import TradingAlgorithm

def main():
    algo = TradingAlgorithm()
    signals = algo.generate_trading_signals()
    
    print("\nNVDA Trading Signals:")
    print("=" * 50)
    print(f"\nCombined Recommendation:")
    print(f"Signal: {signals['combined_recommendation']['signal']}")
    print(f"Confidence: {signals['combined_recommendation']['confidence']:.2f}%")
    
    print("\nTechnical Analysis Reasons:")
    for reason in signals['combined_recommendation']['technical_reasons']:
        print(f"- {reason}")
        
    print("\nOptions Flow Analysis Reasons:")
    for reason in signals['combined_recommendation']['options_reasons']:
        print(f"- {reason}")

if __name__ == "__main__":
    main()
