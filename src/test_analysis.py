from visualizer import StockVisualizer

def main():
    visualizer = StockVisualizer()
    
    # Generate and save technical analysis plot
    plot_path = visualizer.plot_technical_analysis()
    print(f"\nTechnical analysis plot saved to: {plot_path}")
    
    # Generate and print analysis report
    report = visualizer.generate_analysis_report()
    print("\nNVDA Analysis Report:")
    print("=" * 50)
    print(report)

if __name__ == "__main__":
    main()
