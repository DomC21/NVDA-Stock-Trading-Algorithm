import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .data_fetcher import DataFetcher
from .signal_generator import SignalGenerator

class TestPredictionModel(unittest.TestCase):
    def setUp(self):
        self.data_fetcher = DataFetcher()
        self.signal_generator = SignalGenerator()
        self.sample_data = self._generate_sample_data()
        
    def _generate_sample_data(self):
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        return pd.DataFrame({
            'Close': np.linspace(100, 200, 100) + np.random.normal(0, 5, 100),
            'High': np.linspace(105, 205, 100) + np.random.normal(0, 5, 100),
            'Low': np.linspace(95, 195, 100) + np.random.normal(0, 5, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
    @patch('src.data_fetcher.DataFetcher')
    def test_prediction_model(self, mock_fetcher):
        mock_fetcher.return_value.get_polygon_data.return_value = pd.DataFrame({
            'volume': [1000000] * 10,
            'vwap': [150.0] * 10
        })
        
        # Test basic prediction
        signal = self.signal_generator.generate_signal(self.sample_data)
        self.assertIsInstance(signal, dict)
        self.assertIn('signal', signal)
        self.assertTrue(-1 <= signal['signal'] <= 1)
        
    def test_extended_backtest(self):
        # Test prediction over multiple days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        test_data = pd.DataFrame({
            'Close': np.linspace(100, 200, len(dates)) + np.random.normal(0, 5, len(dates)),
            'High': np.linspace(105, 205, len(dates)) + np.random.normal(0, 5, len(dates)),
            'Low': np.linspace(95, 195, len(dates)) + np.random.normal(0, 5, len(dates)),
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        signals = []
        for i in range(len(test_data) - 5):  # 5-day prediction window
            window = test_data.iloc[i:i+5]
            signal = self.signal_generator.generate_signal(window)
            signals.append(signal['signal'])
            
        self.assertEqual(len(signals), len(test_data) - 5)
        self.assertTrue(all(isinstance(s, (int, float)) for s in signals))
        self.assertTrue(all(-1 <= s <= 1 for s in signals))
        
    def test_prediction_consistency(self):
        # Test prediction consistency with same input
        signal1 = self.signal_generator.generate_signal(self.sample_data)
        signal2 = self.signal_generator.generate_signal(self.sample_data)
        
        self.assertEqual(signal1['signal'], signal2['signal'])
        
        # Test prediction changes with different market conditions
        modified_data = self.sample_data.copy()
        modified_data['Close'] = modified_data['Close'] * 1.1  # 10% price increase
        signal3 = self.signal_generator.generate_signal(modified_data)
        
        self.assertNotEqual(signal1['signal'], signal3['signal'])

if __name__ == '__main__':
    unittest.main()
