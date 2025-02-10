import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .signal_generator import SignalGenerator
from .dark_pool_analyzer import DarkPoolAnalyzer

class TestGreeksRiskManager(unittest.TestCase):
    def setUp(self):
        self.signal_generator = SignalGenerator()
        self.dark_pool_analyzer = DarkPoolAnalyzer()
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
    def test_greeks_risk_adjustment(self, mock_fetcher):
        # Mock Greeks data
        mock_fetcher.return_value.fetch_greeks_exposure.return_value = {
            'delta': 0.8,  # High delta exposure
            'gamma': 0.15,  # High gamma exposure
            'theta': -0.002  # Significant theta decay
        }
        
        # Generate signal with high-risk Greeks
        signal = self.signal_generator.generate_signal(self.sample_data)
        self.assertIsInstance(signal, dict)
        self.assertIn('signal', signal)
        
        # Signal should be reduced due to high risk Greeks
        self.assertLess(abs(signal['signal']), 1.0)
        
        # Test with low-risk Greeks
        mock_fetcher.return_value.fetch_greeks_exposure.return_value = {
            'delta': 0.3,
            'gamma': 0.05,
            'theta': -0.0005
        }
        
        signal_low_risk = self.signal_generator.generate_signal(self.sample_data)
        self.assertGreater(abs(signal_low_risk['signal']), abs(signal['signal']))
        
    def test_dark_pool_greek_analysis(self):
        # Test Greek exposure analysis from dark pool data
        sample_greeks = {
            'delta': 0.5,
            'gamma': 0.05,
            'theta': -0.001
        }
        
        analysis = self.dark_pool_analyzer._analyze_greek_exposure(sample_greeks)
        self.assertIsInstance(analysis, dict)
        self.assertIn('signal', analysis)
        self.assertIn('net_delta', analysis)
        self.assertIn('net_gamma', analysis)
        self.assertIn('net_theta', analysis)
        
        # Test extreme Greek values
        extreme_greeks = {
            'delta': 0.9,
            'gamma': 0.2,
            'theta': -0.005
        }
        
        extreme_analysis = self.dark_pool_analyzer._analyze_greek_exposure(extreme_greeks)
        self.assertLess(abs(extreme_analysis['signal']), 1.0)
        
    def test_missing_greeks_handling(self):
        # Test handling of missing Greeks data
        empty_greeks = {}
        analysis = self.dark_pool_analyzer._analyze_greek_exposure(empty_greeks)
        self.assertEqual(analysis['signal'], 0)
        
        partial_greeks = {'delta': 0.5}
        partial_analysis = self.dark_pool_analyzer._analyze_greek_exposure(partial_greeks)
        self.assertIsInstance(partial_analysis['signal'], (int, float))

if __name__ == '__main__':
    unittest.main()
