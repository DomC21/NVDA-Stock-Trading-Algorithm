import unittest
import pandas as pd
import numpy as np
from reality_check import RealityCheck

class TestRealityCheck(unittest.TestCase):
    def setUp(self):
        self.checker = RealityCheck(
            min_volume_threshold=500000,  # Higher for NVDA's typical volume
            max_spread_threshold=0.02,    # Tighter spread for NVDA
            min_confidence=0.7,          # Higher confidence requirement
            max_gamma_exposure=0.15,     # NVDA-specific gamma threshold
            max_vega_exposure=0.15,      # Higher vega tolerance for NVDA
            max_theta_decay=0.002,       # Stricter theta decay limit
            min_delta_hedge=0.4,         # Higher hedge requirement
            high_block_threshold=2_000_000  # NVDA-specific block size
        )
        
        # Sample Greeks data
        self.normal_greeks = {
            'gamma': 0.002,
            'vega': 0.15,
            'theta': -0.05,
            'delta': 0.6,
            'hedge_ratio': 0.8
        }
        
        self.high_risk_greeks = {
            'gamma': 0.05,
            'vega': 0.4,
            'theta': -0.2,
            'delta': 0.9,
            'hedge_ratio': 0.2
        }
        
        # Sample dark pool data
        self.dark_pool_data = pd.DataFrame({
            'price': [100.0, 101.0, 99.0],
            'volume': [500000, 300000, 200000],
            'side': ['buy', 'buy', 'sell']
        })
        
        self.large_block_data = pd.DataFrame({
            'price': [98.0, 97.5, 97.0],
            'volume': [2000000, 1500000, 1000000],
            'side': ['sell', 'sell', 'sell']  # All selling pressure
        })
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        self.market_data = pd.DataFrame({
            'Close': 100 * (1 + np.random.normal(0.0001, 0.02, len(dates))).cumprod(),
            'High': 101 * (1 + np.random.normal(0.0001, 0.02, len(dates))).cumprod(),
            'Low': 99 * (1 + np.random.normal(0.0001, 0.02, len(dates))).cumprod(),
            'Volume': np.random.normal(500000, 50000, len(dates))
        }, index=dates)
        
        self.options_data = pd.DataFrame({
            'call_volume': np.random.normal(10000, 1000, len(dates)),
            'put_volume': np.random.normal(8000, 800, len(dates))
        }, index=dates)
        
    def test_volume_check(self):
        result = self.checker._check_volume(self.market_data)
        self.assertIn('valid', result)
        self.assertIn('reason', result)
        self.assertIn('context', result)
        self.assertIn('volume_ratio', result['context'])
        
    def test_spread_check(self):
        result = self.checker._check_spread(self.market_data)
        self.assertIn('valid', result)
        self.assertIn('reason', result)
        self.assertIn('context', result)
        self.assertIn('spread', result['context'])
        
    def test_trend_alignment(self):
        signal = {'direction': 'buy'}
        result = self.checker._check_trend_alignment(signal, self.market_data)
        self.assertIn('valid', result)
        self.assertIn('reason', result)
        self.assertIn('context', result)
        self.assertIn('strength', result['context'])
        
    def test_nvda_specific_thresholds(self):
        # Verify NVDA-specific thresholds
        self.assertEqual(self.checker.min_option_volume, 10000)
        self.assertEqual(self.checker.max_put_call_ratio, 2.0)
        self.assertEqual(self.checker.min_dark_pool_size, 100000)
        
    def test_options_flow(self):
        signal = {'direction': 'buy'}
        
        # Test normal options flow
        normal_options = pd.DataFrame({
            'call_volume': [15000] * 10,
            'put_volume': [8000] * 10
        })
        result = self.checker._check_options_flow(signal, normal_options)
        self.assertTrue(result['valid'])
        self.assertIn('call_volume', result['context'])
        self.assertIn('put_volume', result['context'])
        
        # Test insufficient volume
        low_volume = pd.DataFrame({
            'call_volume': [5000] * 10,
            'put_volume': [3000] * 10
        })
        result = self.checker._check_options_flow(signal, low_volume)
        self.assertFalse(result['valid'])
        self.assertIn('Insufficient options volume', result['reason'])
        
        # Test extreme options imbalance
        imbalanced = pd.DataFrame({
            'call_volume': [50000] * 10,
            'put_volume': [5000] * 10
        })
        result = self.checker._check_options_flow(signal, imbalanced)
        self.assertFalse(result['valid'])
        self.assertIn('Extreme options imbalance', result['reason'])
        
    def test_volatility_check(self):
        result = self.checker._check_volatility(self.market_data)
        self.assertIn('valid', result)
        self.assertIn('reason', result)
        self.assertIn('context', result)
        self.assertIn('volatility', result['context'])
        
    def test_signal_validation(self):
        signal = {'direction': 'buy'}
        
        # Test with all data sources
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            self.options_data,
            self.dark_pool_data,
            self.normal_greeks
        )
        
        self.assertIsInstance(result.is_valid, bool)
        self.assertTrue(0 <= result.confidence <= 1)
        self.assertIsInstance(result.failure_reasons, list)
        self.assertIsInstance(result.market_context, dict)
        self.assertIn('gamma_exposure', result.market_context)
        self.assertIn('block_sum', result.market_context)
        
        # Test with high risk conditions
        high_risk_result = self.checker.validate_signal(
            signal,
            self.market_data,
            self.options_data,
            self.large_block_data,
            self.high_risk_greeks
        )
        
        self.assertFalse(high_risk_result.is_valid)
        self.assertLess(high_risk_result.confidence, 0.5)
        self.assertTrue(len(high_risk_result.failure_reasons) > 0)
        self.assertTrue(any('gamma' in reason.lower() for reason in high_risk_result.failure_reasons))
        
        self.assertIsInstance(result.is_valid, bool)
        self.assertTrue(0 <= result.confidence <= 1)
        self.assertIsInstance(result.failure_reasons, list)
        self.assertIsInstance(result.market_context, dict)
        
    def test_dark_pool_validation(self):
        signal = {'direction': 'buy'}
        
        # Test with normal dark pool data
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            dark_pool_data=self.dark_pool_data
        )
        
        self.assertTrue(result.is_valid)
        self.assertIn('block_sum', result.market_context)
        self.assertIn('buy_ratio', result.market_context)
        
        # Test with large block data showing selling pressure
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            dark_pool_data=self.large_block_data
        )
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('dark pool' in reason.lower() for reason in result.failure_reasons))
        
        # Test with extreme buy/sell imbalance
        imbalanced_data = pd.DataFrame({
            'price': [100.0] * 5,
            'volume': [200000] * 5,
            'side': ['sell'] * 5  # All selling pressure
        })
        
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            dark_pool_data=imbalanced_data
        )
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('sentiment' in reason.lower() for reason in result.failure_reasons))
        
    def test_greeks_validation(self):
        signal = {'direction': 'buy'}
        
        # Test with normal Greeks for NVDA
        normal_greeks = {
            'gamma': 0.10,    # Below NVDA's 0.15 threshold
            'vega': 0.10,     # Below NVDA's 0.15 threshold
            'theta': -0.001,  # Below NVDA's 0.002 threshold
            'delta': 0.6,
            'hedge_ratio': 0.5
        }
        
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            greeks_data=normal_greeks
        )
        
        self.assertTrue(result.is_valid)
        self.assertIn('gamma_exposure', result.market_context)
        self.assertIn('vega_exposure', result.market_context)
        self.assertIn('theta_decay', result.market_context)
        
        # Test with high risk Greeks (above NVDA thresholds)
        high_risk_greeks = {
            'gamma': 0.16,    # Above NVDA's threshold
            'vega': 0.16,     # Above NVDA's threshold
            'theta': -0.003,  # Above NVDA's threshold
            'delta': 0.9,
            'hedge_ratio': 0.3  # Below minimum hedge ratio
        }
        
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            greeks_data=high_risk_greeks
        )
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('gamma' in reason.lower() for reason in result.failure_reasons))
        self.assertTrue(any('vega' in reason.lower() for reason in result.failure_reasons))
        self.assertTrue(any('theta' in reason.lower() for reason in result.failure_reasons))
        self.assertTrue(any('hedge' in reason.lower() for reason in result.failure_reasons))
        
    def test_edge_cases(self):
        signal = {'direction': 'buy'}
        
        # Test with volume below NVDA threshold
        low_volume_data = self.market_data.copy()
        low_volume_data['Volume'] = 400000  # Below NVDA's 500k threshold
        
        result = self.checker.validate_signal(signal, low_volume_data)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('volume' in reason.lower() for reason in result.failure_reasons))
        
        # Test with spread beyond NVDA threshold
        high_spread_data = self.market_data.copy()
        high_spread_data['High'] = high_spread_data['Close'] * 1.03  # Above NVDA's 2% threshold
        high_spread_data['Low'] = high_spread_data['Close'] * 0.97
        
        result = self.checker.validate_signal(signal, high_spread_data)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('spread' in reason.lower() for reason in result.failure_reasons))
        
        # Test extreme options imbalance
        extreme_options = pd.DataFrame({
            'call_volume': [100000] * 10,  # 10:1 ratio
            'put_volume': [10000] * 10
        })
        
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            options_data=extreme_options
        )
        self.assertFalse(result.is_valid)
        self.assertTrue(any('imbalance' in reason.lower() for reason in result.failure_reasons))
        
        # Test extreme dark pool activity
        extreme_dark_pool = pd.DataFrame({
            'price': [95.0] * 5,  # 5% below market
            'volume': [3000000] * 5,  # Well above threshold
            'side': ['sell'] * 5  # All selling pressure
        })
        
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            dark_pool_data=extreme_dark_pool
        )
        self.assertFalse(result.is_valid)
        self.assertTrue(any('dark pool' in reason.lower() for reason in result.failure_reasons))
        
        # Test extreme Greeks exposure
        extreme_greeks = {
            'gamma': 0.2,     # Above 0.15 threshold
            'vega': 0.2,      # Above 0.15 threshold
            'theta': -0.005,  # Above 0.002 threshold
            'delta': 0.95,    # Very high delta
            'hedge_ratio': 0.2  # Below 0.4 threshold
        }
        
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            greeks_data=extreme_greeks
        )
        self.assertFalse(result.is_valid)
        self.assertTrue(len([r for r in result.failure_reasons if any(g in r.lower() for g in ['gamma', 'vega', 'theta', 'hedge'])]) >= 3)

if __name__ == '__main__':
    unittest.main()
