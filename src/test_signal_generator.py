import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .signal_generator import SignalGenerator
from .market_regime_analyzer import MarketRegimeAnalysis, MarketRegimeAnalyzer
from .sentiment_analyzer import SentimentAnalyzer

class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SignalGenerator()
        self.sample_data = self._generate_sample_data()
        
    def _generate_sample_data(self):
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        data = pd.DataFrame({
            'Close': np.linspace(100, 200, 100) + np.random.normal(0, 5, 100),
            'High': np.linspace(105, 205, 100) + np.random.normal(0, 5, 100),
            'Low': np.linspace(95, 195, 100) + np.random.normal(0, 5, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        return data
        
    @patch('src.dark_pool_analyzer.DarkPoolAnalyzer')
    @patch('src.market_regime_analyzer.MarketRegimeAnalyzer')
    @patch('src.sentiment_analyzer.SentimentAnalyzer')
    def test_generate_signal(self, mock_market, mock_dark_pool, mock_sentiment):
        mock_dark_pool_instance = Mock()
        mock_dark_pool_instance.fetch_unusual_whales_data.return_value = {
            'dark_pool': {'trades': []},
            'option_volume': {'calls': [], 'puts': []},
            'greeks': {'delta': 0.5, 'gamma': 0.05}
        }
        mock_dark_pool_instance.analyze_unusual_whales_data.return_value = {
            'dark_pool_sentiment': {'signal': 0.5},
            'option_flow_signals': {'signal': 0.3},
            'greek_exposure': {'signal': 0.2},
            'composite_signal': 0.4
        }
        mock_dark_pool.return_value = mock_dark_pool_instance
        
        mock_market_instance = Mock()
        mock_market_instance.analyze.return_value = MarketRegimeAnalysis(
            regime='trending',
            trend_strength=0.7,
            volatility_regime='normal',
            momentum_score=0.6,
            support_resistance=[180.0, 220.0],
            confidence=0.8
        )
        mock_market.return_value = mock_market_instance
        
        mock_sentiment_instance = Mock()
        mock_sentiment_instance.fetch_news_sentiment.return_value = [
            Mock(sentiment_score=0.5, relevance_score=0.8, impact_score=0.4)
        ]
        mock_sentiment_instance.analyze_sentiment.return_value = {
            'composite_score': 0.4,
            'recent_sentiment': 0.5,
            'sentiment_momentum': 0.3,
            'confidence': 0.8
        }
        mock_sentiment.return_value = mock_sentiment_instance
        
        signal = self.generator.generate_signal(self.sample_data)
        
        self.assertIsInstance(signal, dict)
        self.assertIn('signal', signal)
        self.assertIn('dark_pool_sentiment', signal)
        self.assertIn('option_flow', signal)
        self.assertIn('greek_exposure', signal)
        self.assertIn('market_regime', signal)
        self.assertIn('volume_analysis', signal)
        self.assertIn('risk_levels', signal)
        
        self.assertTrue(-1 <= signal['signal'] <= 1)
        
    def test_calculate_composite_signal(self):
        dark_pool = {
            'composite_signal': 0.5,
            'dark_pool_sentiment': {'signal': 0.5},
            'option_flow_signals': {'signal': 0.3},
            'greek_exposure': {'signal': 0.2}
        }
        
        regime = MarketRegimeAnalysis(
            regime='trending',
            trend_strength=0.7,
            volatility_regime='normal',
            momentum_score=0.6,
            support_resistance=[180.0, 220.0],
            confidence=0.8
        )
        
        volume = {
            'long': {'volume_confidence': 0.8},
            'short': {'volume_confidence': 0.3}
        }
        
        sentiment = {
            'composite_score': 0.4,
            'recent_sentiment': 0.5,
            'sentiment_momentum': 0.3,
            'confidence': 0.8
        }
        signal = self.generator._calculate_composite_signal(dark_pool, regime, volume, sentiment)
        self.assertTrue(-1 <= signal <= 1)
        
        # Test bearish scenario
        regime.trend_strength = -0.7
        bearish_signal = self.generator._calculate_composite_signal(dark_pool, regime, volume, sentiment)
        self.assertLess(bearish_signal, signal)
        
        # Test high volatility adjustment
        regime.volatility_regime = 'high'
        volatile_signal = self.generator._calculate_composite_signal(dark_pool, regime, volume, sentiment)
        self.assertNotEqual(signal, volatile_signal)

if __name__ == '__main__':
    unittest.main()
