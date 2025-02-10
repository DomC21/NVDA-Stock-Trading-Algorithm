import pandas as pd
import numpy as np
from typing import Dict, Optional
from .dark_pool_analyzer import DarkPoolAnalyzer
from .volume_analyzer import VolumeAnalyzer
from .market_regime_analyzer import MarketRegimeAnalyzer, MarketRegimeAnalysis
from .sentiment_analyzer import SentimentAnalyzer
import os
from dotenv import load_dotenv

load_dotenv()

class SignalGenerator:
    def __init__(self):
        self.dark_pool_analyzer = DarkPoolAnalyzer()
        self.market_analyzer = MarketRegimeAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer(os.getenv('ALPHA_VANTAGE_API_KEY'))
        
    def generate_signal(self, price_data: pd.DataFrame) -> Dict:
        volume_analyzer = VolumeAnalyzer(price_data)
        
        # Fetch and analyze dark pool data
        dark_pool_data = self.dark_pool_analyzer.fetch_unusual_whales_data()
        dark_pool_analysis = self.dark_pool_analyzer.analyze_unusual_whales_data(dark_pool_data)
        
        # Analyze market regime
        regime = self.market_analyzer.analyze_regime(price_data)
        
        # Get volume analysis
        volume_profile = volume_analyzer.calculate_volume_profile()
        volume_signals = volume_analyzer.get_entry_exit_signals(price_data['Close'].iloc[-1])
        
        # Get sentiment analysis
        news_items = self.sentiment_analyzer.fetch_news_sentiment('NVDA')
        sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(news_items)
        
        # Calculate composite signal
        signal = self._calculate_composite_signal(
            dark_pool_analysis,
            regime,
            volume_signals,
            sentiment_analysis
        )
        
        return {
            'signal': signal,
            'dark_pool_sentiment': dark_pool_analysis['dark_pool_sentiment'],
            'option_flow': dark_pool_analysis['option_flow_signals'],
            'greek_exposure': dark_pool_analysis['greek_exposure'],
            'market_regime': {
                'trend': regime.regime,
                'trend_strength': regime.trend_strength,
                'volatility': regime.volatility_regime,
                'momentum': regime.momentum_score,
                'strength': regime.confidence
            },
            'volume_analysis': {
                'support': volume_signals['long']['stop_loss'],
                'resistance': volume_signals['long']['take_profit'],
                'volume_confidence': volume_signals['long']['volume_confidence']
            },
            'risk_levels': {
                'stop_loss': volume_signals['long']['stop_loss'],
                'take_profit': volume_signals['long']['take_profit']
            }
        }
        
    def _calculate_composite_signal(self, dark_pool: Dict, regime: MarketRegimeAnalysis, 
                                  volume: Dict, sentiment: Dict) -> float:
        # Dark pool sentiment (35% weight)
        dark_pool_score = dark_pool['composite_signal'] * 0.35
        
        # News sentiment (5% weight)
        sentiment_score = sentiment.get('composite_score', 0) * 0.05
        
        # Market regime (30% weight)
        regime_score = 0.0
        if regime.regime == 'trending':
            regime_score = regime.trend_strength
        else:
            regime_score = 0.0
        regime_score *= 0.3
        
        # Volume analysis (30% weight)
        volume_score = (volume['long']['volume_confidence'] - 
                       volume['short']['volume_confidence']) * 0.3
        
        # Combine scores
        composite = dark_pool_score + regime_score + volume_score + sentiment_score
        
        # Adjust based on volatility regime
        if regime.volatility_regime == 'high':
            composite *= 0.8  # Reduce signal strength in high volatility
        
        return float(np.clip(composite, -1, 1))
