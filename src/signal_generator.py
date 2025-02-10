import pandas as pd
import numpy as np
from typing import Dict, Optional
from .dark_pool_analyzer import DarkPoolAnalyzer
from .volume_analyzer import VolumeAnalyzer
from .market_regime_analyzer import MarketRegimeAnalyzer, MarketRegimeAnalysis
from .sentiment_analyzer import SentimentAnalyzer
from .data_fetcher import DataFetcher
import os
from dotenv import load_dotenv

load_dotenv()

class SignalGenerator:
    def __init__(self):
        self.dark_pool_analyzer = DarkPoolAnalyzer()
        self.market_analyzer = MarketRegimeAnalyzer()
        self.data_fetcher = DataFetcher()
        alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'M0CEIVI5XJJ40J6Q')
        self.sentiment_analyzer = SentimentAnalyzer(alpha_key)
        
    def generate_signal(self, price_data: pd.DataFrame) -> Dict:
        volume_analyzer = VolumeAnalyzer(price_data)
        
        # Fetch additional market data from multiple sources
        polygon_data = self.data_fetcher.get_polygon_data()
        alpha_vantage_data = self.data_fetcher.get_alpha_vantage_data()
        
        # Merge additional data points if available
        if not polygon_data.empty:
            price_data['polygon_volume'] = polygon_data.get('volume', pd.Series())
            price_data['polygon_vwap'] = polygon_data.get('vwap', pd.Series())
        
        if not isinstance(alpha_vantage_data, type(None)):
            if isinstance(alpha_vantage_data, pd.DataFrame):
                price_data['av_volume'] = alpha_vantage_data.get('volume', pd.Series())
            
        # Fetch and analyze dark pool data
        dark_pool_data = self.dark_pool_analyzer.fetch_unusual_whales_data()
        dark_pool_analysis = self.dark_pool_analyzer.analyze_unusual_whales_data(dark_pool_data)
        
        # Get market context and analyze regime
        market_tide = self.data_fetcher.fetch_market_tide()
        options_data = self.data_fetcher.fetch_option_flow('NVDA')
        regime = self.market_analyzer.analyze(
            price_data=price_data,
            market_tide=market_tide,
            options_data=options_data
        )
        
        # Get volume analysis
        volume_profile = volume_analyzer.calculate_volume_profile()
        volume_signals = volume_analyzer.get_entry_exit_signals(price_data['Close'].iloc[-1])
        
        # Get sentiment analysis
        news_items = self.sentiment_analyzer.fetch_news_sentiment('NVDA')
        sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(news_items)
        
        # Get Greeks exposure data for risk management
        greeks_data = self.data_fetcher.fetch_greeks_exposure('NVDA')
        
        # Calculate composite signal with Greeks risk adjustment
        signal = self._calculate_composite_signal(
            dark_pool_analysis,
            regime,
            volume_signals,
            sentiment_analysis,
            market_tide,
            greeks_data
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
                                  volume: Dict, sentiment: Dict, market_tide: Optional[Dict] = None,
                                  greeks: Optional[Dict] = None) -> float:
        # Dark pool sentiment (30% weight)
        dark_pool_score = dark_pool['composite_signal'] * 0.30
        
        # Market context (10% weight)
        market_context_score = 0.0
        if market_tide and isinstance(market_tide, dict) and 'score' in market_tide:
            market_context_score = (float(market_tide['score']) - 0.5) * 2 * 0.10
        
        # News sentiment (5% weight)
        sentiment_score = sentiment.get('composite_score', 0) * 0.05
        
        # Market regime (30% weight)
        regime_score = 0.0
        if regime.regime == 'trending':
            regime_score = regime.trend_strength
        else:
            regime_score = 0.0
        regime_score *= 0.3
        
        # Volume analysis (25% weight)
        volume_score = (volume['long']['volume_confidence'] - 
                       volume['short']['volume_confidence']) * 0.25
        
        # Combine scores with updated weights
        composite = (
            dark_pool_score +      # 30% Dark pool sentiment
            regime_score +         # 30% Market regime
            volume_score +         # 25% Volume analysis
            sentiment_score +      # 5%  News sentiment
            market_context_score   # 10% Market context
        )
        
        # Adjust based on volatility regime and Greeks exposure
        if regime.volatility_regime == 'high':
            composite *= 0.8  # Reduce signal strength in high volatility
            
        if greeks:
            delta = greeks.get('delta', 0)
            gamma = greeks.get('gamma', 0)
            theta = greeks.get('theta', 0)
            
            # Reduce signal strength if gamma exposure is high
            if abs(gamma) > 0.1:
                composite *= 0.85
                
            # Adjust signal based on delta exposure
            if abs(delta) > 0.7:
                composite *= 0.9
                
            # Consider theta decay impact
            if theta < -0.001:  # Significant negative theta
                composite *= 0.95
        
        return float(np.clip(composite, -1, 1))
