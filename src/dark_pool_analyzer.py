from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DarkPoolData:
    timestamp: datetime
    volume: float
    price: float
    side: str  # 'buy' or 'sell'
    block_size: float

class DarkPoolAnalyzer:
    def __init__(self, polygon_key: Optional[str] = None, unusual_whales_key: Optional[str] = None):
        self.api_key = polygon_key or os.getenv('POLYGON_API_KEY')
        self.unusual_whales_key = unusual_whales_key or os.getenv('UNUSUAL_WHALES_API_KEY')
        
        if not all([self.api_key, self.unusual_whales_key]):
            raise ValueError("Missing required API keys")
            
        self.base_url = "https://api.polygon.io/v2"
        self.unusual_whales_url = "https://api.unusualwhales.com"
        self.symbol = 'NVDA'
        
    def fetch_dark_pool_data(self, days: int = 5) -> List[DarkPoolData]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            "apiKey": self.api_key,
            "limit": 50000,
            "timestamp.gte": start_date.strftime("%Y-%m-%d"),
            "timestamp.lte": end_date.strftime("%Y-%m-%d")
        }
        
        endpoint = f"{self.base_url}/ticks/stocks/nbbo/{self.symbol}"
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        dark_pool_trades = []
        for trade in data.get('results', []):
            if trade.get('conditions', []).intersection({'7', '4', 'D'}):  # Dark pool conditions
                dark_pool_trades.append(DarkPoolData(
                    timestamp=datetime.fromtimestamp(trade['t'] / 1000),
                    volume=float(trade['s']),
                    price=float(trade['p']),
                    side='buy' if trade.get('q', 0) > 0 else 'sell',
                    block_size=float(trade['s']) * float(trade['p'])
                ))
                
        return sorted(dark_pool_trades, key=lambda x: x.timestamp, reverse=True)
        
    def analyze_dark_pool_activity(self, trades: List[DarkPoolData]) -> Dict[str, float]:
        if not trades:
            return {
                'net_flow': 0.0,
                'buy_pressure': 0.0,
                'large_trade_impact': 0.0,
                'volume_distribution': 0.0,
                'composite_score': 0.0
            }
            
        total_volume = sum(trade.volume for trade in trades)
        buy_volume = sum(trade.volume for trade in trades if trade.side == 'buy')
        sell_volume = total_volume - buy_volume
        
        # Net flow (-1 to 1)
        net_flow = float((buy_volume - sell_volume) / total_volume if total_volume > 0 else 0)
        
        # Buy pressure (0 to 1)
        buy_pressure = float(buy_volume / total_volume if total_volume > 0 else 0)
        
        # Large trade impact
        avg_trade_size = float(total_volume / len(trades))
        large_trades = [t for t in trades if t.volume > avg_trade_size * 2]
        large_trade_volume = sum(t.volume for t in large_trades)
        large_trade_impact = float(large_trade_volume / total_volume if total_volume > 0 else 0)
        
        # Volume distribution analysis
        volumes = [float(t.volume) for t in trades]
        volume_std = float(np.std(volumes) if volumes else 0)
        volume_mean = float(np.mean(volumes) if volumes else 0)
        volume_distribution = float(volume_std / volume_mean if volume_mean > 0 else 0)
        
        # Composite score (-1 to 1)
        composite_score = (
            0.4 * net_flow +
            0.3 * (buy_pressure - 0.5) * 2 +  # Convert to -1 to 1 range
            0.2 * (large_trade_impact - 0.5) * 2 +
            0.1 * (volume_distribution - 0.5) * 2
        )
        
        return {
            'net_flow': net_flow,
            'buy_pressure': buy_pressure,
            'large_trade_impact': large_trade_impact,
            'volume_distribution': volume_distribution,
            'composite_score': composite_score
        }
        
    def get_significant_levels(self, trades: List[DarkPoolData]) -> List[Dict[str, float]]:
        if not trades:
            return []
            
        # Group trades by price levels
        price_levels = {}
        for trade in trades:
            price = round(trade.price, 2)
            if price not in price_levels:
                price_levels[price] = 0
            price_levels[price] += trade.volume
            
        # Find significant levels (local maxima in volume)
        significant = []
        prices = sorted(price_levels.keys())
        for i in range(1, len(prices) - 1):
            if (price_levels[prices[i]] > price_levels[prices[i-1]] and 
                price_levels[prices[i]] > price_levels[prices[i+1]]):
                significant.append({
                    'price': prices[i],
                    'volume': price_levels[prices[i]],
                    'strength': price_levels[prices[i]] / sum(price_levels.values())
                })
                
        return sorted(significant, key=lambda x: x['strength'], reverse=True)
        
    def fetch_unusual_whales_data(self) -> Dict:
        headers = {"Authorization": f"Bearer {self.unusual_whales_key}"}
        
        # Fetch dark pool data
        darkpool_url = f"{self.unusual_whales_url}/darkpool/{self.symbol}"
        darkpool_response = requests.get(darkpool_url, headers=headers)
        darkpool_data = darkpool_response.json() if darkpool_response.ok else {}
        
        # Fetch options volume levels
        volume_url = f"{self.unusual_whales_url}/stock/{self.symbol}/option-volume-levels"
        volume_response = requests.get(volume_url, headers=headers)
        volume_data = volume_response.json() if volume_response.ok else {}
        
        # Fetch Greeks exposure
        greeks_url = f"{self.unusual_whales_url}/stock/{self.symbol}/greek-exposure"
        greeks_response = requests.get(greeks_url, headers=headers)
        greeks_data = greeks_response.json() if greeks_response.ok else {}
        
        return {
            'dark_pool': darkpool_data,
            'option_volume': volume_data,
            'greeks': greeks_data
        }
        
    def analyze_unusual_whales_data(self, data: Dict) -> Dict:
        """Analyze data from Unusual Whales API specifically for NVDA."""
        if not data:
            return {}
            
        dark_pool = data.get('dark_pool', {})
        option_volume = data.get('option_volume', {})
        greeks = data.get('greeks', {})
        
        analysis = {
            'dark_pool_sentiment': self._analyze_dark_pool_sentiment(dark_pool),
            'option_flow_signals': self._analyze_option_flow(option_volume),
            'greek_exposure': self._analyze_greek_exposure(greeks)
        }
        
        # Calculate composite signal (-1 to 1)
        signals = [
            analysis['dark_pool_sentiment'].get('signal', 0) * 0.4,
            analysis['option_flow_signals'].get('signal', 0) * 0.4,
            analysis['greek_exposure'].get('signal', 0) * 0.2
        ]
        
        # Add composite signal to analysis dict
        analysis_dict = dict(analysis)
        analysis_dict['composite_signal'] = float(sum(signals))
        
        return analysis_dict
        
    def _analyze_dark_pool_sentiment(self, data: Dict) -> Dict:
        if not data:
            return {'signal': 0}
            
        buy_volume = sum(trade.get('volume', 0) for trade in data.get('trades', [])
                        if trade.get('side') == 'buy')
        sell_volume = sum(trade.get('volume', 0) for trade in data.get('trades', [])
                         if trade.get('side') == 'sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {'signal': 0}
            
        buy_ratio = buy_volume / total_volume
        signal = (buy_ratio - 0.5) * 2  # Convert to -1 to 1 scale
        
        return {
            'buy_ratio': buy_ratio,
            'total_volume': total_volume,
            'signal': signal
        }
        
    def _analyze_option_flow(self, data: Dict) -> Dict:
        if not data:
            return {'signal': 0}
            
        call_volume = sum(level.get('volume', 0) for level in data.get('calls', []))
        put_volume = sum(level.get('volume', 0) for level in data.get('puts', []))
        total_volume = call_volume + put_volume
        
        if total_volume == 0:
            return {'signal': 0}
            
        put_call_ratio = put_volume / call_volume if call_volume > 0 else float('inf')
        signal = -1 if put_call_ratio > 1.5 else 1 if put_call_ratio < 0.67 else 0
        
        return {
            'put_call_ratio': put_call_ratio,
            'total_option_volume': total_volume,
            'signal': signal
        }
        
    def _analyze_greek_exposure(self, data: Dict) -> Dict:
        if not data:
            return {'signal': 0}
            
        delta = data.get('delta', 0)
        gamma = data.get('gamma', 0)
        theta = data.get('theta', 0)
        
        # Positive signal if net delta is positive and gamma exposure is manageable
        signal = 1 if delta > 0 and abs(gamma) < 0.1 else -1 if delta < 0 else 0
        
        return {
            'net_delta': delta,
            'net_gamma': gamma,
            'net_theta': theta,
            'signal': signal
        }
