import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import itertools
import os
from .data_collector import DataCollector

class PricePredictor:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.collector = DataCollector()
        self.feature_columns = None
        self.lstm_model = None
        self.gru_model = None
        self.xgb_model = None
        self.gb_model = None
        
    def prepare_data(self):
        raw_data = self.collector.collect_all_data()
        if 'yfinance' not in raw_data or raw_data['yfinance'] is None:
            raise ValueError("Failed to fetch YFinance data")
            
        df = raw_data['yfinance']
        features = self._extract_features(df)
        return self._create_sequences(features)
        
    def _extract_features(self, df):
        try:
            features = pd.DataFrame()
            window_size = min(5, len(df) // 4)  # Dynamic window size
            
            # Basic price and volume features
            features['Close'] = df['Close']
            features['Volume'] = df['Volume']
            features['RSI'] = self._calculate_rsi(df['Close'], period=min(14, window_size))
            features['SMA_Short'] = df['Close'].rolling(window=window_size).mean()
            features['MACD'] = self._calculate_macd(df['Close'])
            
            # Market context features (if available)
            if 'SPY_Correlation' in df.columns:
                features['SPY_Correlation'] = df['SPY_Correlation']
            if 'Market_RS' in df.columns:
                features['Market_RS'] = df['Market_RS']
            
            # Volatility features
            features['Daily_Return'] = df['Close'].pct_change()
            features['Volatility'] = features['Daily_Return'].rolling(window=window_size).std()
            
            # Sentiment features (if available)
            if 'News_Sentiment' in df.columns:
                features['News_Sentiment'] = df['News_Sentiment'].fillna(0.5)
            
            # Forward fill then backfill missing values
            features = features.ffill().bfill()
            
            print(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
        
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices):
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
        
    def _create_sequences(self, features):
        min_sequence_length = 3  # Reduced minimum sequence length
        
        # Print debug information
        print(f"Creating sequences from {len(features)} data points")
        print(f"Current sequence length: {self.sequence_length}")
        
        # Dynamically adjust sequence length based on available data
        if len(features) < self.sequence_length + 1:
            self.sequence_length = max(min_sequence_length, min(10, len(features) // 3))
            print(f"Adjusted sequence length to {self.sequence_length}")
            
        if len(features) < min_sequence_length + 1:
            print(f"Insufficient data points: {len(features)} < {min_sequence_length + 1}")
            return np.array([]), np.array([])
            
        # Store feature columns for consistent ordering
        self.feature_columns = features.columns.tolist()
        print(f"Features available: {self.feature_columns}")
        
        try:
            # Scale features while preserving column names
            scaled_features = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=self.feature_columns,
                index=features.index
            )
            
            X, y = [], []
            for i in range(len(scaled_features) - self.sequence_length):
                sequence = scaled_features.iloc[i:(i + self.sequence_length)].values
                target = scaled_features.iloc[i + self.sequence_length]['Close']
                X.append(sequence)
                y.append(target)
            
            if not X:
                print("No sequences could be created")
                return np.array([]), np.array([])
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"Error creating sequences: {str(e)}")
            return np.array([]), np.array([])
            
        return np.array(X), np.array(y)
        
    def build_lstm_model(self, input_shape, params=None):
        if params is None:
            params = {
                'lstm_units': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'dense_units': 16
            }
        
        model = Sequential([
            LSTM(params['lstm_units'][0], return_sequences=True, input_shape=input_shape),
            Dropout(params['dropout_rate']),
            LSTM(params['lstm_units'][1], return_sequences=True),
            Dropout(params['dropout_rate']),
            LSTM(params['lstm_units'][2], return_sequences=False),
            Dropout(params['dropout_rate']),
            Dense(params['dense_units'], activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                    loss='huber',
                    metrics=['mae'])
        
        return model
        
    def build_gru_model(self, input_shape, params=None):
        if params is None:
            params = {
                'gru_units': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'dense_units': 16
            }
        
        model = Sequential([
            GRU(params['gru_units'][0], return_sequences=True, input_shape=input_shape),
            Dropout(params['dropout_rate']),
            GRU(params['gru_units'][1], return_sequences=True),
            Dropout(params['dropout_rate']),
            GRU(params['gru_units'][2], return_sequences=False),
            Dropout(params['dropout_rate']),
            Dense(params['dense_units'], activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                    loss='huber',
                    metrics=['mae'])
        
        return model
        
    def build_tree_models(self, X_flat, y):
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror'
        )
        
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.xgb_model.fit(X_flat, y)
        self.gb_model.fit(X_flat, y)
        
    def flatten_sequences(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1] * X.shape[2]
        return X.reshape((n_samples, n_features))
        
    def grid_search_cv(self, X, y, param_grid, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_score = float('inf')
        best_params = None
        cv_results = []
        
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        for params in param_combinations:
            fold_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = self.build_lstm_model(input_shape=(X.shape[1], X.shape[2]), params=params)
                
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                ]
                
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=0
                )
                
                val_pred = model.predict(X_val, verbose=0)
                mae = mean_absolute_error(y_val, val_pred)
                fold_scores.append(mae)
            
            avg_score = np.mean(fold_scores)
            cv_results.append({
                'params': params,
                'mean_mae': avg_score,
                'std_mae': np.std(fold_scores)
            })
            
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
        
        self.best_params = best_params
        self.cv_results = cv_results
        return best_params, cv_results
                         
    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1):
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = Sequential([
                LSTM(32, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(16),
                Dropout(0.2),
                Dense(1)
            ])
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
            
            # Store the model for ensemble predictions
            self.lstm_model = self.model
            
            # Create a simpler model for shorter sequences
            self.short_sequence_model = Sequential([
                LSTM(16, input_shape=input_shape),
                Dense(8, activation='relu'),
                Dense(1)
            ])
            self.short_sequence_model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
        
    def predict_next_day(self, features):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            features_processed = self._extract_features(features)
            if len(features_processed) < 10:  # Minimum required data points
                print(f"Warning: Insufficient data points ({len(features_processed)})")
                return None
                
            # Adjust sequence length for shorter periods
            if len(features_processed) < self.sequence_length:
                self.sequence_length = max(5, len(features_processed) // 4)
                print(f"Adjusted sequence length to {self.sequence_length}")
            
            # Ensure feature columns are in correct order
            if self.feature_columns is None:
                self.feature_columns = features_processed.columns.tolist()
            else:
                features_processed = features_processed[self.feature_columns]
            
            # Scale all features first
            scaled_features = pd.DataFrame(
                self.scaler.fit_transform(features_processed),
                columns=self.feature_columns,
                index=features_processed.index
            )
            
            # Get the last sequence
            last_sequence = scaled_features.iloc[-self.sequence_length:].values
            if len(last_sequence) < self.sequence_length:
                print(f"Warning: Sequence length {len(last_sequence)} less than required {self.sequence_length}")
                return None
                
            sequence_array = last_sequence.reshape(1, self.sequence_length, -1)
            prediction = self.model.predict(sequence_array, verbose=0)
            
            # Prepare for inverse transform
            dummy = np.zeros((1, len(self.feature_columns)))
            dummy[0, features_processed.columns.get_loc('Close')] = prediction[0, 0]
            
            # Convert back to original scale
            return float(self.scaler.inverse_transform(dummy)[0, features_processed.columns.get_loc('Close')])
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
        
    def generate_price_ranges(self, current_price, prediction):
        volatility = 0.02  # 2% assumed volatility
        return {
            'prediction': prediction,
            'confidence_ranges': {
                '90%': (prediction * (1 - volatility), prediction * (1 + volatility)),
                '70%': (prediction * (1 - volatility * 0.7), prediction * (1 + volatility * 0.7)),
                '50%': (prediction * (1 - volatility * 0.5), prediction * (1 + volatility * 0.5))
            }
        }
        
    def backtest(self, test_size=0.2, n_splits=5):
        try:
            raw_data = self.collector.collect_all_data()
            if not raw_data or 'yfinance' not in raw_data:
                raise ValueError("Failed to collect YFinance data")
            
            df = raw_data['yfinance']
            features = self._extract_features(df)
            X, y = self._create_sequences(features)
            
            # Split into train/test
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train all models
            history = self.train(X_train, y_train, epochs=50)
            
            # Prepare test data for tree models
            X_test_flat = self.flatten_sequences(X_test)
            
            # Get predictions from all models
            predictions = {}
            
            if self.lstm_model is not None:
                predictions['lstm'] = self.lstm_model.predict(X_test, verbose=0)
            if self.gru_model is not None:
                predictions['gru'] = self.gru_model.predict(X_test, verbose=0)
            if self.xgb_model is not None:
                predictions['xgb'] = self.xgb_model.predict(X_test_flat).reshape(-1, 1)
            if self.gb_model is not None:
                predictions['gb'] = self.gb_model.predict(X_test_flat).reshape(-1, 1)
        
            # Calculate ensemble prediction
            weights = {'lstm': 0.3, 'gru': 0.3, 'xgb': 0.2, 'gb': 0.2}
            ensemble_pred = sum(weights[k] * v for k, v in predictions.items())
        except Exception as e:
            raise ValueError(f"Error during backtesting: {str(e)}")
        
        # Calculate metrics for each model
        def calculate_metrics(predictions, actuals):
            mae = mean_absolute_error(actuals, predictions)
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            return {'mae': mae, 'mape': mape, 'rmse': rmse}
        
        # Calculate metrics for each model's predictions
        metrics = {
            'lstm': calculate_metrics(predictions['lstm'], y_test),
            'gru': calculate_metrics(predictions['gru'], y_test),
            'xgboost': calculate_metrics(predictions['xgb'], y_test),
            'gradient_boosting': calculate_metrics(predictions['gb'], y_test),
            'ensemble': calculate_metrics(ensemble_pred, y_test)
        }
        
        # Create dummy arrays for inverse transformation
        ensemble_pred_array = np.array(ensemble_pred)
        if len(ensemble_pred_array.shape) == 1:
            ensemble_pred_array = ensemble_pred_array.reshape(-1, 1)
            
        dummy = np.zeros((ensemble_pred_array.shape[0], features.shape[1]))
        
        # Transform predictions back to original scale
        def inverse_transform_preds(preds):
            preds_array = np.array(preds)
            if len(preds_array.shape) == 1:
                preds_array = preds_array.reshape(-1, 1)
            dummy[:, 0] = preds_array.flatten()
            return self.scaler.inverse_transform(dummy)[:, 0]
        
        return {
            'metrics': metrics,
            'predictions': {
                'lstm': inverse_transform_preds(predictions['lstm']),
                'gru': inverse_transform_preds(predictions['gru']),
                'xgboost': inverse_transform_preds(predictions['xgb']),
                'gradient_boosting': inverse_transform_preds(predictions['gb']),
                'ensemble': inverse_transform_preds(ensemble_pred)
            },
            'actuals': inverse_transform_preds(y_test.reshape(-1, 1)),
            'training_history': history
        }
