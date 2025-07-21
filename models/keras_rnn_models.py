"""
Keras-based RNN, LSTM, BI-LSTM, GRU models for regression with hyperparameter tuning.
"""
from scikeras.wrappers import KerasRegressor
from typing import Dict, Any
from .base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, GRU, Input
from tensorflow.keras.optimizers import Adam

# Helper functions to build each model

def build_rnn_model(timesteps=1, input_dim=1, units=32, layers=1, lr=0.001):
    model = Sequential()
    model.add(Input(shape=(timesteps, input_dim)))
    for i in range(layers):
        return_sequences = i < layers - 1
        model.add(SimpleRNN(units, activation='relu', return_sequences=return_sequences))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def build_lstm_model(timesteps=1, input_dim=1, units=32, layers=1, lr=0.001):
    model = Sequential()
    model.add(Input(shape=(timesteps, input_dim)))
    for i in range(layers):
        return_sequences = i < layers - 1
        model.add(LSTM(units, activation='relu', return_sequences=return_sequences))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def build_bilstm_model(timesteps=1, input_dim=1, units=32, layers=1, lr=0.001):
    model = Sequential()
    model.add(Input(shape=(timesteps, input_dim)))
    for i in range(layers):
        return_sequences = i < layers - 1
        model.add(Bidirectional(LSTM(units, activation='relu', return_sequences=return_sequences)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def build_gru_model(timesteps=1, input_dim=1, units=32, layers=1, lr=0.001):
    model = Sequential()
    model.add(Input(shape=(timesteps, input_dim)))
    for i in range(layers):
        return_sequences = i < layers - 1
        model.add(GRU(units, activation='relu', return_sequences=return_sequences))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

# Model classes

class RNNModel(BaseModel):
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "RNN"
    def get_model(self):
        return KerasRegressor(
            model=build_rnn_model,
            timesteps=1,
            input_dim=1,
            units=32,
            layers=1,
            lr=0.001,
            epochs=30,
            batch_size=32,
            verbose=0,
            random_state=self.random_state
        )
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__units': [16, 32, 64],
            'regressor__layers': [1, 2],
            'regressor__lr': [0.001, 0.01],
            'regressor__epochs': [20, 30],
            'regressor__batch_size': [16, 32],
            'regressor__timesteps': [1, 3, 5]
        }
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'units': 32,
            'layers': 1,
            'lr': 0.001,
            'epochs': 30,
            'batch_size': 32,
            'timesteps': 1,
            'input_dim': 1,
            'verbose': 0,
            'random_state': self.random_state
        }

class LSTMModel(BaseModel):
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "LSTM"
    def get_model(self):
        return KerasRegressor(
            model=build_lstm_model,
            timesteps=1,
            input_dim=1,
            units=32,
            layers=1,
            lr=0.001,
            epochs=30,
            batch_size=32,
            verbose=0,
            random_state=self.random_state
        )
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__units': [16, 32, 64],
            'regressor__layers': [1, 2],
            'regressor__lr': [0.001, 0.01],
            'regressor__epochs': [20, 30],
            'regressor__batch_size': [16, 32],
            'regressor__timesteps': [1, 3, 5]
        }
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'units': 32,
            'layers': 1,
            'lr': 0.001,
            'epochs': 30,
            'batch_size': 32,
            'timesteps': 1,
            'input_dim': 1,
            'verbose': 0,
            'random_state': self.random_state
        }

class BiLSTMModel(BaseModel):
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "BI-LSTM"
    def get_model(self):
        return KerasRegressor(
            model=build_bilstm_model,
            timesteps=1,
            input_dim=1,
            units=32,
            layers=1,
            lr=0.001,
            epochs=30,
            batch_size=32,
            verbose=0,
            random_state=self.random_state
        )
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__units': [16, 32, 64],
            'regressor__layers': [1, 2],
            'regressor__lr': [0.001, 0.01],
            'regressor__epochs': [20, 30],
            'regressor__batch_size': [16, 32],
            'regressor__timesteps': [1, 3, 5]
        }
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'units': 32,
            'layers': 1,
            'lr': 0.001,
            'epochs': 30,
            'batch_size': 32,
            'timesteps': 1,
            'input_dim': 1,
            'verbose': 0,
            'random_state': self.random_state
        }

class GRUModel(BaseModel):
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_name = "GRU"
    def get_model(self):
        return KerasRegressor(
            model=build_gru_model,
            timesteps=1,
            input_dim=1,
            units=32,
            layers=1,
            lr=0.001,
            epochs=30,
            batch_size=32,
            verbose=0,
            random_state=self.random_state
        )
    def get_param_distributions(self) -> Dict[str, Any]:
        return {
            'regressor__units': [16, 32, 64],
            'regressor__layers': [1, 2],
            'regressor__lr': [0.001, 0.01],
            'regressor__epochs': [20, 30],
            'regressor__batch_size': [16, 32],
            'regressor__timesteps': [1, 3, 5]
        }
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'units': 32,
            'layers': 1,
            'lr': 0.001,
            'epochs': 30,
            'batch_size': 32,
            'timesteps': 1,
            'input_dim': 1,
            'verbose': 0,
            'random_state': self.random_state
        }