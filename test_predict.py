import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Mock the database connection and fetch_new_matches function
def mock_fetch_new_matches():
    # Create a sample DataFrame that mimics what fetch_new_matches would return
    data = {
        'match_id': [1001, 1001, 1002, 1002],
        'bookmaker_id': [64, 39, 64, 39],
        'first_win_sp': [2.1, 2.2, 1.8, 1.9],
        'first_draw_sp': [3.2, 3.3, 3.4, 3.5],
        'first_lose_sp': [2.8, 2.7, 3.1, 3.0],
        'first_win_kelly_index': [0.95, 0.96, 0.97, 0.98],
        'first_draw_kelly_index': [0.92, 0.93, 0.94, 0.95],
        'first_lose_kelly_index': [0.91, 0.92, 0.93, 0.94],
        'first_handicap': [0, 0, 0, 0],
        'first_back_rate': [0.95, 0.96, 0.97, 0.98],
        'max_first_win_sp': [2.2, 2.3, 1.9, 2.0],
        'max_first_draw_sp': [3.3, 3.4, 3.5, 3.6],
        'max_first_lose_sp': [2.9, 2.8, 3.2, 3.1],
        'min_first_win_sp': [2.0, 2.1, 1.7, 1.8],
        'min_first_draw_sp': [3.1, 3.2, 3.3, 3.4],
        'min_first_lose_sp': [2.7, 2.6, 3.0, 2.9],
        'last_update_time_distance': [120, 120, 120, 120],
        'league_id': [1, 1, 2, 2],
        'bet_time': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
        'host_name': ['Team A', 'Team A', 'Team C', 'Team C'],
        'guest_name': ['Team B', 'Team B', 'Team D', 'Team D']
    }
    return pd.DataFrame(data)

# Mock the create_match_level_future_by_match_group function
def mock_create_match_level_future_by_match_group(df):
    # Create a simplified version that just aggregates by match_id
    result = df.groupby('match_id').agg({
        'first_win_sp': 'mean',
        'first_draw_sp': 'mean',
        'first_lose_sp': 'mean',
        'first_win_kelly_index': 'mean',
        'first_draw_kelly_index': 'mean',
        'first_lose_kelly_index': 'mean',
        'league_id': 'first',
        'host_name': 'first',
        'guest_name': 'first'
    }).reset_index()

    # Add some mock statistics and rename columns to match expected format
    for outcome in ['win', 'draw', 'lose']:
        # Add the mean columns (these are used as feature names)
        result[f'first_{outcome}_sp_mean'] = result[f'first_{outcome}_sp']
        result[f'first_{outcome}_kelly_index_mean'] = result[f'first_{outcome}_kelly_index']

        # Add other statistics
        result[f'first_{outcome}_sp_std'] = 0.1
        result[f'first_{outcome}_sp_max'] = result[f'first_{outcome}_sp'] + 0.1
        result[f'first_{outcome}_sp_min'] = result[f'first_{outcome}_sp'] - 0.1
        result[f'first_{outcome}_sp_range'] = 0.2
        result[f'first_{outcome}_sp_skew'] = 0.0
        result[f'first_{outcome}_sp_kurt'] = 0.0

    return result

# Mock the create_features function
def mock_create_features(df, useless_cols=None):
    # Just return the input DataFrame with all the columns we need
    result = df.copy()

    # Add all the columns that would be in the real feature set
    for col in ['first_win_sp_mean', 'first_draw_sp_mean', 'first_lose_sp_mean',
               'first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']:
        if col not in result.columns:
            # If the column name has '_mean' in it, use the base column without the '_mean'
            if '_mean' in col:
                base_col = col.replace('_mean', '')
                if base_col in result.columns:
                    result[col] = result[base_col]
                else:
                    result[col] = 2.0
            else:
                result[col] = 0.1

    return result

# Mock the model, scaler, and feature_names
class MockModel:
    def predict(self, X):
        # Always predict class 1 (draw)
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        # Return fake probabilities
        probs = np.zeros((len(X), 3))
        probs[:, 0] = 0.2  # lose
        probs[:, 1] = 0.6  # draw
        probs[:, 2] = 0.2  # win
        return probs

class MockScaler:
    def transform(self, X):
        # Just return the input
        return X

# Patch the necessary functions
import service.util.predict_new_matches as predict_module
import service.util.spfTest as spfTest_module
import sys

# Create a mock module for mysql_data
class MockMySQLDataModule:
    def fetch_new_matches(self):
        return mock_fetch_new_matches()

# Create a mock module and add it to sys.modules
mock_mysql_data = MockMySQLDataModule()
sys.modules['service.spf.initData.data.mysql_data'] = mock_mysql_data

# Save the original functions
original_create_match_level_future_by_match_group = spfTest_module.create_match_level_future_by_match_group
original_create_features = spfTest_module.create_features

# Replace with mocks
spfTest_module.create_match_level_future_by_match_group = mock_create_match_level_future_by_match_group
spfTest_module.create_features = mock_create_features

# Create mock model files
mock_model = MockModel()
mock_scaler = MockScaler()
mock_feature_names = ['first_win_sp_mean', 'first_draw_sp_mean', 'first_lose_sp_mean',
                     'first_win_sp_std', 'first_draw_sp_std', 'first_lose_sp_std']

# Save mock model files
os.makedirs('models', exist_ok=True)
joblib.dump(mock_model, 'models/best_model.pkl')
joblib.dump(mock_scaler, 'models/scaler.pkl')
joblib.dump(mock_feature_names, 'models/feature_names.pkl')

# Test the predict_new_matches function
from service.util.predict_new_matches import predict_new_matches

print("Testing predict_new_matches function...")
result = predict_new_matches(
    model_path='models/best_model.pkl',
    scaler_path='models/scaler.pkl',
    feature_names_path='models/feature_names.pkl'
)

print("\nTest completed!")
if result is not None:
    print("Result shape:", result.shape)
    print("Result columns:", result.columns.tolist())
    print("Result preview:")
    print(result.head())
else:
    print("No result returned.")

# Restore original functions
spfTest_module.create_match_level_future_by_match_group = original_create_match_level_future_by_match_group
spfTest_module.create_features = original_create_features

# Remove mock module from sys.modules
if 'service.spf.initData.data.mysql_data' in sys.modules:
    del sys.modules['service.spf.initData.data.mysql_data']
