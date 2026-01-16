#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. LSTM + Attention Model
# --------------------------------------------------
class OutbreakAttentionModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=64):
        super(OutbreakAttentionModel, self).__init__()

        # LSTM layer (learns time patterns)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Attention layer (focuses on important days)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_day = attn_out[:, -1, :]
        risk = self.sigmoid(self.fc(last_day))
        return risk

# --------------------------------------------------
# 2. Data Preprocessing
# --------------------------------------------------
def preprocess_data_row(row):
    """Convert one row of CSV into feature vector"""
    features = [
        row['cases'] / 10000,
        row['hospitalization_rate'] / 100,
        row['death_rate'] / 100,
        row['vaccination_rate'] / 100,

        row['temperature'] / 40,
        row['humidity'] / 100,
        row['air_quality_index'] / 300,

        row['mobility_index'] / 100,
        row['public_transport_usage'] / 100,
        row['gathering_events'] / 100,

        # Padding to reach 15 features
        0, 0, 0, 0, 0
    ]
    return np.array(features, dtype=np.float32)

# --------------------------------------------------
# 3. Risk Prediction
# --------------------------------------------------
def predict_risk_from_df(model, df):
    """Predict outbreak risk from dataframe"""
    sequence = []

    last_days = df.tail(30)
    for _, row in last_days.iterrows():
        features = preprocess_data_row(row)
        sequence.append(features)

    while len(sequence) < 30:
        sequence.insert(0, np.zeros(15, dtype=np.float32))

    # Convert list of arrays → single array
    sequence_array = np.array(sequence, dtype=np.float32)

    # Convert to tensor
    x = torch.FloatTensor(sequence_array).unsqueeze(0)  # (1, 30, 15)

    with torch.no_grad():
        score = model(x).item()

    if score > 0.75:
        level = "CRITICAL"
    elif score > 0.5:
        level = "HIGH"
    elif score > 0.3:
        level = "MODERATE"
    else:
        level = "LOW"

    return score, level

# --------------------------------------------------
# 4. Risk Assessment Functions
# --------------------------------------------------
def calculate_risk_score(cases, hospitalization_rate, death_rate, air_quality, mobility_index, 
                         population_density=5000, max_cases=10000):
    """
    Calculate risk score (0-1) based on multiple factors
    
    Args:
        cases: Number of cases
        hospitalization_rate: Percentage (0-100)
        death_rate: Percentage (0-100)
        air_quality: AQI (0-500)
        mobility_index: Percentage (0-100)
        population_density: People per sq km
        max_cases: Maximum expected cases for normalization
    
    Returns:
        risk_score: Float between 0 and 1
    """
    # Normalize each factor
    case_factor = min(cases / max_cases, 1.0) * 0.30  # 30% weight
    hosp_factor = (hospitalization_rate / 100) * 0.25  # 25% weight
    death_factor = (death_rate / 100) * 0.20  # 20% weight
    air_factor = min(air_quality / 500, 1.0) * 0.10  # 10% weight
    mobility_factor = (mobility_index / 100) * 0.10  # 10% weight
    density_factor = min(population_density / 10000, 1.0) * 0.05  # 5% weight
    
    # Calculate weighted risk score
    risk_score = (case_factor + hosp_factor + death_factor + 
                  air_factor + mobility_factor + density_factor)
    
    # Ensure bounds
    risk_score = max(0.0, min(1.0, risk_score))
    
    return risk_score

def get_risk_level(risk_score):
    """
    Convert risk score to categorical risk level
    
    Args:
        risk_score: Float between 0 and 1
    
    Returns:
        risk_level: String (CRITICAL/HIGH/MODERATE/LOW)
        icon: String emoji for visualization
    """
    if risk_score >= 0.75:
        return "CRITICAL", "🔴"
    elif risk_score >= 0.50:
        return "HIGH", "🟠"
    elif risk_score >= 0.30:
        return "MODERATE", "🟡"
    else:
        return "LOW", "🟢"

def estimate_future_factors(last_row, day_ahead, trend_multiplier=1.02):
    """
    Estimate future values for risk factors based on trends
    
    Args:
        last_row: Last row of actual data
        day_ahead: Number of days in the future
        trend_multiplier: Daily growth rate (1.02 = 2% increase per day)
    
    Returns:
        Dictionary with estimated factors
    """
    # Apply exponential trend for increasing factors
    hosp_rate = last_row['hospitalization_rate'] * (trend_multiplier ** day_ahead)
    death_rate = last_row['death_rate'] * (trend_multiplier ** day_ahead)
    mobility = last_row['mobility_index'] * (trend_multiplier ** (day_ahead * 0.5))  # Slower growth
    
    # Air quality - add some variation
    air_quality = last_row['air_quality_index'] + np.random.normal(0, 5) * day_ahead
    
    # Keep within bounds
    hosp_rate = min(hosp_rate, 100)
    death_rate = min(death_rate, 100)
    mobility = min(mobility, 100)
    air_quality = max(0, min(air_quality, 500))
    
    return {
        'hospitalization_rate': hosp_rate,
        'death_rate': death_rate,
        'mobility_index': mobility,
        'air_quality_index': air_quality,
        'population_density': last_row.get('population_density', 5000)
    }

# --------------------------------------------------
# 5. Main Execution
# --------------------------------------------------
if __name__ == "__main__":
    # Load your CSV
    df = pd.read_csv("synthetic_data.csv")

    # Optional: clean data
    df.fillna(0, inplace=True)

    # Create model
    model = OutbreakAttentionModel()
    model.eval()

    # Predict risk
    risk_score, risk_level = predict_risk_from_df(model, df)
    print("Risk Score:", round(risk_score, 3))
    print("Risk Level:", risk_level)