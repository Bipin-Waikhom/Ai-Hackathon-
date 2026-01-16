#!/usr/bin/env python
# coding: utf-8

# predict_outbreak_with_risk.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from diseasemodel import OutbreakAttentionModel

# -----------------------------
# Risk Assessment Functions
# -----------------------------
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
        color: String for visualization
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


# -----------------------------
# Preprocess a single row
# -----------------------------
def preprocess_data_row(row):
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
        0, 0, 0, 0, 0
    ]
    return np.array(features, dtype=np.float32)


# -----------------------------
# Create sequences from dataset
# -----------------------------
def create_sequences(df, seq_len=30):
    df = df.reset_index(drop=True)
    max_cases = max(df['cases'].max(), 1)
    
    seq_len = min(seq_len, len(df)-1)
    if seq_len < 1:
        raise ValueError("Dataset too small for the given seq_len")
    
    X, y = [], []
    for i in range(len(df) - seq_len):
        seq_df = df.iloc[i:i+seq_len].apply(preprocess_data_row, axis=1)
        seq_array = np.stack(seq_df.values)
        X.append(seq_array)
        
        future_cases = df.iloc[i+seq_len]['cases']
        y.append(future_cases / max_cases)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1,1), max_cases


# -----------------------------
# Predict next N days WITH RISK
# -----------------------------
def predict_next_days_with_risk(model, last_seq, last_row_data, days=7, max_cases=1, device='cpu'):
    """
    Predict cases AND risk assessment for next N days
    
    Args:
        model: Trained PyTorch model
        last_seq: Last sequence of preprocessed data
        last_row_data: Last row of raw data (for risk factors)
        days: Number of days to predict
        max_cases: Maximum cases for denormalization
        device: torch device
    
    Returns:
        List of dictionaries with predictions and risk assessments
    """
    model.eval()
    predictions = []
    seq = last_seq.copy()
    
    with torch.no_grad():
        for day in range(1, days + 1):
            # Predict cases
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(seq_tensor)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            
            pred_cases = pred.item() * max_cases
            
            # Estimate future risk factors
            future_factors = estimate_future_factors(last_row_data, day)
            
            # Calculate risk score
            risk_score = calculate_risk_score(
                cases=pred_cases,
                hospitalization_rate=future_factors['hospitalization_rate'],
                death_rate=future_factors['death_rate'],
                air_quality=future_factors['air_quality_index'],
                mobility_index=future_factors['mobility_index'],
                population_density=future_factors['population_density'],
                max_cases=max_cases
            )
            
            # Get risk level
            risk_level, risk_icon = get_risk_level(risk_score)
            
            # Store prediction
            prediction_data = {
                'day': day,
                'predicted_cases': int(pred_cases),
                'risk_score': round(risk_score, 3),
                'risk_level': risk_level,
                'risk_icon': risk_icon,
                'hospitalization_rate': round(future_factors['hospitalization_rate'], 2),
                'death_rate': round(future_factors['death_rate'], 2),
                'mobility_index': round(future_factors['mobility_index'], 2),
                'air_quality_index': round(future_factors['air_quality_index'], 0)
            }
            predictions.append(prediction_data)
            
            # Update sequence
            new_row = seq[-1].copy()
            new_row[0] = pred.item()  # normalized cases
            seq = np.vstack([seq[1:], new_row])
    
    return predictions


# -----------------------------
# Display Predictions Table
# -----------------------------
def display_predictions_table(predictions):
    """
    Display predictions in a formatted table
    """
    print("\n" + "=" * 100)
    print("📊 7-DAY OUTBREAK PREDICTION WITH RISK ASSESSMENT")
    print("=" * 100)
    print()
    
    # Header
    print(f"{'Day':<6} {'Cases':<12} {'Risk Score':<12} {'Risk Level':<15} {'Hosp%':<10} {'Death%':<10} {'AQI':<8}")
    print("-" * 100)
    
    # Rows
    for pred in predictions:
        print(f"{pred['day']:<6} "
              f"{pred['predicted_cases']:<12} "
              f"{pred['risk_score']:<12.3f} "
              f"{pred['risk_icon']} {pred['risk_level']:<12} "
              f"{pred['hospitalization_rate']:<10.2f} "
              f"{pred['death_rate']:<10.2f} "
              f"{pred['air_quality_index']:<8.0f}")
    
    print("=" * 100)
    
    # Summary Statistics
    avg_risk = np.mean([p['risk_score'] for p in predictions])
    max_risk_day = max(predictions, key=lambda x: x['risk_score'])
    total_predicted_cases = sum([p['predicted_cases'] for p in predictions])
    
    print(f"\n📈 SUMMARY:")
    print(f"   • Average Risk Score: {avg_risk:.3f}")
    print(f"   • Highest Risk: Day {max_risk_day['day']} ({max_risk_day['risk_level']}, Score: {max_risk_day['risk_score']:.3f})")
    print(f"   • Total Predicted Cases (7 days): {total_predicted_cases}")
    print(f"   • Daily Average Cases: {total_predicted_cases / 7:.0f}")
    
    # Risk trend
    if predictions[-1]['risk_score'] > predictions[0]['risk_score']:
        trend = "INCREASING ⚠️"
    elif predictions[-1]['risk_score'] < predictions[0]['risk_score']:
        trend = "DECREASING ✓"
    else:
        trend = "STABLE →"
    
    print(f"   • Risk Trend: {trend}")
    print()


# -----------------------------
# Plot predictions with risk
# -----------------------------
def plot_predictions_with_risk(predictions, df, max_cases):
    """
    Create visualization of predictions with risk levels
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cases prediction
    days = [p['day'] for p in predictions]
    cases = [p['predicted_cases'] for p in predictions]
    
    # Historical data (last 30 days)
    historical_cases = df['cases'].tail(30).values
    historical_days = range(-len(historical_cases), 0)
    
    ax1.plot(historical_days, historical_cases, 'b-', label='Historical Cases', linewidth=2)
    ax1.plot(days, cases, 'r--', label='Predicted Cases', linewidth=2, marker='o')
    ax1.axvline(x=0, color='gray', linestyle=':', label='Today')
    ax1.set_xlabel('Days (negative = past, positive = future)', fontsize=12)
    ax1.set_ylabel('Number of Cases', fontsize=12)
    ax1.set_title('Disease Outbreak Prediction: Cases', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk Score
    risk_scores = [p['risk_score'] for p in predictions]
    colors = [p['risk_icon'] for p in predictions]
    
    bars = ax2.bar(days, risk_scores, color=['red' if s >= 0.75 else 
                                               'orange' if s >= 0.50 else 
                                               'yellow' if s >= 0.30 else 
                                               'green' for s in risk_scores])
    
    # Add risk level zones
    ax2.axhspan(0.75, 1.0, alpha=0.1, color='red', label='CRITICAL')
    ax2.axhspan(0.50, 0.75, alpha=0.1, color='orange', label='HIGH')
    ax2.axhspan(0.30, 0.50, alpha=0.1, color='yellow', label='MODERATE')
    ax2.axhspan(0.0, 0.30, alpha=0.1, color='green', label='LOW')
    
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Risk Score (0-1)', fontsize=12)
    ax2.set_title('Disease Outbreak Risk Assessment (7-Day Forecast)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (day, score) in enumerate(zip(days, risk_scores)):
        ax2.text(day, score + 0.02, f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outbreak_prediction_with_risk.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved as 'outbreak_prediction_with_risk.png'")
    plt.show()


# -----------------------------
# Export to CSV
# -----------------------------
def export_predictions_to_csv(predictions, filename='predictions_with_risk.csv'):
    """
    Export predictions to CSV file
    """
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv(filename, index=False)
    print(f"💾 Predictions exported to '{filename}'")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("=" * 100)
    print("🏥 DISEASE OUTBREAK PREDICTION WITH RISK ASSESSMENT")
    print("=" * 100)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")

    # Load dataset
    df = pd.read_csv("synthetic_data.csv")
    df.fillna(0, inplace=True)
    print(f"📊 Dataset loaded: {len(df)} rows")

    seq_len = 30

    # Load trained model
    print("🤖 Loading trained model...")
    model = OutbreakAttentionModel().to(device)
    model.load_state_dict(torch.load("outbreak_model.pth", map_location=device))
    print("✅ Model loaded successfully")

    # Prepare last sequence
    last_seq_list = df.tail(seq_len).apply(preprocess_data_row, axis=1).tolist()
    last_seq = np.stack(last_seq_list, axis=0).astype(np.float32)
    
    # Get last row for risk factor estimation
    last_row_data = df.iloc[-1]

    # Create sequences for denormalization
    _, _, max_cases = create_sequences(df, seq_len)

    # Predict next 7 days WITH RISK ASSESSMENT
    print("\n🔮 Generating 7-day predictions with risk assessment...")
    predictions = predict_next_days_with_risk(
        model=model,
        last_seq=last_seq,
        last_row_data=last_row_data,
        days=7,
        max_cases=max_cases,
        device=device
    )

    # Display predictions in table format
    display_predictions_table(predictions)

    # Export to CSV
    export_predictions_to_csv(predictions)

    # Create visualizations
    print("\n📈 Generating visualizations...")
    plot_predictions_with_risk(predictions, df, max_cases)

    print("\n✨ Prediction complete!")
    print("=" * 100)