#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --------------------------------------------------
# Disease Model Class
# --------------------------------------------------
class OutbreakAttentionModel(torch.nn.Module):
    def __init__(self, input_size=15, hidden_size=64):
        super(OutbreakAttentionModel, self).__init__()
        
        # LSTM layer (learns time patterns)
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Attention layer (focuses on important days)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        # Output layer
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_day = attn_out[:, -1, :]
        risk = self.sigmoid(self.fc(last_day))
        return risk

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def calculate_risk_score(cases, hospitalization_rate, death_rate, air_quality, mobility_index, 
                         population_density=5000, max_cases=10000):
    """Calculate risk score (0-1) based on multiple factors"""
    case_factor = min(cases / max_cases, 1.0) * 0.30
    hosp_factor = (hospitalization_rate / 100) * 0.25
    death_factor = (death_rate / 100) * 0.20
    air_factor = min(air_quality / 500, 1.0) * 0.10
    mobility_factor = (mobility_index / 100) * 0.10
    density_factor = min(population_density / 10000, 1.0) * 0.05
    
    risk_score = (case_factor + hosp_factor + death_factor + 
                  air_factor + mobility_factor + density_factor)
    risk_score = max(0.0, min(1.0, risk_score))
    
    return risk_score

def get_risk_level(risk_score):
    """Convert risk score to categorical risk level"""
    if risk_score >= 0.75:
        return "CRITICAL", "🔴"
    elif risk_score >= 0.50:
        return "HIGH", "🟠"
    elif risk_score >= 0.30:
        return "MODERATE", "🟡"
    else:
        return "LOW", "🟢"

def estimate_future_factors(last_row, day_ahead, trend_multiplier=1.02):
    """Estimate future values for risk factors based on trends"""
    hosp_rate = last_row['hospitalization_rate'] * (trend_multiplier ** day_ahead)
    death_rate = last_row['death_rate'] * (trend_multiplier ** day_ahead)
    mobility = last_row['mobility_index'] * (trend_multiplier ** (day_ahead * 0.5))
    air_quality = last_row['air_quality_index'] + np.random.normal(0, 5) * day_ahead
    
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
        0, 0, 0, 0, 0  # Padding to reach 15 features
    ]
    return np.array(features, dtype=np.float32)

def create_sequences(df, seq_len=30):
    """Create sequences from dataset"""
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

def predict_risk_from_symptoms(symptoms_text):
    """Simple symptom-based risk assessment"""
    symptoms = symptoms_text.lower()
    risk_factors = {
        'fever': 0.15,
        'cough': 0.10,
        'breathing': 0.20,
        'breath': 0.20,
        'shortness of breath': 0.25,
        'fatigue': 0.05,
        'tired': 0.05,
        'headache': 0.05,
        'chest pain': 0.25,
        'chest': 0.20,
        'loss of taste': 0.15,
        'loss of smell': 0.15,
        'taste': 0.12,
        'smell': 0.12,
        'sore throat': 0.08,
        'throat': 0.08,
        'diarrhea': 0.10,
        'vomiting': 0.12,
        'body ache': 0.07,
        'ache': 0.07,
        'chills': 0.08,
        'nausea': 0.08,
        'congestion': 0.06,
        'runny nose': 0.05,
        'muscle pain': 0.07,
        'covid': 0.20,
        'corona': 0.20,
        'flu': 0.15,
        'cold': 0.08
    }
    
    total_risk = 0.0
    matched_symptoms = []
    
    for symptom, risk in risk_factors.items():
        if symptom in symptoms:
            total_risk += risk
            if symptom not in matched_symptoms:
                matched_symptoms.append(symptom)
    
    # Base risk for any symptom
    if len(matched_symptoms) > 0:
        total_risk += 0.1
    
    # Cap at 0.95
    total_risk = min(total_risk, 0.95)
    
    return total_risk, matched_symptoms

def generate_health_advice(risk_score, matched_symptoms):
    """Generate health advice based on risk score and symptoms"""
    advice = []
    
    if risk_score >= 0.6:
        advice.append("🚨 HIGH RISK DETECTED: Consider immediate medical consultation")
        advice.append("📍 Action: Visit Emergency Department or call hospital helpline")
        advice.append("⚠️ Do not delay seeking medical attention")
    elif risk_score >= 0.3:
        advice.append("⚠️ MODERATE RISK: Schedule an appointment with a doctor")
        advice.append("📞 Action: Book OPD consultation within 24 hours")
        advice.append("📊 Monitor your symptoms closely")
    else:
        advice.append("✅ LOW RISK: Monitor symptoms at home")
        advice.append("🏠 Action: Rest, hydrate, and observe for 24-48 hours")
    
    if 'fever' in matched_symptoms:
        advice.append("🌡️ For Fever: Take paracetamol (as directed), keep hydrated, monitor temperature every 4 hours")
    
    if any(s in matched_symptoms for s in ['cough', 'breathing', 'breath', 'chest']):
        advice.append("😷 Respiratory Symptoms: Use steam inhalation, avoid cold beverages, rest in upright position")
    
    if any(s in matched_symptoms for s in ['covid', 'corona']):
        advice.append("🦠 COVID-19 Suspected: Self-isolate immediately, get tested, wear N95 mask, inform close contacts")
    
    if len(matched_symptoms) >= 3:
        advice.append("📊 Multiple Symptoms: Higher likelihood of infectious disease - isolation recommended")
    
    advice.append("🩺 General Advice: Maintain social distance, wear mask in public, report worsening symptoms immediately")
    advice.append("📞 Emergency: If difficulty breathing, chest pain, or high fever (>103°F), call emergency services")
    
    return advice

# --------------------------------------------------
# API Endpoints
# --------------------------------------------------
@app.route('/api/health/chat', methods=['POST'])
def health_chat():
    """Endpoint for symptom analysis and guidance"""
    try:
        data = request.json
        symptoms = data.get('symptoms', '')
        
        if not symptoms:
            return jsonify({
                'success': False,
                'error': 'No symptoms provided'
            }), 400
        
        # Calculate risk from symptoms
        risk_score, matched_symptoms = predict_risk_from_symptoms(symptoms)
        risk_level, risk_icon = get_risk_level(risk_score)
        
        # Generate advice
        advice = generate_health_advice(risk_score, matched_symptoms)
        
        # Determine recommended department
        symptoms_lower = symptoms.lower()
        if any(word in symptoms_lower for word in ['breathing', 'breath', 'chest pain', 'chest']):
            department = "Cardiology & Pulmonology"
        elif any(word in symptoms_lower for word in ['fever', 'cough', 'covid', 'corona', 'flu']):
            department = "General Medicine & Infectious Diseases"
        elif any(word in symptoms_lower for word in ['stomach', 'diarrhea', 'vomiting', 'nausea']):
            department = "Gastroenterology"
        elif any(word in symptoms_lower for word in ['headache', 'dizziness', 'neurological']):
            department = "Neurology"
        else:
            department = "General Medicine"
        
        # Create detailed response message
        if matched_symptoms:
            symptoms_text = ', '.join(matched_symptoms)
            response_text = f"Based on your symptoms ({symptoms_text}), I've identified a {risk_level} risk level ({risk_score*100:.1f}%). "
        else:
            response_text = f"I've analyzed your input and identified a {risk_level} risk level ({risk_score*100:.1f}%). "
        
        response_text += f"I recommend consulting with our {department} department."
        
        return jsonify({
            'success': True,
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'risk_icon': risk_icon,
            'matched_symptoms': matched_symptoms,
            'department_recommendation': department,
            'advice': advice,
            'response': response_text
        })
        
    except Exception as e:
        print(f"Error in health_chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/epidemic/predict', methods=['GET'])
def predict_outbreak():
    """Endpoint for epidemic outbreak prediction"""
    try:
        # Check if model and data files exist
        if not os.path.exists('outbreak_model.pth'):
            # Create a dummy model for demo
            device = torch.device("cpu")
            model = OutbreakAttentionModel().to(device)
            torch.save(model.state_dict(), 'outbreak_model.pth')
            
            # Create dummy data
            dates = pd.date_range(end=datetime.now(), periods=100).tolist()
            dummy_data = {
                'date': dates,
                'cases': np.random.randint(100, 1000, 100),
                'hospitalization_rate': np.random.uniform(5, 20, 100),
                'death_rate': np.random.uniform(0.5, 5, 100),
                'vaccination_rate': np.random.uniform(60, 90, 100),
                'temperature': np.random.uniform(25, 35, 100),
                'humidity': np.random.uniform(60, 90, 100),
                'air_quality_index': np.random.randint(50, 200, 100),
                'mobility_index': np.random.uniform(70, 95, 100),
                'public_transport_usage': np.random.uniform(60, 90, 100),
                'gathering_events': np.random.uniform(10, 50, 100)
            }
            df = pd.DataFrame(dummy_data)
            df.to_csv('synthetic_data.csv', index=False)
            print("✅ Created dummy model and data for demo")
        else:
            # Load existing model and data
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            df = pd.read_csv("synthetic_data.csv")
            df.fillna(0, inplace=True)
            
            model = OutbreakAttentionModel().to(device)
            model.load_state_dict(torch.load("outbreak_model.pth", map_location=device))
            print("✅ Loaded existing model and data")
        
        # Prepare data for prediction
        seq_len = 30
        last_seq_list = df.tail(seq_len).apply(preprocess_data_row, axis=1).tolist()
        last_seq = np.stack(last_seq_list, axis=0).astype(np.float32)
        last_row_data = df.iloc[-1]
        
        _, _, max_cases = create_sequences(df, seq_len)
        
        # Generate predictions for next 7 days
        predictions = []
        model.eval()
        
        with torch.no_grad():
            seq = last_seq.copy()
            for day in range(1, 8):
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
                
                risk_level, risk_icon = get_risk_level(risk_score)
                
                # Store prediction
                prediction_date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
                predictions.append({
                    'day': day,
                    'date': prediction_date,
                    'predicted_cases': int(pred_cases),
                    'risk_score': round(risk_score, 3),
                    'risk_level': risk_level,
                    'risk_icon': risk_icon,
                    'hospitalization_rate': round(future_factors['hospitalization_rate'], 2),
                    'death_rate': round(future_factors['death_rate'], 2),
                    'mobility_index': round(future_factors['mobility_index'], 2),
                    'air_quality_index': round(future_factors['air_quality_index'], 0)
                })
                
                # Update sequence
                new_row = seq[-1].copy()
                new_row[0] = pred.item()
                seq = np.vstack([seq[1:], new_row])
        
        # Calculate summary statistics
        avg_risk = np.mean([p['risk_score'] for p in predictions])
        max_risk_day = max(predictions, key=lambda x: x['risk_score'])
        total_cases = sum([p['predicted_cases'] for p in predictions])
        
        # Determine trend
        if predictions[-1]['risk_score'] > predictions[0]['risk_score']:
            trend = "INCREASING ⚠️"
        elif predictions[-1]['risk_score'] < predictions[0]['risk_score']:
            trend = "DECREASING ✓"
        else:
            trend = "STABLE →"
        
        # Create visualization
        days = [p['day'] for p in predictions]
        cases = [p['predicted_cases'] for p in predictions]
        risk_scores = [p['risk_score'] for p in predictions]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot cases
        ax1.bar(days, cases, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Days Ahead', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Cases', fontsize=12, fontweight='bold')
        ax1.set_title('7-Day Outbreak Prediction - Manipur Hospital', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (day, case) in enumerate(zip(days, cases)):
            ax1.text(day, case + max(cases)*0.02, str(case), ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot risk scores
        colors = ['red' if s >= 0.75 else 
                 'orange' if s >= 0.5 else 
                 'yellow' if s >= 0.3 else 
                 'green' for s in risk_scores]
        
        bars = ax2.bar(days, risk_scores, color=colors, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Days Ahead', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Risk Score (0-1)', fontsize=12, fontweight='bold')
        ax2.set_title('Epidemic Risk Assessment', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, risk_scores)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close('all')  # Close all figures to free memory
        
        print(f"✅ Generated prediction for {len(predictions)} days")
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'summary': {
                'average_risk': round(avg_risk, 3),
                'highest_risk_day': max_risk_day['day'],
                'highest_risk_level': max_risk_day['risk_level'],
                'total_predicted_cases': total_cases,
                'daily_average_cases': round(total_cases / 7, 0),
                'risk_trend': trend
            },
            'visualization': f"data:image/png;base64,{img_base64}",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        print(f"Error in predict_outbreak: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/epidemic/update', methods=['POST'])
def update_epidemic_data():
    """Endpoint to update epidemic data"""
    try:
        data = request.json
        new_cases = data.get('cases', 0)
        location = data.get('location', 'Unknown')
        
        # Here you would update your database
        # For now, return success message
        
        return jsonify({
            'success': True,
            'message': f'Epidemic data updated for {location}',
            'new_cases': new_cases,
            'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/')
def index():
    return jsonify({
        'name': 'Manipur Hospital Epidemic Detection API',
        'version': '2.0.0',
        'status': 'running',
        'endpoints': {
            '/api/health/chat': 'POST - Analyze symptoms and provide guidance',
            '/api/epidemic/predict': 'GET - Get outbreak predictions',
            '/api/epidemic/update': 'POST - Update epidemic data'
        }
    })

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Starting Manipur Hospital Epidemic Detection System")
    print("=" * 80)
    print("\n📊 API Endpoints:")
    print("   POST /api/health/chat       - Symptom analysis & health guidance")
    print("   GET  /api/epidemic/predict  - 7-day outbreak prediction")
    print("   POST /api/epidemic/update   - Update epidemic data")
    print("\n🌐 Server running on http://localhost:5000")
    print("=" * 80)
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)