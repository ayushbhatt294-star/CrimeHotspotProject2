from typing import List, Optional
from datetime import datetime, timedelta, timezone
from ..core.config import settings
from ..models.schemas import Location, CrimeIncident, PatrolPoint, HotspotZone
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
import os

class RiskAnalysis:
    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_model()
    
    def _load_model(self):
        try:
            model_path = os.path.join(settings.MODEL_DIR, 'crime_risk_model.joblib')
            scaler_path = os.path.join(settings.MODEL_DIR, 'crime_risk_scaler.joblib')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load(model_path)
                self.scaler = load(scaler_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def predict_risk(self, area_features: pd.DataFrame) -> float:
        if self.model is None or self.scaler is None:
            return 0.0
        
        try:
            scaled_features = self.scaler.transform(area_features)
            risk_score = self.model.predict_proba(scaled_features)[0][1]  # probability of high risk
            return float(risk_score * 100)  # convert to percentage
        except Exception as e:
            print(f"Error predicting risk: {str(e)}")
            return 0.0

    def calculate_area_risk(self, 
                          crimes: List[CrimeIncident], 
                          area_name: str, 
                          time_window: Optional[int] = None) -> dict:
        """Calculate risk score for a specific area"""
        if not crimes:
            return {
                'area_name': area_name,
                'risk_score': 0.0,
                'risk_category': 'Low',
                'predicted_crimes': 0,
                'crime_trend': 'stable'
            }
        
        # Filter recent crimes if time window specified
        if time_window:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window)
            crimes = [c for c in crimes if c.timestamp >= cutoff]
        
        # Calculate basic metrics
        crime_count = len(crimes)
        severity_sum = sum(c.severity for c in crimes)
        unique_types = len(set(c.crime_type for c in crimes))
        
        # Create feature vector
        features = pd.DataFrame({
            'crime_count': [crime_count],
            'avg_severity': [severity_sum / max(crime_count, 1)],
            'crime_type_diversity': [unique_types],
            'hours_since_last': [(datetime.now(timezone.utc) - max(c.timestamp for c in crimes)).total_seconds() / 3600]
        })
        
        # Get risk score from model
        risk_score = self.predict_risk(features)
        
        # Determine risk category
        risk_category = 'Low'
        if risk_score >= 75:
            risk_category = 'Very High'
        elif risk_score >= 50:
            risk_category = 'High'
        elif risk_score >= 25:
            risk_category = 'Medium'
        
        # Predict crime count
        avg_crimes_per_hour = crime_count / (time_window if time_window else 24)
        predicted_crimes = avg_crimes_per_hour * 24  # predicted crimes next 24 hours
        
        # Calculate trend
        if time_window and time_window >= 48:
            recent = len([c for c in crimes if c.timestamp >= datetime.now(timezone.utc) - timedelta(hours=24)])
            previous = len([c for c in crimes if datetime.now(timezone.utc) - timedelta(hours=48) <= c.timestamp < datetime.now(timezone.utc) - timedelta(hours=24)])
            
            if recent > previous * 1.2:
                trend = 'increasing'
            elif recent < previous * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'area_name': area_name,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'predicted_crimes': predicted_crimes,
            'crime_trend': trend
        }

risk_analyzer = RiskAnalysis()