"""
Inference script for Cardiovascular Disease Prediction
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src import config


class CardiovascularPredictor:
    """
    Cardiovascular Disease Risk Predictor
    """
    
    def __init__(self):
        """Initialize predictor by loading model artifacts"""
        self.model = None
        self.scaler = None
        self.features = None
        self.metadata = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model, scaler, and features"""
        try:
            # Load model
            self.model = joblib.load(config.MODEL_FILE)
            print(f"✓ Model loaded from {config.MODEL_FILE}")
            
            # Load scaler
            self.scaler = joblib.load(config.SCALER_FILE)
            print(f"✓ Scaler loaded from {config.SCALER_FILE}")
            
            # Load features
            with open(config.MODELS_DIR / 'final_features.json', 'r') as f:
                self.features = json.load(f)
            print(f"✓ Features loaded: {len(self.features)} features")
            
            # Load metadata
            with open(config.MODEL_METADATA_FILE, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Metadata loaded")
            
        except Exception as e:
            raise Exception(f"Error loading model artifacts: {str(e)}")
    
    def preprocess_input(self, input_data):
        """
        Preprocess raw input data
        
        Parameters:
        -----------
        input_data : dict or pd.DataFrame
            Raw input data with original features
            
        Returns:
        --------
        pd.DataFrame : Preprocessed data with engineered features
        """
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # 1. Age transformation (days to years)
        df['age'] = (df['age'] / 365.25).round().astype(int)
        
        # 2. Feature Engineering - Numerical
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
        df['health_risk_score'] = (
            (df['cholesterol'] - 1) * 2 + 
            (df['gluc'] - 1) * 2 + 
            df['smoke'] * 3 + 
            df['alco'] * 2 - 
            df['active'] * 2
        )
        df['gender_age_interaction'] = df['gender'] * df['age']
        df['weight_height_ratio'] = df['weight'] / df['height']
        df['chol_gluc_interaction'] = df['cholesterol'] * df['gluc']
        
        # 3. Feature Engineering - Categorical
        df['age_group'] = pd.cut(df['age'], 
                                 bins=[0, 40, 50, 60, 100],
                                 labels=['young', 'middle', 'senior', 'elderly'])
        
        df['bmi_category'] = pd.cut(df['bmi'],
                                    bins=[0, 18.5, 25, 30, 100],
                                    labels=['underweight', 'normal', 'overweight', 'obese'])
        
        def bp_category(row):
            if row['ap_hi'] < 120 and row['ap_lo'] < 80:
                return 'normal'
            elif row['ap_hi'] < 130 and row['ap_lo'] < 80:
                return 'elevated'
            elif row['ap_hi'] < 140 or row['ap_lo'] < 90:
                return 'hypertension_stage1'
            else:
                return 'hypertension_stage2'
        
        df['bp_category'] = df.apply(bp_category, axis=1)
        
        # 4. One-hot encoding
        df_encoded = pd.get_dummies(df, 
                                    columns=['age_group', 'bmi_category', 'bp_category'], 
                                    drop_first=True)
        
        return df_encoded
    
    def predict(self, input_data):
        """
        Make prediction for input data
        
        Parameters:
        -----------
        input_data : dict or pd.DataFrame
            Raw input data
            
        Returns:
        --------
        dict : Prediction results
        """
        try:
            # Preprocess
            df_processed = self.preprocess_input(input_data)
            
            # Select features
            X = df_processed[self.features]
            
            # Convert to numpy array
            X_array = X.values
            
            # Scale
            X_scaled = self.scaler.transform(X_array)
            
            # Predict
            prediction = int(self.model.predict(X_scaled)[0])
            probability = float(self.model.predict_proba(X_scaled)[0, 1])
            
            # Risk category
            if probability < config.LOW_RISK_THRESHOLD:
                risk_category = config.RISK_CATEGORIES['low']
            elif probability < config.HIGH_RISK_THRESHOLD:
                risk_category = config.RISK_CATEGORIES['medium']
            else:
                risk_category = config.RISK_CATEGORIES['high']
            
            # Confidence
            confidence = float(max(self.model.predict_proba(X_scaled)[0]))
            
            return {
                'prediction': prediction,
                'prediction_label': 'Disease' if prediction == 1 else 'Healthy',
                'probability': probability,
                'risk_category': risk_category,
                'confidence': confidence,
                'model_version': self.metadata.get('test_roc_auc', 'N/A')
            }
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def predict_batch(self, input_data_list):
        """
        Make predictions for multiple inputs
        
        Parameters:
        -----------
        input_data_list : list of dict
            List of input data
            
        Returns:
        --------
        list : List of prediction results
        """
        results = []
        for input_data in input_data_list:
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results


def main():
    """Test inference"""
    # Example input
    example_input = {
        "age": 18393,  # days
        "gender": 2,
        "height": 168,
        "weight": 62,
        "ap_hi": 110,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }
    
    # Initialize predictor
    print("\nInitializing Cardiovascular Predictor...")
    predictor = CardiovascularPredictor()
    
    # Make prediction
    print("\nMaking prediction...")
    result = predictor.predict(example_input)
    
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("="*50)


if __name__ == "__main__":
    main()