"""
Complete ML Pipeline for Cardiovascular Disease Prediction
This script executes the entire ML workflow from data loading to model training
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src import config
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, roc_auc_score, roc_curve,
                            precision_recall_curve, f1_score)
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


class CardiovascularMLPipeline:
    """
    Complete ML Pipeline for Cardiovascular Disease Prediction
    """
    
    def __init__(self, data_path=None, load_existing_model=False):
        """
        Initialize pipeline
        
        Parameters:
        -----------
        data_path : str or Path
            Path to raw data file
        load_existing_model : bool
            If True, load existing model instead of training new one
        """
        self.data_path = data_path or config.RAW_DATA_FILE
        self.load_existing_model = load_existing_model
        
        # Initialize attributes
        self.df_raw = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None
        self.model = None
        self.final_features = None
        self.model_params = None
        self.metadata = {}
        
        print("=" * 70)
        print("CARDIOVASCULAR DISEASE PREDICTION - ML PIPELINE")
        print("=" * 70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def load_data(self):
        """Load raw data"""
        print("\n[1/8] Loading raw data...")
        self.df_raw = pd.read_csv(self.data_path, delimiter=';')
        print(f"  ✓ Loaded {len(self.df_raw):,} records")
        print(f"  ✓ Features: {self.df_raw.shape[1]}")
        return self
    
    def preprocess_data(self):
        """Preprocess and engineer features"""
        print("\n[2/8] Preprocessing and feature engineering...")
        
        df = self.df_raw.copy()
        initial_len = len(df)
        
        # 1. Age transformation
        df['age'] = (df['age'] / 365.25).round().astype(int)
        print("  ✓ Age transformed (days → years)")
        
        # 2. Data cleaning
        df = df[(df['ap_hi'] > 0) & (df['ap_hi'] < 250)]
        df = df[(df['ap_lo'] > 0) & (df['ap_lo'] < 200)]
        df = df[df['ap_hi'] > df['ap_lo']]
        df = df[(df['height'] > 120) & (df['height'] < 220)]
        df = df[(df['weight'] > 30) & (df['weight'] < 200)]
        
        removed = initial_len - len(df)
        print(f"  ✓ Removed {removed:,} outliers ({(removed/initial_len)*100:.2f}%)")
        
        # 3. Feature Engineering - Numerical
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
        
        print("  ✓ Created 7 numerical features")
        
        # 4. Feature Engineering - Categorical
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
        
        # 5. One-hot encoding
        df = pd.get_dummies(df, 
                           columns=['age_group', 'bmi_category', 'bp_category'], 
                           drop_first=True)
        
        print("  ✓ Created categorical features and encoded")
        
        self.df_processed = df
        print(f"  ✓ Final shape: {df.shape}")
        
        return self
    
    def load_final_features(self):
        """Load final feature list"""
        print("\n[3/8] Loading final feature list...")
        
        features_file = config.PROCESSED_DATA_DIR / 'final_features.json'
        
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.final_features = json.load(f)
            print(f"  ✓ Loaded {len(self.final_features)} final features")
        else:
            print("  ⚠ Final features file not found, using all features")
            self.final_features = [col for col in self.df_processed.columns 
                                  if col not in ['id', 'cardio']]
        
        return self
    
    def split_data(self):
        """Split data into train and test sets"""
        print("\n[4/8] Splitting data...")
        
        # Prepare X and y
        X = self.df_processed[self.final_features]
        y = self.df_processed['cardio']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        
        print(f"  ✓ Train set: {self.X_train.shape}")
        print(f"  ✓ Test set: {self.X_test.shape}")
        print(f"  ✓ Class balance maintained")
        
        return self
    
    def scale_features(self):
        """Scale features using StandardScaler"""
        print("\n[5/8] Scaling features...")
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("  ✓ Features scaled (StandardScaler)")
        
        return self
    
    def train_model(self):
        """Train the final model"""
        print("\n[6/8] Training model...")
        
        # Load model parameters if exists, else use defaults
        params_file = config.MODELS_DIR / 'model_params.json'
        
        if params_file.exists():
            with open(params_file, 'r') as f:
                self.model_params = json.load(f)
            print("  ✓ Loaded optimized parameters")
        else:
            self.model_params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.5,
                'random_state': config.RANDOM_STATE,
                'eval_metric': 'logloss'
            }
            print("  ✓ Using default parameters")
        
        # Train model
        self.model = xgb.XGBClassifier(**self.model_params)
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print("  ✓ Model trained successfully")
        
        return self
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n[7/8] Evaluating model...")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, 
                            random_state=config.RANDOM_STATE)
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, 
                                    cv=cv, scoring='roc_auc', n_jobs=-1)
        
        print(f"  ✓ Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Train predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_train_pred_proba = self.model.predict_proba(self.X_train_scaled)[:, 1]
        
        # Test predictions
        y_test_pred = self.model.predict(self.X_test_scaled)
        y_test_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        train_roc_auc = roc_auc_score(self.y_train, y_train_pred_proba)
        
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_roc_auc = roc_auc_score(self.y_test, y_test_pred_proba)
        test_f1 = f1_score(self.y_test, y_test_pred)
        
        # Store metadata
        self.metadata = {
            'model_type': 'XGBoost Classifier',
            'n_features': len(self.final_features),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'train_accuracy': float(train_accuracy),
            'train_roc_auc': float(train_roc_auc),
            'test_accuracy': float(test_accuracy),
            'test_roc_auc': float(test_roc_auc),
            'test_f1': float(test_f1),
            'cv_mean_roc_auc': float(cv_scores.mean()),
            'cv_std_roc_auc': float(cv_scores.std()),
            'random_state': config.RANDOM_STATE,
            'test_size': config.TEST_SIZE,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print results
        print(f"\n  📊 TRAIN METRICS:")
        print(f"     Accuracy: {train_accuracy:.4f}")
        print(f"     ROC-AUC:  {train_roc_auc:.4f}")
        
        print(f"\n  📊 TEST METRICS:")
        print(f"     Accuracy: {test_accuracy:.4f}")
        print(f"     ROC-AUC:  {test_roc_auc:.4f}")
        print(f"     F1-Score: {test_f1:.4f}")
        
        # Classification report
        print(f"\n  📋 CLASSIFICATION REPORT:")
        print(classification_report(self.y_test, y_test_pred, 
                                   target_names=['Healthy', 'Disease'],
                                   digits=4))
        
        return self
    
    def save_artifacts(self):
        """Save model and pipeline artifacts"""
        print("\n[8/8] Saving artifacts...")
        
        # Create models directory
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = config.MODELS_DIR / 'final_model.pkl'
        joblib.dump(self.model, model_path)
        print(f"  ✓ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = config.MODELS_DIR / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✓ Scaler saved: {scaler_path}")
        
        # Save features
        features_path = config.MODELS_DIR / 'final_features.json'
        with open(features_path, 'w') as f:
            json.dump(self.final_features, f, indent=2)
        print(f"  ✓ Features saved: {features_path}")
        
        # Save parameters
        params_path = config.MODELS_DIR / 'model_params.json'
        with open(params_path, 'w') as f:
            json.dump(self.model_params, f, indent=2)
        print(f"  ✓ Parameters saved: {params_path}")
        
        # Save metadata
        metadata_path = config.MODELS_DIR / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"  ✓ Metadata saved: {metadata_path}")
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': self.final_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = config.PROCESSED_DATA_DIR / 'final_feature_importance.csv'
        feature_importance.to_csv(importance_path, index=False)
        print(f"  ✓ Feature importance saved: {importance_path}")
        
        return self
    
    def run(self):
        """Execute complete pipeline"""
        try:
            if self.load_existing_model:
                print("\n⚠ Loading existing model (training skipped)")
                self.model = joblib.load(config.MODEL_FILE)
                self.scaler = joblib.load(config.SCALER_FILE)
                with open(config.MODELS_DIR / 'final_features.json', 'r') as f:
                    self.final_features = json.load(f)
                with open(config.MODEL_METADATA_FILE, 'r') as f:
                    self.metadata = json.load(f)
                print("✓ Model loaded successfully")
            else:
                (self
                 .load_data()
                 .preprocess_data()
                 .load_final_features()
                 .split_data()
                 .scale_features()
                 .train_model()
                 .evaluate_model()
                 .save_artifacts())
            
            print("\n" + "=" * 70)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\n📊 Final Model Performance:")
            print(f"   ROC-AUC: {self.metadata.get('test_roc_auc', 'N/A'):.4f}")
            print(f"   Accuracy: {self.metadata.get('test_accuracy', 'N/A'):.4f}")
            print(f"   F1-Score: {self.metadata.get('test_f1', 'N/A'):.4f}")
            print("=" * 70)
            
            return self
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            raise


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cardiovascular Disease ML Pipeline')
    parser.add_argument('--data', type=str, help='Path to raw data file')
    parser.add_argument('--load-model', action='store_true', 
                       help='Load existing model instead of training')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = CardiovascularMLPipeline(
        data_path=args.data,
        load_existing_model=args.load_model
    )
    pipeline.run()


if __name__ == "__main__":
    main()