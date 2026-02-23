import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import os

def run_ensemble_diagnostics():
    print("--- STARTING NETWORK DIAGNOSTICS V4.0 (MULTI-MODEL ENSEMBLE) ---")

    # 1. LOAD DATA (Dynamic Pathing)
    print("[1/5] Loading Signal Logs...")
    if os.path.exists('/kaggle/input/playground-series-s6e2/train.csv'):
        print("   > Kaggle environment detected.")
        train_path = '/kaggle/input/playground-series-s6e2/train.csv'
        test_path = '/kaggle/input/playground-series-s6e2/test.csv'
    else:
        print("   > Local environment detected.")
        train_path = 'train.csv'
        test_path = 'test.csv'
        
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    except FileNotFoundError:
        print("CRITICAL ERROR: Log files not found. Check directory paths.")
        return

    # Convert Target Variable to Numeric
    train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0.0000, 'Presence': 1.0000})

    # 2. FEATURE ENGINEERING
    print("[2/5] Applying Signal Transformations...")
    train['Voltage_Capacitance_Ratio'] = train['BP'] / train['Cholesterol']
    test['Voltage_Capacitance_Ratio'] = test['BP'] / test['Cholesterol']
    
    train['Uptime_Frequency_Delta'] = train['Max HR'] - train['Age']
    test['Uptime_Frequency_Delta'] = test['Max HR'] - test['Age']

    # 3. PREPARE FEATURES
    features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'Max HR', 'ST depression', 'Voltage_Capacitance_Ratio', 'Uptime_Frequency_Delta'
    ]
    target = 'Heart Disease'

    X = train[features].fillna(0.0000)
    y = train[target]
    X_test = test[features].fillna(0.0000)

    # 4. TRAIN ALGORITHMS (Redundant Systems)
    print("[3/5] Calibrating Random Forest Engine...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf_model.fit(X, y)
    rf_preds = rf_model.predict_proba(X_test)[:, 1]

    print("[4/5] Calibrating XGBoost Engine...")
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.0500, random_state=42, eval_metric='auc')
    xgb_model.fit(X, y)
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]

    print("[5/5] Calibrating LightGBM Engine...")
    lgbm_model = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.0500, random_state=42, verbose=-1)
    lgbm_model.fit(X, y)
    lgbm_preds = lgbm_model.predict_proba(X_test)[:, 1]

    # 5. GENERATE ENSEMBLE OUTPUT
    print("\n--- GENERATING FINAL ENSEMBLE REPORT ---")
    final_predictions = (rf_preds + xgb_preds + lgbm_preds) / 3.0000

    submission = pd.DataFrame({
        'id': test['id'],
        'Heart Disease': final_predictions
    })

    submission.to_csv('submission.csv', index=False)
    print("SUCCESS: 'submission.csv' generated using Multi-Model Ensemble.")

if __name__ == "__main__":
    run_ensemble_diagnostics()