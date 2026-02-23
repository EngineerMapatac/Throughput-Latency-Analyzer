import pandas as pd
import xgboost as xgb

def run_advanced_diagnostics():
    print("--- STARTING NETWORK DIAGNOSTICS V2.0 (XGBOOST) ---")

    # 1. LOAD DATA
    print("[1/4] Loading Signal Logs...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # 2. FEATURE ENGINEERING (Process Improvement)
    print("[2/4] Applying Signal Transformations...")
    
    # Create new metrics from existing telemetry
    # Voltage-to-Capacitance Ratio
    train['Voltage_Capacitance_Ratio'] = train['BP'] / train['Cholesterol']
    test['Voltage_Capacitance_Ratio'] = test['BP'] / test['Cholesterol']
    
    # Uptime to Max Frequency Delta
    train['Uptime_Frequency_Delta'] = train['Max HR'] - train['Age']
    test['Uptime_Frequency_Delta'] = test['Max HR'] - test['Age']

    # 3. TRAIN ALGORITHM (XGBoost)
    print("[3/4] Calibrating XGBoost Prediction Engine...")
    
    features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'Max HR', 'ST depression', 'Voltage_Capacitance_Ratio', 'Uptime_Frequency_Delta'
    ]
    target = 'Heart Disease'

    X = train[features].fillna(0)
    y = train[target]
    X_test = test[features].fillna(0)

    # Gradient boosting model configuration
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.0500,
        random_state=42,
        eval_metric='auc'
    )
    model.fit(X, y)
    print("   > Calibration Complete.")

    # 4. GENERATE OUTPUT
    print("[4/4] Generating Final Report...")
    predictions = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        'id': test['id'],
        'Heart Disease': predictions
    })

    submission.to_csv('submission.csv', index=False)
    print("SUCCESS: 'submission.csv' is ready for upload.")

if __name__ == "__main__":
    run_advanced_diagnostics()