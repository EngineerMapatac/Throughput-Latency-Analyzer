import pandas as pd
import lightgbm as lgb

def run_lightgbm_diagnostics():
    print("--- STARTING NETWORK DIAGNOSTICS V3.0 (LIGHTGBM) ---")

    # 1. LOAD DATA
    print("[1/4] Loading Signal Logs...")
    train = pd.read_csv('/kaggle/input/playground-series-s6e2/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s6e2/test.csv')

    # Convert Target Variable to Numeric
    train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})

    # 2. FEATURE ENGINEERING
    print("[2/4] Applying Signal Transformations...")
    train['Voltage_Capacitance_Ratio'] = train['BP'] / train['Cholesterol']
    test['Voltage_Capacitance_Ratio'] = test['BP'] / test['Cholesterol']
    
    train['Uptime_Frequency_Delta'] = train['Max HR'] - train['Age']
    test['Uptime_Frequency_Delta'] = test['Max HR'] - test['Age']

    # 3. TRAIN ALGORITHM
    print("[3/4] Calibrating LightGBM Prediction Engine...")
    
    features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'Max HR', 'ST depression', 'Voltage_Capacitance_Ratio', 'Uptime_Frequency_Delta'
    ]
    target = 'Heart Disease'

    X = train[features].fillna(0.0000)
    y = train[target]
    X_test = test[features].fillna(0.0000)

    # LightGBM model configuration
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.0500,
        random_state=42
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
    run_lightgbm_diagnostics()