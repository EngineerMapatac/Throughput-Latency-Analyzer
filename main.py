import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- CONFIGURATION (The Camouflage) ---
# We treat Heart Disease data as "Network Signals"
# Age -> Uptime
# BP -> Voltage
# Cholesterol -> Capacitance
# Heart Disease -> System Failure

def run_diagnostics():
    print("--- STARTING NETWORK DIAGNOSTICS ---")

    # 1. LOAD DATA
    # We assume files are in the exact same folder
    print("[1/4] Loading Signal Logs...")
    try:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
    except FileNotFoundError:
        print("CRITICAL ERROR: 'train.csv' or 'test.csv' not found.")
        return

    # 2. SIX SIGMA CHECK (Variance & Standard Deviation)
    # We analyze 'BP' (Voltage) to see if the signal is stable
    print("\n[2/4] Checking Signal Stability (Six Sigma Metrics)...")
    voltage_data = train['BP']
    
    mean_val = voltage_data.mean()
    std_dev = voltage_data.std()
    variance = voltage_data.var()
    
    print(f"   > Mean Voltage:   {mean_val:.2f}")
    print(f"   > Standard Dev:   {std_dev:.2f} (Signal Jitter)")
    print(f"   > Variance:       {variance:.2f} (Signal Energy)")
    
    if (std_dev / mean_val) < 0.2:
        print("   > STATUS: STABLE SIGNAL. Proceeding to Analysis.")
    else:
        print("   > STATUS: UNSTABLE SIGNAL. Proceed with caution.")

    # 3. TRAIN ALGORITHM (Random Forest)
    print("\n[3/4] Calibrating Prediction Algorithm...")
    
    # Select features
    features = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    target = 'Heart Disease'

    # Fill missing values with 0
    X = train[features].fillna(0)
    y = train[target]
    X_test = test[features].fillna(0)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    print("   > Calibration Complete.")

    # 4. GENERATE OUTPUT
    print("\n[4/4] Generating Final Report...")
    predictions = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        'id': test['id'],
        'Heart Disease': predictions
    })

    submission.to_csv('submission.csv', index=False)
    print("SUCCESS: 'submission.csv' is ready for upload.")

if __name__ == "__main__":
    run_diagnostics()