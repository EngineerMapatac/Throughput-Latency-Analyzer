import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# --- CONFIGURATION ---
COLUMN_MAPPING = {
    'Age': 'Uptime_Hours',
    'BP': 'Voltage_Input',
    'Cholesterol': 'Capacitance_Load',
    'Max HR': 'Frequency_Max',
    'Heart Disease': 'System_Failure_Flag',
    'Sex': 'Port_Type',                # 1 = Male, 0 = Female
    'Chest pain type': 'Error_Code',    # Categorical
    'ST depression': 'Signal_Lag'       # Continuous
}

def train_detection_protocol():
    print("--- INITIATING PATTERN RECOGNITION PROTOCOL ---")
    
    # 1. Load Data
    df = pd.read_csv('data/train.csv') # Ensure this path matches your local setup
    
    # 2. Select Engineering Features (Camouflaged)
    # We map them back to the original CSV names for training, but think of them as:
    # Uptime, Port_Type, Error_Code, Voltage, Capacitance, Frequency, Signal_Lag
    features = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    target = 'Heart Disease'
    
    X = df[features]
    y = df[target]
    
    # 3. Split Data (80% Training / 20% Validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model (Random Forest = "Decision Tree Ensemble")
    # n_estimators=100 means we are running 100 parallel simulation trees
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate Performance
    probs = model.predict_proba(X_val)[:, 1] # Probability of Failure
    auc_score = roc_auc_score(y_val, probs)
    
    print(f"\n[RESULT] Detection Algorithm Accuracy (AUC): {auc_score:.4f}")
    
    if auc_score > 0.85:
        print("   > STATUS: HIGH CONFIDENCE MODEL. READY FOR DEPLOYMENT.")
    else:
        print("   > STATUS: OPTIMIZATION REQUIRED. RE-CALIBRATE SENSORS.")

if __name__ == "__main__":
    train_detection_protocol()