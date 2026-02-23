# Throughput-Latency-Analyzer (Kaggle: Predicting Heart Disease)

## Project Overview
This repository documents my participation in the 2026 Kaggle Playground Series (Season 6, Episode 2): **Predicting Heart Disease**. 

The repository uses a hardware/network diagnostic camouflage to apply computer engineering and Lean Six Sigma principles to a data science competition. The primary goal is to predict the likelihood of target failures (heart disease) using a synthetically-generated dataset.

## Methodology & Feature Engineering
Continuous medical variables are mapped to hardware telemetry to evaluate signal stability:
* **Voltage_Input (Blood Pressure):** Mean 130.4974, Jitter 14.9758
* **Capacitance_Load (Cholesterol):** Mean 245.0118, Jitter 33.6816

To improve signal detection, new telemetry metrics were engineered:
* `Voltage_Capacitance_Ratio`: Ratio of power input to capacitor load.
* `Uptime_Frequency_Delta`: Difference between peak clock speeds and total uptime.

## System Architecture (V4.0 Ensemble)
To minimize prediction variance and lower the defect rate, the diagnostic tool utilizes a triple-redundant machine learning ensemble:
1. **Random Forest Engine:** Baseline decision tree logic.
2. **XGBoost Engine:** Gradient boosting for high-precision error correction.
3. **LightGBM Engine:** High-efficiency leaf-wise tree growth.

The final system failure probability is the averaged output of all three models.

## Setup and Execution
The script features dynamic pathing and will automatically detect if it is running locally or in a Kaggle cloud environment.

1. Clone this repository to the local machine.

2. Download `train.csv` and `test.csv` from Kaggle and place them in the root directory.

3. Run the diagnostic pipeline:

   ```
   python main.py
   ```

The script generates a submission.csv report formatted for Kaggle evaluation.


Step 3: Save the `README.md` file.

Step 4: Open your terminal, ensure you are in the project folder, and run these commands to sync the documentation with GitHub:
`git add README.md`
`git commit -m "Update README to document V4.0 multi-model ensemble and feature engineering"`
`git push origin main`