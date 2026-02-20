# Throughput-Latency-Analyzer (Kaggle: Predicting Heart Disease)

## Project Overview
This repository documents my participation in the 2026 Kaggle Playground Series (Season 6, Episode 2): **Predicting Heart Disease**. 

While the repository name sounds like a network hardware tool, it is a thematic approach to apply computer engineering and Lean Six Sigma principles to a data science competition. The primary goal is to predict the likelihood of heart disease using a synthetically-generated dataset.

## Competition Context
* **Objective:** Predict the probability of the `Heart Disease` target variable for each ID in the test set.
* **Evaluation Metric:** Area Under the ROC Curve (AUC).
* **Dataset:** The dataset was generated synthetically from a deep learning model trained on a real-world Heart Disease prediction dataset to create a beginner-friendly sandbox.

## Methodology (The Engineering Approach)
To process the data logically, continuous medical variables were mapped to hardware telemetry data to evaluate signal stability before training the prediction algorithm:

**Voltage_Input (Blood Pressure):**
* Mean Signal: 130.4974
* Signal Jitter (Standard Deviation): 14.9758
* Signal Energy (Variance): 224.2746

**Capacitance_Load (Cholesterol):**
* Mean Signal: 245.0118
* Signal Jitter (Standard Deviation): 33.6816
* Signal Energy (Variance): 1134.4489

Both continuous inputs showed stable variance metrics. A baseline Random Forest ensemble was then calibrated to recognize anomalous patterns and predict target failures.

## Setup and Execution
1. Clone this repository to the local machine.
2. Download `train.csv` and `test.csv` from the Kaggle competition page and place them in the root directory.
3. Run the diagnostic pipeline:
   ```
   python main.py
   ```
