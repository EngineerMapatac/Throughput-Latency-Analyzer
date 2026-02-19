# Throughput-Latency-Analyzer

## Overview
This repository contains experimental scripts designed to analyze signal variance and predict critical system failures based on synthetic training data. The goal is to maximize the Area Under the Curve (AUC) for failure detection algorithms.

## Project Scope
- **Data Source:** Synthetic logs generated from a deep learning model (simulating legacy hardware outputs).
- **Objective:** Predict `System_Failure_Flag` (Target) based on variable inputs like `Voltage_Input` and `Frequency_Max`.
- **Tools:** Python, Pandas, Scikit-Learn.

## Key Metrics
- **Variance Analysis:** Monitoring deviations in `Capacitance_Load` to detect anomalies.
- **Signal Processing:** Filtering noise from `Uptime_Hours` vs `Frequency_Max` correlations.

## Setup
1. Clone the repository.
2. Place raw log files in `/data/raw`.
3. Run `preprocessing.py` to normalize signal inputs.