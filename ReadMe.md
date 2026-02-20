# Throughput-Latency-Analyzer

## Project Overview
This utility analyzes packet loss, signal variance, and hardware degradation in simulated network topologies. The script processes raw sensor logs to detect anomalies and predict impending system failures using a Random Forest ensemble.

## Core Diagnostics
The system monitors the following telemetry data:
* **Uptime_Hours:** Total continuous operation time of the node.
* **Voltage_Input:** Main power rail stability.
* **Capacitance_Load:** Capacitor charge cycles.
* **Frequency_Max:** Peak clock speeds.
* **System_Failure_Flag:** The target variable indicating a critical fault.

## Baseline Metrics
During initial testing, signal stability was verified. Both Voltage and Capacitance logs showed stable standard deviations (under 20% of the mean). Since the signal jitter was low, we bypassed heavy signal filtering and proceeded directly to algorithm calibration.

## Setup and Execution
1. Clone this repository to your local machine.
2. Place the raw sensor logs (`train.csv` and `test.csv`) in the root directory.
3. Run the diagnostic pipeline:

```
   python main.py
```