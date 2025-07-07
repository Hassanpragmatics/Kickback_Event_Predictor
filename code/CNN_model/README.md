# Backlog Prediction with 1D CNN

This project demonstrates a machine learning approach to predict backlog events in industrial machinery using time-series sensor data. The solution uses a 1D Convolutional Neural Network (CNN) to analyze sensor readings and provide early warnings before backlog events occur.

## Features
- Time-series data preprocessing and normalization
- Patch-based feature engineering
- 1D CNN model for backlog prediction
- Early warning system with lead time calculation
- Visualization of prediction results

## Requirements
- Python 3.7+
- Required packages: pandas, numpy, matplotlib, scikit-learn, tensorflow

## Installation
```bash
git clone https://github.com/yourusername/backlog-prediction.git
cd backlog-prediction
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## Usage
- Place your CSV data file in the project directory
- Update the configuration in the script:
  - Set csv_path to your data file
  - Adjust backlog_start and backlog_end indices
- Modify sensor columns as needed

## Run the script:
```bash
python backlog_prediction.py
```
## Configuration
```bash
csv_path = 'path/to/your/data.csv'  # Input data file
backlog_start = 821694             # Start index of known backlog event
backlog_end = 841696               # End index of known backlog event
patch_size = 1000                  # Size of input segments
stride = 100                       # Step size for sliding window
future_window = 2000               # Future window for label determination
sensor_cols = [                    # List of sensor columns to use
    'TRIAX X Zeitsignal',
    'Analog Input #14 P Backhoff Track',
    # ... other sensors
]
warning_window = 1000              # Pre-backlog warning period
threshold = 0.7                    # Prediction probability threshold
```

## Output
The script will:

- Train a 1D CNN model on your data
- Predict backlog events across the entire signal
- Calculate warning lead time if detected
- Generate a visualization plot showing:
- True backlog labels
- Predicted probabilities
- Actual backlog start
- Predicted warning point (if any)
- Lead time information

## Example output:
```bash
🚨 Model predicted backlog at index 820000
🕒 Time before actual backlog: 0 days 00:05:00
```
