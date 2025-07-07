# Industrial Machinery Failure Prediction System

This project predicts machine failures using sensor data analysis with deep learning. It processes time-series sensor data, detects early warning signs, and visualizes predictions with lead time before failures occur.

## Key Features
- Time-series sensor data processing
- Custom 1D CNN + LSTM model with attention mechanism
- Early failure detection with warning buffer
- Optimal threshold tuning
- Detailed visualization of predictions
- Performance metrics (AUC, PR-AUC, F1-score)

## Requirements
- Python 3.8+
- numpy
- pandas
- torch
- scikit-learn
- matplotlib
- tqdm

## Setup
1. Clone repository:
 ```bash
 git clone https://github.com/Hassanpragmatics/Kickback_Event_Predictor.git
 cd Kickback_Event_Predictor
 cd code
 cd Hybrid 1D CNN + LSTM with Attention mechanism
  ```
## Prepare data:
- Place sensor data files in data/ directory
- Prepare annotations CSV (annotations/model_mid.csv)
- Directory Structure
Hybrid 1D CNN + LSTM with Attention mechanism/Used_7_files 
├── data/                   # Sensor data files (CSV)
├── annotations/            # Annotation files
│   └── model_mid.csv       # Main annotation file
├── lstm_model.py                # Main training script
└── best_models/model.pth # Trained model (generated)

or

Hybrid 1D CNN + LSTM with Attention mechanism/Using_half_dataset 
├── data/                   # Sensor data files (CSV)
├── annotations/            # Annotation files
│   └── model_mid.csv       # Main annotation file
├── lstm_model.py                # Main training script
└── best_models/model.pth # Trained model (generated)


## How to Run
Execute the main script:

 ```bash
python lstm_model.py
 ```

## Output includes:

- Training progress logs
- Model performance metrics

## Visualizations in root directory:

- training_history.png
- threshold_selection.png
- precision_recall_curve.png
- event_prediction.png

# Configuration

Key parameters in code:

 ```bash
annotations_path = 'annotations/model_mid.csv'  # Annotation file path
data_root = 'data'               # Sensor data directory
patch_size = 500                 # Input sequence length
stride = 100                     # Sliding window step
future_window = 100              # Prediction horizon
warning_buffer = 0.5             # Warning extension (seconds)
min_positive_samples = 1         # Min positive samples per file
 ```
