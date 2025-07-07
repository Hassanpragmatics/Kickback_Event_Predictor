import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os

# ----------------------------
# Config
# ----------------------------
csv_path = 'data/Parameterset_1/_2024-12-11_13-19-10.csv'
backlog_start = 821694
backlog_end = 841696
patch_size = 1000
stride = 100
future_window = 2000

sensor_cols = [
    'TRIAX X Zeitsignal', 'Analog Input #14 P Backhoff Track',
    'TRIAX Y Zeitsignal', 'TRIAX Z Zeitsignal',
    'Leistung_230', 'U1', 'I1'
]

time_column = 'Time'

# ----------------------------
# Step 1: Read and clean CSV
# ----------------------------
df = pd.read_csv(csv_path, encoding="cp1252", sep=";", skiprows=31)

# Drop unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = [col.strip() for col in df.columns]
df = df.dropna(subset=["Time"])

# Replace commas with periods in string-type columns
df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).apply(lambda x: x.str.replace(',', '.'))

# Convert all except time to numeric
df[df.columns.difference([time_column])] = df[df.columns.difference([time_column])].apply(pd.to_numeric, errors='coerce')
df = df.fillna(0)

# Keep Time column clean
df[time_column] = df[time_column].str.replace(',', '.')
df = df.iloc[1:].reset_index(drop=True)
df[time_column] = pd.to_timedelta(df[time_column])

# ----------------------------
# Step 2: Normalize signals
# ----------------------------
df_normalized = df.copy()
for col in sensor_cols:
    if col in df_normalized.columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val - min_val != 0:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0

# ----------------------------
# Step 3: Add manual backlog label
# ----------------------------
# Step 3: Add pre-backlog warning labels
df_normalized['backlog'] = 0

warning_window = 1000  # Number of samples to predict ahead of time
warn_start = max(0, backlog_start - warning_window)
warn_end = backlog_start

df_normalized.loc[warn_start:warn_end, 'backlog'] = 1

signals = df_normalized[sensor_cols].values.astype(np.float32)
labels = df_normalized['backlog'].values

# ----------------------------
# Step 4: Create data patches
# ----------------------------
def create_patches(signals, labels, patch_size, stride, future_window):
    X, y = [], []
    for i in range(0, len(signals) - patch_size - future_window, stride):
        patch = signals[i:i+patch_size]
        future = labels[i+patch_size : i+patch_size+future_window]
        target = 1 if np.any(future == 1) else 0
        X.append(patch)
        y.append(target)
    return np.array(X), np.array(y)

X, y = create_patches(signals, labels, patch_size, stride, future_window)

# Balance the dataset
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
neg_idx = np.random.choice(neg_idx, size=len(pos_idx)*2, replace=False)
final_idx = np.concatenate([pos_idx, neg_idx])
np.random.shuffle(final_idx)
X = X[final_idx]
y = y[final_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# ----------------------------
# Step 5: Define and train model
# ----------------------------
model = models.Sequential([
    layers.Conv1D(32, 5, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32)

# ----------------------------
# Step 6: Predict on full signal
# ----------------------------
def sliding_predict(model, signals, patch_size, stride):
    preds = []
    for i in range(0, len(signals) - patch_size, stride):
        patch = signals[i:i+patch_size]
        patch = np.expand_dims(patch, axis=0)
        prob = model.predict(patch, verbose=0)[0][0]
        preds.append(prob)
    return preds

pred_probs = sliding_predict(model, signals, patch_size, stride)
# Find first time prediction exceeds the threshold before backlog
threshold = 0.7
predicted_warning_index = None
for i, prob in enumerate(pred_probs):
    global_index = i * stride  # Map back to original signal index
    if global_index >= backlog_start:
        break  # Stop once backlog has started
    if prob >= threshold:
        predicted_warning_index = global_index
        break

if predicted_warning_index is not None:
    time_warning = df_normalized.loc[predicted_warning_index, 'Time']
    time_backlog = df_normalized.loc[backlog_start, 'Time']
    time_diff = time_backlog - time_warning
    print(f"🚨 Model predicted backlog at index {predicted_warning_index}")
    print(f"🕒 Time before actual backlog: {time_diff}")
else:
    print("⚠️ Model did not raise a warning before backlog.")
# ----------------------------
# Step 7: Plot results
# ----------------------------
# Plot with prediction, actual backlog, and warning point
time_axis = range(0, len(signals) - patch_size, stride)

plt.figure(figsize=(15, 5))
plt.plot(labels, label='True Backlog', alpha=0.3)
plt.plot(time_axis, pred_probs, label='Predicted Risk (Prob)', color='red')

# Mark actual backlog start
plt.axvline(x=backlog_start, color='black', linestyle='--', label='Backlog Start')

# Mark predicted warning (if available)
if predicted_warning_index is not None:
    plt.axvline(x=predicted_warning_index, color='green', linestyle='--', label='Predicted Warning')
    plt.text(predicted_warning_index + 20, 0.8, f'Warning\n{time_diff}', color='green')

plt.title("Backlog Prediction with Warning Lead Time")
plt.xlabel("Time Index")
plt.ylabel("Label / Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()