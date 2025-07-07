import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, f1_score
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import warnings
from sklearn.preprocessing import StandardScaler
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ----------------------------
# Config - TUNED PARAMETERS
# ----------------------------
annotations_path = 'annotations/half_lift_up.csv'
data_root = 'data'
patch_size = 500       # Increased patch size for better context
stride = 100           # Balanced stride
future_window = 100    # Reduced future window for near-event warnings
test_size = 0.15       # More data for training
random_state = 42
batch_size = 64
epochs = 200           # Reduced epochs with early stopping
min_positive_samples = 1  # Minimum positive samples per file
min_class_samples = 3     # Minimum samples per class for stratified split
event_duration = 100      # Duration of event in samples
warning_buffer = 0.5      # Seconds to extend warning period before annotation
min_consecutive_predictions = 3  # Required consecutive high-prob predictions

sensor_cols = [
    'TRIAX X Zeitsignal', 'Analog Input #14 P Backhoff Track',
    'TRIAX Y Zeitsignal', 'TRIAX Z Zeitsignal',
    'Leistung_230', 'U1', 'I1', 'Time'
]

# ----------------------------
# Create patches function
# ----------------------------
def create_patches(signals, labels, patch_size, stride, future_window):
    """Create patches with target based on future warning state"""
    X, y = [], []
    total_length = len(signals)
    
    for i in range(0, total_length - patch_size - future_window, stride):
        patch = signals[i:i+patch_size]
        patch_end = i + patch_size
        future_labels = labels[patch_end:patch_end+future_window]
        
        # Positive if any warning state (1) appears in future window
        target = 1 if np.any(future_labels == 1) else 0
        X.append(patch)
        y.append(target)
    
    positive_count = sum(y)
    print(f"  Created {len(X)} patches, {positive_count} warning-positive")
    return np.array(X), np.array(y)

# ----------------------------
# Model definition
# ----------------------------
class EventPredictor(nn.Module):
    def __init__(self, input_features, seq_length):
        super(EventPredictor, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # LSTM layers
        self.lstm1 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        
        # Calculate output size
        self._to_linear = self._get_output_size(seq_length, input_features)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def _get_output_size(self, seq_length, input_features):
        with torch.no_grad():
            x = torch.zeros(1, input_features, seq_length)
            x = self.pool1(self.relu(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu(self.bn2(self.conv2(x))))
            x = self.relu(self.bn3(self.conv3(x)))
            
            # Attention
            attn = self.attention(x)
            x = x * attn
            
            # Permute for LSTM
            x = x.permute(0, 2, 1)
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
            return x.shape[1] * x.shape[2]
    
    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Attention
        attn = self.attention(x)
        x = x * attn
        
        # LSTM layers
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)

# ----------------------------
# Utility functions
# ----------------------------
def parse_time(time_str):
    """Parse time string in various formats to seconds"""
    try:
        # Clean and normalize the string
        time_str = str(time_str).strip().strip(';').replace(',', '.')  # Replace commas with periods
        
        # Handle different formats
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
        else:  # Plain seconds format
            return float(time_str)
            
        return 0.0
    except Exception as e:
        print(f"Time parsing error: {e}, input: '{time_str}'")
        return 0.0

def detect_sampling_rate(time_seconds):
    """Detect sampling rate from time differences"""
    try:
        if len(time_seconds) > 1:
            # Calculate time differences in seconds
            time_diffs = np.diff(time_seconds)
            
            # Remove zero and negative differences
            valid_diffs = time_diffs[time_diffs > 0]
            
            if len(valid_diffs) > 0:
                median_diff = np.median(valid_diffs)
                if median_diff > 0:
                    return 1.0 / median_diff
                
        # Fallback if calculation fails
        print("⚠️ Using default sampling rate of 100 Hz")
        return 100.0
    except Exception as e:
        print(f"⚠️ Error detecting sampling rate: {e}")
        return 100.0

# ----------------------------
# Data loading and preprocessing
# ----------------------------
print("Loading and preprocessing data...")
annotations = pd.read_csv(annotations_path)
print(f"Loaded {len(annotations)} annotations")

all_X = []
all_y = []
file_stats = []

# Global scaler initialization
scaler = StandardScaler()

for idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
    file_path = os.path.join(data_root, row['file_path'])
    print(f"\nProcessing: {file_path}")
    
    try:
        df = pd.read_csv(
            file_path, 
            encoding="cp1252", 
            sep=";", 
            skiprows=31,
            dtype=str,
            on_bad_lines='warn'
        )

        # Clean up dataframe
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [col.strip() for col in df.columns]
        
        # Parse time column - FIXED
        if 'Time' in df.columns:
            # Convert time strings to total seconds
            df['Time_seconds'] = df['Time'].apply(parse_time)
            
            # Ensure time is non-decreasing and fill any gaps
            df['Time_seconds'] = df['Time_seconds'].cummax()
            if df['Time_seconds'].isna().any():
                df['Time_seconds'] = df['Time_seconds'].interpolate(method='linear').fillna(0)
            
            # Calculate actual sampling rate from time differences
            time_values = df['Time_seconds'].values
            sampling_rate = detect_sampling_rate(time_values)
            print(f"  Calculated sampling rate: {sampling_rate:.2f} Hz")
            
            # Print first and last time values for verification
            print(f"  First time: {time_values[0]:.6f}s, Last time: {time_values[-1]:.6f}s")
        else:
            print("⚠️ 'Time' column missing, using index as time")
            sampling_rate = 100.0
            df['Time_seconds'] = np.arange(len(df)) / sampling_rate
        
        # Convert annotation times to indices
        start_time = float(row['start_time'])
        end_time = float(row['end_time'])
        
        # Find the closest index for start_time
        start_idx = np.argmin(np.abs(df['Time_seconds'] - start_time))
        end_idx = np.argmin(np.abs(df['Time_seconds'] - end_time))
        
        # Get actual times at these indices
        actual_start_time = df['Time_seconds'].iloc[start_idx]
        actual_end_time = df['Time_seconds'].iloc[end_idx]
        
        print(f"  Annotation: start_time={start_time:.6f}s, end_time={end_time:.6f}s")
        print(f"  Converted: start_idx={start_idx}, end_idx={end_idx}, "
              f"actual times={actual_start_time:.6f}s to {actual_end_time:.6f}s")
        
        # Process numeric columns
        for col in sensor_cols:
            if col in df.columns and col != 'Time' and col != 'Time_seconds':
                # Clean and convert numeric values
                if df[col].dtype == object:
                    df[col] = df[col].str.replace(r'[^\d.,-]', '', regex=True)
                    df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NA values
        df = df.fillna(method='ffill').fillna(0)
        
        # Create 3-state labels with buffer zone
        labels = np.zeros(len(df), dtype=int)  # 0 = Normal
        
        # 1. Mark warning period with buffer
        buffer_samples = int(warning_buffer * sampling_rate)
        warning_start_idx = max(0, start_idx - buffer_samples)
        labels[warning_start_idx:end_idx+1] = 1  # Warning state
        
        # 2. Mark period after annotation as 2 (Event)
        event_start_idx = end_idx + 1
        event_end_idx = min(event_start_idx + event_duration, len(df) - 1)
        labels[event_start_idx:event_end_idx+1] = 2
        
        # Get actual times for warning and event periods
        warning_start_time = df['Time_seconds'].iloc[warning_start_idx]
        warning_end_time = df['Time_seconds'].iloc[end_idx]
        event_start_time = df['Time_seconds'].iloc[event_start_idx]
        event_end_time = df['Time_seconds'].iloc[event_end_idx]
        
        print(f"  Warning period: {warning_start_idx}-{end_idx} "
              f"({warning_start_time:.6f}s to {warning_end_time:.6f}s)")
        print(f"  Event period: {event_start_idx}-{event_end_idx} "
              f"({event_start_time:.6f}s to {event_end_time:.6f}s)")
        
        # Prepare signals
        signal_cols = [col for col in sensor_cols if col not in ['Time', 'Time_seconds']]
        signals = df[signal_cols].values.astype(np.float32)
        
        # Normalize per-sensor (global scaling)
        if idx == 0:
            scaler.fit(signals)
        signals = scaler.transform(signals)
        
        # Add padding at beginning to capture early events
        pad_amount = patch_size
        padded_signals = np.pad(signals, ((pad_amount, 0), (0, 0)), mode='edge')
        padded_labels = np.pad(labels, (pad_amount, 0), mode='constant')
        
        # Create patches focused on WARNING prediction
        X, y = create_patches(padded_signals, padded_labels, patch_size, stride, future_window)
        
        # Only add if we have enough positive samples
        positive_count = np.sum(y)
        if positive_count >= min_positive_samples:
            all_X.append(X)
            all_y.append(y)
            print(f"  ✅ Added {len(X)} patches ({positive_count} warning-positive)")
            file_stats.append({
                'file': file_path,
                'total_patches': len(X),
                'positive_patches': positive_count,
                'sampling_rate': sampling_rate,
                'original_start_idx': start_idx,
                'original_end_idx': end_idx,
                'warning_start_idx': warning_start_idx,
                'warning_end_idx': end_idx,
                'event_start_idx': event_start_idx,
                'event_end_idx': event_end_idx,
                'pad_amount': pad_amount
            })
        else:
            print(f"  ⚠️ Skipped: Only {positive_count} warning-positive samples (< {min_positive_samples} required)")
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Combine all data
if all_X:
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    total_positive = np.sum(y)
    print(f"\nCombined dataset: {len(X)} samples, {total_positive} warning-positive ({total_positive/len(X)*100:.1f}%)")
    
    # Print file statistics
    print("\nFile Contributions:")
    for stat in file_stats:
        print(f"  {stat['file']}: {stat['total_patches']} patches, {stat['positive_patches']} warning-positive")
    
    # Split into train/test
    if len(np.unique(y)) > 1 and np.min(np.bincount(y.astype(int))) >= min_class_samples:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        print("Using stratified split")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print("Using random split (insufficient samples for stratification)")
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train.transpose(0, 2, 1), dtype=torch.float32)
    X_test = torch.tensor(X_test.transpose(0, 2, 1), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    print(f"Train loader: {len(train_loader.dataset)} samples")
    print(f"Test loader: {len(test_loader.dataset)} samples")
else:
    print("No valid data found")
    exit()

# ----------------------------
# Model training
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

input_features = len(signal_cols)
model = EventPredictor(input_features, patch_size).to(device)

# Weighted loss to handle imbalance
pos_weight = torch.tensor([len(np.where(y == 0)[0]) / max(1, len(np.where(y == 1)[0]))]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
best_val_auc = 0
train_losses = []
val_losses = []
val_aucs = []
val_pr_aucs = []  # Precision-Recall AUC

print("\nTraining model...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for inputs, labels in progress:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        progress.set_postfix({"loss": loss.item()})
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    val_loss = val_loss / len(test_loader.dataset)
    val_losses.append(val_loss)
    
    # Calculate metrics
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    
    if len(np.unique(all_labels)) > 1:
        val_auc = roc_auc_score(all_labels, all_outputs)
        precision, recall, _ = precision_recall_curve(all_labels, all_outputs)
        val_pr_auc = auc(recall, precision)
    else:
        val_auc = 0.5
        val_pr_auc = 0.5
    
    val_aucs.append(val_auc)
    val_pr_aucs.append(val_pr_auc)
    
    # Update scheduler
    scheduler.step(val_loss)
    
    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "event_predictor_model.pth")
        print(f"  ✅ Saved new best model with AUC: {val_auc:.4f}, PR-AUC: {val_pr_auc:.4f}")
    
    # Early stopping
    if epoch > 10 and np.mean(val_aucs[-3:]) < np.mean(val_aucs[-6:-3]):
        print(f"  ⚠️ Early stopping at epoch {epoch+1}")
        break
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Val AUC: {val_auc:.4f}, Val PR-AUC: {val_pr_auc:.4f}")

# Load best model
try:
    model.load_state_dict(torch.load("event_predictor_model.pth"))
    print("✅ Model loaded successfully")
except:
    print("⚠️ Could not load saved model, using final model")

# ----------------------------
# Evaluation and Threshold Optimization
# ----------------------------
model.eval()
all_preds = []
all_probs = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        all_preds.append((outputs > 0.5).float().cpu().numpy())
        all_probs.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)

print("\nClassification Report (0.5 threshold):")
print(classification_report(all_labels, all_preds))

# Find optimal threshold using F1 score
if len(np.unique(all_labels)) > 1:
    print("Finding optimal threshold...")
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    for thresh in thresholds:
        preds = (all_probs > thresh).astype(int)
        f1 = f1_score(all_labels, preds)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f} (F1={f1_scores[optimal_idx]:.4f})")
    
    # Plot threshold vs F1
    plt.figure()
    plt.plot(thresholds, f1_scores, 'b-')
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold Selection')
    plt.legend()
    plt.savefig('threshold_selection.png')
    
    # Re-evaluate with optimal threshold
    optimal_preds = (all_probs > optimal_threshold).astype(int)
    print("\nClassification Report (Optimal Threshold):")
    print(classification_report(all_labels, optimal_preds))
    
    print(f"ROC AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    print(f"PR AUC: {auc(recall, precision):.4f}")
    
    # Plot precision-recall curve
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve.png')
else:
    print("AUC: Undefined (only one class present)")
    optimal_threshold = 0.5

# Plot training history
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(val_aucs, label='Val AUC', color='green')
plt.title('Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(val_pr_aucs, label='Val PR-AUC', color='purple')
plt.title('Validation PR-AUC')
plt.xlabel('Epoch')
plt.ylabel('PR-AUC')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# ----------------------------
# Single File Prediction and Visualization
# ----------------------------
def sliding_predict(model, signals, patch_size, stride, future_window):
    """Make predictions using sliding window approach"""
    preds = []
    end_indices = []  # Store end positions of patches
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(signals) - patch_size - future_window, stride):
            patch = signals[i:i+patch_size]
            patch_tensor = torch.tensor(patch.T[np.newaxis, :, :], dtype=torch.float32).to(device)
            prob = model(patch_tensor).item()
            preds.append(prob)
            end_indices.append(i + patch_size)  # End index of patch
    
    return preds, end_indices

if file_stats:
    # Select a random file that contributed to training
    valid_files = [stat['file'] for stat in file_stats]
    test_file = random.choice(valid_files)
    print(f"\nPredicting file: {test_file}")
    
    # Find corresponding annotation and event start
    test_stat = next(stat for stat in file_stats if stat['file'] == test_file)
    sampling_rate = test_stat['sampling_rate']
    pad_amount = test_stat['pad_amount']
    
    # Original annotation indices (without padding)
    original_start_idx = test_stat['original_start_idx']
    original_end_idx = test_stat['original_end_idx']
    
    # Warning period indices (with buffer, without padding)
    warning_start_idx = test_stat['warning_start_idx']
    warning_end_idx = test_stat['warning_end_idx']
    
    # Event period indices (without padding)
    event_start_idx = test_stat['event_start_idx']
    event_end_idx = test_stat['event_end_idx']
    
    # Find corresponding annotation row
    test_row = None
    for idx, row in annotations.iterrows():
        if os.path.join(data_root, row['file_path']) == test_file:
            test_row = row
            break
    
    if test_row is None:
        print("⚠️ No annotation found for test file")
    else:
        try:
            df = pd.read_csv(
                test_file,
                encoding="cp1252",
                sep=";",
                skiprows=31,
                dtype=str,
                on_bad_lines='warn'
            )
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = [col.strip() for col in df.columns]
            
            # Parse time column
            if 'Time' in df.columns:
                df['Time_seconds'] = df['Time'].apply(parse_time)
                # Ensure time is non-decreasing
                df['Time_seconds'] = df['Time_seconds'].cummax()
                if df['Time_seconds'].isna().any():
                    df['Time_seconds'] = df['Time_seconds'].interpolate(method='linear').fillna(0)
            else:
                print("⚠️ 'Time' column missing, using index as time")
                df['Time_seconds'] = np.arange(len(df)) / sampling_rate
            
            # Process numeric columns
            for col in sensor_cols:
                if col in df.columns and col != 'Time' and col != 'Time_seconds':
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(r'[^\d.,-]', '', regex=True)
                        df[col] = df[col].str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.fillna(method='ffill').fillna(0)
            
            # Prepare signals
            signal_cols = [col for col in sensor_cols if col not in ['Time', 'Time_seconds']]
            signals = df[signal_cols].values.astype(np.float32)
            
            # Apply normalization
            signals = scaler.transform(signals)
            
            # Get time values for actual periods
            original_start_time = df['Time_seconds'].iloc[original_start_idx]
            original_end_time = df['Time_seconds'].iloc[original_end_idx]
            
            warning_start_time = df['Time_seconds'].iloc[warning_start_idx]
            warning_end_time = df['Time_seconds'].iloc[warning_end_idx]
            
            event_start_time = df['Time_seconds'].iloc[event_start_idx]
            event_end_time = df['Time_seconds'].iloc[event_end_idx]
            
            # Add padding for prediction
            signals_padded = np.pad(signals, ((pad_amount, 0), (0, 0)), mode='edge')
            
            # Create time vector for padded signal
            time_step = 1.0 / sampling_rate
            padded_time_seconds = np.concatenate([
                np.arange(-pad_amount * time_step, 0, time_step),
                df['Time_seconds'].values
            ])
            
            # Make predictions
            pred_probs, end_indices = sliding_predict(model, signals_padded, patch_size, stride, future_window)
            prediction_times = padded_time_seconds[end_indices]
            
            # Apply smoothing to predictions
            kernel_size = 5
            smoothing_kernel = np.ones(kernel_size) / kernel_size
            smoothed_probs = np.convolve(pred_probs, smoothing_kernel, mode='valid')
            smoothed_prediction_times = prediction_times[kernel_size-1:]
            
            # Detect warning using sustained high probabilities
            warning_sequence = 0
            predicted_warning_time = None
            for j, prob in enumerate(smoothed_probs):
                time_position = smoothed_prediction_times[j]
                
                # Only consider warnings before actual warning period
                if time_position < warning_start_time and prob >= optimal_threshold:
                    warning_sequence += 1
                    if warning_sequence >= min_consecutive_predictions:
                        # Use start of sequence for warning detection
                        predicted_warning_time = smoothed_prediction_times[j - min_consecutive_predictions + 1]
                        break
                else:
                    warning_sequence = 0  # Reset counter
            
            # Plot results
            plt.figure(figsize=(15, 10))
            
            # 1. Plot sensor data with actual periods
            plt.subplot(2, 1, 1)
            for i in range(min(3, len(signal_cols))):  # First 3 sensors
                plt.plot(df['Time_seconds'], signals[:, i], label=signal_cols[i])
            plt.ylabel('Sensor Value')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Highlight actual periods
            plt.axvspan(original_start_time, original_end_time, 
                        color='purple', alpha=0.2, label='Original Annotation')
            plt.axvspan(warning_start_time, warning_end_time, 
                        color='orange', alpha=0.3, label='Warning Period')
            plt.axvspan(event_start_time, event_end_time, 
                        color='red', alpha=0.3, label='Event Period')
            
            if predicted_warning_time is not None:
                plt.axvline(x=predicted_warning_time, color='green', 
                            linestyle='--', linewidth=2, label='Predicted Warning')
            
            plt.legend(loc='upper right')
            plt.title("Sensor Data with Annotations")
            plt.xlabel("Time (seconds)")
            
            # 2. Plot predictions
            plt.subplot(2, 1, 2)
            plt.plot(prediction_times, pred_probs, 'b-', alpha=0.3, label='Raw Risk')
            plt.plot(smoothed_prediction_times, smoothed_probs, 'b-', label='Smoothed Risk')
            plt.axhline(optimal_threshold, color='r', linestyle='--', label='Threshold')
            plt.ylabel('Probability')
            plt.ylim(-0.05, 1.05)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Draw vertical lines for actual periods
            plt.axvline(x=original_start_time, color='purple', linestyle='-', alpha=0.5, label='Annotation Start')
            plt.axvline(x=warning_start_time, color='orange', linestyle='-', alpha=0.5, label='Warning Start')
            plt.axvline(x=event_start_time, color='red', linestyle='-', alpha=0.5, label='Event Start')
            
            if predicted_warning_time is not None:
                plt.axvline(x=predicted_warning_time, color='g', linestyle='--', label='Predicted Warning')
                lead_time = warning_start_time - predicted_warning_time
                plt.text(predicted_warning_time + 0.1, 0.8, 
                         f'Warning\n{lead_time:.2f}s before', color='green', fontsize=9)
            
            plt.legend(loc='upper right')
            plt.xlabel("Time (seconds)")
            plt.title("Event Prediction Probability")
            
            plt.suptitle(f"Event Prediction: {os.path.basename(test_file)}")
            plt.tight_layout()
            plt.savefig('event_prediction.png', dpi=300)
            plt.show()
            
            # Print timing information
            print("\nEvent Timing Summary:")
            print(f"Original annotation: {original_start_time:.6f}s to {original_end_time:.6f}s")
            print(f"Warning period: {warning_start_time:.6f}s to {warning_end_time:.6f}s")
            print(f"Event period: {event_start_time:.6f}s to {event_end_time:.6f}s")
            if predicted_warning_time is not None:
                print(f"Predicted warning at: {predicted_warning_time:.6f}s")
                print(f"Lead time before warning: {lead_time:.6f}s")
            else:
                print("No warning prediction before event")

        except Exception as e:
            print(f"Error processing test file: {str(e)}")
            import traceback
            traceback.print_exc()
else:
    print("No valid files for prediction")