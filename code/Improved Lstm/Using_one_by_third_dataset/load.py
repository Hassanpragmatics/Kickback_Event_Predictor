import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, 
                           precision_recall_curve, auc, f1_score,
                           average_precision_score, balanced_accuracy_score,
                           cohen_kappa_score)
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import warnings
from sklearn.preprocessing import StandardScaler
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# Suppress warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Config - TUNED PARAMETERS
# ----------------------------
annotations_path = 'annotations/one_by_third_lift_up.csv'
data_root = 'data'
patch_size = 500       # Increased patch size for better context
stride = 100           # Balanced stride
future_window = 100    # Future window for prediction
test_size = 0.15       # Validation size
random_state = 42
batch_size = 64
epochs = 100           # Reduced with early stopping
min_positive_samples = 1  
warning_buffer = 0.5   # Seconds to extend warning period
min_consecutive_predictions = 3  # Required consecutive predictions

sensor_cols = [
    'TRIAX X Zeitsignal', 'Analog Input #14 P Backhoff Track',
    'TRIAX Y Zeitsignal', 'TRIAX Z Zeitsignal',
    'Leistung_230', 'U1', 'I1', 'Time'
]

# ----------------------------
# Enhanced Model Architecture
# ----------------------------
class TemporalAttentionPredictor(nn.Module):
    def __init__(self, input_features, seq_length):
        super().__init__()
        
        # CNN Feature Extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
        )
        
        # Temporal Attention
        self.attention = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(256, 128, num_layers=2, 
                           bidirectional=True, 
                           dropout=0.3,
                           batch_first=True)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, 1))
        
    def forward(self, x):
        # CNN Feature Extraction
        x = self.conv_layers(x)
        
        # Attention Mechanism
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        # LSTM Processing
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        
        # Global Max Pooling
        x = torch.max(x, dim=1)[0]
        
        # Classification
        return torch.sigmoid(self.classifier(x))

# ----------------------------
# Advanced Loss Function
# ----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

# ----------------------------
# Data Processing Functions
# ----------------------------
def parse_time(time_str):
    """Improved time parsing with error handling"""
    try:
        time_str = str(time_str).strip().strip(';').replace(',', '.')
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
            elif len(parts) == 2:  # MM:SS
                return float(parts[0])*60 + float(parts[1])
        return float(time_str)
    except:
        return 0.0

def detect_sampling_rate(time_seconds):
    """Robust sampling rate detection"""
    try:
        if len(time_seconds) > 1:
            time_diffs = np.diff(time_seconds)
            valid_diffs = time_diffs[time_diffs > 0]
            if len(valid_diffs) > 0:
                median_diff = np.median(valid_diffs)
                return 1.0 / median_diff
        return 100.0
    except:
        return 100.0

def create_patches(signals, labels, patch_size, stride, future_window):
    """Enhanced patch creation with better labeling"""
    X, y = [], []
    total_length = len(signals)
    
    for i in range(0, total_length - patch_size - future_window, stride):
        patch = signals[i:i+patch_size]
        future_labels = labels[i+patch_size:i+patch_size+future_window]
        
        # Enhanced target definition
        positive_ratio = np.mean(future_labels == 1)
        if positive_ratio > 0.25:  # At least 25% of future window is positive
            target = 1
        elif positive_ratio > 0:   # Any positive in future
            target = 0.5           # Soft label
        else:
            target = 0
            
        X.append(patch)
        y.append(target)
    
    return np.array(X), np.array(y)

# ----------------------------
# Data Loading and Preparation
# ----------------------------
print("Loading and preprocessing data...")
annotations = pd.read_csv(annotations_path)
print(f"Loaded {len(annotations)} annotations")

all_X, all_y = [], []
scaler = StandardScaler()

for idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
    file_path = os.path.join(data_root, row['file_path'])
    
    try:
        df = pd.read_csv(file_path, encoding="cp1252", sep=";", skiprows=31, dtype=str)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [col.strip() for col in df.columns]
        
        # Time processing
        df['Time_seconds'] = df['Time'].apply(parse_time)
        df['Time_seconds'] = df['Time_seconds'].cummax().fillna(0)
        sampling_rate = detect_sampling_rate(df['Time_seconds'].values)
        
        # Annotation conversion
        start_idx = np.argmin(np.abs(df['Time_seconds'] - float(row['start_time'])))
        end_idx = np.argmin(np.abs(df['Time_seconds'] - float(row['end_time'])))
        
        # Enhanced labeling
        labels = np.zeros(len(df), dtype=float)
        buffer_samples = int(warning_buffer * sampling_rate)
        warning_start = max(0, start_idx - buffer_samples)
        labels[warning_start:end_idx+1] = 1  # Warning period
        
        # Process signals
        signal_cols = [col for col in sensor_cols if col not in ['Time', 'Time_seconds']]
        signals = df[signal_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(r'[^\d.,-]', '', regex=True).str.replace(',', '.'), errors='coerce'))
        signals = signals.fillna(method='ffill').fillna(0).values.astype(np.float32)
        
        # Normalize
        if idx == 0:
            scaler.fit(signals)
        signals = scaler.transform(signals)
        
        # Create patches with padding
        pad_amount = patch_size
        padded_signals = np.pad(signals, ((pad_amount, 0), (0, 0)), mode='edge')
        padded_labels = np.pad(labels, (pad_amount, 0), mode='constant')
        
        X, y = create_patches(padded_signals, padded_labels, patch_size, stride, future_window)
        
        if np.sum(y > 0) >= min_positive_samples:
            all_X.append(X)
            all_y.append(y)
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        continue

# Combine and split data
if all_X:
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    print(f"\nFinal dataset: {len(X)} samples, {np.sum(y > 0.5)} positive ({np.sum(y > 0.5)/len(X)*100:.1f}%)")
    
    # Enhanced stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, 
        stratify=(y > 0.5).astype(int),  # Stratify on binary labels
        random_state=random_state
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train.transpose(0, 2, 1), dtype=torch.float32)
    X_test = torch.tensor(X_test.transpose(0, 2, 1), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
else:
    print("No valid data found")
    exit()

# ----------------------------
# Training Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = TemporalAttentionPredictor(len(signal_cols), patch_size).to(device)

# Enhanced optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Combined scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

# Loss function with class weighting
pos_weight = torch.tensor([len(y_train) / max(1, torch.sum(y_train > 0.5))]).to(device)
criterion = FocalLoss(alpha=0.75, gamma=2)  # Using our custom FocalLoss

# ----------------------------
# Enhanced Training Loop
# ----------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc="Training")
    
    for inputs, labels in progress:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        
        progress.set_postfix({"loss": loss.item()})
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Convert soft labels to binary for certain metrics
    binary_labels = (all_labels > 0.5).astype(int)
    
    metrics = {
        'loss': total_loss / len(loader.dataset),
        'auc': roc_auc_score(binary_labels, all_probs) if len(np.unique(binary_labels)) > 1 else 0.5,
        'pr_auc': average_precision_score(binary_labels, all_probs),
        'balanced_acc': balanced_accuracy_score(binary_labels, all_probs > 0.5),
        'kappa': cohen_kappa_score(binary_labels, all_probs > 0.5),
        'probs': all_probs  # Store for threshold optimization
    }
    
    return metrics

def find_optimal_threshold(y_true, y_probs):
    # Convert soft labels to binary for threshold finding
    binary_y = (y_true > 0.5).astype(int)
    
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (y_probs > thresh).astype(int)
        f1 = f1_score(binary_y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1
'''
print("\nStarting training...")
best_auc = 0
train_history, val_history = [], []

for epoch in range(epochs):
    # Train phase
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    train_history.append(train_loss)
    
    # Validation phase
    val_metrics = evaluate(model, test_loader, criterion, device)
    val_history.append(val_metrics)
    
    # Update scheduler
    scheduler.step()
    
    # Print metrics
    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_metrics['loss']:.4f}")
    print(f"  Val AUC: {val_metrics['auc']:.4f}")
    print(f"  Val PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"  Balanced Acc: {val_metrics['balanced_acc']:.4f}")
    print(f"  Cohen's Kappa: {val_metrics['kappa']:.4f}")
    
    # Save best model
    if val_metrics['auc'] > best_auc:
        best_auc = val_metrics['auc']
        torch.save(model.state_dict(), "best_model.pth")
        print("  ✅ Saved new best model")
    
    # Early stopping
    if epoch > 10 and np.mean([v['auc'] for v in val_history[-3:]]) < np.mean([v['auc'] for v in val_history[-6:-3]]):
        print("  ⚠️ Early stopping triggered")
        break
'''
# Load best model
model.load_state_dict(torch.load("best_model.pth"))
print("Loaded best model for evaluation")

# Final evaluation
final_metrics = evaluate(model, test_loader, criterion, device)
y_probs = final_metrics.pop('probs', None)  # Get probabilities if returned

print("\nFinal Evaluation:")
for metric, value in final_metrics.items():
    print(f"{metric:>15}: {value:.4f}")

# Threshold optimization
optimal_thresh, optimal_f1 = find_optimal_threshold(
    (y_test.cpu().numpy() > 0.5).astype(int),
    model(X_test.to(device)).cpu().detach().numpy()
)
print(f"\nOptimal Threshold: {optimal_thresh:.3f} (F1={optimal_f1:.3f})")
'''
# ----------------------------
# Visualization
# ----------------------------
def plot_training_history(train_history, val_history):
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_history, label='Train Loss')
    plt.plot([v['loss'] for v in val_history], label='Val Loss')
    plt.title('Loss History')
    plt.legend()
    
    # AUC
    plt.subplot(1, 3, 2)
    plt.plot([v['auc'] for v in val_history], label='Val AUC', color='green')
    plt.title('Validation AUC')
    plt.legend()
    
    # PR-AUC
    plt.subplot(1, 3, 3)
    plt.plot([v['pr_auc'] for v in val_history], label='Val PR-AUC', color='purple')
    plt.title('Validation PR-AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(train_history, val_history)
'''
# ----------------------------
# Prediction Visualization
# ----------------------------
def visualize_predictions(model, file_path, scaler, patch_size, stride, future_window, device):
    try:
        df = pd.read_csv(file_path, encoding="cp1252", sep=";", skiprows=31, dtype=str)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [col.strip() for col in df.columns]
        
        # Process time
        df['Time_seconds'] = df['Time'].apply(parse_time)
        df['Time_seconds'] = df['Time_seconds'].cummax().fillna(0)
        
        # Process signals
        signal_cols = [col for col in sensor_cols if col not in ['Time', 'Time_seconds']]
        signals = df[signal_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(r'[^\d.,-]', '', regex=True).str.replace(',', '.'), errors='coerce'))
        signals = signals.fillna(method='ffill').fillna(0).values.astype(np.float32)
        signals = scaler.transform(signals)
        
        # Pad signals
        pad_amount = patch_size
        padded_signals = np.pad(signals, ((pad_amount, 0), (0, 0)), mode='edge')
        padded_time = np.concatenate([
            np.arange(-pad_amount, 0) * (1.0/100),  # Approximate padded time
            df['Time_seconds'].values
        ])
        
        # Make predictions
        model.eval()
        probs, indices = [], []
        with torch.no_grad():
            for i in range(0, len(padded_signals) - patch_size - future_window, stride):
                patch = padded_signals[i:i+patch_size]
                patch_tensor = torch.tensor(patch.T[np.newaxis, :, :], dtype=torch.float32).to(device)
                prob = model(patch_tensor).item()
                probs.append(prob)
                indices.append(i + patch_size)  # End position of patch
        
        # Smooth predictions
        kernel_size = 5
        if len(probs) > kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            smoothed_probs = np.convolve(probs, kernel, mode='valid')
            smoothed_indices = indices[kernel_size//2:-(kernel_size//2)]
        else:
            smoothed_probs = probs
            smoothed_indices = indices
        
        # Plot
        plt.figure(figsize=(15, 8))
        
        # Plot first 3 signals
        for i in range(min(3, len(signal_cols))):
            plt.plot(df['Time_seconds'], signals[:, i], label=signal_cols[i], alpha=0.7)
        
        # Plot predictions
        plt.plot(np.array(smoothed_indices)/100, smoothed_probs, 
                'r-', linewidth=2, label='Predicted Risk')
        plt.axhline(optimal_thresh, color='k', linestyle='--', label='Threshold')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Value / Probability')
        plt.title(f'Predictions for {os.path.basename(file_path)}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('predictions_visualization.png')
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing predictions: {str(e)}")

# Visualize a random file
if all_X:
    test_file = random.choice(annotations['file_path'].apply(lambda x: os.path.join(data_root, x))).tolist()
    visualize_predictions(model, test_file, scaler, patch_size, stride, future_window, device)