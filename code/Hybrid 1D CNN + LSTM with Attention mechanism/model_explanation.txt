The model used in the code is a hybrid 1D CNN + LSTM with Attention mechanism, specifically designed for time-series classification/prediction of events. Here's a detailed breakdown:

Model Architecture Components:
1D Convolutional Layers (CNN):

Conv1d(64, kernel_size=7) → BatchNorm → ReLU → MaxPool

Conv1d(128, kernel_size=5) → BatchNorm → ReLU → MaxPool

Conv1d(256, kernel_size=3) → BatchNorm → ReLU

Attention Mechanism:

python
self.attention = nn.Sequential(
    nn.Conv1d(256, 1, kernel_size=1),
    nn.Sigmoid()
)
Learns to weight important time steps

Bidirectional LSTM Layers:

LSTM(256→128, bidirectional=True)

LSTM(256→128, bidirectional=True)

With dropout (0.4) between LSTM layers

Fully Connected Classifier:

Linear(flattened_features → 128) → ReLU → Dropout

Linear(128 → 1) → Sigmoid (for binary classification)

Key Characteristics:
Input: 3D tensor (batch_size, num_features, sequence_length)

Output: Probability (0-1) of event occurrence

Designed for:

Local feature extraction (CNN)

Temporal dependencies (LSTM)

Important time-step highlighting (Attention)

Imbalanced data handling (pos_weight in BCEWithLogitsLoss)

Why This Architecture?

CNNs excel at extracting local patterns from sensor data

LSTMs capture long-range temporal dependencies

Attention helps focus on critical warning periods

Hybrid approach combines strengths of both CNN and LSTM


