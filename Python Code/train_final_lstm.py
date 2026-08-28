import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import copy
import os

# Fix seeds for academic reproducibility
np.random.seed(42)
torch.manual_seed(42)
os.makedirs('Hydrological', exist_ok=True)

print("--- Step 1: Loading & Engineering Features ---")
df = pd.read_csv('Eleyele_HydroMet_Master.csv')

df['Date'] = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
df['DayOfYear'] = df['Date'].dt.dayofyear
df['Sin_Season'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
df['Cos_Season'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

for i in range(1, 8):
    df[f'Q_lag{i}'] = df['Q_peak_m3s'].shift(i)
df = df.dropna().reset_index(drop=True)

feature_cols = [
    'PRECTOTCORR', 'T2M_MAX', 'T2M_MIN', 'ALLSKY_SFC_SW_DWN',
    'Rain_3Day_Roll', 'Rain_7Day_Roll', 'Sin_Season', 'Cos_Season',
    'Q_lag1', 'Q_lag2', 'Q_lag3', 'Q_lag4', 'Q_lag5', 'Q_lag6', 'Q_lag7',
]

target_col = ['Q_peak_m3s']
LOOKBACK = 14
FORECAST = 1

def create_sequences(X, y, lookback, forecast):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback - forecast + 1):
        X_seq.append(X[i: i + lookback])
        y_seq.append(y[i + lookback: i + lookback + forecast])
    return np.array(X_seq), np.array(y_seq)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

scaler_x = StandardScaler().fit(df[feature_cols].iloc[:train_end])
X_scaled = scaler_x.transform(df[feature_cols])
y_real = df[target_col].values 

X_seq, y_seq = create_sequences(X_scaled, y_real, LOOKBACK, FORECAST)
seq_train_end = max(1, train_end - LOOKBACK)
seq_val_end = max(seq_train_end + 1, val_end - LOOKBACK)

X_train, y_train = X_seq[:seq_train_end], y_seq[:seq_train_end]
X_val, y_val = X_seq[seq_train_end:seq_val_end], y_seq[seq_train_end:seq_val_end]
X_test, y_test = X_seq[seq_val_end:], y_seq[seq_val_end:]
actual_real = y_test.flatten()

# =====================================================================
# --- Step 2: Model Architecture ---
# =====================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1]
        seq_len = encoder_outputs.shape[1]
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.shape[0], 1).unsqueeze(1)
        attention_scores = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention_scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(output_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, x, hidden, cell, encoder_outputs):
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(a, encoder_outputs)
        lstm_input = torch.cat((x, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        return prediction.unsqueeze(1), hidden, cell

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, source, target_len=1):
        batch_size = source.shape[0]
        encoder_outputs, hidden, cell = self.encoder(source)
        decoder_input = torch.zeros(batch_size, 1, 1).to(source.device)
        outputs = []
        for t in range(target_len):
            out, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs.append(out)
            decoder_input = out
        return torch.cat(outputs, dim=1)

hidden_size = 128
encoder = Encoder(input_dim=len(feature_cols), hidden_dim=hidden_size, num_layers=2)
decoder = Decoder(output_dim=1, hidden_dim=hidden_size, num_layers=2)
lstm_model = Seq2SeqAttention(encoder, decoder)

criterion = nn.SmoothL1Loss() 
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=128, shuffle=False)

# =====================================================================
# --- Step 3: Training ---
# =====================================================================
print("\n--- Step 3: Training Encoder-Decoder Attention-LSTM ---")
EPOCHS = 60
best_val_loss = float('inf')
patience = 12
patience_counter = 0
best_model_weights = None

for epoch in range(1, EPOCHS + 1):
    lstm_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = lstm_model(X_batch, target_len=FORECAST)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
        optimizer.step()

    lstm_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = lstm_model(X_batch, target_len=FORECAST)
            val_loss += criterion(output, y_batch).item()
    val_loss /= len(val_loader)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:02d}/{EPOCHS} | Val Loss: {val_loss:.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(lstm_model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch}.")
        break

lstm_model.load_state_dict(best_model_weights)
lstm_model.eval()

with torch.no_grad():
    lstm_preds = lstm_model(torch.tensor(X_test, dtype=torch.float32), target_len=FORECAST).squeeze(-1).numpy().flatten()
    lstm_preds = np.maximum(lstm_preds, 0) # No negative flows

# =====================================================================
# --- Step 4: Publication Visualizations (Hydrograph) ---
# =====================================================================
print("\n--- Step 4: Generating Visualizations ---")
plt.style.use('seaborn-v0_8-whitegrid')
days_to_plot = min(365, len(actual_real))

plt.figure(figsize=(12, 5))
plt.plot(actual_real[-days_to_plot:], label='Observed Peak Flow (m³/s)', color='black', linewidth=1.5)
plt.plot(lstm_preds[-days_to_plot:], label='LSTM Prediction (m³/s)', color='crimson', linestyle='--', linewidth=1.5)
plt.title(f'Temporal Flood Hydrograph (Final 1-Year Test Set)', fontsize=14, fontweight='bold')
plt.xlabel('Time (Days)', fontsize=12, fontweight='bold')
plt.ylabel('Peak Discharge (m³/s)', fontsize=12, fontweight='bold')
plt.legend(frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('Hydrological/Final_Hydrograph_LSTM.png', dpi=300)
plt.close()
print("Saved Hydrograph: Hydrological/Final_Hydrograph_LSTM.png")

# =====================================================================
# --- Step 5: SHAP Explainable AI ---
# =====================================================================
print("\n--- Step 5: Running SHAP Analysis (Explainable AI) ---")

# SHAP needs a simple wrapper to bypass the 'target_len' argument
class SHAPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x, target_len=1).squeeze(-1)

wrapped_model = SHAPWrapper(lstm_model)

# Use a small background dataset to prevent memory crash
background = torch.tensor(X_train[:150], dtype=torch.float32)
test_sample = torch.tensor(X_test[:150], dtype=torch.float32)

explainer = shap.GradientExplainer(wrapped_model, background)
shap_values = explainer.shap_values(test_sample)

# For time-series, SHAP returns (Samples, Sequence_Length, Features)
# We take the mean across the sequence length (axis=1) to see overall feature importance
shap_values_2d = np.mean(shap_values, axis=1)
X_test_2d_mean = np.mean(X_test[:150], axis=1)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_2d, features=X_test_2d_mean, feature_names=feature_cols, show=False)
plt.title('SHAP Feature Importance (Attention-LSTM)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Hydrological/SHAP_Summary_Plot.png', dpi=300, bb
ox_inches='tight')
plt.close()
print("Saved SHAP Explainability Plot: Hydrological/SHAP_Summary_Plot.png")

print("--- ML Phase Complete! ---")