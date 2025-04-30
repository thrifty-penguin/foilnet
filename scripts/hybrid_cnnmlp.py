# =========================
# 0. Imports and Setup
# =========================
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.stats
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import scienceplots
# Style
plt.style.use('science')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

start_time = time.time()

# =========================
# 1. Load and Prepare Data (correct way)
# =========================
# Load raw CSVs
input_csv = r'data\xfoil_results\inp_param.csv'
output_csv = r'data\xfoil_results\out_param.csv'

df_in = pd.read_csv(input_csv)
df_out = pd.read_csv(output_csv)

X_geometry = df_in[[col for col in df_in.columns if col.startswith('X') or col.startswith('Y')]].values
X_flow = df_in[['Ncrit', 'Reynolds', 'Mach', 'Alpha']].values
y = df_out[['CL', 'CD', 'CM', 'CDp', 'Top_xtr', 'Bot_xtr']].values

# Split first
n_total = len(X_geometry)
indices = np.arange(n_total)
np.random.seed(42)
np.random.shuffle(indices)

train_size = int(0.8 * n_total)
val_size = int(0.1 * n_total)
test_size = n_total - train_size - val_size

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

X_geo_train, X_flow_train, y_train = X_geometry[train_idx], X_flow[train_idx], y[train_idx]
X_geo_val, X_flow_val, y_val = X_geometry[val_idx], X_flow[val_idx], y[val_idx]
X_geo_test, X_flow_test, y_test = X_geometry[test_idx], X_flow[test_idx], y[test_idx]

# Fit scalers only on training
scaler_X_geo = StandardScaler().fit(X_geo_train)
scaler_X_flow = StandardScaler().fit(X_flow_train)
scaler_y = StandardScaler().fit(y_train)

X_geo_train_scaled = scaler_X_geo.transform(X_geo_train).reshape(-1, 1, 200)
X_geo_val_scaled = scaler_X_geo.transform(X_geo_val).reshape(-1, 1, 200)
X_geo_test_scaled = scaler_X_geo.transform(X_geo_test).reshape(-1, 1, 200)

X_flow_train_scaled = scaler_X_flow.transform(X_flow_train)
X_flow_val_scaled = scaler_X_flow.transform(X_flow_val)
X_flow_test_scaled = scaler_X_flow.transform(X_flow_test)

y_train_scaled = scaler_y.transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# Convert to torch tensors
X_geo_train_tensor = torch.tensor(X_geo_train_scaled, dtype=torch.float32)
X_flow_train_tensor = torch.tensor(X_flow_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_geo_val_tensor = torch.tensor(X_geo_val_scaled, dtype=torch.float32)
X_flow_val_tensor = torch.tensor(X_flow_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

X_geo_test_tensor = torch.tensor(X_geo_test_scaled, dtype=torch.float32)
X_flow_test_tensor = torch.tensor(X_flow_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create datasets
train_dataset = TensorDataset(X_geo_train_tensor, X_flow_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_geo_val_tensor, X_flow_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_geo_test_tensor, X_flow_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512)
test_loader = DataLoader(test_dataset, batch_size=512)

# =========================
# 2. Define Model
# =========================
class CNN_MLP_Surrogate(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 + 4, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 6)
        )
        
    def forward(self, x_geo, x_flow):
        x = self.cnn(x_geo)
        x = torch.cat([x, x_flow], dim=1)
        return self.fc(x)

model = CNN_MLP_Surrogate(dropout_p=0.1).to(device)

# =========================
# 3. Training
# =========================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = np.inf
patience = 10
counter = 0

for epoch in range(200):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0
    for X_geo_batch, X_flow_batch, y_batch in train_loader:
        X_geo_batch, X_flow_batch, y_batch = X_geo_batch.to(device), X_flow_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_geo_batch, X_flow_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_geo_batch.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_geo_batch, X_flow_batch, y_batch in val_loader:
            X_geo_batch, X_flow_batch, y_batch = X_geo_batch.to(device), X_flow_batch.to(device), y_batch.to(device)
            loss = criterion(model(X_geo_batch, X_flow_batch), y_batch)
            val_loss += loss.item() * X_geo_batch.size(0)
    val_loss /= len(val_loader.dataset)
    epoch_end_time = time.time()

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), r'models\surrogate\best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break

# Save scalers
joblib.dump(scaler_X_geo, r'models\surrogate\scaler_X_geo.pkl')
joblib.dump(scaler_X_flow, r'models\surrogate\scaler_X_flow.pkl')
joblib.dump(scaler_y, r'models\surrogate\scaler_y.pkl')

# =========================
# 4. Evaluation + Metrics + MC Dropout
# =========================
model.load_state_dict(torch.load(r'models\surrogate\best_model.pth'))
model.to(device)
model.train()

n_passes = 50
y_true, y_pred_mean, y_pred_std = [], [], []

with torch.no_grad():
    for X_geo_batch, X_flow_batch, y_batch in test_loader:
        X_geo_batch, X_flow_batch = X_geo_batch.to(device), X_flow_batch.to(device)
        outputs = [model(X_geo_batch, X_flow_batch).cpu().numpy() for _ in range(n_passes)]
        outputs = np.stack(outputs, axis=0)
        y_pred_mean.append(outputs.mean(axis=0))
        y_pred_std.append(outputs.std(axis=0))
        y_true.append(y_batch.numpy())

y_true = scaler_y.inverse_transform(np.vstack(y_true))
y_pred_mean = scaler_y.inverse_transform(np.vstack(y_pred_mean))
y_pred_std = np.vstack(y_pred_std) * scaler_y.scale_

# Save results CSV
results = np.hstack([y_true, y_pred_mean, y_pred_std])
columns = [f"True_{name}" for name in ['CL', 'CD', 'CM', 'CDp', 'Top_xtr', 'Bot_xtr']] + \
          [f"Pred_{name}" for name in ['CL', 'CD', 'CM', 'CDp', 'Top_xtr', 'Bot_xtr']] + \
          [f"Uncert_{name}" for name in ['CL', 'CD', 'CM', 'CDp', 'Top_xtr', 'Bot_xtr']]
df_results = pd.DataFrame(results, columns=columns)
os.makedirs(r'models\surrogate', exist_ok=True)
df_results.to_csv(r'models\surrogate\test_predictions_with_uncertainty.csv', index=False)

# Metrics
labels = ['CL', 'CD', 'CM', 'CDp', 'Top_xtr', 'Bot_xtr']

print("\n--- Metrics on Test Set ---")
for i, label in enumerate(labels):
    mae = mean_absolute_error(y_true[:, i], y_pred_mean[:, i])
    r2 = r2_score(y_true[:, i], y_pred_mean[:, i])
    print(f"{label}: MAE = {mae:.6f}, R2 = {r2:.4f}")

print("\n--- Prediction Interval Coverage Probability (PICP) ---")
lower_bound = y_pred_mean - 1.96 * y_pred_std
upper_bound = y_pred_mean + 1.96 * y_pred_std

for i, label in enumerate(labels):
    inside = np.logical_and(y_true[:, i] >= lower_bound[:, i], y_true[:, i] <= upper_bound[:, i])
    picp = np.mean(inside)
    print(f"{label}: PICP = {picp*100:.2f}%")

# =========================
# 5. Visualization
# =========================
save_dir = r'data\surrogate_plots'
os.makedirs(save_dir, exist_ok=True)
end_time= time.time()
print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
for i, label in enumerate(labels):
    sorted_idx = np.argsort(y_true[:, i])
    x_sorted = y_true[sorted_idx, i]
    y_sorted = y_pred_mean[sorted_idx, i]
    std_sorted = y_pred_std[sorted_idx, i]

    # Predicted vs True with 95% CI
    plt.figure(figsize=(8,6))
    plt.plot(x_sorted, y_sorted, 'b-', label='Prediction')
    plt.fill_between(x_sorted, y_sorted - 1.96*std_sorted, y_sorted + 1.96*std_sorted, color='blue', alpha=0.2, label='95 Percent CI')
    plt.plot([x_sorted.min(), x_sorted.max()], [x_sorted.min(), x_sorted.max()], 'r--', label='Ideal')
    plt.xlabel(f"True {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"Prediction vs True with 95 Percent CI ({label})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{label}_pred_vs_true_ci.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Histogram of uncertainties
    plt.figure(figsize=(8,6))
    plt.hist(y_pred_std[:, i], bins=30, alpha=0.7, color='blue')
    plt.xlabel(f"Predicted Std Dev ({label})")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Uncertainties ({label})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{label}_uncertainty_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Q-Q Plot
    errors = (y_pred_mean[:, i] - y_true[:, i]) / y_pred_std[:, i]
    plt.figure(figsize=(8,6))
    scipy.stats.probplot(errors, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot Normalized Residuals ({label})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{label}_qq_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

print("\n✅ All done! All plots saved to 'data/surrogate_plots'")
