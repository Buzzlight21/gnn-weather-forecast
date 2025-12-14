 # ============================================================
# 1️⃣ IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from math import radians, sin, cos, asin, sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from google.colab import files

# Upload file CSV
uploaded = files.upload()

# ============================================================
# 2️⃣ LOAD DATA
# ============================================================
file_path = list(uploaded.keys())[0]
df = pd.read_csv(file_path)

print("Data Loaded:", df.shape)
print(df.head())

# ============================================================
# 3️⃣ CLEANING & FIX DATETIME
# ============================================================
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['city_name', 'datetime']).reset_index(drop=True)

# ============================================================
# 4️⃣ PIVOT MENJADI MATRIX (tanggal × kota)
# ============================================================
temp_data = df.pivot_table(
    index='datetime',
    columns='city_name',
    values='temperature_2m_mean'
)

# normalisasi suhu
scaler = StandardScaler()
temp_scaled = scaler.fit_transform(temp_data.values)
temp_df = pd.DataFrame(temp_scaled, index=temp_data.index, columns=temp_data.columns)

# ============================================================
# 5️⃣ AMBIL KOORDINAT KOTA
# ============================================================
coords = df.groupby('city_name')[['latitude', 'longitude']].first()
coords_list = coords.reset_index()

city_names = list(coords_list['city_name'])
num_nodes = len(city_names)

print("Jumlah Kota:", num_nodes)
print(coords_list)

# ============================================================
# 6️⃣ FUNGSI HAVERSINE
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

# ============================================================
# 7️⃣ ADJACENCY MATRIX (RADIUS 200 KM)
# ============================================================
radius_km = 200
A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

for i in range(num_nodes):
    for j in range(num_nodes):
        d = haversine(
            coords_list.iloc[i]['latitude'], coords_list.iloc[i]['longitude'],
            coords_list.iloc[j]['latitude'], coords_list.iloc[j]['longitude']
        )
        if d <= radius_km:
            A[i, j] = 1.0

# normalisasi D^-1 A
deg = A.sum(axis=1)
deg_inv = np.zeros_like(deg)
deg_inv[deg > 0] = 1 / deg[deg > 0]
A_norm = (deg_inv.reshape(-1, 1) * A).astype(np.float32)

print("Adjacency Matrix:\n", A_norm)

# ============================================================
# 8️⃣ SLIDING WINDOW (LOOKBACK = 7)
# ============================================================
lookback = 7
X_list, Y_list = [], []
temp_values = temp_df.values

for i in range(len(temp_values) - lookback - 1):
    window = temp_values[i:i+lookback].T
    target = temp_values[i+lookback]
    X_list.append(window.astype(np.float32))
    Y_list.append(target.astype(np.float32))

X = np.array(X_list)
Y = np.array(Y_list)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ============================================================
# 9️⃣ TRAIN TEST SPLIT
# ============================================================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# ============================================================
# 🔟 DATASET
# ============================================================
class WeatherGraphDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_ds = WeatherGraphDataset(X_train, Y_train)
test_ds  = WeatherGraphDataset(X_test,  Y_test)

# ============================================================
# 1️⃣1️⃣ GCN LAYER
# ============================================================
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x, adj):
        agg = torch.matmul(adj, x)
        h = self.linear(agg)
        return torch.relu(h)

# ============================================================
# 1️⃣2️⃣ MODEL GNN
# ============================================================
class GNNForecast(nn.Module):
    def __init__(self, lookback, hidden=32):
        super().__init__()
        self.gcn1 = GCNLayer(lookback, hidden)
        self.gcn2 = GCNLayer(hidden, 1)

    def forward(self, x, adj):
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        return h.squeeze(-1)

# ============================================================
# 1️⃣3️⃣ TRAINING
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
adj_tensor = torch.tensor(A_norm, dtype=torch.float32).to(device)

model = GNNForecast(lookback=lookback, hidden=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch, adj_tensor)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss = {total_loss/len(train_loader):.4f}")

# ============================================================
# 1️⃣4️⃣ EVALUASI
# ============================================================
model.eval()
with torch.no_grad():
    X_t = torch.tensor(X_test).to(device)
    preds = model(X_t, adj_tensor).cpu().numpy()

mse = mean_squared_error(Y_test.flatten(), preds.flatten())
print("\n🔥 Final Test MSE:", mse)

# RMSE tiap kota
print("\nRMSE tiap kota:")
for i, city in enumerate(city_names):
    rmse = np.sqrt(mean_squared_error(Y_test[:, i], preds[:, i]))
    print(f"{city:12}: {rmse:.4f}")

# ============================================================
# 📊 1. Grafik Prediksi vs Aktual per Kota
# ============================================================
plt.figure(figsize=(14, 40))
for i, city in enumerate(city_names):
    plt.subplot(len(city_names), 1, i+1)
    plt.plot(Y_test[:, i], label="Actual")
    plt.plot(preds[:, i], label="Predicted", linestyle="--")
    plt.title(f"Prediksi vs Aktual — {city}")
    plt.xlabel("Hari pada Test Set")
    plt.ylabel("Suhu (Standardized)")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 📊 2. Heatmap Prediksi Semua Kota
# ============================================================
plt.figure(figsize=(14, 6))
sns.heatmap(
    preds.T,
    cmap="RdBu_r",        # lebih kontras
    center=0,
    yticklabels=city_names,
    cbar_kws={'label': 'Temperature (Z-Score)'},
    annot=False
)

plt.title("Heatmap Prediksi GNN — High/Low Temperature Pattern", fontsize=14)
plt.xlabel("Hari Test")
plt.ylabel("Kota")
plt.tight_layout()
plt.show()


# ============================================================
# 📊 3. Grafik Error (Pred - Actual)
# ============================================================
plt.figure(figsize=(12,6))
for i, city in enumerate(city_names):
    error = preds[:, i] - Y_test[:, i]
    plt.plot(error, label=city)

plt.title("Error Prediksi GNN per Kota")
plt.xlabel("Hari pada Test Set")
plt.ylabel("Error")
plt.grid(True)
plt.legend()
plt.show()
