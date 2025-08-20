import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1) 读取与排序
df = pd.read_csv('009049_nav.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date').reset_index(drop=True)

# nav 数值化 & 过滤无效
df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
df = df[df['nav'] > 0].reset_index(drop=True)

# 2) 构造 lag / lead
window = 20
horizon = 7

for i in range(1, window+1):
    df[f'nav_lag{i}'] = df['nav'].shift(i)

for h in range(1, horizon + 1):
    df[f'nav_lead{h}'] = df['nav'].shift(-h)

feature_cols = [f'nav_lag{i}' for i in range(1, window + 1)]
target_cols  = [f'nav_lead{h}' for h in range(1, horizon + 1)]

# —— 训练数据：要求特征+目标都不缺 —— 
df_train = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

X = df_train[feature_cols].to_numpy(dtype=np.float32)
Y = df_train[target_cols].to_numpy(dtype=np.float32)

# 划分
n_train = int(len(X) * 0.8)
X_train, Y_train = X[:n_train], Y[:n_train]
X_test,  Y_test  = X[n_train:], Y[n_train:]

# 转 tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
Y_test_tensor  = torch.tensor(Y_test,  dtype=torch.float32)

# 3) 定义模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, output_dim=7):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=X.shape[1], output_dim=Y.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4) 训练
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, Y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_tensor)
            val_loss = criterion(val_pred, Y_test_tensor).item()
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

# 5) 预测未来 7 天净值
# —— 预测时只需要“最新一行 lag 特征”，不要求 lead 存在 —— 
df_feats_only = df.dropna(subset=feature_cols)  # 仅要求 lag 完整
last_feat = df_feats_only[feature_cols].iloc[-1].to_numpy(dtype=np.float32).reshape(1, -1)
last_row = torch.tensor(last_feat, dtype=torch.float32)

model.eval()
with torch.no_grad():
    pred_future_navs = model(last_row).numpy().flatten()  # ← 修正变量名
print("预测未来7天净值:", [f"{v:.3f}" for v in pred_future_navs])

