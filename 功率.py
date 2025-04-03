import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import rcParams
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# 设置中文字体
rcParams['font.family'] = 'SimHei'  # 设置为黑体
rcParams['axes.unicode_minus'] = False  # 解决负号问题

# 文件夹路径
folder_path = r'C:\Users\月亮姐姐\Documents\WeChat Files\wxid_n80v76fcz9fn22\FileStorage\File\2025-02\Wind\Task 3\Task3_W_Zone1_10'  # 请根据实际路径修改

# 获取文件夹中的所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 创建一个空的DataFrame用于存储所有数据
all_data = pd.DataFrame()

# 遍历文件夹中的每个文件并加载数据
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    print(f'正在加载文件: {file_path}')
    data = pd.read_csv(file_path)
    all_data = pd.concat([all_data, data], ignore_index=True)

print(all_data.head())  # 显示前几行数据

# 特征和标签分离
X = all_data.drop(columns=["TARGETVAR", 'ZONEID'])
y = all_data["TARGETVAR"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 假设每一天的数据有 24 个时间步，进行数据重塑
n_time_steps = 24
n_features = X.shape[1]
n_samples = X_scaled.shape[0] // n_time_steps
X_reshaped = X_scaled.reshape(n_samples, n_time_steps, n_features)
y_reshaped = y.values[:n_samples]

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 定义模型
class RNN_CNN_Model(nn.Module):
    def __init__(self, input_dim):
        super(RNN_CNN_Model, self).__init__()
        self.rnn = nn.LSTM(input_dim, 64, batch_first=True, dropout=0.2)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 模型训练
model = RNN_CNN_Model(input_dim=n_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 500
best_model_path = 'saved_models/best_model.pth'
# 训练模型
# for epoch in range(num_epochs):
#     model.train()
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')
#
# # 保存最佳模型
# torch.save(model.state_dict(), best_model_path)

# 误差校正模块：二次回归
model.load_state_dict(torch.load(best_model_path))
model.eval()

with torch.no_grad():
    train_pred = model(X_train_tensor).squeeze().numpy()
    test_pred = model(X_test_tensor).squeeze().numpy()

# 计算残差
train_residual = y_train_tensor.numpy().flatten() - train_pred

# 使用二次回归来拟合误差
poly_regressor = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_regressor.fit(train_pred.reshape(-1, 1), train_residual)

# 计算矫正后的误差
residual_correction_poly = poly_regressor.predict(test_pred.reshape(-1, 1))
corrected_test_pred_poly = test_pred + residual_correction_poly


# 计算均方误差（MSE）
mse = mean_squared_error(y_test, corrected_test_pred_poly)
print(f'Mean Squared Error on Test Data: {mse:.4f}')
# 计算相对误差
threshold = 1e-0
y_test_safe = np.where(np.abs(y_test) < threshold, threshold, y_test)  # 将小于阈值的值替换为阈值

# 计算相对误差
relative_error = np.abs((y_test - corrected_test_pred_poly) / y_test_safe)  # 使用处理过的 y_test
print(corrected_test_pred_poly.shape)
# 计算相对误差的统计量
mean_error = np.mean(relative_error)  # 均值
std_error = np.std(relative_error)    # 标准差
max_error = np.max(relative_error)    # 最大值
min_error = np.min(relative_error)    # 最小值

# 打印统计指标
print(f'Relative Error Mean: {mean_error:.4f}')
print(f'Relative Error Standard Deviation: {std_error:.4f}')
print(f'Relative Error Maximum: {max_error:.4f}')
print(f'Relative Error Minimum: {min_error:.4f}')
# 可选：绘制预测结果与实际结果的对比图并保存
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='实际值')
plt.plot(corrected_test_pred_poly, label='预测值')
plt.legend()
plt.title('风电场功率预测：实际值与预测值对比')
plt.xlabel('测试编号（70小时预测）')
plt.ylabel('风速')
plt.savefig('output_images/70prediction_comparison.png')

