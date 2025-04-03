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
    # 读取CSV文件
    data = pd.read_csv(file_path)
    # 将当前文件的数据添加到总数据框中
    all_data = pd.concat([all_data, data], ignore_index=True)

print(all_data.head())  # 显示前几行数据

# 特征和标签分离
X = all_data.drop(columns=["V10", 'ZONEID'])
y = all_data["V10"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 假设每一天的数据有 24 个时间步，进行数据重塑
n_time_steps = 24  # 时间步数
n_features = X.shape[1]  # 每个时间步的特征数量
n_samples = X_scaled.shape[0] // n_time_steps  # 样本数量，按天计算

# 重塑X数据，将其转换为 (batch_size, time_steps, features)
X_reshaped = X_scaled.reshape(n_samples, n_time_steps, n_features)

# 将y数据重塑为 (n_samples, 1)，每个样本对应一天的目标值
y_reshaped = y.values[:n_samples]

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 打印张量的形状确认
print(X_train_tensor.shape)  # 输出形状 (batch_size, n_time_steps, n_features)
print(y_train_tensor.shape)  # 输出形状 (batch_size, 1)


# 定义RNN + CNN + Dropout模型
class RNN_CNN_Model(nn.Module):
    def __init__(self, input_dim):
        super(RNN_CNN_Model, self).__init__()

        # RNN层（LSTM）
        self.rnn = nn.LSTM(input_dim, 64, batch_first=True, dropout=0.2)

        # CNN层
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Dropout层
        self.dropout = nn.Dropout(p=0.3)

        # 全连接层
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # RNN层：处理时序数据
        x, _ = self.rnn(x)

        # CNN层：调整形状以适应Conv1d (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)  # 转换成 (batch_size, channels, seq_len)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)

        # 池化和Dropout
        x = torch.max(x, dim=2)[0]  # 最大池化，减少维度
        x = self.dropout(x)

        # 全连接层输出
        x = self.fc(x)
        return x

# 创建目录用于保存图像和模型
if not os.path.exists('output_images'):
    os.makedirs('output_images')

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# 模型实例化
n_features = 6  # 每个时间步的特征数量（U10, V10, U100, V100）
model = RNN_CNN_Model(input_dim=n_features)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 回归问题使用均方误差
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型并记录损失
num_epochs = 600
train_losses = []  # 用于记录每个epoch的损失
best_loss = float('inf')  # 初始化一个极大的损失值
best_model_path = 'saved_models/best_model.pth'  # 保存最佳模型的路径

for epoch in range(num_epochs):
    model.train()
    # 前向传播
    outputs = model(X_train_tensor)
    # 计算损失
    loss = criterion(outputs, y_train_tensor)
    train_losses.append(loss.item())  # 记录损失
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 如果当前损失小于最小损失，保存当前模型
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), best_model_path)  # 保存最佳模型
        print(f'Saving model with loss: {loss.item():.4f}')

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()

# 加载最佳模型
model.load_state_dict(torch.load(best_model_path, weights_only=True))

with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = y_pred.squeeze().numpy()

# 计算均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse:.4f}')
# 计算相对误差
threshold = 1e-0
y_test_safe = np.where(np.abs(y_test) < threshold, threshold, y_test)  # 将小于阈值的值替换为阈值

# 计算相对误差
relative_error = np.abs((y_test - y_pred) / y_test_safe)  # 使用处理过的 y_test
print(y_pred.shape)
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
plt.plot(y_pred, label='预测值')
plt.legend()
plt.title('风电场风速预测（以V10为例）：实际值与预测值对比')
plt.xlabel('测试编号（70小时预测）')
plt.ylabel('风速')
plt.savefig('output_images/风速70prediction_comparison.png')

# 绘制损失曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='训练损失')
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.title('训练损失曲线')
plt.legend()
# plt.savefig('output_images/train_loss_curve.png')


