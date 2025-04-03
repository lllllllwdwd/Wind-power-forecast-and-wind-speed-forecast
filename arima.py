import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import os
# 假设你有一个名为 "data.csv" 的文件，其中包含 "TARGETVAR" 列
file_path = r'C:\Users\月亮姐姐\Documents\WeChat Files\wxid_n80v76fcz9fn22\FileStorage\File\2025-02\Wind\Task 3\Task3_W_Zone1_10\Task3_W_Zone1.csv'  # 你需要将这里替换为实际的文件路径

# 加载CSV文件
df = pd.read_csv(file_path)

# 提取目标变量 "TARGETVAR" 列
power_series = df['TARGETVAR']

# 拆分数据集：80% 训练，20% 测试
train_size = int(len(power_series) * 0.8)
train, test = power_series[:train_size], power_series[train_size:]
# 训练 ARIMA 模型
model = ARIMA(train, order=(5, 1, 0))  # (p, d, q) 参数可以根据数据情况调整
model_fit = model.fit()

# 进行预测
forecast = model_fit.forecast(steps=len(test))

# 计算均方误差（MSE）
mse = mean_squared_error(test, forecast)
print(f'Mean Squared Error: {mse:.4f}')
threshold = 1e+1
y_test_safe = np.where(np.abs(test) < threshold, threshold, test)  # 将小于阈值的值替换为阈
# 计算相对误差
relative_error = np.abs((test - forecast) / y_test_safe) * 100  # 计算相对误差，单位为百分比
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

# 绘制实际值与预测值的对比图
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Predicted', color='red')
plt.legend()
plt.title('ARIMA: Actual vs Predicted Power')
plt.show()
