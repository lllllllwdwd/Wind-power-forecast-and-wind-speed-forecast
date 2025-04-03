import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

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
X = all_data.drop(columns=["TARGETVAR", 'ZONEID'])
y = all_data["TARGETVAR"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建并训练SVR模型
model = SVR(kernel='rbf')  # 使用RBF核函数的支持向量回归
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')

# 计算相对误差
threshold = 1e-1
y_test_safe = np.where(np.abs(y_test) < threshold, threshold, y_test)  # 将小于阈值的值替换为阈值

# 计算相对误差
relative_error = np.abs((y_test - y_pred) / y_test_safe)  # 使用处理过的 y_test
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

# plt.savefig('output_images/prediction_comparison.png')

# 绘制训练损失曲线并保存
# SVR没有损失曲线，所以这里不再绘制损失曲线
