import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 给定的数据点  数值稳定需要15秒时读取较准
data = {                                                   #
    '7.0': {'R': 25, 'G': 60, 'B': 112, 'C': 199, 'IR': 2},#7., 
    '6.5': {'R': 17, 'G': 39, 'B': 42, 'C': 97, 'IR': 1},#6., 
    '6.1': {'R': 19, 'G': 43,  'B': 43,  'C': 104, 'IR': 1},#6., 
    '5.3': {'R': 33, 'G': 66,  'B': 48,  'C': 144, 'IR': 1},#5., 
    '5.0': {'R': 70, 'G': 118, 'B': 62,  'C': 258, 'IR': 2},#5., 
    '4.4': {'R': 248,'G': 291, 'B': 121, 'C': 619, 'IR': 5},#4., 
    '4.7': {'R': 166,'G': 221, 'B': 119, 'C': 485, 'IR': 4},#4., 
    '4.0': {'R': 321, 'G': 567, 'B': 194, 'C': 1032, 'IR': 5},#3., 
    '7.001': {'R': 30, 'G': 55, 'B': 100, 'C': 180, 'IR': 2},#6., 
    '6.501': {'R': 17, 'G': 38, 'B': 41,  'C': 94,  'IR': 1},#5., 
    '6.101': {'R': 18, 'G': 42, 'B': 41,  'C': 100,  'IR': 1},#5., 
    '5.801': {'R': 23, 'G': 51, 'B': 44,  'C': 117,  'IR': 1},#5., 
    '5.301': {'R': 31, 'G': 63, 'B': 47,  'C': 137,  'IR': 1},#5., 
    '5.001': {'R': 64, 'G': 119, 'B': 70,  'C': 245,  'IR': 2},#5., 
    '4.701': {'R': 168, 'G': 222, 'B': 119,  'C': 484,  'IR': 4},#4., 
    '4.401': {'R': 248, 'G': 294, 'B': 124,  'C': 623,  'IR': 5},#4., 
    '4.001': {'R': 323, 'G': 566, 'B': 193,  'C': 1024,  'IR': 5},#4.


}

# 提取特征和标签
X = np.array([[v['R'], v['G'], v['B'], v['C'], v['IR']] for v in data.values()])
y = np.array([float(k) for k in data.keys()])

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_scaled, y)

# 输出标准化前的系数
scaled_coef = model.coef_
intercept = model.intercept_

# 计算标准化前的系数
coef = scaled_coef / scaler.scale_
adjusted_intercept = intercept - np.sum(scaled_coef * scaler.mean_ / scaler.scale_)

# 输出模型系数
print('Coefficients (in original scale):')
print('R:', coef[0])
print('G:', coef[1])
print('B:', coef[2])
print('C:', coef[3])
print('IR:', coef[4])
print('Adjusted Intercept:', adjusted_intercept)

# 构造线性回归公式
formula = f"pH = {adjusted_intercept:.8f}"
for i, feature in enumerate(['R', 'G', 'B', 'C', 'IR']):
    formula += f" + {coef[i]:.8f} * {feature}"

print('Linear Regression Formula:')
print(formula)

# 模型评估（简单展示）
predictions = model.predict(X_scaled)
print('Predictions:', predictions)



# 手动计算预测值
manual_predictions = []
for sample in X:
    prediction = adjusted_intercept + \
                 coef[0] * sample[0] + \
                 coef[1] * sample[1] + \
                 coef[2] * sample[2] + \
                 coef[3] * sample[3] + \
                 coef[4] * sample[4]
    manual_predictions.append(prediction)

# 输出手动计算的预测值
print('Manual Predictions:', manual_predictions)
 
