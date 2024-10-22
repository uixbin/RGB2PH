## 1 VEML3328 数据格式

- R红色成分 G绿色成分 B绿色成分 C透明光 IR红外成分
 
## 2 拟合环境
 
- 安装 numpy
- 安装 scikit-learn
- 安装 scipy

## 3 拟合代码
```
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 给定的数据点
data = {
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
formula = f"pH = {adjusted_intercept:.2f}"
for i, feature in enumerate(['R', 'G', 'B', 'C', 'IR']):
    formula += f" + {coef[i]:.2f} * {feature}"

print('Linear Regression Formula:')
print(formula)

# 模型评估（简单展示）
predictions = model.predict(X_scaled)
print('Predictions:', predictions)

```

## 4 结果输出
```
Coefficients (in original scale):
R: 0.026754034286166854
G: 0.01649461371692296
B: 0.05395864170160024
C: -0.02621599344450697
IR: -0.6604382068759376
Adjusted Intercept: 5.773818938798145
Linear Regression Formula:
pH = 5.77381894 + 0.02675403 * R + 0.01649461 * G + 0.05395864 * B + -0.02621599 * C + -0.66043821 * IR
Predictions: [6.93785538 5.93480084 5.92473405 5.89982011 4.85379882 4.2078567
 4.92486698 3.82519014 6.83975266 5.94299556 5.87843209 5.87685782
 5.92638152 5.48224628 5.02108565 4.3143525  4.0179729 ]
Manual Predictions: [6.937855380338158, 5.9348008370970735, 5.92473404812715, 5.8998201143504385, 4.853798820491276, 4.207856702756229, 4.9248669761426145, 3.8251901431525868, 6.839752658210806, 5.942995562012072, 5.878432090498887, 5.876857821930211, 5.926381517037283, 5.48224627688259, 5.021085651876377, 4.314352495233774, 4.0179729038624545]
```

 
