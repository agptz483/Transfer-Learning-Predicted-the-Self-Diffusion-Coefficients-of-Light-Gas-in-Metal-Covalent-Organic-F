import pandas as pd
import numpy as np
from xgboost import XGBRegressor as xgbr
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('F:/work_github/mof_data.csv')
x = data.iloc[:55705, 0:11].values
y = data.iloc[:55705, 12].values

# 划分训练和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 随机森林模型与贝叶斯优化
xgbr_model = xgbr()
xgbr_search_space = {
    'learning_rate':Real(0.001,0.1),
    'n_estimators':Integer(20,300),
    'max_depth':Integer(5,20),
}

xgbr_bayes_search = BayesSearchCV(
    estimator=xgbr_model,
    search_spaces=xgbr_search_space,
    scoring='neg_mean_squared_error',
    cv=5,
    n_iter=50,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# 训练模型
xgbr_bayes_search.fit(x_train, y_train)
print("\n随机森林最佳参数模型性能:")
print(xgbr_bayes_search.best_params_)
print(xgbr_bayes_search.best_score_)

# 获取最佳模型
xgbr_best_model = xgbr_bayes_search.best_estimator_

# 预测
xgbr_pred_train = xgbr_best_model.predict(x_train)
xgbr_pred_test = xgbr_best_model.predict(x_test)

# 计算性能指标
xgbr_mse_train = mean_squared_error(y_train, xgbr_pred_train)
xgbr_r2_train = r2_score(y_train, xgbr_pred_train)
xgbr_mse_test = mean_squared_error(y_test, xgbr_pred_test)
xgbr_r2_test = r2_score(y_test, xgbr_pred_test)

print("\n训练集性能:")
print(f"均方误差 (MSE): {xgbr_mse_train:.5f}")
print(f"R² 分数: {xgbr_r2_train:.5f}")

print("\n测试集性能:")
print(f"均方误差 (MSE): {xgbr_mse_test:.5f}")
print(f"R² 分数: {xgbr_r2_test:.5f}")

df = pd.DataFrame(x_test)
df['target'] = y_test
df['prediction'] = xgbr_pred_test
df.to_csv('F:/work_github/xgbr/mof_xgbr_pretrain.csv', index=False, encoding='utf-8')