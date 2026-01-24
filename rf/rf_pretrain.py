import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
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
rf_model = RandomForestRegressor(random_state=42)
rf_search_space = {
    'n_estimators': Integer(20, 300),
    'max_depth': Integer(3, 20),
    'min_samples_split': Integer(3, 10),
    'min_samples_leaf': Integer(3, 10),
    'max_features': Categorical(['sqrt', 0.3, 0.5, 0.7]),
}

rf_bayes_search = BayesSearchCV(
    estimator=rf_model,
    search_spaces=rf_search_space,
    scoring='neg_mean_squared_error',
    cv=5,
    n_iter=50,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# 训练模型
rf_bayes_search.fit(x_train, y_train)
print("\n随机森林最佳参数模型性能:")
print(rf_bayes_search.best_params_)
print(rf_bayes_search.best_score_)

# 获取最佳模型
rf_best_model = rf_bayes_search.best_estimator_

# 预测
rf_pred_train = rf_best_model.predict(x_train)
rf_pred_test = rf_best_model.predict(x_test)

# 计算性能指标
rf_mse_train = mean_squared_error(y_train, rf_pred_train)
rf_r2_train = r2_score(y_train, rf_pred_train)
rf_mse_test = mean_squared_error(y_test, rf_pred_test)
rf_r2_test = r2_score(y_test, rf_pred_test)

print("\n训练集性能:")
print(f"均方误差 (MSE): {rf_mse_train:.5f}")
print(f"R² 分数: {rf_r2_train:.5f}")

print("\n测试集性能:")
print(f"均方误差 (MSE): {rf_mse_test:.5f}")
print(f"R² 分数: {rf_r2_test:.5f}")

df = pd.DataFrame(x_test)
df['target'] = y_test
df['prediction'] = rf_pred_test
df.to_csv('F:/work_github/rf/mof_rf_pretrain.csv', index=False, encoding='utf-8')