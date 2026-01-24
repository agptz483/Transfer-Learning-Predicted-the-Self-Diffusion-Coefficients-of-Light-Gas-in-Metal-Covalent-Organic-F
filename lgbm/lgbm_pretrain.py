import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 加载数据
data = pd.read_csv('F:/work_github/mof_data.csv')
x = data.iloc[:55336, 0:11].values
y = data.iloc[:55336, 12].values

# 划分训练和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 随机森林模型与贝叶斯优化
lgbm_model = LGBMRegressor()
lgbm_search_space = {
    'n_estimators': Integer(20, 250),                  # 提升迭代次数
    'max_depth': Integer(5, 20),                       # 最大树深度
    'num_leaves': Integer(20, 150),                   # 单棵树的最大叶子数
    'min_child_samples': Integer(5, 20),              # 叶子节点最小样本数
    'learning_rate': Real(0.01, 0.3, 'log-uniform'),  # 学习率
    'subsample': Real(0.5, 1.0),                      # 训练样本采样比例
    'colsample_bytree': Real(0.3, 1.0),              # 特征采样比例
}

lgbm_bayes_search = BayesSearchCV(
    estimator=lgbm_model,
    search_spaces=lgbm_search_space,
    scoring='neg_mean_squared_error',
    cv=5,
    n_iter=50,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# 训练模型
lgbm_bayes_search.fit(x_train, y_train)
print("\n最佳参数模型性能:")
print(lgbm_bayes_search.best_params_)
print(lgbm_bayes_search.best_score_)

# 获取最佳模型
lgbm_best_model = lgbm_bayes_search.best_estimator_

# 预测
lgbm_pred_train = lgbm_best_model.predict(x_train)
lgbm_pred_test = lgbm_best_model.predict(x_test)
# 计算性能指标
lgbm_mse_train = mean_squared_error(y_train, lgbm_pred_train)
lgbm_r2_train = r2_score(y_train, lgbm_pred_train)
lgbm_mse_test = mean_squared_error(y_test, lgbm_pred_test)
lgbm_r2_test = r2_score(y_test, lgbm_pred_test)

print("\n训练集性能:")
print(f"均方误差 (MSE): {lgbm_mse_train:.5f}")
print(f"R² 分数: {lgbm_r2_train:.5f}")

print("\n测试集性能:")
print(f"均方误差 (MSE): {lgbm_mse_test:.5f}")
print(f"R² 分数: {lgbm_r2_test:.5f}")

df = pd.DataFrame(x_test)
df['target'] = y_test
df['prediction'] = lgbm_pred_test
df.to_csv('F:/work_github/lgbm/mof_lgbm_pretrain.csv', index=False, encoding='utf-8')