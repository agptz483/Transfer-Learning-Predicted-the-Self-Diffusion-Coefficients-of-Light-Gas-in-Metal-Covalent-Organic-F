import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import optuna
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from scipy.special import expit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        # self.fc6 = nn.Linear(hidden_size5, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        # self.bn4 = nn.BatchNorm1d(hidden_size4)
        # self.bn5 = nn.BatchNorm1d(hidden_size5)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        # x = self.fc5(x)
        # x = self.bn5(x)
        # x = self.relu(x)
        # x = self.fc6(x)
        return x


def train_kf(features, target, hidden_size1, hidden_size2, hidden_size3, learning_rate):
    input_size = features.shape[1]
    output_size = 1
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    best_model_state = None
    best_r2 = -float('inf')

    for fold, (train_index, val_index) in enumerate(kf.split(features)):
        X_train, X_val = features[train_index].to(device), features[val_index].to(device)
        y_train, y_val = target[train_index].to(device), target[val_index].to(device)

        model = DNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        epochs = 10000
        best_fold_r2 = -float('inf')
        best_fold_state = None

        for epoch in range(epochs):
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 在每个epoch后评估模型
            if epoch % 100 == 0:  # 每100个epoch评估一次
                model.eval()
                with torch.no_grad():
                    val_predictions = model(X_val)
                    current_r2 = r2_score(y_val.cpu().numpy(), val_predictions.cpu().numpy())
                    if current_r2 > best_fold_r2:
                        best_fold_r2 = current_r2
                        best_fold_state = model.state_dict().copy()

        # 使用最佳状态进行最终评估
        if best_fold_state is not None:
            model.load_state_dict(best_fold_state)

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            r2 = r2_score(y_val.cpu().numpy(), val_predictions.cpu().numpy())
            r2_scores.append(r2)
            if r2 > best_r2:
                best_r2 = r2
                best_model_state = model.state_dict().copy()

    # 创建最终的最佳模型
    best_model = DNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size).to(device)
    if best_model_state is not None:
        best_model.load_state_dict(best_model_state)

    return np.mean(r2_scores), best_model


def fine_tune(model, features, target, learning_rate, freeze_layers=True):
    model = model.to(device)
    features = features.to(device)
    target = target.to(device)

    if freeze_layers:
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = False
        for param in model.fc3.parameters():
            param.requires_grad = False
        # for param in model.fc4.parameters():
        #     param.requires_grad = False
        # for param in model.fc5.parameters():
        #     param.requires_grad = False

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Increase epochs to 1000 for fine-tuning
    epochs = 10000
    best_r2 = -float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        outputs = model(features)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每100个epoch评估一次
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(features)
                current_r2 = r2_score(target.cpu().numpy(), predictions.cpu().numpy())
                if current_r2 > best_r2:
                    best_r2 = current_r2
                    best_state = model.state_dict().copy()

    # 使用最佳状态
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        predictions = model(features)
        r2 = r2_score(target.cpu().numpy(), predictions.cpu().numpy())

    return r2, model


def objective(trial):
    hidden_size1 = trial.suggest_int('hidden_size1', 16, 32)
    hidden_size2 = trial.suggest_int('hidden_size2', 8, 16)
    hidden_size3 = trial.suggest_int('hidden_size3', 4, 8)
    # hidden_size4 = trial.suggest_int('hidden_size4', 1, 8)
    # hidden_size5 = trial.suggest_int('hidden_size5', 1, 8)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)

    r2, _ = train_kf(source_features_train_tensor, source_target_train_tensor, hidden_size1, hidden_size2, hidden_size3,
                     learning_rate)
    return r2


if __name__ == '__main__':

    source_data = pd.read_csv('F:/work_github/mof_data.csv')
    source_features = source_data.iloc[:55705, 0:11].values
    source_target = source_data.iloc[:55705, 12].values

    target_data = pd.read_csv('F:/work_github/cof_data.csv')
    target_features = target_data.iloc[:6200, 0:11].values
    target_target = target_data.iloc[:6200, 12].values

    scaler = MinMaxScaler()
    source_features = scaler.fit_transform(source_features)
    target_features = scaler.transform(target_features)

    source_features_train, source_features_test, source_target_train, source_target_test = train_test_split(
        source_features, source_target, test_size=0.2, random_state=42)
    target_features_train, target_features_test, target_target_train, target_target_test = train_test_split(
        target_features, target_target, train_size=2500, random_state=42)
    df = pd.DataFrame(target_features_train)
    df['target'] = target_target_train
    df.to_csv('F:/work_github/dnn/dnn_cof_train.csv', index=False, encoding='utf-8')

    source_features_train_tensor = torch.tensor(source_features_train, dtype=torch.float32)
    source_target_train_tensor = torch.tensor(source_target_train, dtype=torch.float32).view(-1, 1)
    source_features_test_tensor = torch.tensor(source_features_test, dtype=torch.float32)
    source_target_test_tensor = torch.tensor(source_target_test, dtype=torch.float32).view(-1, 1)

    target_features_train_tensor = torch.tensor(target_features_train, dtype=torch.float32)
    target_target_train_tensor = torch.tensor(target_target_train, dtype=torch.float32).view(-1, 1)
    target_features_test_tensor = torch.tensor(target_features_test, dtype=torch.float32)
    target_target_test_tensor = torch.tensor(target_target_test, dtype=torch.float32).view(-1, 1)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_trial = study.best_trial
    print('Best hyperparameters (source train): ', study.best_params)
    print('Best R2 score (source train): ', study.best_value)

    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    best_params = study.best_params
    source_r2, trained_model = train_kf(source_features_train_tensor, source_target_train_tensor,
                                        best_params['hidden_size1'], best_params['hidden_size2'],
                                        best_params['hidden_size3'], best_params['learning_rate'])

    # 在保存前先评估模型在测试集上的表现
    trained_model.eval()
    source_features_test_tensor = source_features_test_tensor.to(device)
    source_target_test_tensor = source_target_test_tensor.to(device)

    with torch.no_grad():
        test_predictions = trained_model(source_features_test_tensor)
        test_r2 = r2_score(source_target_test_tensor.cpu().numpy(), test_predictions.cpu().numpy())

    torch.save(trained_model, 'F:/work_github/dnn/trained_model.pth')
    loaded_model = torch.load('F:/work_github/dnn/trained_model.pth')
    loaded_model.to(device)
    loaded_model.eval()

    source_features_test_tensor = source_features_test_tensor.to(device)
    source_target_test_tensor = source_target_test_tensor.to(device)

    with torch.no_grad():
        predictions = loaded_model(source_features_test_tensor)
        loaded_r2 = r2_score(source_target_test_tensor.cpu().numpy(), predictions.cpu().numpy())

    predictions_np = predictions.cpu().numpy()
    print("pre_train_Predictions:", predictions_np)
    print("pre_train_r2", loaded_r2)

    fine_tune_r2, fine_tuned_model = fine_tune(trained_model, target_features_train_tensor, target_target_train_tensor,
                                               learning_rate=best_params['learning_rate'], freeze_layers=True)

    torch.save(fine_tuned_model, 'F:/work_github/dnn/fine-tuned_model.pth')
    loaded_model2 = torch.load('F:/work_github/dnn/fine-tuned_model.pth')
    loaded_model2.to(device)
    loaded_model2.eval()

    target_features_test_tensor = target_features_test_tensor.to(device)
    target_target_test_tensor = target_target_test_tensor.to(device)

    with torch.no_grad():
        predictions = loaded_model2(target_features_test_tensor)
        loaded_r2 = r2_score(target_target_test_tensor.cpu().numpy(), predictions.cpu().numpy())

    tl_predictions_np = predictions.cpu().numpy()
    print("tl_Predictions:", tl_predictions_np)
    print("tl_r2", loaded_r2)

    df1 = pd.DataFrame(source_target_test_tensor.cpu().numpy())
    df1['mof_pred'] = predictions_np
    df2 = pd.DataFrame(target_features_test_tensor.cpu().numpy())
    df2['cof_target'] = target_target_test_tensor.cpu().numpy()
    df2['cof_pred'] = tl_predictions_np
    df1.to_csv('F:/work_github/dnn/dnn_mof_output.csv', index=False, encoding='utf-8')
    df2.to_csv('F:/work_github/dnn/dnn_cof_output.csv', index=False, encoding='utf-8')