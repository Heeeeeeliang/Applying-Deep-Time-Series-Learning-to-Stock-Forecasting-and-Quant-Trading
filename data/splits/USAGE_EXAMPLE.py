
# ============================================================================
# 如何使用划分后的数据集
# ============================================================================

import pandas as pd

# 选择股票和时间框架
SYMBOL = 'AAPL'
TIMEFRAME = '1hour'

# 加载三个数据集
train = pd.read_csv(f'data/splits/{SYMBOL}_{TIMEFRAME}/train.csv', 
                    index_col=0, parse_dates=True)
val = pd.read_csv(f'data/splits/{SYMBOL}_{TIMEFRAME}/val.csv', 
                  index_col=0, parse_dates=True)
test = pd.read_csv(f'data/splits/{SYMBOL}_{TIMEFRAME}/test.csv', 
                   index_col=0, parse_dates=True)

print(f"训练集: {len(train):,} 条记录")
print(f"验证集: {len(val):,} 条记录")
print(f"测试集: {len(test):,} 条记录")

# 准备特征和目标
def prepare_data(data):
    """准备X和y"""
    # 目标：预测下一时刻的收盘价
    y = data['close'].shift(-1)
    
    # 特征：所有列除了目标
    X = data.drop(['close'], axis=1)
    
    # 移除最后一行（y为NaN）
    X = X[:-1]
    y = y[:-1]
    
    return X, y

X_train, y_train = prepare_data(train)
X_val, y_val = prepare_data(val)
X_test, y_test = prepare_data(test)

# 标准化（用训练集的统计量）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train
X_val_scaled = scaler.transform(X_val)          # transform val
X_test_scaled = scaler.transform(X_test)        # transform test

# 现在可以训练模型了！
# model.fit(X_train_scaled, y_train)
# model.evaluate(X_val_scaled, y_val)
# model.test(X_test_scaled, y_test)
