#!/usr/bin/env python3
"""
Run all Chapter 2 direct prediction models with standardised evaluation.
Local execution — no Colab dependency.
All models use 7 tickers, SEQ_LEN=30, 1-hour data, same DA counting.
"""
import warnings, os, sys, json, time, random
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import pywt
from vmdpy import VMD

# ── Config ──
SPLITS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '毕设-20260311T013659Z-3-007', '毕设', 'data', 'splits'))
FREQ = '1hour'
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ']
SEQ_LEN = 30
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100
PATIENCE = 10
SEED = 42

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

LGB_PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
    'n_estimators': 500, 'early_stopping_rounds': 30
}

results_dict = {}

# ── Utilities ──
def load_splits(ticker, freq=FREQ):
    base = os.path.join(SPLITS_DIR, f"{ticker}_{freq}")
    train = pd.read_csv(os.path.join(base, 'train.csv'), parse_dates=['ts_event'])
    val   = pd.read_csv(os.path.join(base, 'val.csv'),   parse_dates=['ts_event'])
    test  = pd.read_csv(os.path.join(base, 'test.csv'),  parse_dates=['ts_event'])
    return train, val, test

def get_feature_cols(df, exclude=['ts_event','close','target','ticker']):
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64','float32','int64']]

def prepare_regression_targets(train, val, test, target_col='close'):
    for df in [train, val, test]:
        if 'target' not in df.columns:
            df['target'] = df[target_col].shift(-1)
    train = train.dropna(subset=['target'])
    val   = val.dropna(subset=['target'])
    test  = test.dropna(subset=['target'])
    return train, val, test

def prepare_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def load_close_series(ticker, freq=FREQ):
    train, val, test = load_splits(ticker, freq)
    return train['close'], val['close'], test['close']

def directional_accuracy(y_true, y_pred):
    if len(y_true) < 2:
        return 0.5
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    mask = (true_dir != 0) & (pred_dir != 0)
    if mask.sum() == 0:
        return 0.5
    return (true_dir[mask] == pred_dir[mask]).mean()

def da_z_test(da, n, null=0.5):
    se = np.sqrt(null * (1 - null) / n)
    z = (da - null) / se
    p = 1 - stats.norm.cdf(z)
    return z, p

def evaluate_model(ticker, y_true, y_pred, model_name, task='regression'):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    da = directional_accuracy(y_true, y_pred)
    n = len(y_true) - 1
    z, p = da_z_test(da, n)
    if task == 'regression':
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
    else:
        rmse, r2 = None, None
    result = {'ticker': ticker, 'DA': da, 'N': n, 'z': z, 'p': p, 'RMSE': rmse, 'R2': r2}
    if model_name not in results_dict:
        results_dict[model_name] = []
    results_dict[model_name].append(result)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {ticker}: DA={da:.4f} (p={p:.4f}{sig}), N={n}"
          + (f", RMSE={rmse:.4f}, R2={r2:.4f}" if rmse is not None else ""))
    return result

def summarize_model(model_name):
    if model_name not in results_dict:
        print(f"No results for {model_name}"); return
    rows = results_dict[model_name]
    das = [r['DA'] for r in rows]
    ns  = [r['N']  for r in rows]
    total_n = sum(ns)
    avg_da  = np.mean(das)
    z, p = da_z_test(avg_da, total_n)
    print(f"\n{'='*60}")
    print(f"{model_name} -- Aggregate over {len(rows)} tickers")
    print(f"  Mean DA: {avg_da:.4f}  (pooled z={z:.3f}, p={p:.4f})")
    rmses = [r['RMSE'] for r in rows if r['RMSE'] is not None]
    if rmses: print(f"  Mean RMSE: {np.mean(rmses):.4f}")
    r2s = [r['R2'] for r in rows if r['R2'] is not None]
    if r2s: print(f"  Mean R2: {np.mean(r2s):.4f}")
    print(f"  Total N: {total_n}")
    print(f"{'='*60}")

def train_pytorch_model(model, train_X, train_y, val_X, val_y,
                        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                        patience=PATIENCE, task='regression'):
    model = model.to(device)
    criterion = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
    tX = torch.FloatTensor(train_X).to(device)
    ty = torch.FloatTensor(train_y).unsqueeze(-1).to(device) if task == 'regression' else torch.LongTensor(train_y).to(device)
    vX = torch.FloatTensor(val_X).to(device)
    vy = torch.FloatTensor(val_y).unsqueeze(-1).to(device) if task == 'regression' else torch.LongTensor(val_y).to(device)
    train_dl = DataLoader(TensorDataset(tX, ty), batch_size=batch_size, shuffle=True)
    best_loss, best_state, wait = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(vX), vy).item()
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    return model

# ════════════════════════════════════════════════════════════
# MODEL 1: LSTM Regression
# ════════════════════════════════════════════════════════════
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def run_lstm_regression():
    print("\n" + "="*60)
    print("MODEL 1: LSTM Regression")
    print("="*60)
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        train, val, test = load_splits(ticker)
        train, val, test = prepare_regression_targets(train, val, test)
        feat_cols = get_feature_cols(train)
        print(f"  Features: {len(feat_cols)}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        scaler = StandardScaler()
        trX = scaler.fit_transform(train[feat_cols].values)
        vaX = scaler.transform(val[feat_cols].values)
        teX = scaler.transform(test[feat_cols].values)
        trX_s, trY = prepare_sequences(trX, train['target'].values)
        vaX_s, vaY = prepare_sequences(vaX, val['target'].values)
        teX_s, teY = prepare_sequences(teX, test['target'].values)
        model = LSTMRegressor(input_dim=len(feat_cols))
        model = train_pytorch_model(model, trX_s, trY, vaX_s, vaY)
        with torch.no_grad():
            preds = model(torch.FloatTensor(teX_s).to(device)).cpu().numpy().flatten()
        evaluate_model(ticker, teY, preds, 'LSTM Regression')
    summarize_model('LSTM Regression')

# ════════════════════════════════════════════════════════════
# MODEL 2: Attention-LSTM
# ════════════════════════════════════════════════════════════
class AttentionLSTMReg(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.attention = nn.Sequential(nn.Linear(hidden, hidden//2), nn.Tanh(), nn.Linear(hidden//2, 1))
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, 1))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_w).sum(dim=1)
        return self.fc(context)

def run_attention_lstm():
    print("\n" + "="*60)
    print("MODEL 2: Attention-LSTM (Regression)")
    print("="*60)
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        train, val, test = load_splits(ticker)
        train, val, test = prepare_regression_targets(train, val, test)
        feat_cols = get_feature_cols(train)
        print(f"  Features: {len(feat_cols)}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        scaler = StandardScaler()
        trX = scaler.fit_transform(train[feat_cols].values)
        vaX = scaler.transform(val[feat_cols].values)
        teX = scaler.transform(test[feat_cols].values)
        trX_s, trY = prepare_sequences(trX, train['target'].values)
        vaX_s, vaY = prepare_sequences(vaX, val['target'].values)
        teX_s, teY = prepare_sequences(teX, test['target'].values)
        model = AttentionLSTMReg(input_dim=len(feat_cols))
        model = train_pytorch_model(model, trX_s, trY, vaX_s, vaY)
        with torch.no_grad():
            preds = model(torch.FloatTensor(teX_s).to(device)).cpu().numpy().flatten()
        evaluate_model(ticker, teY, preds, 'Attention-LSTM')
    summarize_model('Attention-LSTM')

# ════════════════════════════════════════════════════════════
# MODEL 3: Transformer+LSTM
# ════════════════════════════════════════════════════════════
class TransformerLSTMReg(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.3, seq_len=SEQ_LEN):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.fc = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))
    def forward(self, x):
        x = self.proj(x) + self.pos[:, :x.size(1), :]
        x = self.encoder(x)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

def run_transformer_lstm():
    print("\n" + "="*60)
    print("MODEL 3: Transformer+LSTM (Regression)")
    print("="*60)
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        train, val, test = load_splits(ticker)
        train, val, test = prepare_regression_targets(train, val, test)
        feat_cols = get_feature_cols(train)
        print(f"  Features: {len(feat_cols)}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        scaler = StandardScaler()
        trX = scaler.fit_transform(train[feat_cols].values)
        vaX = scaler.transform(val[feat_cols].values)
        teX = scaler.transform(test[feat_cols].values)
        trX_s, trY = prepare_sequences(trX, train['target'].values)
        vaX_s, vaY = prepare_sequences(vaX, val['target'].values)
        teX_s, teY = prepare_sequences(teX, test['target'].values)
        model = TransformerLSTMReg(input_dim=len(feat_cols))
        model = train_pytorch_model(model, trX_s, trY, vaX_s, vaY)
        with torch.no_grad():
            preds = model(torch.FloatTensor(teX_s).to(device)).cpu().numpy().flatten()
        evaluate_model(ticker, teY, preds, 'Transformer+LSTM')
    summarize_model('Transformer+LSTM')

# ════════════════════════════════════════════════════════════
# MODEL 4: LightGBM v1 (per-ticker)
# ════════════════════════════════════════════════════════════
def run_lightgbm_v1():
    print("\n" + "="*60)
    print("MODEL 4: LightGBM v1 (Tabular Regression)")
    print("="*60)
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        train, val, test = load_splits(ticker)
        train, val, test = prepare_regression_targets(train, val, test)
        feat_cols = get_feature_cols(train)
        print(f"  Features: {len(feat_cols)}")
        dtrain = lgb.Dataset(train[feat_cols], train['target'])
        dval   = lgb.Dataset(val[feat_cols], val['target'], reference=dtrain)
        callbacks = [lgb.early_stopping(30), lgb.log_evaluation(0)]
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)
        preds = model.predict(test[feat_cols])
        evaluate_model(ticker, test['target'].values, preds, 'LightGBM v1')
    summarize_model('LightGBM v1')

# ════════════════════════════════════════════════════════════
# MODEL 5: LightGBM v2 (Alpha158, cross-sectional)
# ════════════════════════════════════════════════════════════
def compute_alpha158_features(df):
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
    feat = pd.DataFrame(index=df.index)
    feat['close_open'] = c / o - 1
    feat['high_low'] = h / l - 1
    feat['close_high'] = c / h - 1
    feat['close_low'] = c / l - 1
    for w in [5, 10, 20, 30, 60]:
        feat[f'ma_{w}'] = c.rolling(w).mean() / c - 1
        feat[f'std_{w}'] = c.rolling(w).std() / c
        feat[f'ret_{w}'] = c.pct_change(w)
        feat[f'vol_ma_{w}'] = v.rolling(w).mean() / (v + 1e-8) - 1
    for w in [6, 12, 24]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss_val = (-delta.clip(upper=0)).rolling(w).mean()
        feat[f'rsi_{w}'] = gain / (gain + loss_val + 1e-8)
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    feat['macd'] = (ema12 - ema26) / c
    feat['macd_signal'] = feat['macd'].ewm(span=9).mean()
    feat['macd_hist'] = feat['macd'] - feat['macd_signal']
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    feat['bb_upper'] = (ma20 + 2*std20) / c - 1
    feat['bb_lower'] = (ma20 - 2*std20) / c - 1
    feat['bb_width'] = (4*std20) / (ma20 + 1e-8)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean() / c
    return feat

def run_lightgbm_v2():
    print("\n" + "="*60)
    print("MODEL 5: LightGBM v2 (Alpha158)")
    print("="*60)
    all_train, all_val, all_test = [], [], []
    ticker_test_indices = {}
    for ticker in TICKERS:
        train, val, test = load_splits(ticker)
        for df in [train, val, test]:
            alpha_feat = compute_alpha158_features(df)
            for col in alpha_feat.columns:
                df[col] = alpha_feat[col].values
            df['target'] = df['close'].shift(-1)
            df['_ticker'] = ticker
        train = train.dropna(); val = val.dropna(); test = test.dropna()
        start_idx = len(pd.concat(all_test)) if all_test else 0
        ticker_test_indices[ticker] = (start_idx, start_idx + len(test))
        all_train.append(train); all_val.append(val); all_test.append(test)
    pool_train = pd.concat(all_train, ignore_index=True)
    pool_val   = pd.concat(all_val, ignore_index=True)
    pool_test  = pd.concat(all_test, ignore_index=True)
    feat_cols_a158 = [c for c in pool_train.columns
                      if c not in ['ts_event','close','target','ticker','_ticker','open','high','low','volume']
                      and pool_train[c].dtype in ['float64','float32','int64']]
    print(f"Alpha158 features: {len(feat_cols_a158)}")
    dtrain = lgb.Dataset(pool_train[feat_cols_a158], pool_train['target'])
    dval   = lgb.Dataset(pool_val[feat_cols_a158], pool_val['target'], reference=dtrain)
    callbacks = [lgb.early_stopping(30), lgb.log_evaluation(0)]
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)
    all_preds = model.predict(pool_test[feat_cols_a158])
    for ticker in TICKERS:
        start, end = ticker_test_indices[ticker]
        t_true = pool_test.iloc[start:end]['target'].values
        t_pred = all_preds[start:end]
        evaluate_model(ticker, t_true, t_pred, 'LightGBM v2 (Alpha158)')
    summarize_model('LightGBM v2 (Alpha158)')

# ════════════════════════════════════════════════════════════
# MODEL 6: TimesFM 2.5 (skip if unavailable)
# ════════════════════════════════════════════════════════════
def run_timesfm():
    print("\n" + "="*60)
    print("MODEL 6: TimesFM 2.5")
    print("="*60)
    try:
        import timesfm
        TIMESFM_AVAILABLE = True
    except ImportError:
        TIMESFM_AVAILABLE = False

    if not TIMESFM_AVAILABLE:
        print("TimesFM unavailable locally -- using naive persistence baseline instead.")
        print("(This provides a fair N-consistent comparison point)")
        for ticker in TICKERS:
            print(f"\n--- {ticker} ---")
            train, val, test = load_splits(ticker)
            train, val, test = prepare_regression_targets(train, val, test)
            # Naive persistence: predict y(t) = y(t-1)
            y_true = test['target'].values
            y_pred = test['close'].values  # previous close as prediction
            evaluate_model(ticker, y_true, y_pred, 'TimesFM 2.5 (naive proxy)')
        summarize_model('TimesFM 2.5 (naive proxy)')
        return

    # If TimesFM is available, run it properly
    torch.set_float32_matmul_precision('high')
    tfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
    tfm_model.compile(timesfm.ForecastConfig(
        max_context=1024, max_horizon=1, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True,
        infer_is_positive=True, fix_quantile_crossing=True))
    inner = tfm_model.model; inner.eval()
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        train_c, val_c, test_c = load_close_series(ticker)
        context = pd.concat([train_c, val_c]).values
        actuals = test_c.values
        context_len = min(512, len(context))
        full_series = np.concatenate([context, actuals])
        n_ctx = len(context)
        X_raw = []
        for i in range(len(actuals)):
            start = max(0, n_ctx + i - context_len)
            end = n_ctx + i
            X_raw.append(full_series[start:end])
        max_len = max(len(r) for r in X_raw)
        X_raw_padded = np.zeros((len(X_raw), max_len))
        for j, row in enumerate(X_raw):
            X_raw_padded[j, max_len - len(row):] = row
        _device = next(inner.parameters()).device
        X_log = torch.FloatTensor(np.log(X_raw_padded)).to(_device)
        all_preds = []
        for i in range(0, len(X_log), 256):
            xb = X_log[i:i+256]
            bs, seq_l = xb.shape
            patch_len = 32
            pad_len = (patch_len - seq_l % patch_len) % patch_len
            if pad_len > 0:
                x_padded = torch.nn.functional.pad(xb, (pad_len, 0), value=0.0)
                mask = torch.cat([torch.ones(bs, pad_len, device=_device, dtype=torch.bool),
                                  torch.zeros(bs, seq_l, device=_device, dtype=torch.bool)], dim=1)
            else:
                x_padded = xb
                mask = torch.zeros(bs, seq_l, device=_device, dtype=torch.bool)
            num_patches = x_padded.shape[1] // patch_len
            patched = x_padded.reshape(bs, num_patches, patch_len)
            patched_mask = mask.reshape(bs, num_patches, patch_len)
            mu = xb.mean(dim=1, keepdim=True); sigma = xb.std(dim=1, keepdim=True).clamp(min=1e-6)
            normed = (patched - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
            normed = torch.where(patched_mask, 0.0, normed)
            tokenizer_inputs = torch.cat([normed, patched_mask.to(normed.dtype)], dim=-1)
            with torch.no_grad():
                tokens = inner.tokenizer(tokenizer_inputs)
                out = tokens
                for layer in inner.stacked_xf:
                    attn_mask = patched_mask.any(dim=-1)
                    out, _ = layer(out, attn_mask, None)
                point_out = inner.output_projection_point(out)
            last_out = point_out[:, -1, :1]
            pred_log = last_out * sigma + mu
            pred_price = torch.exp(pred_log)
            all_preds.append(pred_price[:, -1].cpu().numpy())
        preds = np.concatenate(all_preds)
        evaluate_model(ticker, actuals, preds, 'TimesFM 2.5')
    summarize_model('TimesFM 2.5')

# ════════════════════════════════════════════════════════════
# MODEL 7: VMD-LSTM Global (LEAKAGE) — AAPL only
# ════════════════════════════════════════════════════════════
class SimpleVMDLSTM(nn.Module):
    def __init__(self, K=5, hidden=64, layers=1):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(1, hidden, layers, batch_first=True) for _ in range(K)])
        self.fc = nn.Linear(hidden * K, 1)
    def forward(self, x):
        outs = []
        for k in range(x.shape[2]):
            o, _ = self.lstms[k](x[:, :, k:k+1])
            outs.append(o[:, -1, :])
        return self.fc(torch.cat(outs, dim=1))

def run_vmd_global():
    print("\n" + "="*60)
    print("MODEL 7: VMD-LSTM Global (LEAKAGE) -- AAPL only")
    print("="*60)
    ticker = 'AAPL'
    print(f"\n--- {ticker} ---")
    train_c, val_c, test_c = load_close_series(ticker)
    full = pd.concat([train_c, val_c, test_c]).values
    u, _, _ = VMD(full, 2000, 0, 5, 0, 1, 1e-7)
    K = u.shape[0]
    n_tr, n_va = len(train_c), len(val_c)
    tr_imfs = u[:, :n_tr].T
    va_imfs = u[:, n_tr:n_tr+n_va].T
    te_imfs = u[:, n_tr+n_va:].T
    trX_s, trY = prepare_sequences(tr_imfs, train_c.values)
    vaX_s, vaY = prepare_sequences(va_imfs, val_c.values)
    teX_s, teY = prepare_sequences(te_imfs, test_c.values)
    model = SimpleVMDLSTM(K=K)
    model = train_pytorch_model(model, trX_s, trY, vaX_s, vaY)
    with torch.no_grad():
        preds = model(torch.FloatTensor(teX_s).to(device)).cpu().numpy().flatten()
    evaluate_model(ticker, teY, preds, 'VMD-LSTM (Global, LEAKAGE)')
    summarize_model('VMD-LSTM (Global, LEAKAGE)')

# ════════════════════════════════════════════════════════════
# MODEL 8: VMD-LSTM Rolling — AAPL only
# ════════════════════════════════════════════════════════════
def run_vmd_rolling():
    print("\n" + "="*60)
    print("MODEL 8: VMD-LSTM Rolling (NO LEAKAGE) -- AAPL only")
    print("="*60)
    ticker = 'AAPL'
    print(f"\n--- {ticker} ---")
    train_c, val_c, test_c = load_close_series(ticker)
    full = pd.concat([train_c, val_c, test_c]).values
    vmd_window = 120
    n_before_test = len(train_c) + len(val_c)
    # Rolling VMD for test
    print("  Computing rolling VMD for test set...")
    test_imfs = []
    for t in range(len(test_c)):
        idx = n_before_test + t
        if idx < vmd_window: continue
        seg = full[idx-vmd_window:idx]
        try:
            u, _, _ = VMD(seg, 2000, 0, 5, 0, 1, 1e-7)
            test_imfs.append(u[:, -1])
        except:
            test_imfs.append(test_imfs[-1] if test_imfs else np.zeros(5))
        if (t+1) % 500 == 0:
            print(f"    {t+1}/{len(test_c)} done")
    test_imfs = np.array(test_imfs)
    test_targets = test_c.values[:len(test_imfs)]
    # Rolling VMD for train (subsample for speed)
    print("  Computing rolling VMD for train set...")
    train_imfs = []
    for t in range(vmd_window, len(train_c)):
        seg = full[t-vmd_window:t]
        try:
            u, _, _ = VMD(seg, 2000, 0, 5, 0, 1, 1e-7)
            train_imfs.append(u[:, -1])
        except:
            train_imfs.append(train_imfs[-1] if train_imfs else np.zeros(5))
        if (t+1) % 1000 == 0:
            print(f"    {t+1}/{len(train_c)} done")
    train_imfs = np.array(train_imfs)
    train_targets = train_c.values[vmd_window:vmd_window+len(train_imfs)]
    # Val
    print("  Computing rolling VMD for val set...")
    val_imfs = []
    for t in range(len(val_c)):
        idx = len(train_c) + t
        if idx < vmd_window: continue
        seg = full[idx-vmd_window:idx]
        try:
            u, _, _ = VMD(seg, 2000, 0, 5, 0, 1, 1e-7)
            val_imfs.append(u[:, -1])
        except:
            val_imfs.append(val_imfs[-1] if val_imfs else np.zeros(5))
    val_imfs = np.array(val_imfs)
    val_targets = val_c.values[:len(val_imfs)]
    if len(train_imfs) < SEQ_LEN + 10:
        print(f"  Skipping: insufficient data"); return
    trX_s, trY = prepare_sequences(train_imfs, train_targets[:len(train_imfs)])
    vaX_s, vaY = prepare_sequences(val_imfs, val_targets[:len(val_imfs)])
    teX_s, teY = prepare_sequences(test_imfs, test_targets[:len(test_imfs)])
    model = SimpleVMDLSTM(K=5)
    model = train_pytorch_model(model, trX_s, trY, vaX_s, vaY)
    with torch.no_grad():
        preds = model(torch.FloatTensor(teX_s).to(device)).cpu().numpy().flatten()
    evaluate_model(ticker, teY, preds, 'VMD-LSTM (Rolling)')
    summarize_model('VMD-LSTM (Rolling)')

# ════════════════════════════════════════════════════════════
# MODEL 9: Wavelet-LSTM
# ════════════════════════════════════════════════════════════
class WaveletLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def sliding_window_wavelet_denoise(series, window=120, wavelet='db4', level=3):
    result = np.full(len(series), np.nan)
    for t in range(window, len(series)):
        segment = series[t-window:t]
        coeffs = pywt.wavedec(segment, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(window))
        coeffs_t = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        denoised = pywt.waverec(coeffs_t, wavelet)
        result[t] = denoised[-1]
    return result

def run_wavelet_lstm():
    print("\n" + "="*60)
    print("MODEL 9: Wavelet-LSTM")
    print("="*60)
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        train_c, val_c, test_c = load_close_series(ticker)
        full = pd.concat([train_c, val_c, test_c]).values
        denoised = sliding_window_wavelet_denoise(full)
        features = np.column_stack([full, np.nan_to_num(denoised)])
        n_tr, n_va = len(train_c), len(val_c)
        tr_feat = features[:n_tr]; va_feat = features[n_tr:n_tr+n_va]; te_feat = features[n_tr+n_va:]
        trX_s, trY = prepare_sequences(tr_feat, train_c.values)
        vaX_s, vaY = prepare_sequences(va_feat, val_c.values)
        teX_s, teY = prepare_sequences(te_feat, test_c.values)
        if len(trX_s) < 10:
            print(f"  Skipping {ticker}: insufficient data"); continue
        model = WaveletLSTM(input_dim=2)
        model = train_pytorch_model(model, trX_s, trY, vaX_s, vaY)
        with torch.no_grad():
            preds = model(torch.FloatTensor(teX_s).to(device)).cpu().numpy().flatten()
        evaluate_model(ticker, teY, preds, 'Wavelet-LSTM')
    summarize_model('Wavelet-LSTM')

# ════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════
def print_summary_table():
    print("\n" + "="*100)
    print("TABLE 2.1 -- DIRECT PREDICTION MODEL COMPARISON (STANDARDISED)")
    print("="*100)
    header = f"{'Model':<35} {'Mean DA':>8} {'N':>7} {'z':>8} {'p':>9} {'RMSE':>9} {'R2':>8} {'Sig':>4}"
    print(header)
    print("-"*len(header))
    model_order = ['LSTM Regression', 'Attention-LSTM', 'Transformer+LSTM',
                    'LightGBM v1', 'LightGBM v2 (Alpha158)',
                    'TimesFM 2.5', 'TimesFM 2.5 (naive proxy)',
                    'VMD-LSTM (Global, LEAKAGE)', 'VMD-LSTM (Rolling)', 'Wavelet-LSTM']
    table_rows = []
    for model_name in model_order:
        if model_name not in results_dict: continue
        rows = results_dict[model_name]
        das = [r['DA'] for r in rows]
        ns = [r['N'] for r in rows]
        avg_da = np.mean(das)
        total_n = sum(ns)
        z, p = da_z_test(avg_da, total_n)
        rmses = [r['RMSE'] for r in rows if r['RMSE'] is not None]
        r2s = [r['R2'] for r in rows if r['R2'] is not None]
        avg_rmse = np.mean(rmses) if rmses else None
        avg_r2 = np.mean(r2s) if r2s else None
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        rmse_str = f"{avg_rmse:.4f}" if avg_rmse is not None else "--"
        r2_str = f"{avg_r2:.4f}" if avg_r2 is not None else "--"
        print(f"{model_name:<35} {avg_da:>8.4f} {total_n:>7d} {z:>8.3f} {p:>9.4f} {rmse_str:>9} {r2_str:>8} {sig:>4}")
        table_rows.append({
            'Model': model_name, 'DA': round(avg_da, 4), 'N': total_n,
            'z': round(z, 4), 'p': round(p, 6),
            'RMSE': round(avg_rmse, 4) if avg_rmse else None,
            'R2': round(avg_r2, 4) if avg_r2 else None,
            'Sig': sig, 'per_ticker': rows
        })
    print()
    print("Key: *** p<0.001, ** p<0.01, * p<0.05 (one-sided z-test: H0: DA <= 0.5)")
    print("Note: VMD-LSTM (Global) DA is artificially inflated due to data leakage")
    print("      VMD-LSTM models are AAPL-only; all others are 7-ticker pooled")
    # Save JSON
    outpath = os.path.join(os.path.dirname(__file__), 'ch2_standardised_results.json')
    with open(outpath, 'w') as f:
        json.dump({'results': table_rows, 'config': {
            'tickers': TICKERS, 'seq_len': SEQ_LEN, 'freq': FREQ, 'seed': SEED
        }}, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"SPLITS_DIR: {SPLITS_DIR}")
    print(f"Tickers: {TICKERS}")
    print(f"SEQ_LEN: {SEQ_LEN}, FREQ: {FREQ}")
    print()

    t_start = time.time()

    # Phase 1: Fast models (< 1 min each)
    run_lstm_regression()
    run_attention_lstm()
    run_transformer_lstm()
    run_lightgbm_v1()
    run_lightgbm_v2()

    # Phase 2: TimesFM (skip if not installed)
    run_timesfm()

    # Phase 3: VMD models (AAPL only -- VMD rolling is slow)
    run_vmd_global()
    # VMD Rolling is VERY slow (thousands of VMD calls) -- run separately if needed
    print("\n" + "="*60)
    print("SKIPPING VMD-LSTM Rolling (too slow for interactive run)")
    print("Estimated time: 30-60 minutes for AAPL alone")
    print("Run separately with: run_vmd_rolling()")
    print("="*60)

    # Phase 4: Wavelet-LSTM
    run_wavelet_lstm()

    total_time = time.time() - t_start
    print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f}m)")

    print_summary_table()
