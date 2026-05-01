import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class DynamicDNN(nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        layers = []
        last_dim = input_dim

        #Build up the layers
        for _ in range(params['n_layers']):
            layers.append(nn.Linear(last_dim, params['h1']))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(params['dropout']))
            last_dim = params['h1']
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)    
    

class DynamicRNN(nn.Module):
    def __init__(self, input_dim, params, r_type="LSTM"):
        super().__init__()
        rnn_class = nn.LSTM if r_type == "LSTM" else nn.GRU
        
        # PyTorch RNNs have a built-in num_layers argument
        self.rnn = rnn_class(
            input_dim, 
            params['h1'], 
            num_layers=params['n_layers'], 
            batch_first=True, 
            dropout=params['dropout'] if params['n_layers'] > 1 else 0
        )
        self.fc = nn.Linear(params['h1'], 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
    
class UniversalTorchWrapper:
    def __init__(self, model_type, params, input_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.params = params
        self.input_dim = input_dim
        # Suggest window_size in Optuna (e.g., 24 for daily sequences)
        self.window_size = params.get("window_size", 24) 
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        if model_type == "DNN":
            self.model = DynamicDNN(input_dim, params).to(self.device)
        else:
            self.model = DynamicRNN(input_dim, params, r_type=model_type).to(self.device)
            
        self.criterion = nn.HuberLoss()

    def create_sequences(self, X, y=None, window_size=96):
        X_seq, y_seq, idx = [], [], []

        for i in range(window_size - 1, len(X)):
            X_seq.append(X[i - window_size + 1:i + 1])
            if y is not None:
                y_seq.append(y[i])
            idx.append(i)

        X_seq = np.array(X_seq)

        if y is not None:
            return X_seq, np.array(y_seq), np.array(idx)

        return X_seq, np.array(idx)

    def fit(self, X, y, sample_weight=None):
        X_np = self.feature_scaler.fit_transform(X.values).astype(np.float32)
        y_np = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten().astype(np.float32)

        if self.model_type in ["LSTM", "GRU"]:
            X_seq, y_seq, idx = self.create_sequences(X_np, y_np, self.window_size)

            if sample_weight is not None:
                w_seq = np.array(sample_weight)[idx]
            else:
                w_seq = np.ones_like(y_seq)

        else:
            X_seq, y_seq = X_np, y_np
            w_seq = np.array(sample_weight) if sample_weight is not None else np.ones_like(y_seq)

        X_tensor = torch.from_numpy(X_seq).to(self.device)
        y_tensor = torch.from_numpy(y_seq).to(self.device)
        w_tensor = torch.from_numpy(w_seq).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, w_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params['lr'])

        self.model.train()
        for _ in range(self.params['epochs']):
            for batch_X, batch_y, batch_w in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X).squeeze()
                loss = (self.criterion(pred, batch_y) * batch_w).mean()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X, target_len=None):
        self.model.eval()

        X_np = self.feature_scaler.transform(X.values).astype(np.float32)

        if self.model_type in ["LSTM", "GRU"]:
            X_seq, idx = self.create_sequences(X_np, window_size=self.window_size)
            X_tensor = torch.from_numpy(X_seq).to(self.device)
        else:
            X_tensor = torch.from_numpy(X_np).to(self.device)

        with torch.no_grad():
            preds = self.model(X_tensor).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)

        preds_np = preds.cpu().numpy()
        preds_final = self.target_scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()

        # ✅ CRITICAL: deterministic slicing
        if target_len is not None:
            return preds_final[-target_len:]

        return preds_final