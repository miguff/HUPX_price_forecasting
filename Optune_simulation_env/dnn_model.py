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
        # The 'window_size' should be a hyperparameter suggested by Optuna
        self.window_size = params.get("window_size", 96) 
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        if model_type == "DNN":
            self.model = DynamicDNN(input_dim, params).to(self.device)
        else:
            # RNN/LSTM now receives (batch, window_size, input_dim)
            self.model = DynamicRNN(input_dim, params, r_type=model_type).to(self.device)
            
        self.criterion = nn.HuberLoss()

    def _create_sequences(self, X_np, y_np=None):
        """
        Converts flat tabular data into sequences of length self.window_size.
        """
        X_seq, y_seq = [], []
        # We need enough history to create the first window
        for i in range(len(X_np) - self.window_size + 1):
            X_seq.append(X_np[i : i + self.window_size])
            if y_np is not None:
                y_seq.append(y_np[i + self.window_size - 1])
        
        X_seq = np.array(X_seq)
        if y_np is not None:
            return X_seq, np.array(y_seq)
        return X_seq

    def fit(self, X, y, sample_weight=None):
        X_scaled = self.feature_scaler.fit_transform(X.values).astype(np.float32)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten().astype(np.float32)

        if self.model_type in ["LSTM", "GRU"]:
            X_t_np, y_t_np = self._create_sequences(X_scaled, y_scaled)
            # Sample weights also need to be aligned to the end of the sequence
            if sample_weight is not None:
                sw_np = np.array(sample_weight)[self.window_size - 1:].astype(np.float32)
            else:
                sw_np = np.ones(len(y_t_np), dtype=np.float32)
        else:
            X_t_np, y_t_np = X_scaled, y_scaled
            sw_np = np.array(sample_weight).astype(np.float32) if sample_weight is not None else np.ones_like(y_t_np)

        X_t = torch.from_numpy(X_t_np).to(self.device)
        y_t = torch.from_numpy(y_t_np).to(self.device)
        w_t = torch.from_numpy(sw_np).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t, w_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params['lr'])

        self.model.train()
        for epoch in range(self.params['epochs']):
            for batch_X, batch_y, batch_w in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X).squeeze()
                loss = (self.criterion(pred, batch_y) * batch_w).mean()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        # To predict a single day, we need the preceding window_size rows from the dataset
        X_scaled = self.feature_scaler.transform(X.values).astype(np.float32)
        
        if self.model_type in ["LSTM", "GRU"]:
            X_t_np = self._create_sequences(X_scaled)
            X_t = torch.from_numpy(X_t_np).to(self.device)
        else:
            X_t = torch.from_numpy(X_scaled).to(self.device)

        with torch.no_grad():
            preds_scaled = self.model(X_t).squeeze()
            if preds_scaled.dim() == 0: preds_scaled = preds_scaled.unsqueeze(0)
                
        preds_np = preds_scaled.cpu().numpy()
        return self.target_scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()