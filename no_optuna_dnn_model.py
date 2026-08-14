import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class DynamicDNN(nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        layers = []
        last_dim = input_dim

        for _ in range(params.get('n_layers', 1)):
            layers.append(nn.Linear(last_dim, params['h1']))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(params.get('dropout', 0.0)))
            last_dim = params['h1']
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)    
    

class DynamicRNN(nn.Module):
    def __init__(self, input_dim, params, r_type="LSTM"):
        super().__init__()
        rnn_class = nn.LSTM if r_type == "LSTM" else nn.GRU
        
        self.rnn = rnn_class(
            input_dim, 
            params['h1'], 
            num_layers=params.get('n_layers', 1), 
            batch_first=True, 
            dropout=params.get('dropout', 0.0) if params.get('n_layers', 1) > 1 else 0
        )
        self.fc = nn.Linear(params['h1'], params.get('pred_horizon', 96))
        
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
    

class UniversalTorchWrapper:
    def __init__(self, model_type, params, input_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.params = params
        self.input_dim = input_dim
        
        # Define the new input and output windows
        self.window_size = params.get("window_size", 96 * 7) # 672 (Input)
        self.pred_horizon = params.get("pred_horizon", 96)   # 96 (Output)
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # NOTE: Your DynamicDNN and DynamicRNN must be updated to output `pred_horizon` values
        if model_type == "DNN":
            self.model = DynamicDNN(input_dim, params).to(self.device)
        else:
            self.model = DynamicRNN(input_dim, params, r_type=model_type).to(self.device)
            
        self.criterion = nn.HuberLoss()

    def create_sequences(self, X, y=None):
        X_seq, y_seq, idx = [], [], []

        # We need enough data points for both the input window AND the prediction horizon
        total_len = self.window_size + self.pred_horizon

        for i in range(len(X) - total_len + 1):
            # Input is the current 672 steps
            X_seq.append(X[i : i + self.window_size])
            
            if y is not None:
                # Target is the NEXT 96 steps
                y_seq.append(y[i + self.window_size : i + total_len])
                
            idx.append(i + self.window_size - 1)

        X_seq = np.array(X_seq, dtype=np.float32)
        
        if y is not None:
            return X_seq, np.array(y_seq, dtype=np.float32), np.array(idx)

        return X_seq, np.array(idx)

    def fit(self, X, y, sample_weight=None):
        X_np = self.feature_scaler.fit_transform(X.values).astype(np.float32)
        y_np = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten().astype(np.float32)

        if self.model_type in ["LSTM", "GRU"]:
            X_seq, y_seq, idx = self.create_sequences(X_np, y_np)

            if sample_weight is not None:
                # Assuming weight applies to the whole sequence
                w_seq = np.array(sample_weight)[idx]
                w_seq = np.expand_dims(w_seq, axis=1) # Shape (batch, 1) to broadcast with (batch, 96)
            else:
                w_seq = np.ones((len(y_seq), 1), dtype=np.float32)

        else:
            X_seq, y_seq = X_np, y_np
            w_seq = np.array(sample_weight) if sample_weight is not None else np.ones_like(y_seq)

        X_tensor = torch.from_numpy(X_seq)
        y_tensor = torch.from_numpy(y_seq)
        w_tensor = torch.from_numpy(w_seq)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, w_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params.get('batch_size', 64), shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.get('lr', 1e-3))

        self.model.train()
        for _ in range(self.params.get('epochs', 10)):
            for batch_X, batch_y, batch_w in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_w = batch_w.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_X) 
                
                # Squeeze can be dangerous if batch_size=1, keep dimensions explicit
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                    
                loss = (self.criterion(pred, batch_y) * batch_w).mean()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        """
        Walk-forward prediction. For RNN models, slides a window of size
        window_size across X in steps of pred_horizon, predicting pred_horizon
        values at each step. For DNN models, feeds all rows directly.
        """
        self.model.eval()

        if self.model_type in ["LSTM", "GRU"]:
            if len(X) < self.window_size:
                raise ValueError(f"Input X must have at least {self.window_size} rows to predict.")

            X_np = self.feature_scaler.transform(X.values).astype(np.float32)
            all_preds = []

            for start in range(0, len(X) - self.window_size + 1, self.pred_horizon):
                window = X_np[start : start + self.window_size]
                X_tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    preds = self.model(X_tensor)

                all_preds.append(preds.cpu().numpy().flatten())

            preds_final = np.concatenate(all_preds)
            preds_final = self.target_scaler.inverse_transform(preds_final.reshape(-1, 1)).flatten()
            return preds_final
        else:
            X_np = self.feature_scaler.transform(X.values).astype(np.float32)
            X_tensor = torch.from_numpy(X_np).to(self.device)

            with torch.no_grad():
                preds = self.model(X_tensor)

            preds_np = preds.cpu().numpy()
            preds_final = self.target_scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()
            return preds_final