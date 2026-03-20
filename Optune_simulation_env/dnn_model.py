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
        
        # Initialize Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        if model_type == "DNN":
            self.model = DynamicDNN(input_dim, params).to(self.device)
        else:
            self.model = DynamicRNN(input_dim, params, r_type=model_type).to(self.device)
            
        self.criterion = nn.HuberLoss()

    def _prepare_data(self, X_np, y_np=None):
        """
        X_np and y_np MUST be numpy arrays, not DataFrames.
        """
        X_tensor = torch.from_numpy(X_np).float().to(self.device)
        
        if self.model_type in ["LSTM", "GRU"]:
            # Reshape to (Batch, Sequence=1, Features)
            X_tensor = X_tensor.view(-1, 1, self.input_dim)
            
        if y_np is not None:
            y_tensor = torch.from_numpy(y_np).float().to(self.device)
            return X_tensor, y_tensor
        return X_tensor

    def fit(self, X, y, sample_weight=None):
        # 1. Scale and convert to NumPy
        X_scaled = self.feature_scaler.fit_transform(X.values).astype(np.float32)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten().astype(np.float32)

        X_t, y_t = self._prepare_data(X_scaled, y_scaled)
        
        # Handle sample weights for synthetic data
        if sample_weight is not None:
            # Ensure sample_weight is a numpy array before tensor conversion
            sw_np = np.array(sample_weight).astype(np.float32)
            w_t = torch.from_numpy(sw_np).to(self.device)
        else:
            w_t = torch.ones_like(y_t)

        dataset = torch.utils.data.TensorDataset(X_t, y_t, w_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params['lr'])

        self.model.train()
        for epoch in range(self.params['epochs']):
            for batch_X, batch_y, batch_w in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X).squeeze()
                
                # Weighted Loss
                loss = (self.criterion(pred, batch_y) * batch_w).mean()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        
        # 1. Scale features using the FIT from the training slice
        X_scaled = self.feature_scaler.transform(X.values).astype(np.float32)
        X_t = self._prepare_data(X_scaled)
        
        with torch.no_grad():
            preds_scaled = self.model(X_t).squeeze()
            
            # Handle single-row prediction edge case
            if preds_scaled.dim() == 0:
                preds_scaled = preds_scaled.unsqueeze(0)
                
        # 2. Convert back to CPU/Numpy
        preds_np = preds_scaled.cpu().numpy()
        
        # 3. Inverse transform to get back to real price units (e.g. EUR/MWh)
        preds_final = self.target_scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()
        
        return preds_final