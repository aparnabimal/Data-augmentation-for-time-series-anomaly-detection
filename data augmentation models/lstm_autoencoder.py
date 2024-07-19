import torch, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(LSTMModel, self).__init__()

        # LSTM Encoder 
        self.lstm1 = nn.LSTM(input_dim, 256, batch_first=True)  
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)

        # LSTM Decoder
        self.lstm2 = nn.LSTM(256, 64, batch_first=True)  
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)

        self.lstm3 = nn.LSTM(64, 64, batch_first=True)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.lstm4 = nn.LSTM(64, 256, batch_first=True) 
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout3(x)

        x, _ = self.lstm4(x)
        x = self.bn4(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout4(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
        

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
