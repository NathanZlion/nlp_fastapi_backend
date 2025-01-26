import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
  
        identity = x
    
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        
        if x.shape == identity.shape:
            x = x + identity
            
        x = self.fc2(x)
        return x