"""
Model definitions for PixStory
"""
import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, embed_size, fine_tune=False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        if fine_tune:
            for param in list(self.features.parameters())[-30:]:
                param.requires_grad = True
        else:
            for param in self.features.parameters():
                param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)

class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, word2idx, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size + embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.word2idx = word2idx
        self.vocab_size = vocab_size
        self.feature_projection = nn.Linear(embed_size, hidden_size) if embed_size != hidden_size else nn.Identity()
    
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        hidden_features = self.feature_projection(features)
        features_expanded = features.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        gru_input = torch.cat([embeddings, features_expanded], dim=2)
        hidden = hidden_features.unsqueeze(0)
        output, _ = self.gru(gru_input, hidden)
        output = self.dropout(output)
        return self.linear(output)
    
    def predict(self, features, max_len=25, device='cuda'):
        batch_size = features.size(0)
        hidden_features = self.feature_projection(features)
        hidden = hidden_features.unsqueeze(0)
        inputs = torch.tensor([[self.word2idx['<start>']]] * batch_size, device=device)
        seq = []
        
        for _ in range(max_len):
            embeddings = self.embed(inputs)
            features_expanded = features.unsqueeze(1)
            gru_input = torch.cat([embeddings, features_expanded], dim=2)
            output, hidden = self.gru(gru_input, hidden)
            predicted = self.linear(output.squeeze(1)).argmax(1)
            seq.append(predicted.unsqueeze(1))
            inputs = predicted.unsqueeze(1)
            if (predicted == self.word2idx['<end>']).all():
                break
        
        return torch.cat(seq, 1)

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, word2idx, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.word2idx = word2idx
        self.vocab_size = vocab_size
        self.feature_projection = nn.Linear(embed_size, hidden_size) if embed_size != hidden_size else nn.Identity()
    
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        hidden_features = self.feature_projection(features)
        features_expanded = features.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        lstm_input = torch.cat([embeddings, features_expanded], dim=2)
        hidden = (hidden_features.unsqueeze(0), torch.zeros_like(hidden_features).unsqueeze(0))
        output, _ = self.lstm(lstm_input, hidden)
        output = self.dropout(output)
        return self.linear(output)
    
    def predict(self, features, max_len=25, device='cuda'):
        batch_size = features.size(0)
        hidden_features = self.feature_projection(features)
        hidden = (hidden_features.unsqueeze(0), torch.zeros_like(hidden_features).unsqueeze(0))
        inputs = torch.tensor([[self.word2idx['<start>']]] * batch_size, device=device)
        seq = []
        
        for _ in range(max_len):
            embeddings = self.embed(inputs)
            features_expanded = features.unsqueeze(1)
            lstm_input = torch.cat([embeddings, features_expanded], dim=2)
            output, hidden = self.lstm(lstm_input, hidden)
            predicted = self.linear(output.squeeze(1)).argmax(1)
            seq.append(predicted.unsqueeze(1))
            inputs = predicted.unsqueeze(1)
            if (predicted == self.word2idx['<end>']).all():
                break
        
        return torch.cat(seq, 1)
