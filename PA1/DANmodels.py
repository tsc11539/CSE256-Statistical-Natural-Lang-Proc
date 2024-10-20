import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from sentiment_data import read_sentiment_examples, WordEmbeddings, read_word_embeddings


# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, Indexer, frozen=True) -> None:
        # Read the pretrained data from the input file
        self.examples = read_sentiment_examples(infile)

        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        # Vectorize the sentences
        self.embeddings = []
        for sts in self.sentences:
            indices  = [Indexer.index_of(s) for s in sts.split()]
            self.embeddings.append(indices)
        
        max_len = max(len(row) for row in self.embeddings)
        self.embeddings = np.array([np.pad(row, (0, max_len-len(row)), 'constant', constant_values=0) for row in self.embeddings])
        
        # Convert embeddings and labls to Pytorch tensors
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.embeddings[idx], self.labels[idx]

class NN2DANModel(nn.Module):
    def __init__(self, embedding_layer, input_size, hidden_size):
        super(NN2DANModel, self).__init__()
        # define fully connected layer
        self.embedding_layer = embedding_layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.5)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = x.to(torch.int64)
        embedded = self.embedding_layer(x)
        # embedded shape: (batch_size, seq_length, embedding_dim)

        # Average the embeddings
        averaged = torch.mean(embedded, dim=1)
        x = F.relu(self.fc1(averaged))
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
    

class NN3DANModel(nn.Module):
    def __init__(self, embedding_layer, input_size, hidden_size):
        super(NN3DANModel, self).__init__()
        # define fully connected layer
        self.embedding_layer = embedding_layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = x.to(torch.int64)
        embedded = self.embedding_layer(x)
        # embedded shape: (batch_size, seq_length, embedding_dim)

        # Average the embeddings
        averaged = torch.mean(embedded, dim=1)
        x = F.relu(self.fc1(averaged))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x
