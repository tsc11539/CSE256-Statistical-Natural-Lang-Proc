import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from sentiment_data import read_sentiment_examples, WordEmbeddings, read_word_embeddings, SentimentExample
import re, collections
from tqdm import tqdm


# Dataset class for handling sentiment analysis data
class SentimentDatasetBPE(Dataset):
    def __init__(self, infile, Indexer, sorted_tokens, frozen=True) -> None:
        # Read the training data
        self.examples = read_sentiment_examples(infile)

        # Extract sentences and labels from the examples
        self.sentences = []
        for ex in tqdm(self.examples, desc=f"Tokenizing {infile}"):
            merged = tokenize_word('</w> '.join(ex.words)+ '</w>', sorted_tokens)
            self.sentences.append(merged)
        self.labels = [ex.label for ex in self.examples]

        # Vectorize the sentences
        self.embeddings = []
        
        for sts in self.sentences:
            indices = []
            for tk in sts:
                indices.append(Indexer.index_of(tk))
            self.embeddings.append(indices)

        max_len = max(len(row) for row in self.embeddings)
        for idx, row in enumerate(self.embeddings):
            self.embeddings[idx].extend([0]*(max_len-len(row)))
        # self.embeddings = np.array([np.pad(row, (0, max_len-len(row)), 'constant', constant_values=0) for row in tqdm(self.embeddings)])
        # Convert embeddings and labls to Pytorch tensors
        self.embeddings = torch.tensor(np.array(self.embeddings), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.embeddings[idx], self.labels[idx]
    
class NN2BPEModel(nn.Module):
    def __init__(self, embedding_layer, input_size, hidden_size):
        super(NN2BPEModel, self).__init__()
        # define fully connected layer
        self.embedding_layer = embedding_layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = x.to(torch.int64)
        embedded = self.embedding_layer(x)
        # embedded shape: (batch_size, seq_length, embedding_dim)

        # Average the embeddings
        averaged = torch.mean(embedded, dim=1)
        x = F.relu(self.fc1(averaged))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
    
class NN3BPEModel(nn.Module):
    def __init__(self, embedding_layer, input_size, hidden_size):
        super(NN3BPEModel, self).__init__()
        # define fully connected layer
        self.embedding_layer = embedding_layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size,2)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = x.to(torch.int64)
        embedded = self.embedding_layer(x)
        # embedded shape: (batch_size, seq_length, embedding_dim)

        # Average the embeddings
        averaged = torch.mean(embedded, dim=1)
        x = F.relu(self.fc1(averaged))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x

def CreateBPEToken(infile, num_merges):
    vocab = get_vocab(infile)
    tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)

    for i in tqdm(range(num_merges), desc=f"merge {num_merges} times"):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)
        
    sorted_tokens_tuple = sorted(tokens_frequencies.items(), key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
    sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]
    return sorted_tokens

def get_vocab(infile):
    vocab = collections.defaultdict(int)
    # words concatenate by space + ' </w>' in the end
    # [low] -> ["l o w </w>": 1] 
    with open(infile, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            if len(line.strip()) > 0:
                fields = line.split("\t")
                if len(fields) != 2:
                    fields = line.split()
                    label = 0 if "0" in fields[0] else 1
                    sent = " ".join(fields[1:]).lower()
                else:
                    # Slightly more robust to reading bad output than int(fields[0])
                    label = 0 if "0" in fields[0] else 1
                    sent = fields[1].lower()
                words = sent.strip().split()
                for word in words:
                    vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    # connect adjecent 2 chars to a pair
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_tokens_from_vocab(vocab):
    tokens_frequencies = collections.defaultdict(int)
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization[''.join(word_tokens)] = word_tokens
    return tokens_frequencies, vocab_tokenization

def measure_token_length(token):
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)

def tokenize_word(string, sorted_tokens, unknown_token='UNK'):
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]

    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token.replace('.', '[.]'))

        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
        break
    return string_tokens