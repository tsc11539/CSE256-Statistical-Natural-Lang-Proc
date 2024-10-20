# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings, WordEmbeddings, random_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, NN2DANModel, NN3DANModel
from BPEmodel import *
from utils import *
from tqdm import tqdm

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set different word vector dimension
    word_vecdim = 300

    print(f"Running {args.model} model...")
    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        # get pretrained_embs vector
        pretrained_embs = read_word_embeddings(f"data/glove.6B.{word_vecdim}d-relativized.txt")
        embedding_layer = pretrained_embs.get_initialized_embedding_layer(True)
        indexer = pretrained_embs.word_indexer
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetDAN("data/train.txt", indexer)
        dev_data = SentimentDatasetDAN("data/dev.txt", indexer)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate DAN
        start_time = time.time()
        print('\Pretrained 2 layers:')
        train_accuracy, test_accuracy = experiment(NN2DANModel(embedding_layer, input_size=word_vecdim, hidden_size=100), train_loader, test_loader)

        
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(train_accuracy, label=f'Train')
        plt.plot(test_accuracy, label=f'Test')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('DAN Training Accuracy')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        accuracy_file = 'DAN_Train_Accuracy.png'
        plt.savefig(accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {accuracy_file}")


    elif args.model == "BPE":
        plt.figure(1, figsize=(8, 6))
        plt.figure(2, figsize=(8, 6))
        for num_merges in [500]:
            sorted_tokens = CreateBPEToken("data/train.txt", num_merges)
            embedding_layer = nn.Embedding(num_embeddings=len(sorted_tokens)+2, embedding_dim=word_vecdim)
            word_indexer = Indexer()
            word_indexer.add_and_get_index("PAD")
            word_indexer.add_and_get_index("UNK")
            for token in sorted_tokens:
                word_indexer.add_and_get_index(token)
            train_data = SentimentDatasetBPE("data/train.txt", word_indexer, sorted_tokens)
            dev_data = SentimentDatasetBPE("data/dev.txt", word_indexer, sorted_tokens)
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
            train_accuracy, test_accuracy = experiment(NN2BPEModel(embedding_layer, input_size=word_vecdim, hidden_size=20), train_loader, test_loader)
            plt.figure(1)
            plt.plot(train_accuracy, label=f'Merge {num_merges} Times')
            plt.figure(2)
            plt.plot(test_accuracy, label=f'Merge {num_merges} Times')
        
        for idx, type in enumerate(['Training', 'Testing'], start=1):
            plt.figure(idx)
            plt.xlabel('Epochs')
            plt.ylabel(f'{type} Accuracy')
            plt.title(f'{type} Accuracy for diff merges')
            plt.legend()
            plt.grid()
            # Save the accuracy figure
            accuracy_file = f'{type}_accuracy for diff merges.png'
            plt.savefig(accuracy_file)
            print(f"\n\n{type} accuracy plot saved as {accuracy_file}")

        
if __name__ == "__main__":
    main()
