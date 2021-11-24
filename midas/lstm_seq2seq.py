# %%
import time
from typing import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.utils.data as data_utils
from sklearn.metrics import roc_auc_score
from argparse import ArgumentParser
from midas_cores import CMSCounter, MidasR
from online_autoencoder import lstmautoencoder
import midas

batch_size = 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

edges = pd.read_csv("../data/darpa_processed.csv")
labels = pd.read_csv("../data/darpa_ground_truth.csv")

ed_tens = torch.tensor(edges[["source"]].values)
label_tens = torch.tensor(edges["dest"].values)

(ed_tens.size(), label_tens.size())

train = data_utils.TensorDataset(ed_tens, label_tens)
train_loader = data_utils.DataLoader(
    train, batch_size=batch_size, shuffle=False)


class LSTMs2s(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, node_count):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.edge_embed = nn.Embedding(node_count, embedding_dim)

        # takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 3)

        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(hidden_dim, embedding_dim)

    def embed(self, node):
        return self.edge_embed(node)

    def forward(self, node):

        embeds = F.relu(self.edge_embed(node))
        hidden, _ = self.lstm(embeds)
        output = self.output(hidden)
        return output
# %% seq2seq source to dest


losses = np.array([])
model = LSTMs2s(4, 50, 100000).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss(reduction='none')
for source, dest in train_loader:
    source, dest = source.view(source.size(0), -1).to(device), dest.to(device)
    optimizer.zero_grad()
    print(dest.size())
    y_pred = model(source).view(source.size(0), -1)
    y = model.embed(dest)
    print(y.size())

    # no reduction so we can get raw loss scores for AUC later. sum so that we get a single score vs. 4
    loss = loss_function(y_pred, y).sum(1)
    print("loss:{}".format(loss.mean().data.cpu().numpy()))
    losses = np.append(losses, loss.data.cpu().numpy().flatten())

    a = loss.mean()
    a.backward()
    optimizer.step()

roc_auc_score(labels, -losses)

"""Note - due to attacker nodes communicating with single target at a time they are much easier tolearn
than the patterns in legit nodes which communicate with multiple targets. 
Guessing source from dest is the same issue but losses are higher across the board

50 hidden dim/3 layers gives AUROC of 0.86"""

# %%
