# %%
from typing import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.utils.data as data_utils
from sklearn.metrics import roc_auc_score

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

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 1)

        # map from hidden dim back to embeddings
        self.output = nn.Linear(hidden_dim, embedding_dim)

    def embed(self, node):
        return self.edge_embed(node)

    def forward(self, node):

        embeds = F.relu(self.edge_embed(node))
        hidden, _ = self.lstm(embeds)
        output = self.output(hidden)
        return output


losses = np.array([])
model = LSTMs2s(4, 50, 100_000).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss(reduction='none')
for source, dest in train_loader:
    source, dest = source.to(device), dest.to(device)
    optimizer.zero_grad()
    y_pred = model(source).squeeze()
    y = model.embed(dest)

    # no reduction so we can get raw loss scores for AUC later. sum so that we get a single score vs. 4
    loss = loss_function(y_pred, y).sum(1)
    print("loss:{}".format(loss.mean().data.cpu().numpy()))
    losses = np.append(losses, loss.data.cpu().numpy().flatten())

    a = loss.mean()
    a.backward()
    optimizer.step()


"""Note - due to attacker nodes communicating with single target at a time they are much easier tolearn
than the patterns in legit nodes which communicate with multiple targets. 
Guessing source from dest is the same issue but losses are higher across the board

50 hidden dim/4 embedding dim/1 layer lstm gives AUROC of 0.88"""
print(roc_auc_score(labels, -losses))


# %%
