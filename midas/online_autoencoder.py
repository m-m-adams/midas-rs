import torch
import torch.nn as nn
import torch.optim as optim


class lstmautoencoder:

    class AE(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.encoder_hidden_layer = nn.Linear(
                in_features=1, out_features=128
            )
            self.encoder_output_layer = nn.Linear(
                in_features=128, out_features=128
            )
            self.decoder_hidden_layer = nn.Linear(
                in_features=128, out_features=128
            )
            self.decoder_output_layer = nn.Linear(
                in_features=128, out_features=1
            )

        def forward(self, features):
            activation = self.encoder_hidden_layer(features)
            activation = torch.relu(activation)
            code = self.encoder_output_layer(activation)
            code = torch.relu(code)
            activation = self.decoder_hidden_layer(code)
            activation = torch.relu(activation)
            activation = self.decoder_output_layer(activation)
            reconstructed = torch.relu(activation)
            return reconstructed

    def __init__(self, input_dim=1, latent_dim=20, num_layers=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.AE().to(self.device)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def run(self, edges: list[(int, int)]):
        edges = torch.tensor(edges).float().to(self.device)
        scores = []
        for edge in edges:
            x = edge[0].reshape(1, 1, 1)
            y = edge[1].reshape(1, 1, 1)
            y_pred = self.model(x)
            self.optimizer.zero_grad()
            loss = self.loss_function(y_pred, y)
            scores.append(loss)
            loss.backward()
            self.optimizer.step()
            print(y_pred.data, y.data, loss.data)
