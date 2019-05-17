import torch.nn as nn
import torch
import numpy as np
import math

class MomentLoss(nn.Module):

    def __init__(self, moments):
        super(MomentLoss, self).__init__()
        self.moments = moments

    def forward(self, x, y):
        xm = torch.zeros(self.moments)
        ym = torch.zeros(self.moments)

        for i in range(self.moments):
            xm[i] = torch.mul(torch.mean(torch.pow(x, i+1)), i+1)
            ym[i] = torch.mul(torch.mean(torch.pow(y, i+1)), i+1)
        loss = nn.MSELoss()
        return loss(xm, ym)


class AutoEncoder(nn.Module):

    def __init__(self, input, output=64):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, output),
        )

        self.decoder = nn.Sequential(
            nn.Linear(output, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, input),
        )

    def forward(self, data):
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return encoded, decoded



def create_embedding(wavelet_coefficients, d_in, d_out=64, n_moment=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.from_numpy(wavelet_coefficients)
    data.requires_grad_(True)
    data = data.to(device)

    model = AutoEncoder(d_in, d_out).to(device)
    criterion = MomentLoss(n_moment).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    epochs = 100
    for i in range(epochs):
        _, decoded = model(data)
        loss = criterion(decoded, data)
        if i % 10 == 0:
            print(i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    embedding = []
    for i in range(len(wavelet_coefficients)):
        encoded, _ = model(data[i, :])
        encoded = encoded.cpu()
        embedding.append(encoded.detach().numpy())
    return np.array(embedding)


if __name__ == '__main__':
    x = torch.ones(3, requires_grad=True)
    y = torch.tensor([2, 2, 2], dtype=torch.float32, requires_grad=True)
    loss = MomentLoss(5)
    out = loss(x, y)
    out.backward()
    print(x.grad)
    print(y.grad)


