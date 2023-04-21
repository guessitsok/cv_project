import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms as tfs
from torchvision.datasets import MNIST

from PIL import Image


data_tfs = tfs.Compose([
    tfs.Resize((28, 28)),
    tfs.ToTensor(),
    tfs.Normalize((0.5), (0.5))
])

root = './'
train_dataset = MNIST(root, train=True, transform=data_tfs, download=True)
valid_dataset = MNIST(root, train=False, transform=data_tfs, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=6,
                              shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=19,
                              shuffle=False, num_workers=4)


activation = nn.ELU()

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    activation,
    nn.Linear(128, 128),
    activation,
    nn.Linear(128, 10)
)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

loaders = {'train': train_dataloader,
           'valid': valid_dataloader}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_epochs = 50
accuracy = {'train': [],
            'valid': []}
for epoch in range(max_epochs):
    for k, dataloader in loaders.items():
        epoch_correct = 0
        epoch_all = 0
        for x_batch, y_batch in dataloader:
            if k == 'train':
                model.train()
                optimizer.zero_grad()
                output = model(x_batch)
                loss = loss_func(output, y_batch)
                loss.backward()
                optimizer.step()
            else:
                model.eval()
                with torch.no_grad():
                    output = model(x_batch)

            predictions = output.argmax(-1)
            correct = len([pred for i,
                          pred in enumerate(predictions) if pred == y_batch[i]])
            _all = y_batch.shape[0]
            epoch_correct += correct
            epoch_all += _all
        if k == 'train':
            print(f"Epoch: {epoch + 1}")
        print(f"Loader: {k}. Accuracy: {epoch_correct / epoch_all}")
        accuracy[k].append(epoch_correct / epoch_all)

# torch.save(model, '../models/MNIST_model.pt')
