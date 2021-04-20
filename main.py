import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from layer import GCN

dropout=0.5
epochs=200
hidden_dim=16 
lr=0.01
weight_decay=0.0005


torch.manual_seed(0)

adj, features, labels = load_data()

idx_train = range(200)
idx_val = range(200, 500)
idx_test = range(500, 1500)


model = GCN(features.shape[1],
            hidden_dim,
            labels.max().item() + 1,
            dropout)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

"Training the model"

for epoch in range(epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()

#     print('features/adj',features,adj)

    output = model(features, adj)
    
#     print('output',output, output.shape)

#     print(output[idx_train], labels[idx_train])
#     print(output[idx_train].shape, labels[idx_train].shape)

    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])

    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))

"Testing the model"


model.eval()
output = model(features, adj)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])

print("Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()))