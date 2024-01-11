import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from datetime import datetime

import numpy as np
import random
from tqdm import tqdm

from model import Transformer, get_pad_mask
from dataloader import ModelNetDataset

d_model = 512
d_hidden = 2048
n_head = 8
n_layer = 6
dropout = 0.1

checkpoint_epoch = 10
checkpoint = torch.load("./models/transformer_epoch_" + str(checkpoint_epoch) + ".pt")

device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
transformer = Transformer(d_model, d_hidden, n_head, n_layer, dropout).to(device=device)
transformer.load_state_dict(checkpoint['model_state_dict'])

test_dataset = ModelNetDataset(data_type='val')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

with torch.no_grad():
    # Iterate over batches
    for data in tqdm(test_dataloader):
        # Get the source and target sequences from the batch
        x_seq, y_seq, edge = data
        x_seq = x_seq.to(device=device)
        y_seq = y_seq.to(device=device)
        edge = edge.to(device=device)

        # Get the model's predictions
        output = transformer(x_seq, y_seq.detach(), edge)

        # Compute the losses using the generated sequence
        pad_mask = get_pad_mask(y_seq.detach(), -1)
        loss = self.cross_entropy_loss(output.detach(), y_seq.detach(), pad_mask.detach())

        loss_sum += loss.detach()
        true_sum, problem_sum = self.get_accuracy(output.detach(), y_seq.detach(), pad_mask.detach())
        true_sums += true_sum
        problem_sums += problem_sum