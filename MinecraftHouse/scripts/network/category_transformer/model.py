import numpy as np
import torch
import torch.nn as nn
from layer import EncoderLayer

class TrnasformerEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout):
        super(TrnasformerEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, enc_input, enc_mask):
        enc_output = enc_input
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, enc_mask)

        return enc_output

class CategoryTransformer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, n_layer, dropout):
        super().__init__()

        self.position_encoding = nn.Linear(3, int(d_model / 2))
        self.block_id_embedding = nn.Embedding(253, int(d_model / 4))
        self.block_semantic_embedding = nn.Embedding(33 + 3, int(d_model / 4))

        self.encoder = TrnasformerEncoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                                 d_inner=d_hidden, dropout=dropout)

        self.category_decoding = nn.Linear(d_model, 33 + 3)

    def get_subsequent_mask(self, seq, diagonal):
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=diagonal)).bool()
        return subsequent_mask

    def forward(self, position_sequence, block_id_sequence, block_semantic_sequence, pad_mask_sequence):
        position_sequence = self.position_encoding(position_sequence)
        block_id_sequence = self.block_id_embedding(block_id_sequence)
        block_semantic_sequence = self.block_semantic_embedding(block_semantic_sequence)

        pad_mask_sequence = pad_mask_sequence.unsqueeze(1)
        sub_mask_sequence = self.get_subsequent_mask(block_id_sequence[:, :, 0], diagonal=1)
        mask = pad_mask_sequence & sub_mask_sequence

        enc_input = torch.cat((position_sequence, block_id_sequence, block_semantic_sequence), dim=-1)
        enc_output = self.encoder(enc_input, mask)

        category_output = self.category_decoding(enc_output)
        category_output = torch.softmax(category_output, dim=-1)

        return category_output