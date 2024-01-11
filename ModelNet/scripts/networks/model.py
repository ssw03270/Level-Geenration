import torch
import torch.nn as nn
from layer import EncoderLayer, DecoderLayer

def get_pad_mask(seq, pad_idx):
    mask = (seq != pad_idx)
    return mask

class TrnasformerEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout):
        super(TrnasformerEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, enc_input):
        enc_output = self.dropout(enc_input)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output

class TransformerDecoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
           DecoderLayer(d_model, d_inner, n_head, dropout)
           for _ in range(n_layer)
        ])
        self.d_model = d_model

    def forward(self, dec_input, enc_output, dec_mask):
        dec_output = self.dropout(dec_input)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_output, dec_mask)

        return dec_output

class Transformer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, n_layer, dropout):
        super().__init__()

        self.voxel_encoding = nn.Linear(3, d_model)
        self.dir_encoding = nn.Embedding(6, d_model)

        self.encoder = TrnasformerEncoder(n_layer=n_layer, n_head=n_head, d_model=d_model, d_inner=d_hidden, dropout=dropout)
        self.decoder = TransformerDecoder(n_layer=n_layer, n_head=n_head, d_model=d_model, d_inner=d_hidden, dropout=dropout)

        self.decoding = nn.Linear(d_model, 6)

    def get_subsequent_mask(self, seq):
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    def forward(self, x_seq, y_seq, edge):
        dir = torch.tensor([0, 1, 2, 3, 4, 5])
        dir = dir.unsqueeze(0).repeat(x_seq.shape[0], 1).to(x_seq.device)
        dir = self.dir_encoding(dir)
        dir = self.encoder(dir)

        x_seq = self.voxel_encoding(x_seq)
        pad_mask = get_pad_mask(y_seq[:, :, 0], -1).unsqueeze(1)
        sub_mask = self.get_subsequent_mask(x_seq[:, :, 0])
        dec_mask = pad_mask & sub_mask
        output = self.decoder(x_seq, dir, dec_mask)

        output = self.decoding(output)
        output = torch.sigmoid(output)

        return output