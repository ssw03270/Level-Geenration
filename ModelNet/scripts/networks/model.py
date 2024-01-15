import torch
import torch.nn as nn
from layer import EncoderLayer, DecoderLayer

class TrnasformerEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout):
        super(TrnasformerEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, enc_input, enc_mask):
        enc_output = self.dropout(enc_input)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, enc_mask)

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

        self.parent_embedding = nn.Embedding(256, d_model)
        self.child_encoding = nn.Linear(3, d_model)

        self.parent_encoder = TrnasformerEncoder(n_layer=n_layer, n_head=n_head, d_model=d_model, d_inner=d_hidden, dropout=dropout)
        self.parent_decoder = TransformerDecoder(n_layer=n_layer, n_head=n_head, d_model=d_model, d_inner=d_hidden, dropout=dropout)

        self.child_encoder = TrnasformerEncoder(n_layer=n_layer, n_head=n_head, d_model=d_model, d_inner=d_hidden, dropout=dropout)
        self.child_decoder = TransformerDecoder(n_layer=n_layer, n_head=n_head, d_model=d_model, d_inner=d_hidden, dropout=dropout)

        self.parent_decoding = nn.Linear(d_model, 256)
        self.dir_decoding = nn.Linear(d_model, 6)

    def get_subsequent_mask(self, seq, diagonal):
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=diagonal)).bool()
        return subsequent_mask

    def forward(self, parent_seq, child_seq, pad_mask):
        parent_embedd = self.parent_embedding(parent_seq)
        child_encode = self.child_encoding(child_seq)

        pad_mask = pad_mask.unsqueeze(1)

        child_mask = pad_mask & self.get_subsequent_mask(parent_seq, diagonal=1)
        parent_mask = pad_mask & self.get_subsequent_mask(parent_seq, diagonal=1)
        parent_encoder_output = self.parent_encoder(child_encode, child_mask)
        parent_decoder_output = self.parent_decoder(parent_embedd, parent_encoder_output, parent_mask)

        parent_mask = pad_mask & self.get_subsequent_mask(parent_seq, diagonal=2)
        child_mask = pad_mask & self.get_subsequent_mask(parent_seq, diagonal=1)
        child_encoder_output = self.parent_encoder(parent_embedd, parent_mask)
        child_decoder_output = self.parent_decoder(child_encode, child_encoder_output, child_mask)

        output_parent = self.parent_decoding(parent_decoder_output)
        output_dir = self.dir_decoding(child_decoder_output)

        output_parent = torch.softmax(output_parent, dim=-1)
        output_dir = torch.softmax(output_dir, dim=-1)

        return output_parent, output_dir