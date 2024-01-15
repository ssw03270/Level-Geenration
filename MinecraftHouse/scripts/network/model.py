import torch
import torch.nn as nn
from layer import EncoderLayer
from sub_layer import MultiHeadAttention, PositionwiseFeedForward

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

class Transformer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, n_layer, dropout):
        super().__init__()

        self.position_encoding = nn.Linear(3, d_model)
        self.block_id_embedding = nn.Embedding(253, d_model)
        self.block_semantic_embedding = nn.Embedding(701, d_model)

        self.encoder = TrnasformerEncoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                          d_inner=d_hidden, dropout=dropout)
        self.attention = MultiHeadAttention(n_head=1, d_model=d_model, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_hidden, dropout=dropout)
        self.dir_decoding = nn.Linear(d_model, 26)

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

        enc_input = position_sequence + block_id_sequence + block_semantic_sequence
        enc_output = self.encoder(enc_input, mask)

        dir_output, parent_output = self.attention(enc_output, enc_output, enc_output, mask)

        dir_output = self.ffn(dir_output)
        dir_output = self.dir_decoding(dir_output)
        dir_output = torch.softmax(dir_output, dim=-1)

        parent_output = parent_output.squeeze()

        return parent_output, dir_output