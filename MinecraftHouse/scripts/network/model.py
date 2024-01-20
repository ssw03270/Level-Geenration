import numpy as np
import torch
import torch.nn as nn
from layer import EncoderLayer
from sub_layer import MultiHeadAttention, PositionwiseFeedForward

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, seq_length):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(seq_length, d_hid))

    def _get_sinusoid_encoding_table(self, seq_length, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(seq_length)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()

class TrnasformerEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout):
        super(TrnasformerEncoder, self).__init__()

        # self.pos_enc = PositionalEncoding(d_model, 2048)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, enc_input, enc_mask):
        enc_output = enc_input # + self.pos_enc(enc_input)
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, enc_mask)

        return enc_output

class Transformer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, n_layer, dropout):
        super().__init__()

        self.position_encoding = nn.Linear(3, int(d_model / 2))
        self.block_id_embedding = nn.Embedding(253, int(d_model / 4))
        self.block_semantic_embedding = nn.Embedding(33 + 3, int(d_model / 4))

        self.encoder = TrnasformerEncoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                          d_inner=d_hidden, dropout=dropout)
        self.attention = MultiHeadAttention(n_head=1, d_model=d_model, dropout=0.0)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_hidden, dropout=dropout)
        self.dir_decoding = nn.Linear(d_model, 26)
        self.id_decoding = nn.Linear(d_model, 253)
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

        enc_input = torch.cat((position_sequence + block_id_sequence + block_semantic_sequence), dim=-1)
        enc_output = self.encoder(enc_input, mask)

        sub_output, parent_output = self.attention(enc_output, enc_output, enc_output, mask)
        sub_output = self.ffn(sub_output)

        dir_output = self.dir_decoding(sub_output)
        dir_output = torch.softmax(dir_output, dim=-1)

        id_output = self.id_decoding(sub_output)
        id_output = torch.softmax(id_output, dim=-1)

        category_output = self.category_decoding(sub_output)
        category_output = torch.softmax(category_output, dim=-1)

        parent_output = parent_output.squeeze()

        return parent_output, dir_output, id_output, category_output