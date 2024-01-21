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

    def forward(self, enc_input, global_mask, category_mask):
        enc_output = enc_input
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, global_mask, category_mask)

        return enc_output

class IDTransformer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, n_layer, dropout):
        super().__init__()

        self.position_encoding = nn.Linear(3, int(d_model / 2))
        self.block_id_embedding = nn.Embedding(253, int(d_model / 4))
        self.block_category_embedding = nn.Embedding(33 + 3, int(d_model / 4))

        self.encoder = TrnasformerEncoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                                 d_inner=d_hidden, dropout=dropout)

        self.id_decoding = nn.Linear(d_model, 250 + 3)

    def get_subsequent_mask(self, seq, diagonal):
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=diagonal)).bool()
        return subsequent_mask

    def get_category_mask(self, block_category_sequence, next_category_sequence):
        # a와 b에 새로운 차원을 추가하여 각 행을 확장
        a_expanded = block_category_sequence.unsqueeze(2)  # a의 shape가 (batch, seq, 1)이 됩니다.
        b_expanded = next_category_sequence.unsqueeze(1)  # b의 shape가 (batch, 1, seq)이 됩니다.

        # a_expanded와 b_expanded의 크기를 맞춰 비교
        mask = a_expanded == b_expanded  # 결과적으로 mask의 shape는 (batch, seq, seq)가 됩니다.

        # mask를 동일한 디바이스로 이동
        mask = mask.to(block_category_sequence.device)

        return mask

    def forward(self, position_sequence, block_id_sequence, block_category_sequence, next_category_sequences, pad_mask_sequence):
        category_mask = self.get_category_mask(block_category_sequence, next_category_sequences)

        position_sequence = self.position_encoding(position_sequence)
        block_id_sequence = self.block_id_embedding(block_id_sequence)
        block_category_sequence = self.block_category_embedding(block_category_sequence)

        pad_mask_sequence = pad_mask_sequence.unsqueeze(1)
        sub_mask_sequence = self.get_subsequent_mask(block_id_sequence[:, :, 0], diagonal=1)
        global_mask = pad_mask_sequence & sub_mask_sequence

        category_mask = category_mask & global_mask

        enc_input = torch.cat((position_sequence, block_id_sequence, block_category_sequence), dim=-1)
        enc_output = self.encoder(enc_input, global_mask, category_mask)

        id_output = self.id_decoding(enc_output)
        id_output = torch.softmax(id_output, dim=-1)

        return id_output