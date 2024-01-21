import numpy as np
import torch
import torch.nn as nn
from layer import EncoderLayer, DecoderLayer
from sub_layer import MultiHeadAttention
from transformers import BertModel


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
        enc_output = enc_input  # + self.pos_enc(enc_input)
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, enc_mask)

        return enc_output


class TransformerDecoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout):
        super(TransformerDecoder, self).__init__()

        self.position_encoding = nn.Linear(3, int(d_model / 2))
        self.id_embedding = nn.Embedding(253, int(d_model / 4))
        self.category_embedding = nn.Embedding(33 + 3, int(d_model / 4))

        # self.pos_enc = PositionalEncoding(d_model, 2048)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, enc_input, position_sequence, id_sequence, category_sequence, dec_mask, enc_mask):
        position_sequence = self.position_encoding(position_sequence)
        id_sequence = self.id_embedding(id_sequence)
        category_sequence = self.category_embedding(category_sequence)

        dec_input = torch.cat((position_sequence, id_sequence, category_sequence), dim=-1)
        dec_output = self.dropout(dec_input)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(enc_input, dec_output, dec_mask, enc_mask)

        return dec_output


class Transformer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, n_layer, dropout):
        super().__init__()
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')

        self.category_decoder = TransformerDecoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                                   d_inner=d_hidden, dropout=dropout)
        self.id_decoder = TransformerDecoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                             d_inner=d_hidden, dropout=dropout)
        self.parent_decoder = TransformerDecoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                                 d_inner=d_hidden, dropout=dropout)
        self.direction_decoder = TransformerDecoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                                    d_inner=d_hidden, dropout=dropout)

        self.category_decoding = nn.Linear(d_model, 33 + 3)
        self.id_decoding = nn.Linear(d_model, 253)
        self.parent_decoding = MultiHeadAttention(n_head=1, d_model=d_model, dropout=0.0)
        self.direction_decoding = nn.Linear(d_model, 26)

    def get_index_mask(self, category_sequence, next_category_sequence):
        a_expanded = category_sequence.unsqueeze(2)
        b_expanded = next_category_sequence.unsqueeze(1)

        mask = a_expanded == b_expanded
        mask = mask.to(category_sequence.device)
        return mask

    def get_subsequent_mask(self, seq, diagonal):
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=diagonal)).bool()
        return subsequent_mask

    def select_mask_with_indices(self, mask, indices):
        batch_indices = torch.arange(mask.shape[0]).unsqueeze(1).expand_as(indices)
        selected_mask = mask[batch_indices, indices, :]
        return selected_mask

    def calculate_distances_with_mask(self, tensor, distance=3):
        tensor_expanded_1 = tensor.unsqueeze(2)
        tensor_expanded_2 = tensor.unsqueeze(1)

        differences = tensor_expanded_1 - tensor_expanded_2
        distances = torch.sqrt(torch.sum(differences ** 2, dim=-1))

        mask = distances <= distance
        mask.to(tensor.device)
        return mask

    def forward(self, text_sequence, position_sequence, id_sequence, category_sequence, real_position_sequence,
                pad_mask_sequence):
        bert_input = text_sequence['input_ids']
        bert_mask = text_sequence['attention_mask']
        bert_output = self.bert_encoder(bert_input, attention_mask=bert_mask)
        bert_output = bert_output['last_hidden_state']
        bert_mask = bert_mask.unsqueeze(1)

        pad_mask_sequence = pad_mask_sequence.unsqueeze(1)
        sub_mask_sequence = self.get_subsequent_mask(id_sequence, diagonal=1)
        global_mask = pad_mask_sequence & sub_mask_sequence

        category_output = self.category_decoder(bert_output, position_sequence, id_sequence, category_sequence, global_mask, bert_mask)
        decoded_category = self.category_decoding(category_output)
        decoded_category = torch.softmax(decoded_category, dim=-1)
        decoded_category_index = torch.argmax(decoded_category, dim=-1)
        category_mask = self.get_index_mask(category_sequence, decoded_category_index)
        category_mask = category_mask & global_mask

        id_output = self.id_decoder(category_output, position_sequence, id_sequence, category_sequence, category_mask, global_mask)
        decoded_id = self.id_decoding(id_output)
        decoded_id = torch.softmax(decoded_id, dim=-1)
        decoded_id_index = torch.argmax(decoded_id, dim=-1)
        id_mask = self.get_index_mask(id_sequence, decoded_id_index)
        id_mask = id_mask & global_mask

        parent_output = self.parent_decoder(id_output, position_sequence, id_sequence, category_sequence, id_mask, global_mask)
        _, decoded_parent = self.parent_decoding(parent_output, parent_output, parent_output, mask=global_mask)
        decoded_parent_index = torch.argmax(decoded_parent, dim=-1)
        local_mask = self.calculate_distances_with_mask(real_position_sequence, distance=3)
        local_mask = self.select_mask_with_indices(local_mask, decoded_parent_index)
        local_mask = local_mask & global_mask

        direction_output = self.direction_decoder(parent_output, position_sequence, id_sequence, category_sequence, local_mask, global_mask)
        decoded_direction = self.direction_decoding(direction_output)
        decoded_direction = torch.softmax(decoded_direction, dim=-1)

        return decoded_category, decoded_id, decoded_parent, decoded_direction