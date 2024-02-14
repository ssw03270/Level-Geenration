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
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout, use_additional_global_attn=False):
        super(TransformerDecoder, self).__init__()

        self.position_encoding = nn.Linear(3, int(d_model / 2))
        self.id_embedding = nn.Embedding(257, int(d_model / 4))
        self.category_embedding = nn.Embedding(37, int(d_model / 4))

        self.pos_enc = PositionalEncoding(d_model, 2048)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout,
                         use_additional_global_attn=use_additional_global_attn)
            for _ in range(n_layer)
        ])

    def forward(self, enc_input, position_sequence, id_sequence, category_sequence,
                enc_mask, dec_mask=None, local_mask=None, category_mask=None, id_mask=None):
        position_sequence = self.position_encoding(position_sequence)
        id_sequence = self.id_embedding(id_sequence)
        category_sequence = self.category_embedding(category_sequence)

        dec_input = torch.cat((position_sequence, id_sequence, category_sequence), dim=-1)
        dec_output = dec_input + self.pos_enc(dec_input)
        dec_output = self.dropout(dec_output)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(enc_input, dec_output, enc_mask=enc_mask,
                                   dec_mask=dec_mask, local_mask=local_mask,
                                   category_mask=category_mask, id_mask=id_mask)

        return dec_output

class Transformer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, n_layer, dropout):
        super().__init__()
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoding = nn.Linear(768, d_model)

        self.block_decoder = TransformerDecoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                                d_inner=d_hidden, dropout=dropout, use_additional_global_attn=False)

        self.category_decoding = nn.Linear(d_model, d_model)
        self.category_fc = nn.Linear(d_model, 37)

        self.id_decoding = nn.Linear(d_model, d_model)
        self.id_fc = nn.Linear(d_model, 257)

        self.position_decoding = nn.Linear(d_model, d_model)
        self.position_fc = nn.Linear(d_model, 3)

    def get_subsequent_mask(self, seq, diagonal):
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=diagonal)).bool()
        return subsequent_mask

    def forward(self, text_sequence, position_sequence, id_sequence, category_sequence,
                pad_mask_sequence):
        bert_input = text_sequence['input_ids']
        bert_mask = text_sequence['attention_mask']
        bert_output = self.bert_encoder(bert_input, attention_mask=bert_mask)
        bert_output = bert_output['last_hidden_state']
        bert_output = self.bert_encoding(bert_output)
        bert_mask = bert_mask.unsqueeze(1)

        pad_mask_sequence = pad_mask_sequence.unsqueeze(1)
        sub_mask_sequence = self.get_subsequent_mask(id_sequence, diagonal=1)
        global_mask = pad_mask_sequence & sub_mask_sequence

        dec_output = self.block_decoder(bert_output, position_sequence, id_sequence, category_sequence,
                                        enc_mask=bert_mask, dec_mask=global_mask)

        decoded_category = self.category_decoding(dec_output)
        decoded_category = torch.relu(decoded_category)
        decoded_category = self.category_fc(decoded_category)
        decoded_category = torch.softmax(decoded_category, dim=-1)

        decoded_id = self.id_decoding(dec_output)
        decoded_id = torch.relu(decoded_id)
        decoded_id = self.id_fc(decoded_id)
        decoded_id = torch.softmax(decoded_id, dim=-1)

        decoded_position = self.position_decoding(dec_output)
        decoded_position = torch.relu(decoded_position)
        decoded_position = self.position_fc(decoded_position)

        return decoded_category, decoded_id, decoded_position
