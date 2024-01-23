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

        self.dir_embedding = nn.Embedding(27, int(d_model / 4))
        self.position_encoding = nn.Linear(3, int(d_model / 4))
        self.id_embedding = nn.Embedding(253, int(d_model / 4))
        self.category_embedding = nn.Embedding(33 + 3, int(d_model / 4))

        # self.pos_enc = PositionalEncoding(d_model, 2048)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout,
                         use_additional_global_attn=use_additional_global_attn)
            for _ in range(n_layer)
        ])

    def forward(self, enc_input, direction_sequence, position_sequence, id_sequence, category_sequence,
                enc_mask, dec_mask=None, local_mask=None, category_mask=None, id_mask=None):
        direction_sequence = self.dir_embedding(direction_sequence)
        position_sequence = self.position_encoding(position_sequence)
        id_sequence = self.id_embedding(id_sequence)
        category_sequence = self.category_embedding(category_sequence)

        dec_input = torch.cat((direction_sequence, position_sequence, id_sequence, category_sequence), dim=-1)
        dec_output = self.dropout(dec_input)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(enc_input, dec_output, enc_mask,
                                   dec_mask=dec_mask, local_mask=local_mask,
                                   category_mask=category_mask, id_mask=id_mask)

        return dec_output


class Transformer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, n_layer, dropout):
        super().__init__()
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_encoder.parameters():
            param.requires_grad = False

        self.bert_encoding = nn.Linear(768, d_model)

        self.parent_decoder = TransformerDecoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                                 d_inner=d_hidden, dropout=dropout)
        self.block_decoder = TransformerDecoder(n_layer=n_layer, n_head=n_head, d_model=d_model,
                                                d_inner=d_hidden, dropout=dropout, use_additional_global_attn=True)

        self.parent_decoding = MultiHeadAttention(n_head=1, d_model=d_model, dropout=0.0)

        self.category_decoding = nn.Linear(d_model, 33 + 3)
        self.id_decoding = nn.Linear(d_model, 253)
        self.direction_decoding = nn.Linear(d_model, 27)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)

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

    def get_voxel_with_mask(self, real_position_sequence, category_sequence, id_sequence, mask, distance=3):
        batch_size = real_position_sequence.shape[0]
        seq_length = real_position_sequence.shape[1]
        device = real_position_sequence.device

        real_position_sequence = real_position_sequence.cpu().detach().numpy()
        category_sequence = category_sequence.cpu().detach().numpy()
        id_sequence = id_sequence.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()

        voxel_size = 2 * distance + 1
        voxel_grid = np.zeros((batch_size, seq_length, voxel_size, voxel_size, voxel_size, 2))
        for b in range(batch_size):
            for s in range(seq_length):
                center_position = real_position_sequence[b, s]
                cx, cy, cz = center_position

                for m in range(seq_length):
                    if mask[b, s, m]:
                        position = real_position_sequence[b, m]
                        category = category_sequence[b, m]
                        id = id_sequence[b, m]

                        x, y, z = position
                        x, y, z = x - cx + distance, y - cy + distance, y - cy + distance
                        if not 0 <= x < voxel_size and not 0 <= y < voxel_size and not 0 <= z < voxel_size:
                            print(center_position, position)
                        voxel_grid[b, s, x, y, z] = [category, id]

        voxel_grid = torch.Tensor(voxel_grid, dtype=torch.long).to(device)
        return voxel_grid

    def forward(self, text_sequence, direction_sequence, position_sequence, id_sequence, category_sequence,
                real_position_sequence, pad_mask_sequence, next_category_sequence, next_id_sequence,
                next_parent_sequence):

        batch_size = position_sequence.shape[0]
        seq_length = position_sequence.shape[1]

        bert_input = text_sequence['input_ids']
        bert_mask = text_sequence['attention_mask']
        bert_output = self.bert_encoder(bert_input, attention_mask=bert_mask)
        bert_output = bert_output['last_hidden_state']
        bert_output = self.bert_encoding(bert_output)
        bert_mask = bert_mask.unsqueeze(1)

        pad_mask_sequence = pad_mask_sequence.unsqueeze(1)
        sub_mask_sequence = self.get_subsequent_mask(id_sequence, diagonal=1)
        global_mask = pad_mask_sequence & sub_mask_sequence

        parent_output = self.parent_decoder(bert_output, direction_sequence, position_sequence, id_sequence, category_sequence,
                                            enc_mask=bert_mask, dec_mask=global_mask)
        _, decoded_parent = self.parent_decoding(parent_output, parent_output, parent_output, mask=global_mask)

        if self.training:
            decoded_parent_index = next_parent_sequence
        else:
            decoded_parent_index = torch.argmax(decoded_parent, dim=-1)

        local_mask = self.calculate_distances_with_mask(real_position_sequence, distance=3)
        local_mask = self.select_mask_with_indices(local_mask, decoded_parent_index)
        local_mask = local_mask & global_mask

        print(real_position_sequence[0])
        print(category_sequence[0])
        print(id_sequence[0])
        print(self.get_voxel_with_mask(real_position_sequence, category_sequence, id_sequence, local_mask, distance=3)[0, 0])
        print('----')
        category_mask = self.get_index_mask(category_sequence, category_sequence)
        category_mask = self.select_mask_with_indices(category_mask, decoded_parent_index)
        category_mask = category_mask & global_mask

        id_mask = self.get_index_mask(id_sequence, id_sequence)
        id_mask = self.select_mask_with_indices(id_mask, decoded_parent_index)
        id_mask = id_mask & global_mask

        block_output = self.block_decoder(bert_output, direction_sequence, position_sequence, id_sequence, category_sequence,
                                          enc_mask=bert_mask, category_mask=category_mask,
                                          id_mask=id_mask, local_mask=local_mask)

        decoded_category = self.category_decoding(block_output)
        decoded_category = torch.softmax(decoded_category, dim=-1)

        decoded_id = self.id_decoding(block_output)
        decoded_id = torch.softmax(decoded_id, dim=-1)

        decoded_direction = self.direction_decoding(block_output)
        decoded_direction = decoded_direction.view(-1, 3, 3, 3).unsqueeze(1)

        decoded_direction = self.conv1(decoded_direction)
        decoded_direction = torch.relu(decoded_direction)
        decoded_direction = self.conv2(decoded_direction)
        decoded_direction = torch.relu(decoded_direction)
        decoded_direction = self.conv3(decoded_direction)
        decoded_direction = torch.relu(decoded_direction)

        decoded_direction = decoded_direction.view(batch_size, seq_length, 27)
        mask = torch.zeros_like(decoded_direction, dtype=torch.bool)
        mask[:, :, 13] = True
        decoded_direction = decoded_direction.masked_fill(mask, -1e9)

        decoded_direction = torch.softmax(decoded_direction, dim=-1)

        return decoded_category, decoded_id, decoded_parent.squeeze(0), decoded_direction
