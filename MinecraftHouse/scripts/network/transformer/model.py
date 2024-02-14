import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np
from layer import EncoderLayer

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

class Conv3DBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Conv3DBNReLU, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class LocalEncoder(nn.Module):
    def __init__(self, d_model, grid_size):
        super(LocalEncoder, self).__init__()

        self.d_model = d_model
        self.grid_size = grid_size

        self.id_embedding = nn.Embedding(300, d_model)

        self.layer1 = Conv3DBNReLU(d_model, d_model, kernel_size=3)
        self.layer2 = Conv3DBNReLU(d_model, d_model, kernel_size=3)
        self.layer3 = Conv3DBNReLU(d_model, d_model, kernel_size=3)
        self.layer4 = Conv3DBNReLU(d_model, d_model, kernel_size=3)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.reshape(batch_size, -1)
        x = torch.relu(self.id_embedding(x))
        x = x.reshape(batch_size, self.grid_size, self.grid_size, self.grid_size, -1)
        x = x.permute(0, 4, 1, 2, 3)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.permute(0, 2, 3, 4, 1)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, d_model, n_head=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.position_encoding = nn.Linear(3, d_model)
        self.id_embedding = nn.Embedding(300, d_model)

        self.pos_enc = PositionalEncoding(d_model, 4000)
        self.dropout = nn.Dropout(dropout)

        self.fc_layer = nn.Linear(d_model * 3, d_model)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_model, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, position_features, id_features, pad_mask):
        position_feature = F.relu(self.position_encoding(position_features))
        id_feature = F.relu(self.id_embedding(id_features)).squeeze(1)

        enc_input = torch.cat([position_feature, id_feature, self.pos_enc(id_features)], dim=-1)
        enc_input = self.fc_layer(enc_input)

        enc_output = self.dropout(enc_input)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, pad_mask)

        return enc_output[:, 0]

class GenerativeModel(nn.Module):
    def __init__(self, n_layer, d_model):
        super(GenerativeModel, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model
        self.grid_size = 7

        self.local_encoder = LocalEncoder(d_model, self.grid_size)
        self.transformer_encoder = TransformerEncoder(n_layer, d_model)

        self.conv = Conv3DBNReLU(d_model * 2, d_model, kernel_size=1, padding=0)

        self.pos_conv = Conv3DBNReLU(d_model, 1, kernel_size=1, padding=0)

        self.id_fc = nn.Linear(d_model, 300)

    def forward(self, local_grids, position_features, id_features, pad_mask, gt_grid):
        enc_local = self.local_encoder(local_grids)
        enc_transformer, attn = self.transformer_encoder(position_features, id_features, pad_mask)

        batch_size = enc_local.shape[0]
        enc_transformer = enc_transformer.view(batch_size, 1, 1, 1, self.d_model).expand(-1, self.grid_size, self.grid_size, self.grid_size, -1)

        enc_output = torch.cat((enc_local, enc_transformer), dim=-1)
        enc_output = enc_output.permute(0, 4, 1, 2, 3).contiguous()
        enc_output = self.conv(enc_output)

        pos_output = self.pos_conv(enc_output).squeeze()

        torch.autograd.set_detect_anomaly(True)
        if self.training:
            id_idx = gt_grid.reshape(batch_size, -1)
            id_idx = torch.argmax(id_idx, dim=-1)
        else:
            id_idx = pos_output.reshape(batch_size, -1)
            id_idx = torch.argmax(id_idx, dim=-1)

        pos_output = pos_output.reshape(batch_size, -1)
        pos_output = torch.softmax(pos_output, dim=-1)

        id_output = self.id_fc(enc_output.reshape(batch_size, self.d_model, -1)[:, :, id_idx])
        id_output = torch.softmax(id_output, dim=-1)

        if self.training:
            return pos_output, id_output
        else:
            return pos_output, id_output, attn
