import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np

def attention_mask(batch):
    """batch 벡터를 기반으로 마스크 행렬을 생성합니다."""
    mask = torch.eq(batch[:, None], batch[None, :])  # 같은 배치 내의 노드들에 대해서만 True
    return mask

class MaskedGlobalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, batch):
        mask = attention_mask(batch).to(x.device)  # 마스크 생성

        # attn_mask 대신 key_padding_mask를 사용할 경우, 마스크 반전이 필요하지 않음
        attn_output, _ = self.attention(x, x, x, key_padding_mask=~mask)

        out = self.linear(attn_output)
        return out

class Conv3DBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Conv3DBNReLU, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class LocalEncoder(nn.Module):
    def __init__(self, d_model, batch_size, grid_size):
        super(LocalEncoder, self).__init__()

        self.d_model = d_model
        self.batch_size = batch_size
        self.grid_size = grid_size

        self.id_embedding = nn.Embedding(300, d_model)

        self.layer1 = Conv3DBNReLU(d_model, d_model, kernel_size=3)
        self.layer2 = Conv3DBNReLU(d_model, d_model, kernel_size=3)
        self.layer3 = Conv3DBNReLU(d_model, d_model, kernel_size=3)
        self.layer4 = Conv3DBNReLU(d_model, d_model, kernel_size=3)

    def forward(self, x):
        x = x.reshape(self.batch_size, -1)
        x = torch.relu(self.id_embedding(x))
        x = x.reshape(self.batch_size, self.grid_size, self.grid_size, self.grid_size, -1)
        x = x.permute(0, 4, 1, 2, 3)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.permute(0, 2, 3, 4, 1)
        return x

class GraphEncoder(nn.Module):
    def __init__(self, n_layer, d_model, batch_size):
        super(GraphEncoder, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model

        self.position_encoding = nn.Linear(3, d_model)
        self.id_embedding = nn.Embedding(300, d_model)

        self.node_encoding = nn.Linear(d_model * 2, d_model)

        self.conv_gcn = torch_geometric.nn.GCNConv
        self.global_pool = torch_geometric.nn.global_max_pool

        self.layer_stack = nn.ModuleList()
        for _ in range(self.n_layer):
            self.layer_stack.append(self.conv_gcn(d_model, d_model))
            self.layer_stack.append(MaskedGlobalAttention(d_model, d_model))

        self.aggregate = nn.Linear(int(d_model * (1.0 + self.n_layer)), d_model)

    def positional_encoding(self, data):
        sinusoid_tables = []
        for seq_length in data.each_num_nodes:
            def get_position_angle_vec(position):
                return [position / np.power(10000, 2 * (hid_j // 2) / self.d_model) for hid_j in range(self.d_model)]

            sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(seq_length)])
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

            if len(sinusoid_tables) != 0:
                sinusoid_tables = np.concatenate((sinusoid_tables, sinusoid_table), axis=0)
            else:
                sinusoid_tables = sinusoid_table

        return torch.Tensor(sinusoid_tables).to(device=data.id_feature.device)

    def forward(self, data):
        edge_index = data.edge_index

        position_feature = data.position_feature
        id_feature = data.id_feature

        position_feature = F.relu(self.position_encoding(position_feature))
        id_feature = F.relu(self.id_embedding(id_feature)).squeeze(1)

        node_feature = F.relu(self.node_encoding(torch.cat([position_feature, id_feature], dim=1)))
        node_feature += self.positional_encoding(data)

        n_embed_t = node_feature
        g_embed = self.global_pool(n_embed_t, data.batch)

        for layer_idx in range(0, len(self.layer_stack), 2):
            n_embed_t = F.relu(self.layer_stack[layer_idx](n_embed_t, edge_index))
            n_embed_t = F.relu(self.layer_stack[layer_idx + 1](n_embed_t, data.batch))
            g_embed_t = self.global_pool(n_embed_t, data.batch)

            g_embed = torch.cat((g_embed, g_embed_t), dim=1)

        latent = self.aggregate(g_embed)

        return latent

class GenerativeModel(nn.Module):
    def __init__(self, n_layer, d_model, batch_size):
        super(GenerativeModel, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model
        self.batch_size = batch_size
        self.grid_size = 7

        self.local_encoder = LocalEncoder(d_model, batch_size, self.grid_size)
        self.graph_encoder = GraphEncoder(n_layer, d_model, batch_size)

        self.conv = Conv3DBNReLU(d_model * 2, d_model, kernel_size=1, padding=0)

        self.pos_conv = Conv3DBNReLU(d_model, 1, kernel_size=1, padding=0)

        self.id_fc = nn.Linear(d_model, 300)

    def forward(self, data):
        enc_local = self.local_encoder(data.local_grid)
        enc_graph = self.graph_encoder(data)

        enc_graph = enc_graph.view(self.batch_size, 1, 1, 1, self.d_model).expand(-1, self.grid_size, self.grid_size, self.grid_size, -1)

        enc_output = torch.cat((enc_local, enc_graph), dim=-1)
        enc_output = enc_output.permute(0, 4, 1, 2, 3)
        enc_output = self.conv(enc_output)

        pos_output = self.pos_conv(enc_output).squeeze()

        if self.training:
            id_idx = torch.flatten(data.gt_grid, 1)
            id_idx = torch.argmax(id_idx, dim=-1)
        else:
            id_idx = torch.flatten(pos_output, 1)
            id_idx = torch.argmax(id_idx, dim=-1)

        pos_output = torch.flatten(pos_output, 1)
        pos_output = torch.softmax(pos_output, dim=-1)

        id_output = self.id_fc(enc_output[:, id_idx])
        id_output = torch.softmax(id_output, dim=-1)

        return pos_output, id_output