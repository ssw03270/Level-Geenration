import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np

def attention_mask(batch):
    """batch 벡터를 기반으로 마스크 행렬을 생성합니다."""
    mask = torch.eq(batch[:, None], batch[None, :])  # 같은 배치 내의 노드들에 대해서만 True
    return mask.unsqueeze(0)

class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward sub-layer.

    Args:
    - d_in (int): Input dimensionality.
    - d_hid (int): Dimensionality of the hidden layer.
    - dropout (float, optional): Dropout rate. Default is 0.1.
    """

    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_inner)
        self.w2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w1(x)
        x = torch.relu(x)
        x = self.w2(x)
        x += residual
        x = self.layer_norm(x)

        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=4, d_model=512, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_model // n_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, batch):
        mask = attention_mask(batch).to(x.device)  # 마스크 생성
        q = x
        k = x
        v = x

        d_k, d_v, n_head = self.d_model // self.n_head, self.d_model // self.n_head, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        q = self.w_q(q).view(1, sz_b, n_head, d_k)
        k = self.w_k(k).view(1, sz_b, n_head, d_k)
        v = self.w_v(v).view(1, sz_b, n_head, d_k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, -1)
        q = self.fc(q)
        q = self.dropout(q)
        q += residual
        q = self.layer_norm(q)

        return q, attn
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
        batch_size = x.shape[0] // self.grid_size

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

class GraphEncoder(nn.Module):
    def __init__(self, n_layer, d_model):
        super(GraphEncoder, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model

        self.position_encoding = nn.Linear(3, d_model)
        self.id_embedding = nn.Embedding(300, d_model)

        self.node_encoding = nn.Linear(d_model * 3, d_model)

        self.conv_gcn = torch_geometric.nn.GCNConv
        self.global_pool = torch_geometric.nn.global_max_pool

        self.layer_stack = nn.ModuleList()
        for _ in range(self.n_layer):
            # self.layer_stack.append(self.conv_gcn(d_model, d_model))
            self.layer_stack.append(MultiHeadAttention(d_model=d_model))
            self.layer_stack.append(PositionwiseFeedForward(d_model=d_model, d_inner=d_model * 4))

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
        positional_feature = self.positional_encoding(data)

        node_feature = torch.cat([position_feature, id_feature, positional_feature], dim=1)
        node_feature = F.relu(self.node_encoding(node_feature))

        n_embed_t = node_feature
        # g_embed = self.global_pool(n_embed_t, data.batch)

        for layer_idx in range(0, len(self.layer_stack), 2):
            # n_embed_t = F.relu(self.layer_stack[layer_idx](n_embed_t, edge_index))
            n_embed_t, attn = self.layer_stack[layer_idx](n_embed_t, data.batch)
            n_embed_t = self.layer_stack[layer_idx + 1](n_embed_t)

            # g_embed_t = self.global_pool(n_embed_t, data.batch)
            #
            # g_embed = torch.cat((g_embed, g_embed_t), dim=1)

        # latent = self.aggregate(g_embed)

        # return latent, attn
        return n_embed_t, attn
class GenerativeModel(nn.Module):
    def __init__(self, n_layer, d_model):
        super(GenerativeModel, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model
        self.grid_size = 7

        self.local_encoder = LocalEncoder(d_model, self.grid_size)
        self.graph_encoder = GraphEncoder(n_layer, d_model)

        self.conv = Conv3DBNReLU(d_model * 2, d_model, kernel_size=1, padding=0)

        self.pos_conv = Conv3DBNReLU(d_model, 1, kernel_size=1, padding=0)

        self.id_fc = nn.Linear(d_model, 300)
        # self._init_params()

    def forward(self, data):
        batch_size = data.local_grid.shape[0] // self.grid_size
        enc_local = self.local_encoder(data.local_grid)
        enc_graph, attn = self.graph_encoder(data)

        enc_graph = enc_graph.view(batch_size, 1, 1, 1, self.d_model).expand(-1, self.grid_size, self.grid_size, self.grid_size, -1)

        enc_output = torch.cat((enc_local, enc_graph), dim=-1)
        enc_output = enc_output.permute(0, 4, 1, 2, 3).contiguous()
        enc_output = self.conv(enc_output)

        pos_output = self.pos_conv(enc_output).squeeze()

        torch.autograd.set_detect_anomaly(True)
        if self.training:
            id_idx = data.gt_grid.reshape(batch_size, -1)
            id_idx = torch.argmax(id_idx, dim=-1)
        else:
            id_idx = pos_output.reshape(batch_size, -1)
            id_idx = torch.argmax(id_idx, dim=-1)

        pos_output = pos_output.reshape(batch_size, -1)
        pos_output = torch.softmax(pos_output, dim=-1)

        batch_indices = torch.arange(0, batch_size, device=enc_output.device)
        id_output = self.id_fc(enc_output.reshape(batch_size, self.d_model, -1)[batch_indices, :, id_idx])
        id_output = torch.softmax(id_output, dim=-1)

        if self.training:
            return pos_output, id_output
        else:
            return pos_output, id_output, attn
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is None:
                    # Normal Conv3d layers
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                else:
                    # Last layers of coords and types predictions
                    nn.init.normal_(m.weight, mean=0, std=0.001)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)