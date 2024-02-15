import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

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

        self.node_encoding = nn.Linear(d_model * 2, d_model)

        self.conv_gcn = torch_geometric.nn.GCNConv
        self.global_pool = torch_geometric.nn.global_max_pool

        self.layer_stack = nn.ModuleList()
        for _ in range(self.n_layer):
            self.layer_stack.append(self.conv_gcn(d_model, d_model))
            self.layer_stack.append(self.conv_gcn(d_model, d_model))

        self.aggregate = nn.Linear(int(d_model * (1.0 + self.n_layer)), d_model)

    def forward(self, data):
        edge_index = data.edge_index
        temporal_edge_index = data.temporal_edge_index

        position_feature = data.position_feature
        id_feature = data.id_feature

        position_feature = F.relu(self.position_encoding(position_feature))
        id_feature = F.relu(self.id_embedding(id_feature)).squeeze(1)

        node_feature = torch.cat([position_feature, id_feature], dim=1)
        node_feature = F.relu(self.node_encoding(node_feature))

        n_embed_t = node_feature
        g_embed = self.global_pool(n_embed_t, data.batch)

        for layer_idx in range(0, len(self.layer_stack), 2):
            n_embed_t = F.relu(self.layer_stack[layer_idx](n_embed_t, edge_index))
            n_embed_t = F.relu(self.layer_stack[layer_idx + 1](n_embed_t, temporal_edge_index))

            g_embed_t = self.global_pool(n_embed_t, data.batch)

            g_embed = torch.cat((g_embed, g_embed_t), dim=1)

        latent = self.aggregate(g_embed)

        return latent

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

    def forward(self, data):
        batch_size = data.local_grid.shape[0] // self.grid_size
        enc_local = self.local_encoder(data.local_grid)
        enc_graph = self.graph_encoder(data)

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
            return pos_output, id_output