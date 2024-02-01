import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing

class GraphModel(nn.Module):
    def __init__(self, n_layer, d_model):
        super(GraphModel, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model

        self.position_encoding = nn.Linear(3, d_model)
        self.id_embedding = nn.Embedding(256, d_model)
        self.category_embedding = nn.Embedding(34, d_model)
        self.idx_embedding = nn.Embedding(3060, d_model)

        self.node_encoding = nn.Linear(d_model * 4, d_model)

        self.conv_gcn = torch_geometric.nn.GCNConv
        self.conv_gin = lambda in_channels, out_channels: torch_geometric.nn.GINConv(
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
        )
        self.global_pool = torch_geometric.nn.global_max_pool

        self.t_conv1 = self.conv_gcn(d_model, d_model)
        self.s_conv1 = self.conv_gin(d_model, d_model)
        self.layer_stack = nn.ModuleList()
        for _ in range(self.n_layer - 1):
            t_conv = self.conv_gcn(d_model, d_model)
            s_conv = self.conv_gin(d_model, d_model)
            self.layer_stack.extend([t_conv, s_conv])

        self.aggregate = nn.Linear(int(d_model * (1.0 + self.n_layer)), d_model)

        self.position_decoding = nn.Linear(d_model, d_model)
        self.position_fc = nn.Linear(d_model, 3)

        self.id_decoding = nn.Linear(d_model, d_model)
        self.id_fc = nn.Linear(d_model, 256)

        self.category_decoding = nn.Linear(d_model, d_model)
        self.category_fc = nn.Linear(d_model, 34)

    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        temporal_edges = edge_index[:, edge_attr == 1]
        spatial_edges = edge_index[:, edge_attr == 0]

        position_feature = data.position_feature
        id_feature = data.id_feature
        category_feature = data.category_feature
        idx_feature = data.idx_feature

        position_feature = torch.relu(self.position_encoding(position_feature))
        id_feature = torch.relu(self.id_embedding(id_feature))
        category_feature = torch.relu(self.category_embedding(category_feature))
        idx_feature = torch.relu(self.idx_embedding(idx_feature))

        node_feature = F.relu(self.node_encoding(torch.cat([position_feature, id_feature,
                                                            category_feature, idx_feature], dim=1)))

        n_embed_0 = node_feature
        g_embed_0 = self.global_pool(n_embed_0, data.batch)

        n_embed_t = F.relu(self.t_conv1(n_embed_0, temporal_edges))
        n_embed_t = F.relu(self.s_conv1(n_embed_t, spatial_edges))
        g_embed_t = self.global_pool(n_embed_t, data.batch)

        g_embed = torch.cat((g_embed_0, g_embed_t), dim=1)

        for i in range(0, len(self.layer_stack), 2):
            n_embed_t = F.relu(self.layer_stack[i](n_embed_t, temporal_edges))
            n_embed_t = F.relu(self.layer_stack[i + 1](n_embed_t, spatial_edges))
            g_embed_t = self.global_pool(n_embed_t, data.batch)

            g_embed = torch.cat((g_embed, g_embed_t), dim=1)

        latent = self.aggregate(g_embed)

        position_output = torch.relu(self.position_decoding(latent))
        position_output = torch.sigmoid(self.position_fc(position_output))

        id_output = torch.relu(self.id_decoding(latent))
        id_output = torch.softmax(self.id_fc(id_output), dim=-1)

        category_output = torch.relu(self.category_decoding(latent))
        category_output = torch.softmax((self.category_fc(category_output)), dim=-1)

        return position_output, id_output, category_output