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
    def __init__(self, d_model):
        super(LocalEncoder, self).__init__()

        self.d_model = d_model

        self.id_embedding = nn.Embedding(256, d_model)

        self.layer1 = Conv3DBNReLU(d_model, d_model, kernel_size=3)
        self.layer2 = Conv3DBNReLU(d_model, d_model, kernel_size=3)
        self.layer3 = Conv3DBNReLU(d_model, d_model, kernel_size=3)
        self.layer4 = Conv3DBNReLU(d_model, d_model, kernel_size=3)

    def forward(self, x):
        print(x.shape)
        x = self.id_embedding(x)
        print(x.shape)
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
        self.id_embedding = nn.Embedding(256, d_model)

        self.node_encoding = nn.Linear(d_model * 2, d_model)

        self.conv_gcn = torch_geometric.nn.GCNConv
        self.global_pool = torch_geometric.nn.global_max_pool

        self.conv1 = self.conv_gcn(d_model, d_model)
        self.layer_stack = nn.ModuleList([
            self.conv_gcn(d_model, d_model)
            for _ in range(self.n_layer - 1)
        ])
        self.aggregate = nn.Linear(int(d_model * (1.0 + self.n_layer)), d_model)

    def forward(self, data):
        edge_index = data.edge_index

        position_feature = data.position_feature
        id_feature = data.id_feature

        position_feature = torch.relu(self.position_encoding(position_feature))
        id_feature = torch.relu(self.id_embedding(id_feature))

        node_feature = F.relu(self.node_encoding(torch.cat([position_feature, id_feature], dim=1)))

        n_embed_0 = node_feature
        g_embed_0 = self.global_pool(n_embed_0, data.batch)

        n_embed_t = F.relu(self.conv1(n_embed_0, edge_index))
        g_embed_t = self.global_pool(n_embed_t, data.batch)

        g_embed = torch.cat((g_embed_0, g_embed_t), dim=1)

        for e_conv_t in self.layer_stack:
            n_embed_t = F.relu(e_conv_t(n_embed_t, edge_index))
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

        self.local_encoder = LocalEncoder(d_model)
        self.graph_encoder = GraphEncoder(n_layer, d_model)

        self.conv = Conv3DBNReLU(d_model * 2, d_model, kernel_size=1, padding=0)

        self.pos_conv = Conv3DBNReLU(d_model, 1, kernel_size=1, padding=0)

        self.id_fc = nn.Linear(d_model, 256)

    def forward(self, data):
        enc_local = self.local_encoder(data.local_grid)
        enc_graph = self.graph_encoder(data)

        batch_size = enc_graph.shape[0]
        enc_graph = enc_graph.view(batch_size, 1, 1, 1, self.d_model).expand(-1, self.grid_size, self.grid_size, self.grid_size, -1)

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