import argparse
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from model import GenerativeModel
from dataloader import GraphDataset

from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def create_cube(center, size=1):
    # 정육면체의 중심에서 꼭지점으로의 방향 벡터
    dirs = np.array([[1, 1, -1],
                     [1, -1, -1],
                     [-1, -1, -1],
                     [-1, 1, -1],
                     [1, 1, 1],
                     [1, -1, 1],
                     [-1, -1, 1],
                     [-1, 1, 1]])
    # 꼭지점 계산
    return center + size * 0.5 * dirs


def create_mesh(vertices_list, id, color):
    faces = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    quad_to_tri = lambda quad: [(quad[0], quad[1], quad[2]), (quad[2], quad[3], quad[0])]
    vertices = []
    i_faces = []
    for i, cube_center in enumerate(vertices_list):
        cube_vertices = create_cube(cube_center)
        vertices.append(cube_vertices)
        for quad in faces:
            i_faces.extend(quad_to_tri(quad + 8 * i))

    vertices = np.concatenate(vertices, axis=0)
    i_faces = np.array(i_faces)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = i_faces[:, 0], i_faces[:, 1], i_faces[:, 2]

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=1, name=str(id), showlegend=True)
    return mesh


def rendering(position_sequence, id_sequence):
    position_sequence = position_sequence.cpu().detach().numpy()
    id_sequence = id_sequence.squeeze(1).cpu().detach().numpy()

    block_ids = []
    for block_id in id_sequence:
        block_ids.append(block_id)

    block_id_colors = {
        block_id: f'rgb({(hash(block_id) & 0xFF)}, {(hash(block_id) >> 8) & 0xFF}, {(hash(block_id) >> 16) & 0xFF})'
        for block_id in block_ids
    }

    id_mesh_data = {id: {'coords': []} for id in block_id_colors}
    fig = go.Figure()
    for id, pos in zip(block_ids, position_sequence):
        cur_coord = [pos[0], pos[1], pos[2]]
        id_mesh_data[id]['coords'].append([cur_coord[0], cur_coord[2], cur_coord[1]])

    for id, coords in id_mesh_data.items():
        coord = coords['coords']
        if len(coord) == 0:
            continue

        color = block_id_colors[id]
        mesh = create_mesh(coord, id, color)

        fig.add_trace(mesh)

    # Update the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-30, 30]),
            yaxis=dict(range=[-30, 30]),
            zaxis=dict(range=[-30, 30])
        )
    )

    fig.show()

def rendering_attn(position_sequence, id_sequence, attn_map):
    position_sequence = position_sequence.cpu().detach().numpy()
    id_sequence = id_sequence.cpu().detach().numpy()

    colors_hex = ['#%02x%02x%02x' % (255 - int(score * 255), 255 - int(score * 255), 255 - int(score * 255)) for score
                  in attn_map]
    colors_hex.append('#FF0000')  # 빨간색 추가
    print(position_sequence.shape, id_sequence.shape, len(colors_hex))

    fig = go.Figure()
    for coord, id, color in zip(position_sequence, id_sequence, colors_hex):
        if len(coord) == 0:
            continue

        mesh = create_mesh([[coord[0], coord[2], coord[1]]], id, color)

        fig.add_trace(mesh)

    # Update the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-30, 30]),
            yaxis=dict(range=[-30, 30]),
            zaxis=dict(range=[-30, 30])
        )
    )

    fig.show()

if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a graph model with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--d_model", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--n_layer", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=100, help="Maximum number of epochs for training.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_dir_path", type=str, default="graph", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")
    parser.add_argument("--local-rank", type=int)

    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    test_dataset = GraphDataset(data_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=8, pin_memory=True)

    # Initialize the Transformer model
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generative_model = GenerativeModel(opt.n_layer, opt.d_model).to(device)

    checkpoint = torch.load(
        "./models/" + opt.save_dir_path + "/graph_epoch_" + str(opt.checkpoint_epoch) + ".pth")
    generative_model.load_state_dict(checkpoint['model_state_dict'])

    generative_model.eval()

    with torch.no_grad():
        # Iterate over batches
        iter = 5
        for data in tqdm(test_dataloader):
            data = data.to(device=device)
            rendering(data.position_feature, data.id_feature)

            for idx in range(100):
                position_output, id_output = generative_model(data)
                position_output = torch.argmax(position_output, dim=-1).cpu().detach().numpy()

                grid_size = 7
                z = position_output % grid_size
                x = position_output // (grid_size * grid_size)
                y = (position_output - z - x * (grid_size * grid_size)) // grid_size
                new_pos = [x - grid_size // 2, y - grid_size // 2, z - grid_size // 2]
                new_pos = np.array(new_pos)
                new_pos = np.reshape(new_pos, (-1))
                new_pos = torch.tensor([new_pos]).to(device=device)

                new_id = torch.argmax(id_output, dim=-1)
                new_id = new_id.unsqueeze(0)

                data.position_feature = torch.cat((data.position_feature, new_pos), dim=0)
                data.id_feature = torch.cat((data.id_feature, new_id), dim=0)

                data.position_feature -= new_pos

                differences = torch.diff(data.position_feature, dim=0)
                data.direction_feature = torch.cat((torch.zeros(1, 3, dtype=torch.float32).to(device=device), differences), dim=0)

                local_grid = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.long).to(device=device)
                center_offset = torch.tensor([grid_size // 2, grid_size // 2, grid_size // 2],
                                             dtype=torch.long).to(device=device)

                # 각 좌표를 voxel grid에 맞게 조정
                adjusted_coords = data.position_feature + center_offset

                # voxel grid에 할당 가능한 좌표만 필터링
                mask = (adjusted_coords >= 0) & (adjusted_coords < grid_size)
                mask = mask.all(dim=1)  # 모든 차원에 대해 조건이 참인지 확인

                # 할당 가능한 좌표와 해당 id만 추출
                valid_coords = adjusted_coords[mask].long()
                valid_ids = data.id_feature[mask]

                # voxel grid에 id 할당
                for coord, id in zip(valid_coords, valid_ids):
                    local_grid[coord[0], coord[1], coord[2]] = id

                data.local_grid = local_grid

                for jdx, pos in enumerate(data.position_feature[:-1]):
                    if torch.sum(torch.abs(pos)) == 1:
                        last_jdx = len(data.position_feature) - 1
                        new_edge = torch.tensor([[jdx, jdx, last_jdx], [jdx, last_jdx, jdx]], dtype=torch.long).to(device=device)
                        data.edge_index = torch.cat((data.edge_index, new_edge), dim=1)

                new_temporal_edge = torch.tensor([[len(data.position_feature - 1)], [len(data.position_feature)]], dtype=torch.long).to(device=device)
                data.temporal_edge_index = torch.cat((data.temporal_edge_index, new_temporal_edge), dim=1)

                data.each_num_nodes += 1
                data.num_nodes += 1

                data.batch = torch.cat((data.batch, torch.tensor([0], dtype=torch.long).to(device)), dim=0)

            # rendering_attn(data.position_feature, data.id_feature, attn)
            rendering(data.position_feature, data.id_feature)
            iter -= 1

            if iter <= 0:
                break