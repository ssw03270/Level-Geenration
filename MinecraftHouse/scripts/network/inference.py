import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from datetime import datetime

import plotly.graph_objects as go
import numpy as np
import random
from tqdm import tqdm

from model import Transformer
from dataloader import CraftAssistDataset
from train_single_gpu import get_accuracy, cross_entropy_loss

dir_dictionary = {
    0: [-1, -1, -1],
    1: [-1, -1, 0],
    2: [-1, -1, 1],
    3: [-1, 0, -1],
    4: [-1, 0, 0],
    5: [-1, 0, 1],
    6: [-1, 1, -1],
    7: [-1, 1, 0],
    8: [-1, 1, 1],
    9: [0, -1, -1],
    10: [0, -1, 0],
    11: [0, -1, 1],
    12: [0, 0, -1],
    13: [0, 0, 1],
    14: [0, 1, -1],
    15: [0, 1, 0],
    16: [0, 1, 1],
    17: [1, -1, -1],
    18: [1, -1, 0],
    19: [1, -1, 1],
    20: [1, 0, -1],
    21: [1, 0, 0],
    22: [1, 0, 1],
    23: [1, 1, -1],
    24: [1, 1, 0],
    25: [1, 1, 1]
}
dir_list = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]]

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

def create_mesh(vertices_list, category, color):
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

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=1, name=category, showlegend=True)
    return mesh

def rendering(position_sequence, semantic_sequence):
    position_sequence = position_sequence.squeeze().cpu().detach().numpy()
    position_sequence = train_dataset.restore_min_max_scaling(position_sequence)
    semantic_sequence = semantic_sequence.squeeze().cpu().detach().numpy()

    categorys = []
    for semantic in semantic_sequence:
        category = ''
        if semantic == 0:
            category = 'sos'
        elif semantic == 1:
            category = 'eos'
        elif semantic == '2':
            categorys = 'pad'
        else:
            category = train_dataset.sorted_block_semantic_values[semantic - 3]

        categorys.append(category)
        print(category)

    category_colors = {
        category: f'rgb({(hash(category) & 0xFF)}, {(hash(category) >> 8) & 0xFF}, {(hash(category) >> 16) & 0xFF})'
        for category in categorys
    }

    category_mesh_data = {category: {'coords': []} for category in category_colors}
    fig = go.Figure()
    for category, coord in zip(categorys, position_sequence):
        category_mesh_data[category]['coords'].append([coord[0], coord[2], coord[1]])

    for category, coords in category_mesh_data.items():
        coord = coords['coords']
        if len(coord) == 0:
            continue

        color = category_colors[category]
        mesh = create_mesh(coord, category, color)

        fig.add_trace(mesh)

    # Update the layout
    fig.update_layout(
        title='3D Data Visualization with Categories',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        showlegend=True
    )

    fig.show()


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--d_model", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--d_hidden", type=int, default=2048, help="Batch size for training.")
    parser.add_argument("--n_head", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--n_layer", type=int, default=6, help="Batch size for training.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--save_dir_path", type=str, default="transformer", help="save dir path")

    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    # Set the device for training (either GPU or CPU based on availability)
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Only the first dataset initialization will load the full dataset from disk
    train_dataset = CraftAssistDataset(data_type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    # Initialize the Transformer model
    transformer = Transformer(opt.d_model, opt.d_hidden, opt.n_head, opt.n_layer, opt.dropout).to(device=device)
    checkpoint = torch.load("./models/" + opt.save_dir_path + "/transformer_epoch_" + str(opt.checkpoint_epoch) + ".pth")
    transformer.load_state_dict(checkpoint['model_state_dict'])

    transformer.eval()

    with torch.no_grad():
        # Iterate over batches
        for idx, data in enumerate(tqdm(train_dataloader)):
            # Get the source and target sequences from the batch
            position_sequence, block_id_sequence, block_semantic_sequence, \
                dir_sequence, parent_sequence, pad_mask_sequence, terrain_mask_sequence = data

            position_sequence = position_sequence.to(device=device)
            block_id_sequence = block_id_sequence.to(device=device)
            block_semantic_sequence = block_semantic_sequence.to(device=device)
            dir_sequence = dir_sequence.to(device=device)
            parent_sequence = parent_sequence.to(device=device)
            pad_mask_sequence = pad_mask_sequence.to(device=device)
            terrain_mask_sequence = terrain_mask_sequence.to(device=device)

            real_position_sequence = []
            real_block_id_sequence = []
            real_block_semantic_sequence = []
            real_dir_sequence = []
            real_parent_sequence = []
            real_pad_mask_sequence = []
            real_terrain_mask_sequence = []

            for idx in range(len(position_sequence[0])):
                if terrain_mask_sequence[0, idx].cpu().detach().numpy():
                    break

                real_position_sequence.append(position_sequence[0, idx].cpu().detach().numpy().tolist())
                real_block_id_sequence.append(block_id_sequence[0, idx].cpu().detach().numpy().tolist())
                real_block_semantic_sequence.append(block_semantic_sequence[0, idx].cpu().detach().numpy().tolist())
                real_dir_sequence.append(dir_sequence[0, idx].cpu().detach().numpy().tolist())
                real_parent_sequence.append(parent_sequence[0, idx].cpu().detach().numpy().tolist())
                real_pad_mask_sequence.append(pad_mask_sequence[0, idx].cpu().detach().numpy().tolist())
                real_terrain_mask_sequence.append(terrain_mask_sequence[0, idx].cpu().detach().numpy().tolist())

            real_position_sequence = torch.tensor(real_position_sequence, dtype=torch.float32).to(device).unsqueeze(0)
            real_block_id_sequence = torch.tensor(real_block_id_sequence, dtype=torch.long).to(device).unsqueeze(0)
            real_block_semantic_sequence = torch.tensor(real_block_semantic_sequence, dtype=torch.long).to(device).unsqueeze(0)
            real_dir_sequence = torch.tensor(real_dir_sequence, dtype=torch.long).to(device).unsqueeze(0)
            real_parent_sequence = torch.tensor(real_parent_sequence, dtype=torch.long).to(device).unsqueeze(0)
            real_pad_mask_sequence = torch.tensor(real_pad_mask_sequence, dtype=torch.bool).to(device).unsqueeze(0)
            real_terrain_mask_sequence = torch.tensor(real_terrain_mask_sequence, dtype=torch.bool).to(device).unsqueeze(0)

            print(real_position_sequence.shape, position_sequence.shape)
            print(real_block_id_sequence.shape, block_id_sequence.shape)
            print(real_block_semantic_sequence.shape, block_semantic_sequence.shape)
            print(real_dir_sequence.shape, dir_sequence.shape)
            print(real_parent_sequence.shape, parent_sequence.shape)
            print(real_pad_mask_sequence.shape, pad_mask_sequence.shape)
            print(real_terrain_mask_sequence.shape, terrain_mask_sequence.shape)
            print(real_position_sequence)

            jdx = 0
            while True:
                # Get the model's predictions
                parent_output, dir_output, id_output, category_output = transformer(real_position_sequence,
                                                                                    real_block_id_sequence,
                                                                                    real_block_semantic_sequence,
                                                                                    real_pad_mask_sequence)

                dir_list = torch.tensor(dir_list).float().to(device)
                parent_output = parent_output.unsqueeze(0)
                cur_parent_idx = torch.argmax(parent_output[:, -1], dim=-1).unsqueeze(0)
                cur_dir = torch.argmax(dir_output[:, -1], dim=-1).unsqueeze(0)
                # print(parent_output[:, -1].cpu().detach().numpy().tolist(), cur_parent_idx, parent_output.shape)

                real_dir_sequence = torch.cat((real_dir_sequence, cur_dir), dim=1)

                new_dir = np.array(dir_dictionary[cur_dir.cpu().detach().numpy()[0, 0]])
                new_position = real_position_sequence[0, cur_parent_idx.cpu().detach().numpy()[0, 0]].cpu().detach().numpy()
                new_position = train_dataset.restore_min_max_scaling(new_position) + new_dir
                new_position = train_dataset.min_max_scaling(new_position)
                new_position = torch.tensor(new_position).float()
                new_position = new_position.to(device).unsqueeze(0).unsqueeze(0)
                real_position_sequence = torch.cat((real_position_sequence, new_position), dim=1)

                cur_block_id = torch.argmax(id_output[:, -1], dim=-1).unsqueeze(0)
                real_block_id_sequence = torch.cat((real_block_id_sequence, cur_block_id), dim=1)

                cur_block_semantic = torch.argmax(category_output[:, -1], dim=-1).unsqueeze(0)
                real_block_semantic_sequence = torch.cat((real_block_semantic_sequence, cur_block_semantic), dim=1)

                real_pad_mask_sequence = torch.cat((real_pad_mask_sequence, torch.tensor([[1]]).bool().to(device)), dim=1)
                real_terrain_mask_sequence = torch.cat((real_terrain_mask_sequence, torch.tensor([[1]]).bool().to(device)), dim=1)

                print(jdx, cur_block_semantic.cpu().detach().numpy()[0])
                jdx += 1
                if cur_block_semantic.cpu().detach().numpy()[0] == 1 or jdx > 1500:
                    break

            rendering(real_position_sequence, real_block_semantic_sequence)
            break
            # Compute the losses
            # mask = real_pad_mask_sequence & real_terrain_mask_sequence
            # loss_parent = cross_entropy_loss(real_parent_sequence, parent_sequence.detach(), mask.detach())
            # loss_dir = cross_entropy_loss(real_dir_sequence, dir_sequence.detach(), mask.detach())
            # loss_id = cross_entropy_loss(real_block_id_sequence[:, :-1], block_id_sequence[:, 1:].detach(), mask[:, 1:].detach())
            # loss_category = cross_entropy_loss(real_block_semantic_sequence[:, :-1], block_semantic_sequence[:, 1:].detach(), mask[:, 1:].detach())
            #
            # true_parent_sum, problem_parent_sum = get_accuracy(real_parent_sequence.detach(), parent_sequence.detach(),
            #                                                    mask.detach())
            # true_dir_sum, problem_dir_sum = get_accuracy(real_dir_sequence.detach(), dir_sequence.detach(), mask.detach())
            # true_id_sum, problem_id_sum = get_accuracy(real_block_id_sequence[:, :-1].detach(),
            #                                            block_id_sequence[:, 1:].detach(), mask[:, 1:].detach())
            # true_category_sum, problem_category_sum = get_accuracy(real_block_semantic_sequence[:, :-1].detach(),
            #                                                        block_semantic_sequence[:, 1:].detach(),
            #                                                        mask[:, 1:].detach())
            #
            # print(f"Epoch {idx + 1} - Inference Loss CE parent: {loss_parent:.4f}")
            # print(f"Epoch {idx + 1} - Inference Loss CE dir: {loss_dir:.4f}")
            # print(f"Epoch {idx + 1} - Inference Loss CE id: {loss_id:.4f}")
            # print(f"Epoch {idx + 1} - Inference Loss CE category: {loss_category:.4f}")
            #
            # print(f"Epoch {idx + 1} - Inference accuracy parent: {true_parent_sum / problem_parent_sum:.4f}")
            # print(f"Epoch {idx + 1} - Inference accuracy dir: {true_dir_sum / problem_dir_sum:.4f}")
            # print(f"Epoch {idx + 1} - Inference accuracy id: {true_id_sum / problem_id_sum:.4f}")
            # print(f"Epoch {idx + 1} - Inference accuracy category: {true_category_sum / problem_category_sum:.4f}")