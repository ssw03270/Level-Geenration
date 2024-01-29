import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from datetime import datetime
from transformers import BertTokenizer, BertModel

import plotly.graph_objects as go
import numpy as np
import random
from tqdm import tqdm

from model import Transformer
from dataloader import CraftAssistDataset
from train_single_gpu import get_accuracy, cross_entropy_loss

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


def rendering(direction_sequence, semantic_sequence):
    print(direction_sequence)
    direction_sequence = direction_sequence.squeeze().cpu().detach().numpy()
    direction_sequence = train_dataset.restore_min_max_scaling(direction_sequence)
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
            category = train_dataset.sorted_category_values[semantic - 3]

        categorys.append(category)
        print(category)

    category_colors = {
        category: f'rgb({(hash(category) & 0xFF)}, {(hash(category) >> 8) & 0xFF}, {(hash(category) >> 16) & 0xFF})'
        for category in categorys
    }

    category_mesh_data = {category: {'coords': []} for category in category_colors}
    fig = go.Figure()
    cur_coord = [0, 0, 0]
    for category, dir in zip(categorys, direction_sequence):
        cur_coord = [cur_coord[0] + dir[0], cur_coord[1] + dir[1], cur_coord[2] + dir[2]]
        category_mesh_data[category]['coords'].append([cur_coord[0], cur_coord[2], cur_coord[1]])

    for category, coords in category_mesh_data.items():
        coord = coords['coords']
        if len(coord) == 0:
            continue

        color = category_colors[category]
        mesh = create_mesh(coord, category, color)

        fig.add_trace(mesh)

    # Update the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-2, 30]),
            yaxis=dict(range=[-2, 30]),
            zaxis=dict(range=[-2, 30])
        )
    )

    fig.show()


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--d_model", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--d_hidden", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--n_head", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--n_layer", type=int, default=3, help="Batch size for training.")
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Initialize the Transformer model
    transformer = Transformer(opt.d_model, opt.d_hidden, opt.n_head, opt.n_layer, opt.dropout).to(device=device)
    checkpoint = torch.load(
        "./models/" + opt.save_dir_path + "/transformer_epoch_" + str(opt.checkpoint_epoch) + ".pth")
    transformer.load_state_dict(checkpoint['model_state_dict'])

    transformer.eval()

    with torch.no_grad():
        # Iterate over batches
        for idx, data in enumerate(tqdm(train_dataloader)):
            # Get the source and target sequences from the batch
            direction_sequence, id_sequence, category_sequence, \
                next_direction_sequence, next_category_sequence, next_id_sequence, \
                pad_mask_sequence, text_sequence = data

            text_sequence = tokenizer(text_sequence, padding=True, truncation=True, return_tensors="pt")
            text_sequence = text_sequence.to(device=device)

            direction_sequence = direction_sequence.to(device=device)
            id_sequence = id_sequence.to(device=device)
            category_sequence = category_sequence.to(device=device)

            next_category_sequence = next_category_sequence.to(device=device)
            next_id_sequence = next_id_sequence.to(device=device)
            next_direction_sequence = next_direction_sequence.to(device=device)

            pad_mask_sequence = pad_mask_sequence.to(device=device)

            real_direction_sequence = []
            real_id_sequence = []
            real_category_sequence = []
            real_pad_mask_sequence = []

            for idx in range(len(direction_sequence[0])):
                if idx > 50:
                    break

                real_direction_sequence.append(direction_sequence[0, idx].cpu().detach().numpy().tolist())
                real_id_sequence.append(id_sequence[0, idx].cpu().detach().numpy().tolist())
                real_category_sequence.append(category_sequence[0, idx].cpu().detach().numpy().tolist())
                real_pad_mask_sequence.append(pad_mask_sequence[0, idx].cpu().detach().numpy().tolist())

            real_direction_sequence = torch.tensor(real_direction_sequence, dtype=torch.float32).to(device).unsqueeze(0)
            real_id_sequence = torch.tensor(real_id_sequence, dtype=torch.long).to(device).unsqueeze(0)
            real_category_sequence = torch.tensor(real_category_sequence, dtype=torch.long).to(
                device).unsqueeze(0)
            real_pad_mask_sequence = torch.tensor(real_pad_mask_sequence, dtype=torch.bool).to(device).unsqueeze(0)

            print(real_direction_sequence.shape, direction_sequence.shape)
            print(real_id_sequence.shape, id_sequence.shape)
            print(real_category_sequence.shape, category_sequence.shape)
            print(real_pad_mask_sequence.shape, pad_mask_sequence.shape)

            jdx = 0
            while True:
                category_output, id_output, direction_output = transformer(text_sequence,
                                                                           real_direction_sequence,
                                                                           real_id_sequence,
                                                                           real_category_sequence,
                                                                           real_pad_mask_sequence)

                cur_dir = torch.round(direction_output[:, -1]).unsqueeze(0)
                print(cur_dir, direction_output.shape)
                real_direction_sequence = torch.cat((real_direction_sequence, cur_dir), dim=1)

                cur_id = torch.argmax(id_output[:, -1], dim=-1).unsqueeze(0)
                real_id_sequence = torch.cat((real_id_sequence, cur_id), dim=1)

                cur_category = torch.argmax(category_output[:, -1], dim=-1).unsqueeze(0)
                real_category_sequence = torch.cat((real_category_sequence, cur_category), dim=1)

                real_pad_mask_sequence = torch.cat((real_pad_mask_sequence, torch.tensor([[1]]).bool().to(device)),
                                                   dim=1)

                print(jdx, cur_category.cpu().detach().numpy()[0])
                jdx += 1
                if cur_category.cpu().detach().numpy()[0] == 1 or jdx > 1500:
                    break

            rendering(real_direction_sequence, real_category_sequence)

            # Compute the losses
            # mask = real_pad_mask_sequence & real_terrain_mask_sequence
            # loss_parent = cross_entropy_loss(real_parent_sequence, parent_sequence.detach(), mask.detach())
            # loss_dir = cross_entropy_loss(real_direction_sequence, direction_sequence.detach(), mask.detach())
            # loss_id = cross_entropy_loss(real_id_sequence[:, :-1], id_sequence[:, 1:].detach(), mask[:, 1:].detach())
            # loss_category = cross_entropy_loss(real_category_sequence[:, :-1], category_sequence[:, 1:].detach(), mask[:, 1:].detach())
            #
            # true_parent_sum, problem_parent_sum = get_accuracy(real_parent_sequence.detach(), parent_sequence.detach(),
            #                                                    mask.detach())
            # true_direction_sum, problem_direction_sum = get_accuracy(real_direction_sequence.detach(), direction_sequence.detach(), mask.detach())
            # true_id_sum, problem_id_sum = get_accuracy(real_id_sequence[:, :-1].detach(),
            #                                            id_sequence[:, 1:].detach(), mask[:, 1:].detach())
            # true_category_sum, problem_category_sum = get_accuracy(real_category_sequence[:, :-1].detach(),
            #                                                        category_sequence[:, 1:].detach(),
            #                                                        mask[:, 1:].detach())
            #
            # print(f"Epoch {idx + 1} - Inference Loss CE parent: {loss_parent:.4f}")
            # print(f"Epoch {idx + 1} - Inference Loss CE dir: {loss_dir:.4f}")
            # print(f"Epoch {idx + 1} - Inference Loss CE id: {loss_id:.4f}")
            # print(f"Epoch {idx + 1} - Inference Loss CE category: {loss_category:.4f}")
            #
            # print(f"Epoch {idx + 1} - Inference accuracy parent: {true_parent_sum / problem_parent_sum:.4f}")
            # print(f"Epoch {idx + 1} - Inference accuracy dir: {true_direction_sum / problem_direction_sum:.4f}")
            # print(f"Epoch {idx + 1} - Inference accuracy id: {true_id_sum / problem_id_sum:.4f}")
            # print(f"Epoch {idx + 1} - Inference accuracy category: {true_category_sum / problem_category_sum:.4f}")
