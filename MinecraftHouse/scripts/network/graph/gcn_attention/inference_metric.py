import argparse
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from model import GenerativeModel
from dataloader import GraphDataset

from tqdm import tqdm

def get_accuracy(pred, trg):
    # 가장 높은 확률을 갖는 클래스 인덱스를 얻음
    pred = torch.argmax(pred, dim=-1)
    # 정확도 계산
    correct = (trg == pred).sum().item()
    total = len(pred)

    return correct, total

if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a graph model with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--d_model", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--n_layer", type=int, default=3, help="Batch size for training.")
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
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True,
                                 num_workers=8, pin_memory=True)

    # Initialize the Transformer model
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generative_model = GenerativeModel(opt.n_layer, opt.d_model).to(device)

    checkpoint = torch.load(
        "./models/" + opt.save_dir_path + "/graph_epoch_" + str(opt.checkpoint_epoch) + ".pth")
    generative_model.load_state_dict(checkpoint['model_state_dict'])

    generative_model.eval()

    with torch.no_grad():

        true_pos_sums = torch.Tensor([0.0]).to(device)
        true_id_sums = torch.Tensor([0.0]).to(device)

        problem_pos_sums = torch.Tensor([0.0]).to(device)
        problem_id_sums = torch.Tensor([0.0]).to(device)

        # Iterate over batches
        for data in tqdm(test_dataloader):
            data = data.to(device=device)

            position_output, id_output, attn = generative_model(data)

            gt_grid = data['gt_grid'].reshape(1, -1)
            gt_grid = torch.argmax(gt_grid, dim=-1)

            true_id_sum, problem_id_sum = get_accuracy(id_output.detach(), data['gt_id'].detach())
            true_id_sums += true_id_sum
            problem_id_sums += problem_id_sum

            true_pos_sum, problem_pos_sum = get_accuracy(position_output.detach(),
                                                         gt_grid.detach())
            true_pos_sums += true_pos_sum
            problem_pos_sums += problem_pos_sum

        true_id_mean = true_id_sums.item() / (problem_id_sums.item())
        true_pos_mean = true_pos_sums.item() / (problem_pos_sums.item())

        print(f"Test accuracy id: {true_id_mean:.4f}")
        print(f"Test accuracy position: {true_pos_mean:.4f}")