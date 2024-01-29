import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.distributed as dist
from datetime import datetime
import torch.nn as nn

import numpy as np
import random
from tqdm import tqdm

from model import GraphModel
from dataloader import GraphDataset

import wandb


def mse_loss(pred, trg):
    trg = trg.reshape(-1, 3)
    loss = F.mse_loss(pred, trg)
    return loss


def cross_entropy_loss(pred, trg):
    """
    Compute the binary cross-entropy loss between predictions and targets.

    Args:
    - pred (torch.Tensor): Model predictions.
    - trg (torch.Tensor): Ground truth labels.

    Returns:
    - torch.Tensor: Computed BCE loss.
    """
    loss = F.cross_entropy(pred, trg)

    return loss


def get_accuracy(pred, trg):
    # 가장 높은 확률을 갖는 클래스 인덱스를 얻음
    pred = torch.argmax(pred, dim=-1)
    # 정확도 계산
    correct = (trg == pred).sum().item()
    total = len(pred)

    return correct, total


def get_position_accuracy(pred, trg):
    # 예측값을 반올림
    pred = torch.round(pred * 32).int()
    trg = (trg.reshape(-1, 3) * 32).int()

    # (batch, seq, 3) 텐서에서 모든 요소가 일치하는지 확인
    correct_elements = (pred == trg).all(dim=-1)

    # 마스크 적용 및 일치하는 요소의 개수 계산
    correct = correct_elements.sum().item()

    # 마스크에 의해 선택된 요소의 총 개수
    total = len(pred)

    return correct, total

class Trainer:
    def __init__(self, d_model, n_layer, batch_size, max_epoch,
                 use_checkpoint, checkpoint_epoch, use_wandb,
                 val_epoch, save_epoch, lr, save_dir_path):
        """
        Initialize the trainer with the specified parameters.

        Args:
        - batch_size (int): Size of each training batch.
        - max_epoch (int): Maximum number of training epochs.
        - pad_idx (int): Padding index for sequences.
        - d_model (int): Dimension of the model.
        - n_layer (int): Number of transformer layers.
        - n_head (int): Number of multi-head attentions.
        """

        # Initialize trainer parameters
        self.d_model = d_model
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.use_checkpoint = use_checkpoint
        self.checkpoint_epoch = checkpoint_epoch
        self.use_wandb = use_wandb
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.save_dir_path = save_dir_path
        self.lr = lr

        # Set the device for training
        self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Dataset and Dataloader
        self.train_dataset = GraphDataset(data_type='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataset = GraphDataset(data_type='validation')
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the Transformer model
        self.graph_model = GraphModel(d_model, n_layer).to(self.device)
        self.optimizer = torch.optim.Adam(self.graph_model.parameters(),
                                          lr=self.lr,
                                          betas=(0.9, 0.98))

    def train(self):
        """Training loop for the transformer model."""
        epoch_start = 0

        for epoch in range(epoch_start, self.max_epoch):
            loss_pos_sum = torch.Tensor([0.0]).to(self.device)
            loss_id_sum = torch.Tensor([0.0]).to(self.device)
            loss_category_sum = torch.Tensor([0.0]).to(self.device)

            true_pos_sums = torch.Tensor([0.0]).to(self.device)
            true_id_sums = torch.Tensor([0.0]).to(self.device)
            true_category_sums = torch.Tensor([0.0]).to(self.device)

            problem_pos_sums = torch.Tensor([0.0]).to(self.device)
            problem_id_sums = torch.Tensor([0.0]).to(self.device)
            problem_category_sums = torch.Tensor([0.0]).to(self.device)

            # Iterate over batches
            for data in tqdm(self.train_dataloader):
                # Zero the gradients
                self.optimizer.zero_grad()

                data = data.to(device=self.device)
                position_output, id_output, category_output = self.graph_model(data)

                # Compute the losses
                loss_category = cross_entropy_loss(category_output, data['next_category'].detach())
                loss_id = cross_entropy_loss(id_output, data['next_id'].detach())
                loss_pos = mse_loss(position_output, data['next_position'].detach())
                loss = loss_pos + loss_id + loss_category

                # Backpropagation and optimization step
                loss.backward()
                self.optimizer.step()

                loss_pos_sum += loss_pos.detach()
                loss_id_sum += loss_id.detach()
                loss_category_sum += loss_category.detach()

                true_category_sum, problem_category_sum = get_accuracy(category_output.detach(), data['next_category'].detach())
                true_category_sums += true_category_sum
                problem_category_sums += problem_category_sum

                true_id_sum, problem_id_sum = get_accuracy(id_output.detach(), data['next_id'].detach())
                true_id_sums += true_id_sum
                problem_id_sums += problem_id_sum

                true_pos_sum, problem_pos_sum = get_position_accuracy(position_output.detach(), data['next_position'].detach())
                true_pos_sums += true_pos_sum
                problem_pos_sums += problem_pos_sum

            loss_category_mean = loss_category_sum.item() / (len(self.train_dataloader))
            loss_id_mean = loss_id_sum.item() / (len(self.train_dataloader))
            loss_pos_mean = loss_pos_sum.item() / (len(self.train_dataloader))

            true_category_mean = true_category_sums.item() / (problem_category_sums.item())
            true_id_mean = true_id_sums.item() / (problem_id_sums.item())
            true_pos_mean = true_pos_sums.item() / (problem_pos_sums.item())

            print(f"Epoch {epoch + 1}/{self.max_epoch} - Train Loss CE category: {loss_category_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Train Loss CE id: {loss_id_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Train Loss CE pos: {loss_pos_mean:.4f}")

            print(f"Epoch {epoch + 1}/{self.max_epoch} - Train accuracy category: {true_category_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Train accuracy id: {true_id_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Train accuracy pos: {true_pos_mean:.4f}")

            if self.use_wandb:
                wandb.log({"Train ce category": loss_category_mean}, step=epoch + 1)
                wandb.log({"Train ce id": loss_id_mean}, step=epoch + 1)
                wandb.log({"Train ce pos": loss_pos_mean}, step=epoch + 1)

                wandb.log({"Train accuracy category": true_category_mean}, step=epoch + 1)
                wandb.log({"Train accuracy id": true_id_mean}, step=epoch + 1)
                wandb.log({"Train accuracy pos": true_pos_mean}, step=epoch + 1)

            if (epoch + 1) % self.save_epoch == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.graph_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }

                save_path = os.path.join("models", self.save_dir_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(checkpoint, os.path.join(save_path, "transformer_epoch_" + str(epoch + 1) + ".pth"))


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--d_model", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--n_layer", type=int, default=3, help="Batch size for training.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=1000, help="Maximum number of epochs for training.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=50, help="Use checkpoint index.")
    parser.add_argument("--save_dir_path", type=str, default="graph", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")

    opt = parser.parse_args()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt.save_dir_path = f"CraftAssist_{current_time}"

    wandb.login(key='0f272b4978c0b450c3765b24b8abd024d7799e80')
    wandb.init(project='level-generation', entity='ssw03270', config=vars(opt), name=opt.save_dir_path)

    for key, value in wandb.config.items():
        setattr(opt, key, value)

    save_path = os.path.join("models", opt.save_dir_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config_file_path = os.path.join(save_path, "config.txt")
    with open(config_file_path, "w") as f:
        for arg in vars(opt):
            f.write(f"{arg}: {getattr(opt, arg)}\n")

    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # Create a Trainer instance and start the training process
    trainer = Trainer(d_model=opt.d_model, n_layer=opt.n_layer,
                      batch_size=opt.batch_size, max_epoch=opt.max_epoch,
                      use_wandb=opt.use_wandb, use_checkpoint=opt.use_checkpoint,
                      checkpoint_epoch=opt.checkpoint_epoch, val_epoch=opt.val_epoch, save_epoch=opt.save_epoch,
                      lr=opt.lr, save_dir_path=opt.save_dir_path)

    trainer.train()

    wandb.finish()
