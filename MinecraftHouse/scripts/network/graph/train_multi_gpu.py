import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from datetime import datetime
import torch.nn as nn

import numpy as np
import random
from tqdm import tqdm

from model import GenerativeModel
from dataloader import GraphDataset

import wandb


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


class Trainer:
    def __init__(self, d_model, n_layer, batch_size, max_epoch,
                 use_checkpoint, checkpoint_epoch, use_wandb,
                 val_epoch, save_epoch, lr, save_dir_path, local_rank):
        """
        Initialize the trainer with the specified parameters.

        Args:
        - batch_size (int): Size of each training batch.
        - max_epoch (int): Maximum number of training epochs.
        - pad_idx (int): Padding index for sequences.
        - d_model (int): Dimension of the model.
        - n_layer (int): Number of model layers.
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
        self.lr = lr * torch.distributed.get_world_size()
        self.local_rank = local_rank

        # Set the device for training
        self.device = torch.device(f'cuda:{self.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

        # Dataset and Dataloader
        self.train_dataset = GraphDataset(data_type='train')
        self.train_sampler = DistributedSampler(dataset=self.train_dataset,
                                                num_replicas=torch.distributed.get_world_size(), rank=self.local_rank,
                                                shuffle=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler,
                                           num_workers=8, pin_memory=True)

        self.val_dataset = GraphDataset(data_type='val')
        self.val_sampler = DistributedSampler(dataset=self.val_dataset,
                                              num_replicas=torch.distributed.get_world_size(), rank=self.local_rank,
                                              shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=self.val_sampler,
                                         num_workers=8, pin_memory=True)

        # Initialize the Transformer model
        self.generative_model = GenerativeModel(n_layer, d_model).to(self.device)
        self.generative_model = nn.parallel.DistributedDataParallel(self.generative_model, device_ids=[self.local_rank])
        self.optimizer = torch.optim.Adam(self.generative_model.module.parameters(),
                                          lr=self.lr, betas=(0.9, 0.98))

    def train(self):
        epoch_start = 0

        for epoch in range(epoch_start, self.max_epoch):
            loss_pos_sum = torch.Tensor([0.0]).to(self.device)
            loss_id_sum = torch.Tensor([0.0]).to(self.device)

            true_pos_sums = torch.Tensor([0.0]).to(self.device)
            true_id_sums = torch.Tensor([0.0]).to(self.device)

            problem_pos_sums = torch.Tensor([0.0]).to(self.device)
            problem_id_sums = torch.Tensor([0.0]).to(self.device)

            # Iterate over batches
            for data in tqdm(self.train_dataloader):
                # Zero the gradients
                self.optimizer.zero_grad()

                data = data.to(device=self.device)
                position_output, id_output = self.generative_model(data)

                batch_size = data['gt_grid'].shape[0] // 7
                gt_grid = data['gt_grid'].reshape(batch_size, -1)
                gt_grid = torch.argmax(gt_grid, dim=-1)

                # Compute the losses
                loss_id = cross_entropy_loss(id_output, data['gt_id'].detach())
                loss_pos = cross_entropy_loss(position_output, gt_grid.detach())
                loss = loss_pos + loss_id

                # Backpropagation and optimization step
                loss.backward()
                self.optimizer.step()

                dist.all_reduce(loss_pos, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_id, op=dist.ReduceOp.SUM)

                loss_pos_sum += loss_pos.detach()
                loss_id_sum += loss_id.detach()

                true_id_sum, problem_id_sum = get_accuracy(id_output.detach(), data['gt_id'].detach())
                dist.all_reduce(torch.tensor(true_id_sum).to(self.device), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(problem_id_sum).to(self.device), op=dist.ReduceOp.SUM)
                true_id_sums += true_id_sum
                problem_id_sums += problem_id_sum

                true_pos_sum, problem_pos_sum = get_accuracy(position_output.detach(),
                                                             gt_grid.detach())
                dist.all_reduce(torch.tensor(true_pos_sum).to(self.device), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(problem_pos_sum).to(self.device), op=dist.ReduceOp.SUM)
                true_pos_sums += true_pos_sum
                problem_pos_sums += problem_pos_sum

            if self.local_rank == 0:
                loss_id_mean = loss_id_sum.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_pos_mean = loss_pos_sum.item() / (len(self.train_dataloader) * dist.get_world_size())

                true_id_mean = true_id_sums.item() / (problem_id_sums.item())
                true_pos_mean = true_pos_sums.item() / (problem_pos_sums.item())

                print(f"Epoch {epoch + 1}/{self.max_epoch} - Train Loss CE id: {loss_id_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Train Loss CE position: {loss_pos_mean:.4f}")

                print(f"Epoch {epoch + 1}/{self.max_epoch} - Train accuracy id: {true_id_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Train accuracy position: {true_pos_mean:.4f}")

                if self.use_wandb:
                    wandb.log({"Train ce id": loss_id_mean}, step=epoch + 1)
                    wandb.log({"Train ce position": loss_pos_mean}, step=epoch + 1)

                    wandb.log({"Train accuracy id": true_id_mean}, step=epoch + 1)
                    wandb.log({"Train accuracy position": true_pos_mean}, step=epoch + 1)

                if (epoch + 1) % self.val_epoch == 0:
                    self.generative_model.module.eval()

                    with torch.no_grad():
                        val_loss_pos_sum = torch.Tensor([0.0]).to(self.device)
                        val_loss_id_sum = torch.Tensor([0.0]).to(self.device)

                        val_true_pos_sums = torch.Tensor([0.0]).to(self.device)
                        val_true_id_sums = torch.Tensor([0.0]).to(self.device)

                        val_problem_pos_sums = torch.Tensor([0.0]).to(self.device)
                        val_problem_id_sums = torch.Tensor([0.0]).to(self.device)
                        print(1)
                        # Iterate over batches
                        for data in tqdm(self.val_dataloader):
                            data = data.to(device=self.device)
                            print(2)
                            position_output, id_output = self.generative_model(data)
                            print(3)

                            batch_size = data['gt_grid'].shape[0] // 7
                            gt_grid = data['gt_grid'].reshape(batch_size, -1)
                            gt_grid = torch.argmax(gt_grid, dim=-1)

                            print(4)
                            # Compute the losses
                            val_loss_id = cross_entropy_loss(id_output.detach(), data['gt_id'].detach())
                            val_loss_pos = cross_entropy_loss(position_output.detach(), gt_grid.detach())
                            print(5)

                            dist.all_reduce(val_loss_pos, op=dist.ReduceOp.SUM)
                            dist.all_reduce(val_loss_id, op=dist.ReduceOp.SUM)
                            print(6)

                            val_loss_pos_sum += val_loss_pos.detach()
                            val_loss_id_sum += val_loss_id.detach()

                            val_true_id_sum, val_problem_id_sum = get_accuracy(id_output.detach(),
                                                                               data['gt_id'].detach())
                            dist.all_reduce(torch.tensor(val_true_id_sum).to(self.device), op=dist.ReduceOp.SUM)
                            dist.all_reduce(torch.tensor(val_problem_id_sum).to(self.device), op=dist.ReduceOp.SUM)
                            val_true_id_sums += val_true_id_sum
                            val_problem_id_sums += val_problem_id_sum
                            print(7)

                            val_true_pos_sum, val_problem_pos_sum = get_accuracy(position_output.detach(),
                                                                                 gt_grid.detach())
                            dist.all_reduce(torch.tensor(val_true_pos_sum).to(self.device), op=dist.ReduceOp.SUM)
                            dist.all_reduce(torch.tensor(val_problem_pos_sum).to(self.device), op=dist.ReduceOp.SUM)
                            val_true_pos_sums += val_true_pos_sum
                            val_problem_pos_sums += val_problem_pos_sum
                            print(8)

                        val_loss_id_mean = val_loss_id_sum.item() / (len(self.val_dataloader) * dist.get_world_size())
                        val_loss_pos_mean = val_loss_pos_sum.item() / (len(self.val_dataloader) * dist.get_world_size())

                        val_true_id_mean = val_true_id_sums.item() / (val_problem_id_sums.item())
                        val_true_pos_mean = val_true_pos_sums.item() / (val_problem_pos_sums.item())

                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss CE id: {val_loss_id_mean:.4f}")
                        print(
                            f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss CE position: {val_loss_pos_mean:.4f}")

                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation accuracy id: {val_true_id_mean:.4f}")
                        print(
                            f"Epoch {epoch + 1}/{self.max_epoch} - Validation accuracy position: {val_true_pos_mean:.4f}")

                        if self.use_wandb:
                            wandb.log({"Validation ce id": val_loss_id_mean}, step=epoch + 1)
                            wandb.log({"Validation ce position": val_loss_pos_mean}, step=epoch + 1)

                            wandb.log({"Validation accuracy id": val_true_id_mean}, step=epoch + 1)
                            wandb.log({"Validation accuracy position": val_true_pos_mean}, step=epoch + 1)

                        self.generative_model.module.train()

                if (epoch + 1) % self.save_epoch == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.generative_model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }

                    save_path = os.path.join("models", self.save_dir_path)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(checkpoint, os.path.join(save_path, "graph_epoch_" + str(epoch + 1) + ".pth"))


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a graph model with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--d_model", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--n_layer", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
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

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt.save_dir_path = f"CraftAssist_{current_time}"

    if opt.local_rank == 0:
        wandb.login(key='0f272b4978c0b450c3765b24b8abd024d7799e80')
        wandb.init(project='level-generation', entity='city_team', config=vars(opt), name=opt.save_dir_path)

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

    rank = opt.local_rank
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        if torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') == "cuda:0":
            dist.init_process_group("gloo")

        else:
            dist.init_process_group('nccl')

    # Create a Trainer instance and start the training process
    trainer = Trainer(d_model=opt.d_model, n_layer=opt.n_layer,
                      batch_size=opt.batch_size, max_epoch=opt.max_epoch,
                      use_wandb=opt.use_wandb, use_checkpoint=opt.use_checkpoint,
                      checkpoint_epoch=opt.checkpoint_epoch, val_epoch=opt.val_epoch, save_epoch=opt.save_epoch,
                      lr=opt.lr, save_dir_path=opt.save_dir_path, local_rank=opt.local_rank)

    trainer.train()

    if opt.local_rank == 0:
        wandb.finish()
